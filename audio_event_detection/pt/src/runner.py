"""Train/evaluate the original BC-ResNet with native PyTorch checkpoints."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .dataset import GSCDataset
from .models import get_custom_model


def validate_config(cfg):
    if cfg.operation_mode not in ("training", "evaluation"):
        raise ValueError("PyTorch AED supports training/evaluation; TFLite chains are not supported")
    if cfg.model.framework != "pytorch" or cfg.model.model_name != "custom":
        raise ValueError("Set model.framework=pytorch and model.model_name=custom")
    if cfg.model.pretrained:
        raise ValueError("No bundled pretrained weights; use model.model_path with a PyTorch state_dict")
    if cfg.training.get("resume_training"):
        raise ValueError("Exact training resume is not supported; use model.model_path for fine-tuning")
    if cfg.training.fine_tune and not cfg.model.model_path:
        raise ValueError("fine_tune requires a PyTorch model.model_path")
    if cfg.operation_mode == "evaluation" and not cfg.model.model_path:
        raise ValueError("Evaluation requires model.model_path")
    if cfg.training.num_workers != 0:
        raise ValueError("Use num_workers=0 to preserve the waveform augmentation RNG sequence")
    if not cfg.preprocessing.gsc_default or cfg.dataset.multi_label:
        raise ValueError("This adapter supports single-label GSC preprocessing only")
    if len(set(cfg.dataset.class_names)) != len(cfg.dataset.class_names):
        raise ValueError("class_names must be unique and ordered")
    f = cfg.feature_extraction
    if list(cfg.model.input_shape) != [1, f.n_mels, f.patch_length]:
        raise ValueError("model.input_shape must match [1, n_mels, patch_length]")
    if f.to_db or f.include_last_patch:
        raise ValueError("This GSC adapter uses natural log and the first complete patch")
    if cfg.training.epochs < 1 or cfg.training.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 <= cfg.training.warmup_epochs < cfg.training.epochs:
        raise ValueError("warmup_epochs must be nonnegative and less than epochs")


def load_weights(model, path, cfg):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        if checkpoint.get("class_names") != list(cfg.dataset.class_names):
            raise ValueError("Checkpoint class ordering differs from dataset.class_names")
        if checkpoint.get("tau") != cfg.model.tau:
            raise ValueError("Checkpoint tau differs from model.tau")
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)


def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            correct += (logits.argmax(1) == targets).sum().item()
            count += targets.numel()
    if not count:
        raise ValueError("Cannot evaluate an empty dataset")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}


def learning_rate(step, total_steps, warmup_steps, initial_lr):
    if step < warmup_steps:
        return initial_lr * step / warmup_steps
    return initial_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))


def run(cfg, output_dir):
    validate_config(cfg)
    seed = int(cfg.general.global_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(bool(cfg.general.deterministic_ops))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.general.device == "auto" else torch.device(cfg.general.device)
    model = get_custom_model(cfg.model.input_shape, len(cfg.dataset.class_names), cfg.model.tau)
    if cfg.model.model_path:
        load_weights(model, cfg.model.model_path, cfg)
    model.to(device)
    print(f"BC-ResNet-{cfg.model.tau}: {sum(p.numel() for p in model.parameters())} parameters on {device}")

    def loader(split, shuffle=False):
        return DataLoader(GSCDataset(cfg, split), batch_size=cfg.training.batch_size,
                          shuffle=shuffle, num_workers=0)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml")
    valid = loader("validation")
    history = []
    if cfg.operation_mode == "training":
        train = loader("training", shuffle=cfg.dataset.shuffle)
        opt_cfg = cfg.training.optimizer.SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=opt_cfg.learning_rate,
                                    momentum=opt_cfg.momentum, weight_decay=opt_cfg.weight_decay)
        total_steps = len(train) * cfg.training.epochs
        warmup_steps = len(train) * cfg.training.warmup_epochs
        saved = output / cfg.general.saved_models_dir
        saved.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        step = 0
        for epoch in range(cfg.training.epochs):
            model.train()
            loss_sum = correct = count = 0
            for inputs, targets in train:
                step += 1
                lr = learning_rate(step, total_steps, warmup_steps, opt_cfg.learning_rate)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * targets.numel()
                correct += (logits.argmax(1) == targets).sum().item()
                count += targets.numel()
            metrics = {"epoch": epoch + 1, "lr": lr, "train_loss": loss_sum / count,
                       "train_accuracy": correct / count, "validation": evaluate(model, valid, device)}
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
            checkpoint = {"state_dict": model.state_dict(), "class_names": list(cfg.dataset.class_names),
                          "tau": cfg.model.tau, "epoch": epoch + 1,
                          "input_shape": list(cfg.model.input_shape)}
            torch.save(checkpoint, saved / "last_model.pth")
            if metrics["validation"]["loss"] < best_loss:
                best_loss = metrics["validation"]["loss"]
                torch.save(checkpoint, saved / "best_model.pth")
            (output / "history.json").write_text(json.dumps(history, indent=2))
        load_weights(model, saved / "best_model.pth", cfg)
    results = {"validation": evaluate(model, valid, device)}
    if cfg.dataset.get("test_csv_path"):
        results["test"] = evaluate(model, loader("test"), device)
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results
