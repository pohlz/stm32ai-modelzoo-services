"""Generate STM figures for a completed PyTorch run without retraining.

Run from audio_event_detection:
python tools/plot_bcresnet_pt_run.py --run-dir pt/src/experiments_outputs/RUN
"""

import argparse
import json
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

AED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AED_ROOT))
from pt.src.dataset import GSCDataset
from pt.src.models import get_custom_model
from pt.src.reporting import plot_training_history
from pt.src.runner import evaluate, load_weights


def generate_reports(run_dir, data_base=AED_ROOT, device="auto", threads=4):
    """Leave original config/history/metrics/checkpoints intact; add reports."""
    run_dir, data_base = Path(run_dir).resolve(), Path(data_base).resolve()
    cfg = OmegaConf.load(run_dir / "config.yaml")
    history_path = run_dir / "history.json"
    if history_path.exists():
        plot_training_history(json.loads(history_path.read_text()), run_dir)
    # Saved paths were relative to the original launch directory, not run_dir.
    for key, value in cfg.dataset.items():
        if key.endswith("_path") and value and not Path(value).is_absolute():
            cfg.dataset[key] = str(data_base / value)
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    model = get_custom_model(cfg.model.input_shape, len(cfg.dataset.class_names), cfg.model.tau)
    checkpoint = run_dir / cfg.general.saved_models_dir / "best_model.pth"
    load_weights(model, checkpoint, cfg)
    model.to(device)
    results = {}
    for split in ("validation", "test"):
        if not cfg.dataset.get(f"{split}_csv_path"):
            continue
        print(f"Evaluating {checkpoint.name} on {split} ({device})...", flush=True)
        loader = DataLoader(GSCDataset(cfg, split), batch_size=cfg.training.batch_size, num_workers=0)
        results[split] = evaluate(model, loader, device, output_dir=run_dir,
                                  class_names=list(cfg.dataset.class_names), split=split)
    (run_dir / "report_metrics.json").write_text(json.dumps(
        {"checkpoint": str(checkpoint), "results": results}, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-base", type=Path, default=AED_ROOT,
                        help="Original launch directory for relative dataset paths")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    generate_reports(args.run_dir, args.data_base, args.device, args.threads)
