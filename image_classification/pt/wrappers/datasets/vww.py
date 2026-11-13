# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from os.path import expanduser
import random
from torch.utils.data import DataLoader, Subset
import torch

import torch
from pyvww.pytorch import VisualWakeWordsClassification

from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from common.utils import LOGGER
from image_classification.pt.src.datasets.augmentations.augs import (
    DEFAULT_CROP_PCT, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
    get_imagenet_transforms, get_vanilla_transforms)
from image_classification.pt.src.datasets.dataset_utils import create_loader, PredictionDataset
from image_classification.pt.src.datasets import prepare_kwargs_for_dataloader

__all__ = ['get_vww']

@DATASET_WRAPPER_REGISTRY.register(framework='torch', dataset_name='vww', use_case="image_classification")
def get_vww(cfg):
    args = prepare_kwargs_for_dataloader(cfg)
    args["augmentation_mode"] = "imagenet"
    if isinstance(args["device"], str):
        args["device"] = torch.device(args["device"])

    #data_root = os.path.join(cfg.dataset.data_dir, "vww")
    train_loader = val_loader = pred_loader = None
    root_directory = getattr(cfg.dataset,"data_dir", None) # dataset folder
    if root_directory:
        data_root = os.path.join(cfg.dataset.data_dir, "vww") # imagenet folder
    else:
        data_root = None
    #data_root = cfg.dataset.data_dir
    if data_root:
        if args["augmentation_mode"] not in ("vanilla", "imagenet"):
            raise ValueError(
                f'Wrong value of augmentation_mode arg: {args["augmentation_mode"]}. Choices: "vanilla", "imagenet"'
            )

        re_num_splits = 0
        if args["re_split"]:
            # apply RE to second half of batch if no aug split otherwise line up with aug split
            re_num_splits = args["num_aug_splits"] or 2
        if args["augmentation_mode"] == "imagenet":
            default_train_transforms, default_val_transforms = get_imagenet_transforms(
                args["img_size"],
                mean=args.get("mean") or IMAGENET_DEFAULT_MEAN,
                std=args.get("std") or IMAGENET_DEFAULT_STD,
                crop_pct=args.get("crop_pct") or DEFAULT_CROP_PCT,
                scale=args["scale"],
                ratio=args["ratio"],
                hflip=args["hflip"],
                vflip=args["vflip"],
                color_jitter=args["color_jitter"],
                auto_augment=args["auto_augment"],
                train_interpolation=args["train_interpolation"],
                test_interpolation=args["test_interpolation"],
                re_prob=args["re_prob"],
                re_mode=args["re_mode"],
                re_count=args["re_count"],
                re_num_splits=re_num_splits,
                use_prefetcher=args["use_prefetcher"],
            )
        else:
            default_train_transforms, default_val_transforms = get_vanilla_transforms(
                args["img_size"],
                hflip=args["hflip"],
                jitter=args["color_jitter"],
                mean=args.get("mean") or IMAGENET_DEFAULT_MEAN,
                std=args.get("std") or IMAGENET_DEFAULT_STD,
                crop_pct=args.get("crop_pct") or DEFAULT_CROP_PCT,
                use_prefetcher=args["use_prefetcher"],
            )

        dataset_train = VisualWakeWordsClassification(
            root=os.path.join(data_root, "all"),
            annFile=os.path.join(data_root, "annotations/instances_train.json"),
            transform=args.get("train_transforms", default_train_transforms),
        )

        dataset_eval = VisualWakeWordsClassification(
            root=os.path.join(data_root, "all"),
            annFile=os.path.join(data_root, "annotations/instances_val.json"),
            transform=args.get("val_transforms", default_val_transforms),
        )

        train_loader = create_loader(
            dataset_train,
            input_size=args["img_size"],
            batch_size=args["batch_size"],
            is_training=True,
            use_prefetcher=args["use_prefetcher"],
            no_aug=args["no_aug"],
            re_prob=args["re_prob"],
            re_mode=args["re_mode"],
            re_count=args["re_count"],
            num_aug_repeats=args["num_aug_repeats"],
            re_num_splits=re_num_splits,
            mean=args.get("mean") or IMAGENET_DEFAULT_MEAN,
            std=args.get("std") or IMAGENET_DEFAULT_STD,
            num_workers=args["num_workers"],
            distributed=args["distributed"],
            collate_fn=args["collate_fn"],
            pin_memory=args["pin_memory"],
            device=args["device"],
            use_multi_epochs_loader=args["use_multi_epochs_loader"],
            worker_seeding=args["worker_seeding"],
        )

        val_loader = create_loader(
            dataset_eval,
            input_size=args["img_size"],
            batch_size=args.get("test_batch_size", args["batch_size"]),
            is_training=False,
            use_prefetcher=args["use_prefetcher"],
            mean=args.get("mean") or IMAGENET_DEFAULT_MEAN,
            std=args.get("std") or IMAGENET_DEFAULT_STD,
            num_workers=args["num_workers"],
            distributed=args["distributed"],
            pin_memory=args["pin_memory"],
            device=args["device"],
        )
    else:
        LOGGER.info("No path available for training and validation data")
        train_loader = None
        val_loader = None
    # ------------------------------------
    # Create QUANTIZATION & PREDICTION DS
    # ------------------------------------
    # Quantization: small subset of train data
    # use quantization split from cfg file
    if train_loader and getattr(cfg, "quantization", None):
        # quantization_split defines the fraction of training data to use, e.g. 0.1 for 10%, default 1 i.e. 100%
        quantization_split = args.get("quantization_split", 1.0)
        if quantization_split == 1.0:
            LOGGER.info("100 percent data is being used for quantization")

        # Compute number of samples for quantization subset
        num_quant_samples = int(len(dataset_train) * quantization_split)
        quant_indices = random.sample(range(len(dataset_train)), min(num_quant_samples, len(dataset_train)))
        quant_subset = Subset(dataset_train, quant_indices)
        quant_loader = DataLoader(
            quant_subset,
            batch_size=32,
            shuffle=False,
            num_workers=args["num_workers"],
            pin_memory=args["pin_memory"],
        ) 
    else:
        LOGGER.info("No path available for quantization data")
        quant_loader = None
        
    if getattr(cfg.dataset, "prediction_path", None):
        dataset_pred = PredictionDataset(cfg.dataset.prediction_path, default_val_transforms)
        pred_loader = DataLoader(
            dataset_pred,
            batch_size=1,
            shuffle=False,
            num_workers=args["num_workers"],
            pin_memory=args["pin_memory"],
        )
    else: 
        LOGGER.info("No path available for prediction data")
        pred_loader = None
    return {'train': train_loader, 'valid': val_loader, 'test': val_loader, 'quantization': quant_loader, 'predict': pred_loader}
