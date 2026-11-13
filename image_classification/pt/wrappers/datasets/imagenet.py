# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import random

import torch
from timm.data import create_dataset
from timm.data.transforms_factory import (transforms_imagenet_eval,
                                          transforms_imagenet_train)
from torch.utils.data import DataLoader, Subset

from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from common.utils import LOGGER
from image_classification.pt.src.datasets import prepare_kwargs_for_dataloader
from image_classification.pt.src.datasets.augmentations.augs import (
    DEFAULT_CROP_PCT, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
from image_classification.pt.src.datasets.dataset_utils import (
    PredictionDataset, create_loader)

__all__ = ['get_imagenet']    


@DATASET_WRAPPER_REGISTRY.register(framework='torch', dataset_name='imagenet16', use_case="image_classification")
@DATASET_WRAPPER_REGISTRY.register(framework='torch', dataset_name='imagenet10', use_case="image_classification")
@DATASET_WRAPPER_REGISTRY.register(framework='torch', dataset_name='imagenet', use_case="image_classification")
def get_imagenet(cfg):
    args = prepare_kwargs_for_dataloader(cfg)
    # args is dict after this point
    
    if isinstance(args["device"], str):
        args["device"] = torch.device(args["device"])

    train_loader = test_loader = val_loader = pred_loader = None
    root_directory = getattr(cfg.dataset,"data_dir", None) # dataset folder
    if root_directory:
        args["data_dir"] = data_root = os.path.join(cfg.dataset.data_dir, "imagenet") # imagenet folder
    else:
        args["data_dir"] = data_root = None
    if data_root: # imagenet folder
        if args.get("train_split", None):
            LOGGER.info(f"Loading training data from {data_root}/{args['train_split']}")
            train_loader = create_training_dataset(args)
        if args.get("val_split", None):
            LOGGER.info(f"Loading validation data from {data_root}/{args['val_split']}")
            val_loader = create_validation_dataset(args)
    else:
        LOGGER.info("No path available for training and validation data")
        
    if getattr(cfg.dataset, "test_path", None):
        LOGGER.info(f"Loading test data from {cfg.dataset.test_path}")
        test_loader = create_test_dataset(args)
    else:
        LOGGER.info("No path available for test data")
        
    quant_loader = create_quantization_dataset(args)
    
    if getattr(cfg.dataset, "prediction_path", None):
        LOGGER.info(f"Loading prediction data from {cfg.dataset.prediction_path}")
        pred_loader = create_prediction_dataset(args)
    else:
        LOGGER.info("No path available for prediction data")

    return {'train': train_loader, 'valid': val_loader, 'test': test_loader, 'quantization': quant_loader, 'predict': pred_loader}    
    
def create_training_dataset(args):
    data_root = args["data_dir"]
    re_num_splits = 0
    if args["re_split"]:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = args["num_aug_splits"] or 2
    img_size = args["img_size"]
    
    if isinstance(img_size, (tuple, list)):
        img_size = img_size[-1]
    default_train_transforms = transforms_imagenet_train(
        img_size,
        mean=args["mean"] or IMAGENET_DEFAULT_MEAN,
        std=args["std"] or IMAGENET_DEFAULT_STD,
        scale=args["scale"],
        ratio=args["ratio"],
        hflip=args["hflip"],
        vflip=args["vflip"],
        color_jitter=args["color_jitter"],
        auto_augment=args["auto_augment"],
        interpolation=args["train_interpolation"],
        re_prob=args["re_prob"],
        re_mode=args["re_mode"],
        re_count=args["re_count"],
        re_num_splits=re_num_splits,
        use_prefetcher=args["use_prefetcher"],
    )
    dataset_train = create_dataset(
        'imagenet',
        root=data_root,
        split=args["train_split"],
        is_training=True,
        class_map=args["class_map"],
        download=args["download"],
        batch_size=args["batch_size"],
        seed=args["seed"],
        repeats=args["repeats"],
    )
    dataset_train.transform = args.get("train_transforms", default_train_transforms)
    dataset_train.classes = range(args["num_classes"])
    
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
    return train_loader

def create_validation_dataset(args):
    data_root = args["data_dir"]
    img_size = args["img_size"]
    if isinstance(img_size, (tuple, list)):
        img_size = img_size[-1]
        
    default_val_transforms = transforms_imagenet_eval(
        img_size,
        mean=args["mean"] or IMAGENET_DEFAULT_MEAN,
        std=args["std"] or IMAGENET_DEFAULT_STD,
        crop_pct=args.get("crop_pct") or DEFAULT_CROP_PCT,
        interpolation=args["test_interpolation"],
        use_prefetcher=args["use_prefetcher"],
    )
    dataset_val = create_dataset(
        'imagenet',
        root=data_root,
        split=args["val_split"],
        is_training=False,
        class_map=args["class_map"],
        download=args["download"],
        batch_size=args["batch_size"],
    )

    dataset_val.transform=args.get("val_transforms", default_val_transforms)
    val_loader = create_loader(
        dataset_val,
        input_size=args["img_size"],
        batch_size=args.get("val_batch_size", args["batch_size"]),
        is_training=False,
        use_prefetcher=args["use_prefetcher"],
        mean=args.get("mean") or IMAGENET_DEFAULT_MEAN,
        std=args.get("std") or IMAGENET_DEFAULT_STD,
        num_workers=args["num_workers"],
        distributed=args["distributed"],
        pin_memory=args["pin_memory"],
        device=args["device"],
    )
    return val_loader

def create_test_dataset(args):
    #data_root = args["data_dir"]
    img_size = args["img_size"]
    if isinstance(img_size, (tuple, list)):
        img_size = img_size[-1]
        
    default_test_transforms = transforms_imagenet_eval(
        img_size,
        mean=args["mean"] or IMAGENET_DEFAULT_MEAN,
        std=args["std"] or IMAGENET_DEFAULT_STD,
        crop_pct=args.get("crop_pct") or DEFAULT_CROP_PCT,
        interpolation=args["test_interpolation"],
        use_prefetcher=args["use_prefetcher"],
    )
    dataset_test = create_dataset(
        'imagenet',
        root=args["test_path"],
        #split=args["test_split"],
        search_split=False,
        is_training=False,
        class_map=args["class_map"],
        download=args["download"],
        batch_size=args["batch_size"],
    )

    dataset_test.transform=args.get("test_transforms", default_test_transforms)
    test_loader = create_loader(
        dataset_test,
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
    return test_loader

def create_quantization_dataset(args):
    re_num_splits = 0
    if args["re_split"]:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = args["num_aug_splits"] or 2
    img_size = args["img_size"]

    if isinstance(img_size, (tuple, list)):
        img_size = img_size[-1]
    default_train_transforms = transforms_imagenet_train(
        img_size,
        mean=args["mean"] or IMAGENET_DEFAULT_MEAN,
        std=args["std"] or IMAGENET_DEFAULT_STD,
        scale=args["scale"],
        ratio=args["ratio"],
        hflip=args["hflip"],
        vflip=args["vflip"],
        color_jitter=args["color_jitter"],
        auto_augment=args["auto_augment"],
        interpolation=args["train_interpolation"],
        re_prob=args["re_prob"],
        re_mode=args["re_mode"],
        re_count=args["re_count"],
        re_num_splits=re_num_splits,
        use_prefetcher=args["use_prefetcher"],
        )
    
    if args.get("quantization_path"):
        data_path = args["quantization_path"]
        LOGGER.info(f"Loading quantization data from {data_path}")
    elif args.get("data_dir") and args.get("train_split"):# and getattr(cfg, "quantization", None) is not None:
        data_path = os.path.join(args["data_dir"],args["train_split"])
        LOGGER.info(f"Loading quantization data from training data at: {data_path}")
    else:
        LOGGER.info("No path available for quantization data")
        return None

    dataset_train = create_dataset(
        'imagenet',
        root=data_path,
        #split=args["train_split"],
        search_split=False,
        is_training=True, #Should this be false
        class_map=args["class_map"],
        download=args["download"],
        batch_size=args["batch_size"],
        seed=args["seed"],
        repeats=args["repeats"],
    )
    #print(type(dataset_train))
    dataset_train.transform = args.get("train_transforms", default_train_transforms)
    dataset_train.classes = range(args["num_classes"])
    
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
        batch_size=1, # what shud be the batch size 
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=args["pin_memory"],
    ) #returns (img, target)

    return quant_loader

def create_prediction_dataset(args):
    img_size = args["img_size"]
    
    if isinstance(img_size, (tuple, list)):
        img_size = img_size[-1]
    default_val_transforms = transforms_imagenet_eval(
        img_size,
        mean=args["mean"] or IMAGENET_DEFAULT_MEAN,
        std=args["std"] or IMAGENET_DEFAULT_STD,
        crop_pct=args.get("crop_pct") or DEFAULT_CROP_PCT,
        interpolation=args["test_interpolation"],
        use_prefetcher=args["use_prefetcher"],
    )
    dataset_pred = PredictionDataset(args["prediction_path"], default_val_transforms)
    pred_loader = DataLoader(
        dataset_pred,
        batch_size=1,
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=args["pin_memory"],
    )
    return pred_loader