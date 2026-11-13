# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.

#  * Loader Factory, Fast Collate, CUDA Prefetcher
#  * Prefetcher and Fast Collate inspired by NVIDIA APEX example at
#  * https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf
#  *  Copyright 2019, Ross Wightman
#  *--------------------------------------------------------------------------------------------*/

from functools import partial

import torch
from timm.data.dataset import IterableImageDataset
from timm.data.distributed_sampler import (OrderedDistributedSampler,
                                           RepeatAugSampler)
from timm.data.loader import (MultiEpochsDataLoader, PrefetchLoader,
                              _worker_init, fast_collate)

from image_classification.pt.src.datasets.augmentations.augs import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os

class PredictionDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_path

    def __len__(self):
        return len(self.image_paths)
    
def create_loader(
    dataset,
    input_size,
    batch_size,
    is_training=False,
    use_prefetcher=True,
    no_aug=False,
    re_prob=0.,
    re_mode='const',
    re_count=1,
    re_num_splits=0,
    num_aug_repeats=0,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    num_workers=1,
    distributed=False,
    collate_fn=None,
    pin_memory=False,
    fp16=False,  # deprecated, use img_dtype
    img_dtype=torch.float32,
    device=torch.device('cuda'),
    use_multi_epochs_loader=False,
    persistent_workers=True,
    worker_seeding='all',
):
    if isinstance(input_size, int):
        input_size = (3, input_size, input_size)

    if isinstance(dataset, IterableImageDataset):
        # give Iterable datasets early knowledge of num_workers so that sample estimates
        # are correct before worker processes are launched
        dataset.set_loader_cfg(num_workers=num_workers)

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader_args = {
        'batch_size': batch_size,
        'shuffle': not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        'num_workers': num_workers,
        'sampler': sampler,
        'collate_fn': collate_fn,
        'pin_memory': pin_memory,
        'drop_last': is_training,
        'worker_init_fn': partial(_worker_init, worker_seeding=worker_seeding),
        'persistent_workers': persistent_workers
    }
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(  # pylint: disable=unexpected-keyword-arg
            loader,
            mean=mean,
            std=std,
            channels=input_size[0],
            device=device,
            fp16=fp16,  # deprecated, use img_dtype
            img_dtype=img_dtype,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )
    return loader


from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler as DS


def get_dataloader(
    dataset, batch_size=32, num_workers=4, fp16=False, distributed=False, shuffle=False,
    collate_fn=None, device="cuda"
):
    if collate_fn is None:
        collate_fn = default_collate
    def half_precision(x):
        x = collate_fn(x)
        x = [_x.half() if isinstance(_x, torch.FloatTensor) else _x for _x in x]
        return x
 
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn= half_precision if fp16 else collate_fn,
        sampler=DS(dataset) if distributed else None,
    )
    return dataloader


from timm import utils
from timm.data import FastCollateMixup


def prepare_kwargs_for_dataloader(cfg):
    #####################################
    # Hydra config to flat config with last keys
    #####################################
    from types import SimpleNamespace

    from common.utils import flatten_config

    # convert hydra heirarchical config to flat config
    args_raw = flatten_config(cfg)
    # convert dict based config to argument parser type
    if isinstance(args_raw, dict):
        args = SimpleNamespace(**args_raw)
    else:
        args = args_raw
    #####################################
    args.prefetcher = not getattr(args, 'no_prefetcher', False)
    if not hasattr(args, "batch_size") or args.batch_size is None:
        args.batch_size = 128
    if not hasattr(args, "validation_batch_size") or args.validation_batch_size is None:
        args.validation_batch_size = args.batch_size
    num_aug_splits = 0
    collate_fn = None
    args.mixup = getattr(args, "mixup", 0)
    args.cutmix = getattr(args, "cutmix", 0)
    args.cutmix_minmax = getattr(args, "cutmix_minmax", None)

    mixup_active = (args.mixup > 0) or (args.cutmix > 0) or (args.cutmix_minmax is not None)
    # with current config following "if" is false
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=getattr(args, "mixup_prob", 1.0),
            switch_prob=getattr(args, "mixup_switch_prob", 0.5),
            mode=getattr(args, "mixup_mode", 'batch'),
            label_smoothing=getattr(args, "smoothing", 0.1),
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
    
    device = args.device # already set in the main py
    
    if getattr(args, 'aug_splits', 0) > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits
    extra_loader_kwargs = {}
   
    if getattr(args, 'dataset_name', '') in ['imagenet','custom']:
        extra_loader_kwargs = {
            'train_split': getattr(args, 'train_split', 'train'),
            'val_split': getattr(args, 'val_split', 'val'),
            'class_map': getattr(args, 'class_map', None),
            'seed': getattr(args, 'seed', None),
            'repeats': getattr(args, 'epoch_repeats', None),
        }
    # create data loaders w/ augmentation pipeline
    test_interpolation = 'bicubic'
    train_interpolation = getattr(args, 'interpolation', "random")
    args.no_aug = getattr(args, 'no_aug', False)
    if args.no_aug or not train_interpolation:
        train_interpolation = test_interpolation
    
    args.mean= getattr(args, 'mean', [0.485, 0.456, 0.406]) 
    args.std= getattr(args, 'std', [0.229, 0.224, 0.225]) 
    batch_size = getattr(args, 'batch_size', 128)
    validation_batch_size = getattr(args, 'validation_batch_size', batch_size)
    loader_kwargs = dict(
        img_size=tuple(args.input_shape),
        batch_size=batch_size,
        test_batch_size=validation_batch_size,
        download=getattr(args, 'data_download', False),
        distributed=getattr(args, 'distributed', False), # created same way as before 
        use_prefetcher=args.prefetcher, # created same way as before
        no_aug=args.no_aug,
        re_prob=getattr(args, 'reprob', 0),
        re_mode=getattr(args, 'remode', 'pixel'),
        re_count=getattr(args, 'recount', 1),
        re_split=getattr(args, 'resplit', False),
        scale=getattr(args, 'crop_range', [0.08, 1.0]), # names are different as scale is used for other purpose in the config
        ratio=getattr(args, 'ratio', [0.75, 1.33]),
        hflip=getattr(args, 'hflip', 0.5),
        vflip=getattr(args, 'vflip', 0.0),
        color_jitter=getattr(args, 'color_jitter', 0.4),
        auto_augment=getattr(args, 'aa', None),
        num_aug_repeats=getattr(args, 'aug_repeats', 0),
        num_aug_splits=num_aug_splits,
        train_interpolation=train_interpolation,
        test_interpolation=test_interpolation,
        mean=args.mean,
        std=args.std,
        num_workers=getattr(args, 'workers', 4),
        collate_fn=collate_fn,
        pin_memory=getattr(args, 'pin_mem', False),
        device=device, # multi gpu
        use_multi_epochs_loader=getattr(args, 'use_multi_epochs_loader', False),
        worker_seeding=getattr(args, 'worker_seeding', False),
        test_path=getattr(args, 'test_path', None),
        prediction_path=getattr(args, 'prediction_path', None),
        quantization_path=getattr(args, 'quantization_path', None),
        quantization_split = getattr(args, 'quantization_split', 0.1),
        num_classes = args.num_classes, # has to be there
        **extra_loader_kwargs)
    return loader_kwargs

def get_from_config(cfg, key_path, default=None):
    """
    Safely get a nested value from cfg using dot-separated keys.
    Works for OmegaConf DictConfig, dict, or objects with attributes.
    
    Example:
        get_from_config(cfg, "a.b.c", default=42)
    """
    keys = key_path.split(".")
    current = cfg

    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        elif hasattr(current, key):
            current = get_from_config(current, key)
        else:
            return default
    return current

