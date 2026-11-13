# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.model_utils.torch_utils import load_pretrained_weights
from common.registries.model_registry import MODEL_WRAPPER_REGISTRY
from common.model_utils.torch_utils import load_state_dict_partial
from image_classification.pt.wrappers.models.checkpoints import CHECKPOINT_STORAGE_URL, MODEL_CHECKPOINTS
from urllib.parse import urljoin

NUM_IMAGENET_CLASSES = 1000


from pathlib import Path
import torch

def load_checkpoint_ic(model, cfg):
    """
    Load pretrained weights into an already-defined model.
    Handles:
        - Direct path in cfg.model.model_path
        - Custom datasets (food101, flowers102)
        - Imagenet handled externally
    """
    dataset = cfg.model.pretrained_dataset.lower()
    model_name = cfg.model.model_name

    # Direct model path — highest priority
    if getattr(cfg.model, "model_path", None):
        ckpt_path = cfg.model.model_path
        model = load_pretrained_weights(model, str(ckpt_path))
        print(f"Loaded {model_name} pretrained on mode_path you provided")
        return model

    # Custom datasets (Food101 / Flowers102)
    elif dataset in ["food101", "flowers102", "imagenet", "vww"]:
        checkpoint_key = f"{model_name}_dataset{dataset}_res{cfg.model.input_shape[1]}"
        if checkpoint_key not in MODEL_CHECKPOINTS:
            print(f"No checkpoint found for {checkpoint_key}")
            return model
        ckpt_path = urljoin(CHECKPOINT_STORAGE_URL + "/", MODEL_CHECKPOINTS[checkpoint_key])
        model = load_pretrained_weights(model, str(ckpt_path))
        print(f"Loaded {model_name} pretrained on {dataset}")
        return model
    else:
        raise ValueError(
            f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset}. \n'
            'Use pretrained=False if you want to create a untrained model.'
        )

def load_checkpoint(model, model_name, dataset_name, model_urls, device='cpu'):
    if f'{model_name}_{dataset_name}' not in model_urls:
        raise ValueError(
            f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. \n'
            'Use pretrained=False if you want to create a untrained model.'
        )
    model = load_pretrained_weights(model, model_urls[f'{model_name}_{dataset_name}'], device)
    return model