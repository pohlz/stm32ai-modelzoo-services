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
from image_classification.pt.src.models.mobilenetv1.mobilenetv1 import MobileNetV1
from image_classification.pt.wrappers.models.utils import load_checkpoint_ic
from image_classification.pt.wrappers.models.checkpoints import MODEL_CHECKPOINTS

__all__ = ["mobilenet_v1_vww", 'mobilenet_v1_025_vww', 'mobilenet_v1_025_96px_vww']


def mobilenetv1_vww(cfg, last_pooling_size=7, width_mult=1.0,
):
    model = MobileNetV1(
        num_classes=cfg.dataset.num_classes,
        width_mult=width_mult,
        last_pooling_size=last_pooling_size,
    )
    if cfg.model.pretrained:
        model = load_checkpoint_ic(model, cfg)
    return model


@MODEL_WRAPPER_REGISTRY.register(
        framework='torch',
        model_name='mobilenet_v1_pt',
        use_case="image_classification",
        has_checkpoint = any(k.startswith('mobilenet_v1_pt') for k in MODEL_CHECKPOINTS)
)
def mobilenet_v1_vww(cfg):
    return mobilenetv1_vww(cfg)


@MODEL_WRAPPER_REGISTRY.register(
        framework='torch',
        model_name='mobilenet_v1_0.25_pt',
        use_case="image_classification",
        has_checkpoint = any(k.startswith('mobilenet_v1_0.25_pt') for k in MODEL_CHECKPOINTS)
)
def mobilenet_v1_025_vww(cfg):
    return mobilenetv1_vww(cfg, width_mult=0.25)


@MODEL_WRAPPER_REGISTRY.register(
        framework='torch',
        model_name='mobilenet_v1_0.25_96px_pt',
        use_case="image_classification",
        has_checkpoint = any(k.startswith('mobilenet_v1_0.25_96px_pt') for k in MODEL_CHECKPOINTS)
)
def mobilenet_v1_025_96px_vww(cfg):
    return mobilenetv1_vww(cfg, width_mult=0.25, last_pooling_size=3)
