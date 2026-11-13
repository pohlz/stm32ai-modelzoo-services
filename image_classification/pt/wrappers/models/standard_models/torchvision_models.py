# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import torchvision

from common.model_utils.torch_utils import load_state_dict_partial
from common.registries.model_registry import MODEL_WRAPPER_REGISTRY
from common.utils import LOGGER
from image_classification.pt.wrappers.models.utils import NUM_IMAGENET_CLASSES, load_checkpoint_ic
from image_classification.pt.wrappers.models.model_implementation_dict import FINAL_SMALL_VALIDATED_MODELS


def make_wrapper_func(model_name_zoo, model_name_library):
    @MODEL_WRAPPER_REGISTRY.register(
        framework='torch',
        model_name=model_name_zoo,
        use_case="image_classification",
        has_checkpoint=True
    )
    def wrapper_func(cfg):
        # Load imagenet model for this exceptional case
        if cfg.model.pretrained_dataset == "imagenet" and cfg.model.pretrained:
            # Loading model with imagenet weights
            model = torchvision.models.__dict__[model_name_library](
                pretrained=cfg.model.pretrained, num_classes=NUM_IMAGENET_CLASSES)
            # Warning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
            if cfg.dataset.num_classes != NUM_IMAGENET_CLASSES:
                pretrained_dict = model.state_dict()
                # Loading model without weights
                model = torchvision.models.__dict__[model_name_library](
                    pretrained=False, num_classes=cfg.dataset.num_classes)
                # Warning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
                load_state_dict_partial(model, pretrained_dict)
                LOGGER.info(f"Loaded {model_name_zoo} pretrained on imagenet")
        else:
            # Loading model without weights
            model = torchvision.models.__dict__[model_name_library](
                pretrained=False, num_classes=cfg.dataset.num_classes)
            if cfg.model.pretrained:
                # Loading weights partial or full based on matching of key and size
                model = load_checkpoint_ic(model, cfg)
        return model

    wrapper_func.__name__ = model_name_zoo
    return wrapper_func
    
for model_name_zoo, model_name_library in FINAL_SMALL_VALIDATED_MODELS["torchvision"].items():
     globals()[model_name_zoo] = make_wrapper_func(model_name_zoo, model_name_library)
    