# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from common.utils import LOGGER
from torchvision.models import MobileNetV2

from common.registries.model_registry import MODEL_WRAPPER_REGISTRY
from image_classification.pt.src.models.edgevit.edgevit import (edgevit_s, edgevit_xs,
                                                        edgevit_xxs)
from image_classification.pt.src.models.fasternet.fasternet import (fasternet_l,
                                                          fasternet_m,
                                                          fasternet_s,
                                                          fasternet_t0,
                                                          fasternet_t1,
                                                          fasternet_t2)
from image_classification.pt.src.models.stresnet import STResNetMicro, STResNetMilli, STResNetNano ,STResNetPico, STResNetTiny
from image_classification.pt.wrappers.models.utils import load_checkpoint_ic
from image_classification.pt.wrappers.models.checkpoints import MODEL_CHECKPOINTS

__all__ = []


MODEL_FNS = {
    'edgevit_s_pt': (edgevit_s, {}),
    'edgevit_xs_pt': (edgevit_xs, {}),
    'edgevit_xxs_pt': (edgevit_xxs, {}),
    'fasternet_t0_pt': (fasternet_t0, {}),
    'fasternet_t1_pt': (fasternet_t1, {}),
    'fasternet_t2_pt': (fasternet_t2, {}),
    'fasternet_s_pt': (fasternet_s, {}),
    'fasternet_m_pt': (fasternet_m, {}),
    'fasternet_l_pt': (fasternet_l, {}),
    'mobilenetv2_w035_pt': (MobileNetV2, {'width_mult': 0.35}),
    'st_resnetmicro_actrelu_pt': (STResNetMicro, {}),
    'st_resnetmilli_actrelu_pt': (STResNetMilli, {}),
    'st_resnetnano_actrelu_pt': (STResNetNano, {}),
    'st_resnetpico_actrelu_pt': (STResNetPico, {}),
    'st_resnettiny_actrelu_pt': (STResNetTiny, {}),
}

def register_model_wrapper(model_fn, model_name, **model_init_kwargs):

    def get_model(cfg):
        model = model_fn(num_classes=cfg.dataset.num_classes, **model_init_kwargs)
        if cfg.model.pretrained:
            # Loading weights partial or full based on matching of key and size
            model = load_checkpoint_ic(model, cfg)
        return model

    get_model = MODEL_WRAPPER_REGISTRY.register(
        framework='torch',
        model_name=model_name,
        use_case="image_classification",
        has_checkpoint = any(k.startswith(model_name) for k in MODEL_CHECKPOINTS)
    )(get_model)

    get_model.__name__ = f'{model_name}'
    return get_model


for _model_name, (_model_fn, _model_kwargs) in MODEL_FNS.items():
    globals()[_model_name] = register_model_wrapper(_model_fn, _model_name, **_model_kwargs)
    __all__.append(_model_name)