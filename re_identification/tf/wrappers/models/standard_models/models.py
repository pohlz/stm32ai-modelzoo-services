# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from common.registries.model_registry import MODEL_WRAPPER_REGISTRY
from re_identification.tf.src.models import *



re_identification_MODEL_FNS = {
    # Key: model name, Value: (wrapper function, dict of static kwargs)
    'efficientnetv2b0': (get_efficientnetv2, {'model_type': 'B0' }),
    'efficientnetv2b1': (get_efficientnetv2, {'model_type': 'B1' }),
    'efficientnetv2b2': (get_efficientnetv2, {'model_type': 'B2' }),
    'efficientnetv2b3': (get_efficientnetv2, {'model_type': 'B3' }),
    'efficientnetv2s':  (get_efficientnetv2, {'model_type': 'S' }),
    'mobilenetv1_a025': (get_mobilenetv1,    {'alpha': 0.25 }),
    'mobilenetv1_a050': (get_mobilenetv1,    {'alpha': 0.50 }),
    'mobilenetv1_a075': (get_mobilenetv1,    {'alpha': 0.75 }),
    'mobilenetv1_a100': (get_mobilenetv1,    {'alpha': 1.0 }),
    'mobilenetv2_a035': (get_mobilenetv2,    {'alpha': 0.35 }),
    'mobilenetv2_a050': (get_mobilenetv2,    {'alpha': 0.50 }),
    'mobilenetv2_a075': (get_mobilenetv2,    {'alpha': 0.75 }),
    'mobilenetv2_a100': (get_mobilenetv2,    {'alpha': 1.0 }),
    'mobilenetv2_a130': (get_mobilenetv2,    {'alpha': 1.3 }),
    'mobilenetv2_a140': (get_mobilenetv2,    {'alpha': 1.4 }),
    'resnet50v2':       (get_resnet50v2,     {}),
    'resnet8':          (get_resnet,         {'depth': 8}),
    'resnet20':         (get_resnet,         {'depth': 20}),
    'resnet32':         (get_resnet,         {'depth': 32}),
    'osnet_a025':       (get_osnet,          {'alpha': 0.25}),
    'osnet_a050':       (get_osnet,          {'alpha': 0.50}),
    'osnet_a075':       (get_osnet,          {'alpha': 0.75}),
    'osnet_a100':       (get_osnet,          {'alpha': 1.0})
}

def register_tf_model_wrapper(model_fn, model_name, **model_init_kwargs):
    """
    Register a TensorFlow model wrapper in the global registry.

    Args:
        model_fn (callable): The model-building function.
        model_name (str): Name to register the model under.
        **model_init_kwargs: Static keyword arguments for the model_fn.

    Returns:
        function: The registered build_model_fn function.
    """
    def build_model_fn(cfg):
        """
        Build and return the model instance for the registry.

        Args:
            cfg (dict): top level config

        Returns:
            keras.Model: The constructed model.
        """
        model_kwargs = prepare_kwargs_for_model(cfg)
        merged_kwargs = {**model_kwargs, **model_init_kwargs}
        return model_fn(**merged_kwargs)
    # Register the model in the global registry
    build_model_fn = MODEL_WRAPPER_REGISTRY.register(
        framework='tf',
        model_name=model_name,
        use_case='re_identification',
    )(build_model_fn)
    # Set a unique function name for clarity
    build_model_fn.__name__ = f'{model_name}_tf'
    return build_model_fn

# Register all models defined in TF_MODEL_FNS
for _model_name, (_model_fn, _model_kwargs) in re_identification_MODEL_FNS.items():
    wrapper_fn = register_tf_model_wrapper(_model_fn, _model_name, **_model_kwargs)
    # Expose the wrapper in the module's global namespace
    globals()[wrapper_fn.__name__] = wrapper_fn
