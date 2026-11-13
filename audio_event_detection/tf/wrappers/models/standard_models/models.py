# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from common.registries.model_registry import MODEL_WRAPPER_REGISTRY
from audio_event_detection.tf.src.models import *



TF_MODEL_FNS = {
    'miniresnetv2_s1_ap': (get_miniresnetv2, {"n_stacks": 1, "pooling": 'avg'}),
    'miniresnetv2_s2_ap': (get_miniresnetv2, {"n_stacks": 2, "pooling": 'avg'}),
    'miniresnetv2_s1': (get_miniresnetv2, {"n_stacks": 1, "pooling": None}),
    'miniresnetv2_s2': (get_miniresnetv2, {"n_stacks": 2, "pooling": None}),
    'miniresnetv1_s1_ap': (get_miniresnetv1, {"n_stacks": 1, "pooling": 'avg'}),
    'miniresnetv1_s2_ap': (get_miniresnetv1, {"n_stacks": 2, "pooling": 'avg'}),
    'miniresnetv1_s1': (get_miniresnetv1, {"n_stacks": 1, "pooling": None}),
    'miniresnetv1_s2': (get_miniresnetv1, {"n_stacks": 2, "pooling": None}),
    'custom':(get_custom_model, {}),
    'yamnet_e256': (get_yamnet, {"embedding_size": 256, "pretrained": True}),
    'yamnet_e512': (get_yamnet, {"embedding_size": 512, "pretrained": True}),
    'yamnet_e1024': (get_yamnet, {"embedding_size": 1024, "pretrained": True}),
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
            num_classes (int): Number of output classes.
            pretrained (bool): Use pretrained weights if True.
            **model_kwargs: Additional arguments for the model_fn.

        Returns:
            keras.Model: The constructed model.
        """
        kwargs = prepare_kwargs_for_model(cfg)
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        merged_model_kwargs = {**kwargs, **filtered_kwargs, **model_init_kwargs}
        return model_fn(**merged_model_kwargs)

    # Register the model in the global registry
    build_model_fn = MODEL_WRAPPER_REGISTRY.register(
        framework='tf',
        model_name=model_name, 
        use_case='audio_event_detection',
    )(build_model_fn)
    # Set a unique function name for clarity
    build_model_fn.__name__ = f'{model_name}_tf'
    return build_model_fn

# Register all models defined in TF_MODEL_FNS
for _model_name, (_model_fn, _model_kwargs) in TF_MODEL_FNS.items():
    wrapper_fn = register_tf_model_wrapper(_model_fn, _model_name, **_model_kwargs)
    # Expose the wrapper in the module's global namespace
    globals()[wrapper_fn.__name__] = wrapper_fn
