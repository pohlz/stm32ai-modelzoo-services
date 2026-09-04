# Adding a new trainable model to Audio Event Detection

This guide describes the model-loading path used by the current TensorFlow Audio
Event Detection (AED) service and the smallest safe way to add a new architecture
as a `model.model_name` option.

## What ST already provides

The service already has the extension machinery needed for a new model:

- `tf/src/models/custom_model.py` is an intentionally editable starter model.
- `tf/src/models/model_utils.py` provides `add_head()` for attaching a classifier
  to a backbone and `prepare_kwargs_for_model()` for translating the service
  configuration into builder arguments.
- `tf/wrappers/models/standard_models/models.py` maps model names to builders and
  registers them in ST's global `MODEL_WRAPPER_REGISTRY`.
- The common training service supplies optimizers, callbacks, learning-rate
  schedulers, augmentation, checkpointing, evaluation, quantization, benchmarking,
  and deployment. A new Keras model does not need a new trainer.
- Existing MiniResNet v1/v2 and YAMNet implementations are useful reference
  implementations for scratch training and pretrained-backbone transfer learning.

For a single local experiment, editing `get_custom_model()` and selecting
`model_name: custom` is the shortest route. For a permanent, separately named
option, follow the steps below.

## How model creation flows

1. `stm32ai_main.py` parses the YAML and calls `api.get_model(cfg)`.
2. Importing `api/api.py` imports the AED model wrapper module, which performs
   registration as an import side effect.
3. `api.get_model()` looks up `(model_name, audio_event_detection, tf)` in the
   global registry.
4. The wrapper calls `prepare_kwargs_for_model(cfg)`, adds any fixed arguments
   from `TF_MODEL_FNS`, and invokes the architecture builder.
5. The resulting Keras model is saved, passed to `AEDTrainer`, wrapped with any
   configured augmentation layers, compiled, trained, and evaluated.

If `model.model_path` points to `.keras`, `.h5`, `.tflite`, or `.onnx`, the registry
is bypassed. Therefore `model_name` is for constructing an architecture; `model_path`
is for loading an already-built model.

## Builder contract

Create a module, for example:

`audio_event_detection/tf/src/models/my_aed/my_aed.py`

Its public builder should return an **uncompiled** `tf.keras.Model` and accept the
arguments supplied by the wrapper. Accepting `**kwargs` keeps it compatible with
the shared wrapper:

```python
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers


def get_my_aed(
    num_classes: int,
    input_shape: Tuple[int, int, int],
    dropout: Optional[float] = 0.0,
    use_garbage_class: bool = False,
    multi_label: bool = False,
    **kwargs,
) -> tf.keras.Model:
    if use_garbage_class:
        num_classes += 1

    # The current trainer does not implement multi-label loss. Fail explicitly
    # until that service limitation is removed.
    if multi_label:
        raise NotImplementedError("AED multi-label training is not implemented")

    inputs = tf.keras.Input(shape=tuple(input_shape), name="log_mel_patch")
    x = layers.Conv2D(24, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="my_aed")
```

Important contract details:

- With `dataset.expand_last_dim: true`, samples have shape
  `(feature_extraction.n_mels, feature_extraction.patch_length, 1)`. The YAML
  `model.input_shape` and the model input must match it exactly.
- The output width must equal `dataset.num_classes`; add one only when honoring
  `use_garbage_class`.
- Current single-label training uses categorical cross-entropy, so the output is
  expected to be a probability vector (normally softmax, not logits).
- Do not compile the model in the builder; `AEDTrainer` compiles it with the YAML
  optimizer and the service loss.
- Prefer standard serializable Keras layers. The trainer saves and reloads a full
  `.keras` checkpoint before producing `best_model.keras`.

Add `audio_event_detection/tf/src/models/my_aed/__init__.py`:

```python
from .my_aed import get_my_aed
```

Then export it from `audio_event_detection/tf/src/models/__init__.py`:

```python
from .my_aed import get_my_aed
```

## Register the new option

In `audio_event_detection/tf/wrappers/models/standard_models/models.py`, add an
entry to `TF_MODEL_FNS`:

```python
TF_MODEL_FNS = {
    # Existing entries...
    "my_aed": (get_my_aed, {}),
}
```

The dictionary supports fixed variants without duplicating builders:

```python
"my_aed_small": (get_my_aed, {"width_multiplier": 0.5}),
"my_aed_large": (get_my_aed, {"width_multiplier": 1.0}),
```

Static values in this dictionary override values assembled from the YAML. This is
the same mechanism ST uses for MiniResNet stack/pooling variants and YAMNet
embedding sizes.

## Configure and train it

Start from `config_file_examples/training_config.yaml` and change the model block:

```yaml
operation_mode: training

model:
  model_name: my_aed
  input_shape: (64, 96, 1)
  pretrained: false

dataset:
  expand_last_dim: true

feature_extraction:
  n_mels: 64
  patch_length: 96
```

Keep the remaining dataset, preprocessing, feature-extraction, augmentation, and
training settings appropriate for the data. From the AED directory run:

```powershell
python stm32ai_main.py --config-path=config_file_examples --config-name=training_config.yaml
```

Hydra may also accept the config name without `.yaml`, depending on the installed
Hydra version. Running `python stm32ai_main.py` uses `user_config.yaml`.

## Architecture parameters from YAML

The current AED adapter does **not** forward arbitrary keys from `model:`. The AED
config parser permits only `framework`, `model_path`, `model_name`, `input_shape`,
`pretrained`, and `model_type`, while `prepare_kwargs_for_model()` forwards a fixed
set of model/dataset/training/feature arguments.

For a fixed family variant, prefer static registry arguments as shown above. To
make a new parameter such as `width_multiplier` user-configurable, both changes are
required:

1. Add `"width_multiplier"` to the model-section `legal` list in
   `audio_event_detection/tf/src/utils/parse_config.py`.
2. Add it to the dictionary returned by
   `audio_event_detection/tf/src/models/model_utils.py`:

```python
"width_multiplier": getattr(cfg.model, "width_multiplier", None),
```

Then declare it in the builder and use it in YAML. Merely adding a builder argument
or a YAML key is insufficient.

## Pretrained backbone option

For transfer learning, follow the MiniResNet/YAMNet pattern:

1. Build or load a backbone with Keras.
2. Replace its original classification head using `add_head()`.
3. Pass `trainable_backbone=cfg.training.fine_tune` (received as `fine_tune`) so
   head-only training freezes the backbone and fine-tuning unfreezes it.
4. Make `pretrained: false` construct the scratch variant, or reject it explicitly
   if the architecture is meaningful only with supplied weights.
5. Keep preprocessing and log-mel geometry identical to that used for the
   pretrained weights.

Bundled weights can be resolved relative to `__file__`, as the existing models do.
Do not silently download weights during a training run if reproducibility or an
offline build matters.

## Custom layers and checkpoint reload

If the architecture contains a custom Keras layer, loss, activation, or other
serialized object, register it in `AED_CUSTOM_OBJECTS` in
`audio_event_detection/tf/src/utils/models_mgt.py`. Otherwise the trainer may fit
successfully and then fail while reloading `best_augmented_model.keras`.

For new layers, also use Keras serialization support (for example
`@tf.keras.utils.register_keras_serializable`) and implement `get_config()` where
needed. Avoid anonymous `Lambda` functions in a model intended for quantization or
deployment.

## Verification checklist

Before a full run:

1. Instantiate through `api.get_model(cfg)`, not only by calling the builder, to
   prove registration and config forwarding work.
2. Confirm `model.input_shape[1:]` equals one dataset element's feature shape.
3. Confirm `model.output_shape[-1]` equals the effective class count.
4. Run one forward pass and a one-step/dry-run training.
5. Confirm `best_augmented_model.keras` reloads and `best_model.keras` is written.
6. Run evaluation, then quantization and STM32 benchmarking before treating the
   architecture as deployable. A valid Keras model is not automatically supported
   or efficient on every STM32 target.

## Known AED limitations relevant to a new model

- Multi-label training currently raises `NotImplementedError` in `get_loss()`.
- The public config currently cannot express arbitrary architecture parameters
  without the two adapter changes described above.
- The checkpoint cleanup assumes the last nested model in the augmentation
  `Sequential` container is the actual classifier. Keep the builder's return value
  as one top-level Keras model.
- Deployment expects a static, generally channels-last input shape and supported
  TensorFlow Lite/STM32 operators. Dynamic axes and exotic TensorFlow operations
  should be avoided.

## Relevant ST files

- `tf/src/models/custom_model.py` — ready-made custom model hook
- `tf/src/models/model_utils.py` — config-to-builder adapter and `add_head()`
- `tf/wrappers/models/standard_models/models.py` — name-to-builder registry
- `common/registries/model_registry.py` — registry implementation
- `api/api.py` — model lookup/load/save behavior
- `tf/src/training/aed_trainer.py` — compile, fit, checkpoint, and reload behavior
- `docs/README_TRAINING.md` — training configuration and output documentation
- `docs/README_MODELS.md` — links to ST's supported AED model families
- `common/training/` — optimizer, freezing, and learning-rate scheduler helpers

