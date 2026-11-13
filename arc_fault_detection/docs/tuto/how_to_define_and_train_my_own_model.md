# Define and train your own model (AFD)

This tutorial walks through the complete path from a custom TensorFlow model to a successful training run: define your network, register it, configure the YAML, launch training, and review the results.

A key requirement is that your model’s input_shape matches the data structure produced by the AFD loader and preprocessing. The second section explains in detail how that structure is built.

## 1) Define your model

In the AFD project, TensorFlow models live under `arc_fault_detection/tf/src/models/`.
Create a new file (example: `arc_fault_detection/tf/src/models/my_model.py`) and define a model-building function.

Here’s a clean example you can adapt:

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation, Flatten


def get_my_model(input_shape=(4, 256, 1), num_classes=2):
    inputs = keras.Input(shape=input_shape)
    # reshape to (n_channels, seq_len)
    x = layers.Reshape((input_shape[0], input_shape[1]))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, kernel_initializer='random_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.05)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="st_dense")
    return model
```

### Register the model

Make it discoverable by the pipeline:

1) Import it in `arc_fault_detection/tf/src/models/__init__.py`.
2) Register it in `arc_fault_detection/tf/wrappers/models/custom_models/models.py` by adding it to `TF_MODEL_FNS`.

```python
from arc_fault_detection.tf.src.models.my_model import get_my_model

TF_MODEL_FNS = {
  # existing models...
  "my_model": get_my_model,
}
```

## 2) Understand the input shape and data structure

The AFD loader **does not** produce a flat vector. It reshapes the CSV into a **multi-channel** tensor, then optionally applies downsampling, FFT, and normalization.

### The required input shape

Your `model.input_shape` in the YAML must be `(n_channels, seq_len, 1)` for Dense-style and Conv-style models

The loader will always reshape to **`(batch, n_channels, seq_len, 1)`** before feeding the model, so a single integer like `(2048,)` is **not valid** for AFD.

### How `seq_len` is built

The raw CSV width must be **large enough** for the preprocessing pipeline:

- If `time_domain: True`, the pipeline keeps `seq_len` samples per channel.
- If `time_domain: False`, the pipeline takes an FFT and keeps `seq_len` frequency bins. That requires **`2 x seq_len` raw samples** per channel before the FFT.

### What about downsampling?

If `downsampling: True`, the loader can downsample the CSV width to the target length. But it **still requires the raw width to be at least the target length**. In other words:

- Time domain: raw width must be `>= seq_len`
- Frequency domain: raw width must be `>= 2 x seq_len`

If the CSV width is smaller than required, data loading will fail.

### Channel grouping matters

The loader groups rows into blocks of `n_channels`. That means:

- Your total number of rows must be divisible by `n_channels`.
- If not, **extra rows are truncated** (with a warning).

If you see unexpected data loss, this is often the reason.

## 3) Update your YAML configuration

Set `model_name` to your registered model and choose an `input_shape` that matches your preprocessing settings.

Example for a **time-domain** model with 4 channels and 256 samples per channel:

```yaml
# user_config.yaml

general:
  project_name: afd
  logs_dir: logs
  saved_models_dir: saved_models
  display_figures: True
  global_seed: 123
  gpu_memory_limit: 4
  deterministic_ops: True

model:
  model_name: my_model
  input_shape: (4, 256, 1)

operation_mode: training
# choices=['benchmarking', 'evaluation', 'training', 'quantization', 'prediction',
#          'chain_tb','chain_tbqeb','chain_tqe','chain_eqe','chain_qb','chain_eqeb']

dataset:
  dataset_name: afd_test_bench
  class_names: [normal, arc]
  training_path: ./datasets/afd_test_bench/Arc_and_Normal_train.csv
  validation_path: ./datasets/afd_test_bench/Arc_and_Normal_val.csv
  test_split: 0.2
  to_cache: True
  seed: 123

preprocessing:
  downsampling: False
  normalization: True
  time_domain: True

training:
  initialize_parameters_from:
  batch_size: 256
  epochs: 5
  optimizer:
    Adam:
      learning_rate: 0.001
  callbacks:
    ReduceLROnPlateau:
      monitor: val_accuracy
      mode: max
      factor: 0.5
      patience: 40
      min_lr: 1.0e-05
    EarlyStopping:
      monitor: val_accuracy
      mode: max
      restore_best_weights: true
      patience: 60

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

If you want to use **frequency-domain** features, set `time_domain: False` and make sure your CSV width supports `2 x seq_len`.

## 4) Train the model

Run from the AFD root (the folder that contains `stm32ai_main.py`):

```powershell
python stm32ai_main.py
```

Or use a specific YAML file:

```powershell
# Replace 'path_to_yaml_folder' with the directory containing your YAML file,
# and 'name_of_your_yaml_file.yaml' with the actual YAML filename
python stm32ai_main.py --config-path=path_to_yaml_folder --config-name=name_of_your_yaml_file.yaml
```

## 5) Review the results

After training completes, you can review:

- **Console logs**: training and validation metrics printed during the run.
- **Saved models**: the trained model is stored under the folder specified by `saved_models_dir` in your YAML.
- **Experiment outputs**: run artifacts are created under the `hydra.run.dir` path (by default in `tf/src/experiments_outputs/`).
- **MLflow tracking** : runs are logged to the `mlflow.uri` directory so you can browse metrics and parameters.

Tip: if you want consistent comparisons, keep `global_seed` and preprocessing settings identical between runs.

## 6) Need more details?

- [Training documentation](../README_TRAINING.md)
- [Benchmark documentation](../README_BENCHMARKING.md)
- Examples for different operation modes are available in [../../config_file_examples](../../config_file_examples)