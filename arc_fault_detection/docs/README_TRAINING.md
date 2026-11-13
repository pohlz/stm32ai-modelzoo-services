# Arc Fault Detection (AFD) STM32 model training

This document describes a reproducible training workflow for the Arc Fault Detection (AFD) use case, which consists of binary classification between electrical *normal* and *arc* conditions.

The workflow begins by ensuring that the dataset follows the required format and structure. Next, the YAML configuration file is checked to confirm that it includes all necessary parameters for training. Once these prerequisites are met, the Python training script is launched with the appropriate arguments. After training, the results can be evaluated using various tools, which are introduced at the end of this tutorial.

## Table of Contents

<details open><summary><a href="#1"><b>1. Prepare the dataset</b></a></summary></details>
<details open><summary><a href="#2"><b>2. Create your configuration file</b></a></summary>
  <ul>
    <li><a href="#2-1">2.1 Overview</a></li>
    <li><a href="#2-2">2.2 General settings</a></li>
    <li><a href="#2-3">2.3 Model section settings</a></li>
    <li><a href="#2-4">2.4 Dataset specification</a></li>
    <li><a href="#2-5">2.5 Dataset preprocessing</a></li>
    <li><a href="#2-6">2.6 Training parameters</a></li>
    <li><a href="#2-7">2.7 Hydra and MLflow settings</a></li>
  </ul>
</details>
<details open><summary><a href="#3"><b>3. Train your model</b></a></summary></details>
<details open><summary><a href="#4"><b>4. Model evaluation</b></a></summary></details>
<details open><summary><a href="#5"><b>5. Visualize training results</b></a></summary></details>
<details open><summary><a href="#6"><b>6. Run TensorBoard</b></a></summary></details>
<details open><summary><a href="#7"><b>7. Run MLflow</b></a></summary></details>

---

<details open><summary><a href="#1"><b>1. Prepare the dataset</b></a></summary><a id="1"></a>

The AFD dataset is provided as CSV files where each file has the following format:
- Columns 0..N-2 = time series samples
- Column N-1 = label (0 = normal, 1 = arc)

Typical files are:
- Arc_and_Normal_train.csv
- Arc_and_Normal_val.csv (optional; if not provided, a validation split is created according to `validation_split`)
- Arc_and_Normal_test.csv (optional; if not provided, a test split is created according to `test_split`)

All dataset files are expected under `./datasets/`.

Basic sanity checks:
- No NaN/Inf
- Correct label set {0,1}
- Consistent sampling length. If `preprocessing.downsampling` is enabled, samples are downsampled to a target length derived from `model.input_shape` (target length = `model.input_shape[1]`). If `preprocessing.time_domain` is disabled (frequency-domain pipeline), the raw waveform length must be at least `2 * model.input_shape[1]` before the FFT step.

</details>

<details open><summary><a href="#2"><b>2. Create your configuration file</b></a></summary><a id="2"></a>
<details open><summary><a href="#2-1">2.1 Overview</a></summary><a id="2-1"></a>

Training is driven by a YAML configuration file. A ready-to-use example is provided as `training_config.yaml` in `config_file_examples/`. The main sections are:

- `general`, describes your project, including project name, directory where to save models, etc.
- `model`, provides information about the model, like the model path and the model input shape 
- `operation_mode`, a string describing the operation mode of the model zoo. It is set to "training" for this tutorial.
- `dataset`, describes the used dataset, including directory paths, class names, etc.
- `preprocessing`, parameters used to perform some processing on the waveform data
- `training`, specifies the training setup, including batch size, number of epochs, optimizer, callbacks, etc.
- `mlflow`, specifies the folder to save MLflow logs.
- `hydra`, specifies the folder to save Hydra logs.

In the following, we give a brief description of each of these sections.

</details>

<details open><summary><a href="#2-2">2.2 General settings</a></summary><a id="2-2"></a>

The first section of the configuration file is the `general` section that provides information about the project.

```yaml
general:
  project_name: afd
  logs_dir: logs
  saved_models_dir: saved_models
  display_figures: True
  global_seed: 123
  gpu_memory_limit: 4
  deterministic_ops: True

```
- `project_name`: experiment name used to track runs in MLflow.
- `logs_dir`: output subdirectory containing logs (TensorBoard traces, CSV metrics, etc.).
- `saved_models_dir`: output subdirectory containing exported models (best and last-epoch checkpoints).
- `display_figures`: whether to display figures at the end of the run (training curves, confusion matrices, etc.).
- `global_seed`: random seed used to improve reproducibility.
- `gpu_memory_limit`: maximum GPU memory allocated to the process (in GB).
- `deterministic_ops`: enable deterministic TensorFlow operations when available.

The `logs_dir` attribute is the name of the directory where the MLflow and TensorBoard files are saved. The `saved_models_dir` attribute is the name of the directory where models are saved, which includes the trained model. Both `logs_dir` and `saved_models_dir` are typically located under the experiment output directory (e.g., `./tf/src/experiments_outputs/<date_time_of_your_run>/`), while the `.hydra` directory contains configuration logs for the run.

</details>

<details open><summary><a href="#2-3">2.3 Model section settings</a></summary><a id="2-3"></a>

The `model` section specifies which neural network architecture is used and what input tensor shape it expects.

```yaml
model:
  #model_path: ../../stm32ai-modelzoo/arc_fault_detection/st_conv/ST_pretrainedmodel_custom_dataset/afd_test_bench_dataset/st_conv_freq_4channels_512/st_conv_freq_4channels_512.keras
  model_name: st_conv
  input_shape: (4,512,1)
```

- `model_path`: (Optional) Path to a pretrained model file (for example a `.keras` file) to fine-tune.
- `model_name`: Name of a registered architecture (for AFD TensorFlow models, common values are `st_conv` and `st_dense`).
- `input_shape`: Expected input tensor shape. For the AFD reference pipeline, inputs are shaped as `(n_channels, seq_len, 1)`.

**Usage notes:**
- Use `model_path` if you want to load and fine-tune an existing model.
- Use `model_name` if you want to train a model from scratch using a predefined architecture.
- Ensure that `input_shape` matches the shape of your preprocessed data.

This section allows you to flexibly choose between training a new model or leveraging an existing one, depending on your project requirements.

</details>

<details open><summary><a href="#2-4">2.4 Dataset specification</a></summary><a id="2-4"></a>

Dataset paths and split strategy are defined in the `dataset` section. Below is the reference configuration used by `training_config.yaml`:

```yaml
dataset:
  dataset_name: afd_test_bench  # AFD dataset name
  class_names: [normal,arc]
  training_path: ./datasets/afd_test_bench/Arc_and_Normal_train.csv
  validation_path: ./datasets/afd_test_bench/Arc_and_Normal_val.csv
  test_path: ./datasets/afd_test_bench/Arc_and_Normal_test.csv
  quantization_path:
  prediction_path: 
  test_split:
  validation_split:
  quantization_split:
  to_cache: True  # Optional, use it to cache the dataset in memory for faster access
  seed: 123
```
- `dataset_name`: Name of the dataset.
- `class_names`: List of class labels, e.g., [normal, arc].
- `training_path`: Path to the training CSV file.
- `validation_path`: Path to the validation CSV file (optional, used if provided).
- `test_path`: Path to the test CSV file (optional, used if provided).
- `quantization_path`: Path to the quantization CSV file (optional).
- `test_split`: Fraction of data to use for testing (float between 0 and 1).
- `validation_split`: Fraction of data to use for validation if no validation file is provided.
- `quantization_split`: Fraction of data to use for quantization.
- `to_cache`: Boolean flag to cache the dataset in memory for faster access.
- `seed`: Integer seed for reproducible data splits.

Split precedence during training is:
- if `validation_path` is provided, it is used; otherwise, `validation_split` is applied to the training set;
- if `test_path` is provided, it is used; otherwise, `test_split` is applied to the remaining training set.

In other words, explicit `*_path` values take priority over `*_split` values.

</details>

<details open><summary><a href="#2-5">2.5 Dataset preprocessing</a></summary><a id="2-5"></a>

Each CSV row is interpreted as a waveform sample. Before feeding data to the network, the loader applies preprocessing so that tensors match `model.input_shape`.
For the AFD reference pipeline, preprocessing consists of:
- Optional downsampling to a target length  
  `target_length` = `model.input_shape[1]` in the time domain, and  
  `target_length` = 2 x `model.input_shape[1]` in the frequency domain,
- Optional conversion to the frequency domain via FFT magnitude (enabled when `time_domain: False`) and keeping only `model.input_shape[1]` frequency bins.
- Optional normalization (recommended for training stability).

Example YAML configuration:
```yaml
preprocessing:
  downsampling: True
  normalization: True
  time_domain: False
```
- `downsampling`: If `True`, the input waveform is downsampled to the target length.
- `time_domain`: If `False`, the pipeline computes the magnitude of the FFT and keeps `seq_len` frequency bins; if `True`, it stays in time domain.
- `normalization`: If `True`, normalization parameters are estimated from the training set and applied consistently across train/val/test.

</details>

<details open><summary><a href="#2-6">2.6 Training parameters</a></summary><a id="2-6"></a>

The `training` section of the configuration file specifies the parameters for the training process, including batch size, number of epochs, optimizer, callbacks, and model saving options. Here after, an example configuration snippet for the `training` section (AFD classification):

```yaml
training:
  initialize_parameters_from:
  batch_size: 256
  epochs: 40
  optimizer:
    Adam:
      learning_rate: 0.01
  callbacks:          # Optional section
    ReduceLROnPlateau:
      monitor: val_accuracy
      mode: max
      factor: 0.5
      patience: 10
      min_lr: 1.0e-05
    EarlyStopping:
      monitor: val_accuracy
      mode: max
      restore_best_weights: true
      patience: 70
```

- `initialize_parameters_from`: (Optional) Path to a previously saved model to be used to initialize the parameters of the loaded or created model. 
- `batch_size`: Number of samples per gradient update.
- `epochs`: Number of training epochs.
- `optimizer`: Optimizer configuration (e.g., Adam with a specified learning rate).
- `callbacks`: (Optional) TensorFlow callbacks for training control, such as learning rate scheduling and early stopping.

`initialize_parameters_from` is a convenient way to warm-start training in the Model Zoo. It initializes the loaded model with trained weights (even if the number of channels differs), as long as the architecture is compatible (same backbone and layer names). If a layer shape does not match, the initialization with the given parameters is skipped and either the original loaded parameters or a random initialization are considered. Provide a `.keras` or `.h5` checkpoint path.

All TensorFlow optimizers can be used in the `optimizer` subsection. All TensorFlow callbacks can be used in the `callbacks` subsection, except for `ModelCheckpoint`, `TensorBoard`, and `CSVLogger`, which are built-in and cannot be redefined.

Refer to the [learning rate schedulers README](../../common/training/lr_schedulers_README.md) for more information on available callbacks and utilities.

</details>

<details open><summary><a href="#2-7">2.7 Hydra and MLflow settings</a></summary><a id="2-7"></a>

The `mlflow` and `hydra` sections must always be present in the YAML configuration file. The `hydra` section can be used to specify the name of the directory where experiment directories are saved and/or the pattern used to name experiment directories. With the YAML code below, every time you run the Model Zoo, an experiment directory is created that contains all the directories and files created during the run. The names of experiment directories are all unique as they are based on the date and time of the run.

```yaml
hydra:
   run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

The `mlflow` section is used to specify the location and name of the directory where MLflow files are saved, as shown below:

```yaml
mlflow:
   uri: ./tf/src/experiments_outputs/mlruns
```

</details>
</details>

<details open><summary><a href="#3"><b>3. Train your model</b></a></summary><a id="3"></a>

To launch your model training using a real dataset, run the following command from the AFD folder:

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config.yaml
```
The trained model can be found under `./tf/src/experiments_outputs/` (in the run's `saved_models/` folder).

</details>

<details open><summary><a href="#4"><b>4. Model evaluation</b></a></summary><a id="4"></a>

After training completes, the best checkpoint is evaluated on the validation set and (if configured) on the test set.
If `general.display_figures` is set to `True`, the run also produces and/or displays confusion matrices and training curves.

</details>

<details open><summary><a href="#5"><b>5. Visualize training results</b></a></summary><a id="5"></a>

All training artifacts, figures, and models are saved under the output directory specified in the config file, for example:

```yaml
hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```
By default, the output directory is `./tf/src/experiments_outputs/<date_time_of_your_run>/`. Note that this directory will not exist until you run the model zoo at least once.

This directory typically contains:
- `.hydra/`: Hydra configuration logs
- `logs/`: TensorBoard logs and `train_metrics.csv`
- `saved_models/`: best and last-epoch model checkpoints
- `stm32ai_main.log`: run log
- exported plots (training curves and confusion matrices)

</details>

<details open><summary><a href="#6"><b>6. Run TensorBoard</b></a></summary><a id="6"></a>

To visualize the training curves logged by TensorBoard, go to the output directory (by default, `./tf/src/experiments_outputs/<date_time_of_your_run>/`) and run:

```bash
cd tf/src/experiments_outputs/<date_time_of_your_run>/
tensorboard --logdir logs
```

And open the URL `http://localhost:6006` in your browser.

</details>

<details open><summary><a href="#7"><b>7. Run MLflow</b></a></summary><a id="7"></a>

MLflow logs parameters, metrics, and artifacts for each run and provides a web UI to compare experiments.
To browse results across runs, start the MLflow UI:
```bash
cd tf/src/experiments_outputs
mlflow ui
```
And open the given IP address in your browser.

</details>
