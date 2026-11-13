# Arc Fault Detection STM32 model zoo

This Arc Fault Detection (AFD) model zoo provides a reproducible workflow to train, evaluate, quantize and benchmark classifiers that separate **normal** from **arc** events. The services are driven by a single YAML configuration file, allowing to run experiments with consistent preprocessing and traceable artifacts.

Minimal YAML examples are available in [config_file_examples](../config_file_examples/). Pretrained AFD models and their training configs are available in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/), which are excellent starting points for new experiments.

## Table of Contents

<details open><summary><a href="#1"><b>1. Model Zoo Overview</b></a></summary>
	<ul>
		<li><a href="#1-1">1.1 YAML configuration file</a></li>
		<li><a href="#1-2">1.2 Output directory structure</a></li>
		<li><a href="#1-3">1.3 MLflow run</a></li>
	</ul>
</details>
<details open><summary><a href="#2"><b>2. Quickstart tutorial</b></a></summary>
	<ul>
		<li><a href="#2-1">2.1 Choose the operation mode</a></li>
		<li><a href="#2-2">2.2 Minimal YAML example</a></li>
		<li><a href="#2-3">2.3 Run the service</a></li>
	</ul>
</details>
<details open><summary><a href="#3"><b>3. Supported dataset format</b></a></summary></details>
<details open><summary><a href="#4"><b>4. Configuration file</b></a></summary>
	<ul>
		<li><a href="#4-1">4.1 YAML syntax extensions</a></li>
		<li><a href="#4-2">4.2 Operation mode</a></li>
		<li><a href="#4-3">4.3 Top-level sections</a></li>
		<li><a href="#4-4">4.4 Global settings and model path</a></li>
		<li><a href="#4-5">4.5 Datasets</a></li>
		<li><a href="#4-6">4.6 Preprocessing</a></li>
		<li><a href="#4-7">4.7 Training</a></li>
		<li><a href="#4-8">4.8 Quantization</a></li>
		<li><a href="#4-9">4.9 Evaluation</a></li>
		<li><a href="#4-10">4.10 Prediction</a></li>
		<li><a href="#4-11">4.11 Benchmarking</a></li>
		<li><a href="#4-12">4.12 Chained modes</a></li>
		<li><a href="#4-13">4.13 Learning rate schedule</a></li>
	</ul>
</details>
<details open><summary><a href="#5"><b>5. Visualize results</b></a></summary>
	<ul>
		<li><a href="#5-1">5.1 Saved artifacts</a></li>
		<li><a href="#5-2">5.2 Run TensorBoard</a></li>
		<li><a href="#5-3">5.3 Run MLflow</a></li>
	</ul>
</details>

<details open><summary><a href="#1"><b>1. Model Zoo Overview</b></a></summary><a id="1"></a>
<ul><details open><summary><a href="#1-1">1.1 YAML configuration file</a></summary><a id="1-1"></a>

The AFD workflow is configured from [user_config.yaml](../user_config.yaml). You can also start from minimal service configs under [config_file_examples](../config_file_examples/) and progressively customize them.

For detailed service guidance, see the linked docs in the configuration sections below.

</details></ul>
<ul><details open><summary><a href="#1-2">1.2 Output directory structure</a></summary><a id="1-2"></a>

By default, outputs are written to `tf/src/experiments_outputs/<timestamp>/`. Each run typically contains:
- `.hydra/` (Hydra logs and resolved configs)
- `logs/` (TensorBoard traces and CSV metrics)
- `saved_models/` (best and last checkpoints for float models for services that generate models)
- `quantized_models/` (quantized outputs, when applicable)
- `stm32ai_main.log` (run log)

</details></ul>
<ul><details open><summary><a href="#1-3">1.3 MLflow run</a></summary><a id="1-3"></a>

MLflow logs are stored under `tf/src/experiments_outputs/mlruns/`. Launch the UI from the experiments_outputs root:

```bash
mlflow ui
```

</details></ul>
</details>

<details open><summary><a href="#2"><b>2. Quickstart tutorial</b></a></summary><a id="2"></a>

This short tutorial demonstrates a minimal training run using the example dataset shipped with the repository. It is deliberately compact; for advanced settings, refer to the service‑specific READMEs linked in section <a href="#4">4</a>.

<ul><details open><summary><a href="#2-1">2.1 Choose the operation mode</a></summary><a id="2-1"></a>

Set the `operation_mode` to `training` to learn a baseline model. You can combine many services by using chained modes which are covered in [README_CHAINED_MODES.md](README_CHAINED_MODES.md).

```yaml
operation_mode: training
```

</details></ul>
<ul><details open><summary><a href="#2-2">2.2 Minimal YAML example</a></summary><a id="2-2"></a>

Below is a compact configuration based on the default training example. Update the dataset paths if you place your CSVs elsewhere.

```yaml
model:
  model_name: st_conv
  input_shape: (4,512,1)

operation_mode: training

dataset:
  dataset_name: afd_test_bench  # AFD dataset name
  class_names: [normal,arc]
  training_path: ./datasets/afd_test_bench/Arc_and_Normal_train.csv
  validation_path: ./datasets/afd_test_bench/Arc_and_Normal_val.csv
  test_path: ./datasets/afd_test_bench/Arc_and_Normal_test.csv
  seed: 123

preprocessing:
  downsampling: True
  normalization: True
  time_domain: False

training:
  initialize_parameters_from:
  batch_size: 256
  epochs: 4
  optimizer:
    Adam:
      learning_rate: 0.01

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

</details></ul>
<ul><details open><summary><a href="#2-3">2.3 Run the service</a></summary><a id="2-3"></a>

Run the main entry point from the AFD root folder:

```bash
python stm32ai_main.py
```

</details></ul>
<ul><details open><summary><a href="#2-4">2.4 Expected outputs</a></summary><a id="2-4"></a>

The run creates a time‑stamped experiment directory under `tf/src/experiments_outputs/` containing the trained model, metrics, and logs. Section <a href="#5">5</a> describes how to visualize these results.

</details></ul>

</details>

<details open><summary><a href="#3"><b>3. Supported dataset format</b></a></summary><a id="3"></a>

AFD datasets are CSV‑based. Each row is a sample represented by fixed‑length features, with the last column containing the label (`0 = normal`, `1 = arc`). CSVs can be provided directly or packaged in a `.zip` archive.

**Minimal CSV example:**

```csv
feature_1,feature_2,feature_3,feature_4,label
0.12,0.48,0.33,0.95,0
0.87,0.05,0.66,0.14,1
```

**Typical dataset layout:**

```
datasets/
  afd_test_bench/
    Arc_and_Normal_train.csv
    Arc_and_Normal_val.csv
    Arc_and_Normal_test.csv
```

See [README_DATASETS.md](README_DATASETS.md) for full details:
- Train/validation/test split strategy and automatic split options.
- Preprocessing consistency requirements across services.

</details>

<details open><summary><a href="#4"><b>4. Configuration file</b></a></summary><a id="4"></a>
<ul><details open><summary><a href="#4-1">4.1 YAML syntax extensions</a></summary><a id="4-1"></a>

The model zoo supports standard YAML plus a few extensions:

```yaml
# Equivalent syntaxes for attributes with no value
attribute_1:
attribute_2: null
attribute_2: None

# Equivalent syntaxes for boolean values
attribute_1: true
attribute_2: false
attribute_3: True
attribute_4: False

# Syntax for environment variables
model_path: ${PROJECT_ROOT}/models/mymodel.keras
```

</details></ul>
<ul><details open><summary><a href="#4-2">4.2 Operation mode</a></summary><a id="4-2"></a>

The `operation_mode` attribute specifies which service(s) to run. Supported values are:

| `operation_mode` attribute | Operations |
|:---------------------------|:-----------|
| `training`     | Train a model |
| `evaluation`   | Evaluate a float or quantized model |
| `quantization` | Quantize a float model |
| `prediction`   | Predict labels for CSV data |
| `benchmarking` | Benchmark a float or quantized model on an STM32 board |
| `chain_tb`     | Training → Benchmarking |
| `chain_tbqeb`  | Training → Benchmarking → Quantization → Evaluation → Benchmarking |
| `chain_tqe`    | Training → Quantization → Evaluation |
| `chain_eqe`    | Evaluation → Quantization → Evaluation |
| `chain_qb`     | Quantization → Benchmarking |
| `chain_eqeb`   | Evaluation → Quantization → Evaluation → Benchmarking |

See [README_CHAINED_MODES.md](README_CHAINED_MODES.md) for details:
- Required config sections per chain mode.
- End-to-end examples and run commands.

</details></ul>
<ul><details open><summary><a href="#4-3">4.3 Top-level sections</a></summary><a id="4-3"></a>

Common top-level sections include:

| Attribute name | Mandatory | Usage |
|:---------|:------|:-----|
| `general` | no | Project name, output directories, reproducibility settings |
| `model` | yes | Model path/name and input shape |
| `operation_mode` | yes | operation mode to be executed |
| `dataset` | depends on operation mode | CSV paths, class names, split controls |
| `preprocessing` | depends on operation mode  | Time/frequency-domain pipeline options |
| `training` | depends on operation mode  | Training hyperparameters and callbacks |
| `quantization` | depends on operation mode  | PTQ settings and representative data |
| `benchmarking` | depends on operation mode  | Board selection and benchmarking settings |
| `mlflow` | yes | MLflow tracking configuration |
| `hydra` | yes | Output directory naming |

</details></ul>
<ul><details open><summary><a href="#4-4">4.4 Global settings and model path</a></summary><a id="4-4"></a>

The `general` and `model` sections define run metadata and the model definition. See [README_TRAINING.md](README_TRAINING.md) for:
- Required fields for `general` (logging, seeds, GPU limits).
- `model_path` vs. `model_name` usage and `input_shape` expectations.

</details></ul>
<ul><details open><summary><a href="#4-5">4.5 Datasets</a></summary><a id="4-5"></a>

See [README_DATASETS.md](README_DATASETS.md):
- CSV schema and label conventions.
- Splits, representative sets, and quality checks.

</details></ul>
<ul><details open><summary><a href="#4-6">4.6 Preprocessing</a></summary><a id="4-6"></a>

Preprocessing must be consistent across services (training, evaluation, quantization, prediction, benchmarking). This is essential for fair comparisons because the classifier is sensitive to shifts in feature distributions. See [README_TRAINING.md](README_TRAINING.md) for the reference preprocessing settings and how they map to `input_shape`.

</details></ul>
<ul><details open><summary><a href="#4-7">4.7 Training</a></summary><a id="4-7"></a>

See [README_TRAINING.md](README_TRAINING.md):
- End-to-end training flow for the arc/normal classifier.
- YAML sections for `training`, callbacks, and output artifacts.
- TensorBoard and MLflow usage.

</details></ul>
<ul><details open><summary><a href="#4-8">4.8 Quantization</a></summary><a id="4-8"></a>

See [README_QUANTIZATION.md](README_QUANTIZATION.md):
- PTQ workflow with TFLite converter or ONNX quantizer.
- Representative dataset guidance.
- Accuracy comparison and troubleshooting.

</details></ul>
<ul><details open><summary><a href="#4-9">4.9 Evaluation</a></summary><a id="4-9"></a>

See [README_EVALUATION.md](README_EVALUATION.md):
- Float vs. quantized evaluation setup.
- Required dataset fields and preprocessing alignment.
- Confusion matrices and result visualization.

</details></ul>
<ul><details open><summary><a href="#4-10">4.10 Prediction</a></summary><a id="4-10"></a>

See [README_PREDICTION.md](README_PREDICTION.md):
- Prediction config fields and example CSVs.
- Output probabilities and class mapping.

</details></ul>
<ul><details open><summary><a href="#4-11">4.11 Benchmarking</a></summary><a id="4-11"></a>

See [README_BENCHMARKING.md](README_BENCHMARKING.md):
- STEdgeAI Developper Cloud vs. local benchmarking.
- Supported model types and board selection.
- MLflow logging and artifacts.

</details></ul>
<ul><details open><summary><a href="#4-12">4.12 Chained modes</a></summary><a id="4-12"></a>

See [README_CHAINED_MODES.md](README_CHAINED_MODES.md):
- Supported chain modes and their required sections.
- Single-command execution with example configs.

</details></ul>
<ul><details open><summary><a href="#4-13">4.13 Learning rate schedule</a></summary><a id="4-13"></a>

See [README_LR_SCHEDULE.md](README_LR_SCHEDULE.md):
- Built-in schedulers and Keras callbacks.
- Plotting a schedule before training.

</details></ul>
</details>

<details open><summary><a href="#5"><b>5. Visualize results</b></a></summary><a id="5"></a>

AFD produces standard artifacts (metrics, logs, and model checkpoints) that support careful model inspection. These files live under each experiment directory in `tf/src/experiments_outputs/<timestamp>/`.

<ul><details open><summary><a href="#5-1">5.1 Saved artifacts</a></summary><a id="5-1"></a>

Typical outputs include:
- `saved_models/` with the best and last float checkpoints.
- `quantized_models/` with quantized outputs (if quantization was run).
- `logs/` with TensorBoard traces and CSV metrics.
- `stm32ai_main.log` with a detailed run log.

</details></ul>
<ul><details open><summary><a href="#5-2">5.2 Run TensorBoard</a></summary><a id="5-2"></a>

Navigate to your experiment directory and run:

```bash
tensorboard --logdir logs
```

</details></ul>
<ul><details open><summary><a href="#5-3">5.3 Run MLflow</a></summary><a id="5-3"></a>

From the experiments_outputs root, launch the MLflow UI:

```bash
mlflow ui
```

</details></ul>

</details>
