# Arc Fault Detection (AFD) STM32 Model Quantization

This guide explains how to quantize the AFD classification model for STM32 deployment. Quantization reduces Flash/RAM usage and accelerates inference, typically with minimal accuracy loss. The AFD module supports post-training quantization (PTQ) using int8 weights via TFLite converter or ONNX quantizer.

 **Tip:** Read   [README_TRAINING.md](../docs/README_TRAINING.md)  before proceeding.

---

## Table of Contents

<details open><summary><a href="#1"><b>1. Configuration</b></a></summary>
  <ul>
    <li><a href="#operation-mode">Operation Mode</a></li>
    <li><a href="#model-specification">Model Specification</a></li>
    <li><a href="#general-settings">General Settings</a></li>
    <li><a href="#dataset-specification">Dataset Specification</a></li>
    <li><a href="#preprocessing">Preprocessing</a></li>
    <li><a href="#quantization-parameters">Quantization Parameters</a></li>
  </ul>
</details>
<details open><summary><a href="#2"><b>2. Running Quantization</b></a></summary></details>
<details open><summary><a href="#3"><b>3. Results & Artifacts</b></a></summary></details>
<details open><summary><a href="#4"><b>4. MLflow Integration</b></a></summary></details>
<details open><summary><a href="#5"><b>5. Representative Dataset Guidelines</b></a></summary></details>
<details open><summary><a href="#6"><b>6. Accuracy Comparison</b></a></summary></details>
<details open><summary><a href="#7"><b>7. Troubleshooting</b></a></summary></details>

---


<details open><summary><b><a href="#1">1. Configuration</a></b></summary><a id="1"></a>

To update your configuration yaml file, you can start from the configuration used in the training process and update it, so you guarantee using the same preprocessing input shape during quantization. Changing preprocessing or model architecture affects quantization performance.

Each training run stores its Hydra config under:

```
experiments_outputs/<timestamp>/.hydra/
```

Trained models are saved in:

```
experiments_outputs/<timestamp>/saved_models/
```

To quantize:
1. Copy or recreate the run’s parameters in `user_config.yaml`.
2. Set `operation_mode: quantization`.
3. Specify the quantization dataset if needed, and the quantization parameters. 

In the following we provide an example on how to update these parameters.

<details open><summary><a href="#operation-mode">Operation Mode</a></summary><a id="operation-mode"></a>

Set the quantization service in your config:

```yaml
operation_mode: quantization
```

<details open><summary><a href="#model-specification">Model Specification</a></summary><a id="model-specification"></a>

```yaml
model:
  model_path: ../../stm32ai-modelzoo/arc_fault_detection/st_conv/ST_pretrainedmodel_custom_dataset/afd_test_bench_dataset/st_conv_freq_4channels_512/st_conv_freq_4channels_512.keras
  model_name:
  input_shape: (4,512,1)   # (n_channels, seq_len, 1)


```
`model_path` is required for quantization if the run is not chained with training. The reference AFD model expects tensors of shape `(n_channels, seq_len, 1)` corresponding to a multi-channel representation. Keep `input_shape` consistent with the preprocessing and with the model used during training.

</details>

<details open><summary><a href="#general-settings">General Settings</a></summary><a id="general-settings"></a>

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

Hydra output and mlflow root:

```yaml
mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

<details open><summary><a href="#dataset-specification">Dataset Specification</a></summary><a id="dataset-specification"></a>

```yaml
dataset:
  dataset_name: afd_test_bench  # AFD dataset name
  class_names: [normal,arc]
  training_path: ./datasets/afd_test_bench/Arc_and_Normal_train.csv
  validation_path: 
  test_path: 
  quantization_path: ./datasets/afd_test_bench/Arc_and_Normal_train.csv
  prediction_path: 
  test_split:
  validation_split: 
  quantization_split:
  to_cache: True  # Optional, use it to cache the dataset in memory for faster access
  seed: 123

```

The `quantization_path` field specifies the CSV file used as representative dataset for calibration. If it is left empty but `training_path` is defined, a stratified subset of the training set can be drawn according to `quantization_split`. In all cases, ensure that both classes (*normal* and *arc*) are present and well balanced.

</details>

<details open><summary><a href="#preprocessing">Preprocessing</a></summary><a id="preprocessing"></a>
Use identical preprocessing for float and quantized inference. For the reference AFD configuration:

```yaml
preprocessing:
  downsampling: False
  normalization: True
  time_domain: False
```

<details open><summary><a href="#quantization-parameters">Quantization Parameters</a></summary><a id="quantization-parameters"></a>

```yaml
quantization:
  quantizer: TFlite_converter   # onnx_quantizer for ONNX quantization, TFlite_converter for TFLite quantization
  quantization_type: PTQ
  quantization_input_type: int8
  quantization_output_type: int8
  granularity: per_channel            # Optional, defaults to "per_channel". 
  optimize: False                     # Optional, defaults to False.
  target_opset: 17                   # Optional, defaults to 17. Only used for ONNX quantization
  export_dir: quantized_models

```

Notes:
- Weights always quantized to int8.
- Use int8 input/output for a fully integer pipeline on STM32.
- Representative dataset: balanced arc/normal samples (200–1000 typical).

</details>

</details>

<details open><summary><b><a href="#2">2. Running Quantization</a></b></summary><a id="2"></a>

From the AFD root directory:

```bash
python stm32ai_main.py
```
If you chose to update the [quantization_config.yaml](../config_file_examples/quantization_config.yaml) and use it, then run the following command from the AFD folder: 

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name quantization_config.yaml
```


</details>

<details open><summary><b><a href="#3">3. Results & Artifacts</a></b></summary><a id="3"></a>

Hydra output directory:

```
./tf/src/experiments_outputs/<timestamp>/
  saved_models/
  quantized_models/
  logs/
  .hydra/
```

Artifacts:
- `quantized_models/model_quantized.tflite`
- Logs (MLflow, TensorBoard)
- Original float model for comparison

</details>

<details open><summary><b><a href="#4">4. MLflow Integration</a></b></summary><a id="4"></a>

To inspect and compare runs:

```bash
cd tf/src/experiments_outputs
mlflow ui
```

Compare float vs quantized metrics (Accuracy, Precision, Recall, F1, size).

</details>

<details open><summary><b><a href="#5">5. Representative Dataset Guidelines</a></b></summary><a id="5"></a>

- Use >100 samples
- Ensure class balance (arc / normal)
- Cover amplitude and waveform diversity
- Avoid test leakage

If the F1 score of the quantized model drops by more than approximately 3 percentage points compared to the float (original) model, increase the representative set size or verify preprocessing consistency.

</details>

<details open><summary><b><a href="#6">6. Accuracy Comparison</a></b></summary><a id="6"></a>

1. Evaluate float model (`operation_mode: evaluation`)
2. Quantize
3. Evaluate quantized model (load `.tflite`)
4. Record metrics and model size

</details>

<details open><summary><b><a href="#7">7. Troubleshooting</a></b></summary><a id="7"></a>

| Issue                  | Cause                              | Action                                 |
|------------------------|------------------------------------|----------------------------------------|
| Large F1 drop          | Weak representative diversity      | Add samples, balance classes           |
| Converter error        | Unsupported op                     | Simplify architecture (Conv/Relu/Dense)|
| MCU vs PC mismatch     | Different normalization            | Mirror scaling constants on device     |
| Always one class       | Bad calibration / saturated int8   | Check input dtype & scaling            |

</details>
