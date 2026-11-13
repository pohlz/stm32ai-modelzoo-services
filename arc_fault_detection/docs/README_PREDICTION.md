# Prediction with AFD model

This document provides details on how to use a pretrained AFD model for prediction (inference) on new data.
The prediction service enables users to apply their trained model to new samples and obtain predicted classes and probabilities. The inputs are the pretrained model and the data to predict on.

The details on how to use this service are provided below.

## Table of Contents

<details open><summary><a href="#configure-the-yaml-file"><b>1. Configure the YAML file</b></a></summary>
  <ul>
    <li><a href="#general-settings">1.1 General settings</a></li>
    <li><a href="#setting-the-model-and-the-operation-mode">1.2 Setting the model and the operation mode</a></li>
    <li><a href="#prepare-the-dataset">1.3 Prepare the dataset</a></li>
    <li><a href="#apply-preprocessing">1.4 Apply preprocessing</a></li>
    <li><a href="#mlflow-and-hydra">1.5 MLflow and Hydra</a></li>
  </ul>
</details>
<details open><summary><a href="#run-prediction"><b>2. Run prediction</b></a></summary></details>
<details open><summary><a href="#visualize-the-prediction-results"><b>3. Get the prediction results</b></a></summary></details>

---

<details open><summary><a href="#configure-the-yaml-file"><b>1. Configure the YAML file</b></a></summary><a id="configure-the-yaml-file"></a>

To use the prediction service, users can edit the parameters provided in the main [user_config.yaml](../user_config.yaml) file, or alternatively directly update the configuration file provided for the prediction service [prediction_config.yaml](../config_file_examples/prediction_config.yaml).

Below is a breakdown of the main sections in the updated example config file:

<ul><details open><summary><a href="#general-settings">1.1 General settings</a></summary><a id="general-settings"></a>

General settings for the project, logging, and reproducibility:

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

</details></ul>
<ul><details open><summary><a href="#setting-the-model-and-the-operation-mode">1.2 Setting the model and the operation mode</a></summary><a id="setting-the-model-and-the-operation-mode"></a>

Set the path to the pretrained model and the operation mode:

```yaml
model:
  model_path: ../../stm32ai-modelzoo/arc_fault_detection/st_conv/ST_pretrainedmodel_custom_dataset/afd_test_bench_dataset/st_conv_freq_4channels_512/st_conv_freq_4channels_512_int8.tflite
  model_name:
  input_shape: (4,512,1)   # (n_channels, seq_len, 1)

operation_mode: prediction
```

</details></ul>
<ul><details open><summary><a href="#prepare-the-dataset">1.3 Prepare the dataset</a></summary><a id="prepare-the-dataset"></a>

Specify the path to the CSV file containing the data for prediction using `prediction_path`, and ensure that `class_names` matches the classes used during training:

```yaml
dataset:
  dataset_name: afd_test_bench  # AFD dataset name
  class_names: [normal,arc]
  training_path: ./datasets/afd_test_bench/Arc_and_Normal_train.csv
  validation_path: 
  test_path: 
  quantization_path: 
  prediction_path: ./datasets/afd_test_bench/20Arc_20Normal_predict.csv
  test_split:
  validation_split: 
  quantization_split:
  to_cache: True  # Optional, use it to cache the dataset in memory for faster access
  seed: 123
```

</details></ul>
<ul><details open><summary><a href="#apply-preprocessing">1.4 Apply preprocessing</a></summary><a id="apply-preprocessing"></a>

Preprocessing options for the input data. Ensure that you use the same preprocessing as in the training:

```yaml
preprocessing:
  downsampling: True
  normalization: True
  time_domain: False
```

The `downsampling` parameter controls sequence downsampling to match the model input length. The `normalization` parameter enables standard normalization of the frame. With `time_domain: False`, the pipeline applies an FFT-based frequency-domain transformation consistent with training and quantization.

</details></ul>
<ul><details open><summary><a href="#mlflow-and-hydra">1.5 MLflow and Hydra</a></summary><a id="mlflow-and-hydra"></a>

Experiment tracking and run directory configuration:

```yaml
mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

</details></ul>
</details>
<details open><summary><a href="#run-prediction"><b>2. Run prediction</b></a></summary><a id="run-prediction"></a>

If you chose to modify the [user_config.yaml](../user_config.yaml), you can run prediction by executing the following command from the AFD folder:

```bash
python stm32ai_main.py 
```
If you chose to update the [prediction_config.yaml](../config_file_examples/prediction_config.yaml) and use it, then run the following command from the AFD folder: 

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name prediction_config.yaml
```

</details>
<details open><summary><a href="#visualize-the-prediction-results"><b>3. Get the prediction results</b></a></summary><a id="visualize-the-prediction-results"></a>


Once the prediction is complete, the predicted classes or probabilities are printed in the terminal 

</details>
