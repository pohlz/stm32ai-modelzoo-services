# How can I compare the accuracy after quantization?

The quantization process optimizes the model for efficient deployment on embedded devices by reducing its memory usage (Flash/RAM) and accelerating its inference time, with minimal degradation in model accuracy. This is done by representing weights and activation with lower bit precision, in our case going from 32-bit floating point to 8 bit integer. With ST Model Zoo, you can easily check the accuracy of your model, quantize your model and compare this accuracy after quantization. You can also simply do one of these actions alone.

## Operation modes:

Depending on what you want to do, you can use the operation modes below:

- Chain_eqe:
    - To evaluate a model, quantize it and evaluate it again after quantization for comparison.
- Chain_eqeb:
    - To also add a benchmark of the quantized model.

For any details regarding the parameters of the config file, you can look at the main [readme](../README_OVERVIEW.md).

## User_config.yaml:

The way the ST Model Zoo works is that you edit the user_config.yaml available for each use case and run the stm32ai_main.py python script.

Here is an example where we evaluate an .h5 or a .keras model before quantizing it and evaluate it again for comparison.

The most important parts here are to define:
- The path to the float model
- The operation mode to chain_eqe
- The data paths in [ESC format](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/audio_event_detection/src/README.md#2)
- The class names
- The preprocessing and feature extraction options (same as training)
- The quantization options

It is highly recommended to use real data for the final quantization.


```yaml
general:
  project_name: aed_project
  logs_dir: logs
  saved_models_dir: saved_models
  global_seed: 120
  gpu_memory_limit: 5
  display_figures: True 
  batch_size: 16 # This is used to batch the eval dataset

operation_mode: chain_eqe 

model:
  # Change to the path of the model you wish to quantize and evaluate
  model_path: ../../stm32ai-modelzoo/audio_event_detection/yamnet/ST_pretrainedmodel_public_dataset/esc10/yamnet_e256_64x96_tl/yamnet_e256_64x96_tl.keras

dataset:
  dataset_name: esc10
  class_names: ['dog', 'chainsaw', 'crackling_fire', 'helicopter', 'rain', 'crying_baby', 'clock_tick', 'sneezing', 'rooster', 'sea_waves']
  file_extension: '.wav'

  training_audio_path: ./datasets/ESC-50/audio 
  training_csv_path:   ./datasets/ESC-50/meta/esc50.csv 

  # Optional but recommended, you can use the training dataset
  quantization_audio_path: <quantization-audio-root-directory>
  quantization_csv_path: <quantization-csv-root-directory>
  quantization_split: 0.1 # To use a fraction of the train dataset instead

  # evaluation dataset
  test_audio_path: <test-audio-root-directory>
  test_csv_path: <test-csv-root-directory>

  multi_label: False 
  use_garbage_class: False 
  n_samples_per_garbage_class: 2
  expand_last_dim: True
  seed: 120 # Optional, there is a default seed
  to_cache: True
  shuffle: True

preprocessing:
  min_length: 1
  max_length : 10
  target_rate: 16000
  top_db: 60
  frame_length: 3200
  hop_length: 3200
  trim_last_second: False
  lengthen : 'after'

feature_extraction:
  patch_length: 96
  n_mels: 64
  overlap: 0.25
  n_fft: 512
  hop_length: 160
  window_length: 400
  window: hann
  center: False
  pad_mode: constant
  power: 1.0
  fmin: 125
  fmax: 7500
  norm: None
  htk : True
  to_db : False

quantization:
  quantizer: TFlite_converter
  quantization_type: PTQ
  quantization_input_type: int8
  quantization_output_type: float
  export_dir: quantized_models

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

You can look at user_config.yaml examples [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/audio_event_detection/config_file_examples) for other operation modes.

## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```

## Clip-level and patch-level accuracies

When evaluating an audio event detection model, you will get clip-level and patch-level confusion matrixes. A clip contains multiple patches, so you get both the performances of your model on every patch or on full clips. You could have a low patch-level accuracy while having a very good clip-level accuracy. Both of these metrics help you to evaluate your model.