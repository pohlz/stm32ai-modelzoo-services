# How can I quickly benchmark a model using the ST Model Zoo ?

With ST Model Zoo, you can easily benchmark the memory footprints and inference time of a model on multiple hardwares using the [ST Edge AI Developer Cloud](https://stm32ai.st.com/st-edge-ai-developer-cloud/)

## Operation modes

Depending on the model format you have, you can use the operation modes below:
- Benchmarking:
    - To benchmark a quantized model (.tflite or QDQ onnx), or a floating point model (.h5, .keras).
    - Note that if you are benchmarking a float model on STM32N6, it will not use the NPU and instead run entirely on CPU.
- Chain_qb:
    - To quantize and benchmark a float model (.h5, .keras or .onnx) in one pass
<div align="left" style="width:100%; margin: auto;">

![image.png](../img/chain_qb.png)
</div>
For any details regarding the parameters of the config file, you can look here:

- [Quantization documentation](../README_QUANTIZATION.md)
- [Benchmark documentation](../README_OVERVIEW.md)


## User_config.yaml

The way ST Model Zoo works is that you edit the user_config.yaml available for each use case and run the stm32ai_main.py python script. 

Here is an example where we quantize a .h5 or a .keras model from the model zoo, before benchmarking it.

The most important parts here are to define:
- The path to the model
- The operation mode
- The quantization parameters
- The benchmarking parameters (online or locally, see below)
- The benchmarking target board

```yaml

general:
  project_name: aed_project
  # Change to the path of the model you wish to evaluate
  logs_dir: logs
  saved_models_dir: saved_models
  global_seed: 120
  # If you use the seed indicated in the pretrained model config files,
  # you can guarantee that the validation and quantization sets are the same.
  gpu_memory_limit: 5
  display_figures: True 

operation_mode: chain_qb 

model:
  model_path: ../../stm32ai-modelzoo/audio_event_detection/yamnet/ST_pretrainedmodel_public_dataset/esc10/yamnet_e256_64x96_tl/yamnet_e256_64x96_tl.keras

dataset:
  dataset_name: esc10
  class_names: ['dog', 'chainsaw', 'crackling_fire', 'helicopter', 'rain', 'crying_baby', 'clock_tick', 'sneezing', 'rooster', 'sea_waves']
  file_extension: '.wav'

  # Note : It is not strictly necessary to provide a quantization dataset, but not doing so 
  # can greatly reduce quantized model performance

  quantization_audio_path: ./datasets/ESC-50/audio 
  quantization_csv_path: ./datasets/ESC-50/meta/esc50.csv
  quantization_split:  # Optional

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
  include_last_patch: False

quantization:
  quantizer: TFlite_converter
  quantization_type: PTQ
  quantization_input_type: int8
  quantization_output_type: float
  export_dir: quantized_models

tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

benchmarking:
  board: B-U585I-IOT02A

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
  
```

When evaluating the model, it is highly recommended to use real data for the quantization.

You can look at user_config.yaml examples for any operation mode [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/audio_event_detection/config_file_examples)

## Run the script

Edit the user_config.yaml then open a terminal (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```

## Local benchmarking

You can use the [STEdgeAI developer cloud](https://stedgeai-dc.st.com/home) to access the STEdgeAI Core functionalities without installing the software. This requires internet connection and making a free account. Or, alternatively, you can install [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally. In addition to this you will also need to install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) for building the embedded project.
 
For local installation :
 
- Download and install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html).
- If opting for using [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally, download it then run the installer.
The detailed instructions on installation are available in this [wiki article](https://wiki.st.com/stm32mcu/index.php?title=AI:How_to_install_STM32_model_zoo).

## Available boards for benchmark

'STM32N6570-DK', 'STM32H747I-DISCO', 'STM32H7B3I-DK', 'STM32H573I-DK', 'NUCLEO-H743ZI2', 'STM32F769I-DISCO', 'STM32H735G-DK', 'STM32H7S78-DK', 'STM32F469I-DISCO', 'STM32F746G-DISCO', 'B-U585I-IOT02A', 'STM32L4R9I-DISCO', 'NUCLEO-F401RE', 'NUCLEO-G474RE', 'STM32MP257F-EV1', 'STM32MP135F-DK' and 'STM32MP157F-DK2'


