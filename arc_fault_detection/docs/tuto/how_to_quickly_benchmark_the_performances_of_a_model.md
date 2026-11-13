# How can I quickly benchmark a model using ST Model Zoo?

With ST Model Zoo, you can easily evaluate the memory footprints and inference time of a model on multiple hardwares using the [STEdgeAI Development Cloud](https://stm32ai.st.com/st-edge-ai-developer-cloud/).

## Operation modes:

Depending on the model format you have, you can use the operation modes below:
- Benchmarking:
    - To benchmark a quantized model (.tflite or QDQ onnx)
- Chain_qb:
    - To quantize and benchmark a float model (.h5, .keras or .onnx) in one pass
<div align="left" style="width:100%; margin: auto;">

![image.png](../img/chain_qb.png)
</div>
For any details regarding the parameters of the config file, you can look here:

- [Quantization documentation](../README_QUANTIZATION.md)
- [Benchmark documentation](../README_BENCHMARKING.md)


## User_config.yaml:

The way ST Model Zoo works is that you edit the user_config.yaml available for each use case and run the stm32ai_main.py python script. 

Here is an example where we quantize an .keras model from model zoo, before benchmarking it.

The most important parts here are to define:
- The path to the model
- The operation mode
- The quantization parameters
- The benchmarking parameters (online or locally, see below)
- The benchmarking hardware target

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
  model_path: ../../stm32ai-modelzoo/arc_fault_detection/st_conv/ST_pretrainedmodel_custom_dataset/afd_test_bench_dataset/st_conv_freq_4channels_512/st_conv_freq_4channels_512.keras
  model_name: 
  input_shape: (4,512,1)

operation_mode: chain_qb

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

preprocessing:
  downsampling: False
  normalization: True
  time_domain: False

quantization:
  quantizer: TFlite_converter   # onnx_quantizer for ONNX quantization, TFlite_converter for TFLite quantization
  quantization_type: PTQ
  quantization_input_type: int8
  quantization_output_type: int8
  granularity:                       # Optional, defaults to "per_channel". 
  optimize:                          # Optional, defaults to False.
  target_opset:                      # Optional, defaults to 17. Only used for ONNX quantization
  export_dir: quantized_models

tools:
   stedgeai:
      optimization: balanced
      on_cloud: True
      path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
   path_to_cubeIDE: C:/ST/STM32CubeIDE_1.17.0/STM32CubeIDE/stm32cubeide.exe

benchmarking:
  board: B-U585I-IOT02A

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

```

When evaluating the model, it is highly recommended to use real data for the quantization.

You can look at user_config.yaml examples for any operation mode [here](../../config_file_examples)

## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```

## Local benchmarking:

You can use the [STEdgeAI Developer Cloud](https://stedgeai-dc.st.com/home) to access the STEdgeAI functionalities without installing the software. This requires internet connection and making a free account. Or, alternatively, you can install [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally. In addition to this you will also need to install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) for building the embedded project.
 
For local installation :
 
- Download and install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html).
- If opting for using [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally, download it then extract both `'.zip'` and `'.pack'` files.
The detailed instructions on installation are available in this [wiki article](https://wiki.st.com/stm32mcu/index.php?title=AI:How_to_install_STM32_model_zoo).

## Available boards for benchmark:

'STM32N6570-DK', 'STM32H747I-DISCO', 'STM32H7B3I-DK', 'STM32H573I-DK', 'NUCLEO-H743ZI2', 'STM32F769I-DISCO', 'STM32H735G-DK', 'STM32H7S78-DK', 'STM32F469I-DISCO', 'STM32F746G-DISCO', 'B-U585I-IOT02A', 'STM32L4R9I-DISCO', 'NUCLEO-F401RE', 'NUCLEO-G474RE', 'STM32MP257F-EV1', 'STM32MP135F-DK' and 'STM32MP157F-DK2'
