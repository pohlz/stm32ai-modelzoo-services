# How can I quickly benchmark a model using the ST Model Zoo?

With ST Model Zoo, you can easily evaluate the memory footprints and inference time of a model on multiple hardwares using the [ST Edge AI Development Cloud](https://stm32ai.st.com/st-edge-ai-developer-cloud/)

## Operation modes:

Depending on the model format you have, you can use the operation modes below:
- Benchmarking:
    - To benchmark a quantized model (.tflite or QDQ onnx)
- Chain_qb:
    - To quantize and benchmark a float model (.h5 or .onnx) in one pass
<div align="left" style="width:100%; margin: auto;">

![image.png](../img/chain_qb.png)
</div>

For any details regarding the parameters of the config file, you can look here:
- [Benchmark documentation](../README_BENCHMARKING.md)
- [Evaluation documentation](../README_EVALUATION.md)


## Available boards for benchmark:

'STM32N6570-DK', 'STM32H747I-DISCO', 'STM32H7B3I-DK', 'STM32F469I-DISCO', 'B-U585I-IOT02A', 'STM32L4R9I-DISCO', 'NUCLEO-H743ZI2', 'STM32H747I-DISCO', 'STM32H735G-DK', 'STM32F769I-DISCO', 'NUCLEO-G474RE', 'NUCLEO-F401RE' and 'STM32F746G-DISCO'

## User_config.yaml:

The way ST Model Zoo works is that you edit the user_config.yaml available for each use case and run the stm32ai_main.py python script. 

Here is an example where we benchmark a model already quantized available in ST model zoo.

The most important parts here are to define:
- The path to the model
- The operation mode
- The benchmarking parameters (online or locally, see below)
- The benchmarking hardware target

```yaml
# user_config.yaml 

model:
   model_path: ../../stm32ai-modelzoo/semantic_segmentation/deeplabv3/ST_pretrainedmodel_public_dataset/person_coco_2017_pascal_voc_2012/st_deeplabv3_mnv2_a050_s16_asppv2_320/st_deeplabv3_mnv2_a050_s16_asppv2_320_qdq_int8.onnx

operation_mode: benchmarking

preprocessing:
   rescaling:
      scale: 1/127.5
      offset: -1
   resizing:
      interpolation: bilinear
      aspect_ratio: fit
   color_mode: rgb

tools:
   stedgeai:
      optimization: balanced
      on_cloud: True
      path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
   path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

benchmarking:
   board: STM32N6570-DK


mlflow:
   uri: ./tf/src/experiments_outputs/mlruns


hydra:
   run:
      dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

```

Here is another example where we quantize the model first, then we launch the benchmark:

```yaml
# user_config.yaml 

model:
   model_path: ../../stm32ai-modelzoo/semantic_segmentation/deeplabv3/ST_pretrainedmodel_public_dataset/person_coco_2017_pascal_voc_2012/st_deeplabv3_mnv2_a050_s16_asppv2_320/st_deeplabv3_mnv2_a050_s16_asppv2_320.keras

operation_mode: chain_qb

preprocessing:
   rescaling:
      scale: 1/127.5
      offset: -1
   resizing:
      aspect_ratio: fit
      interpolation: bilinear
   color_mode: rgb

quantization:
  quantizer: onnx_quantizer
  target_opset: 17
  granularity: per_channel
  quantization_type: PTQ
  quantization_input_type: float
  quantization_output_type: float
  onnx_extra_options: 
    CalibMovingAverage: True
  export_dir: quantized_models

tools:
   stedgeai:
      optimization: balanced
      on_cloud: True
      path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
   path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe


benchmarking:
   board: STM32N6570-DK


mlflow:
   uri: ./tf/src/experiments_outputs/mlruns


hydra:
   run:
      dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

```

Here the quantization is made with random data as no data were provided and main goal was to have a quick insight on the performances of a quantized model for a specific HW. When evaluating the model, it is highly recommended to use real data for the final quantization of course.

You can also find examples of user_config.yaml for any operation modes [here](../../config_file_examples)


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

You can use the [STM32 developer cloud](https://stedgeai-dc.st.com/home) to access the STEdgeAI functionalities without installing the software. This requires internet connection and making a free account. Or, alternatively, you can install [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally. In addition to this you will also need to install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) for building the embedded project.
 
For local installation :
 
- Download and install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html).
- If opting for using [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally, download it then extract the files.
The detailed instructions on installation are available in this [wiki article](https://wiki.st.com/stm32mcu/index.php?title=AI:How_to_install_STM32_model_zoo).

