# How to check the accuracy of my model after quantization?

The quantization process optimizes the model for efficient deployment on embedded devices by reducing its memory usage (Flash/RAM) and accelerating its inference time, with minimal degradation in model accuracy. With ST Model Zoo, you can easily check the accuracy of your model, quantize your model and compare this accuracy after quantization. You can also simply do one of these actions alone.

## Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Evaluate:
    - To evaluate a model, quantized or not (.keras, .tflite or QDQ onnx)
- Chain_eqe:
    - To evaluate a model, quantize it and evaluate it again after quantization for comparison.
- Chain_eqeb:
    - To also add a benchmark of the quantized model.

For any details regarding the parameters of the config file, you can look here:
- [Evaluation documentation](../README_EVALUATION.md)
- [Quantization documentation](../README_QUANTIZATION.md)
- [Benchmark documentation](../README_BENCHMARKING.md)


## User_config.yaml:

The way ST Model Zoo works is that you edit the user_config.yaml available for each use case and run the stm32ai_main.py python script. 

Here is an example where we evaluate a .keras model before quantizing it and evaluate it again for comparison:

The most important parts to define are:
- The model path
- The operation mode
- The dataset for quantization and test
- The preprocessing (usually the same used in training)

```yaml
# user_config.yaml

model:
   model_path: ../../stm32ai-modelzoo/pose_estimation/movenet/ST_pretrainedmodel_custom_dataset/custom_coco_person_17kpts/st_movenet_lightning_a100_heatmaps_192/st_movenet_lightning_a100_heatmaps_192.keras
   model_type: heatmaps_spe

operation_mode: chain_eqe

dataset:
   dataset_name: coco
   quantization_path: ./datasets/coco_quant_single_pose

preprocessing:
   rescaling:
      scale: 1/127.5
      offset: -1
   resizing:
      aspect_ratio: fit
      interpolation: nearest
   color_mode: rgb

quantization:
   quantizer: TFlite_converter  # onnx_quantizer
   quantization_type: PTQ
   quantization_input_type: uint8
   quantization_output_type: float
   export_dir: quantized_models
   optimize: True
   granularity: per_tensor

mlflow:
   uri: ./tf/src/experiments_outputs/mlruns

hydra:
   run:
      dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```
When evaluating the model, it is highly recommended to use real data for the final quantization.

You can also find examples of user_config.yaml for any operation mode [here](../../config_file_examples)


## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```
