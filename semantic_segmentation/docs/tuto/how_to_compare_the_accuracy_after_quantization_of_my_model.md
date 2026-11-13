# How to check the accuracy of my model after quantization?

The quantization process optimizes the model for efficient deployment on embedded devices by reducing its memory usage (Flash/RAM) and accelerating its inference time, with minimal degradation in model accuracy. With ST Model Zoo, you can easily check the accuracy of your model, quantize your model and compare this accuracy after quantization. You can also simply do one of these actions alone.

## Operation modes:

Depending on what you want to do, you can use the operation modes below:

- Evaluate:
    - To evaluate a model, quantized or not (.h5, .tflite or QDQ onnx)
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

Here is an example where we evaluate an .h5 model before quantizing it and evaluate it again for comparison:

The most important parts to define are:
- The model path
- The operation mode
- The dataset for quantization and test
- The preprocessing (usually the same used in training)

```yaml
# user_config.yaml

model:
   model_path: ../../stm32ai-modelzoo/semantic_segmentation/deeplabv3/ST_pretrainedmodel_public_dataset/person_coco_2017_pascal_voc_2012/st_deeplabv3_mnv2_a050_s16_asppv2_320/st_deeplabv3_mnv2_a050_s16_asppv2_320.keras

operation_mode: chain_eqe

dataset:
   dataset_name: person_coco_2017_pascal_voc_2012
   class_names: [ "background", "person" ]
   test_path: ./datasets/person_coco_2017_pascal_voc_2012/JPEGImages
   test_masks_path: ./datasets/person_coco_2017_pascal_voc_2012/SegmentationClassAug
   test_files_path: ./datasets/person_coco_2017_pascal_voc_2012/val.txt
   quantization_path: ./datasets/person_coco_2017_pascal_voc_2012/JPEGImages
   quantization_masks_path: ./datasets/person_coco_2017_pascal_voc_2012/SegmentationClassAug
   quantization_files_path: ./datasets/person_coco_2017_pascal_voc_2012/train.txt
   quantization_split: 0.2

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
