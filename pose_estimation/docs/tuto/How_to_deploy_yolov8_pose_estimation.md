# How to quantize, evaluate and deploy Yolov8 pose estimation model for STM32N6

## Notice

Notice regarding usage of Ultralytics software [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics):

If You combine this software ("Software") with other software from STMicroelectronics ("ST Software"),  
to generate a software or software package ("Combined Software"), for instance for use in or in  
combination with STM32 products, You must comply with the license terms under which ST distributed   
such ST Software ("ST Software Terms"). Since this Software is provided to You under AGPL-3.0-only  
license terms, in most cases (such as, but not limited to, ST Software delivered under the terms of   
SLA0044, SLA0048, or SLA0078), ST Software Terms contain restrictions which will strictly forbid any   
distribution or non-internal use of the Combined Software. You are responsible for compliance with  
applicable license terms for any Software You use, and as such, You must limit your use of this  
software and any Combined Software accordingly.

## ST Ultralytics fork 

The STMicroelectronics Ultralytics fork: [https://github.com/stm32-hotspot/ultralytics/tree/main/examples/YOLOv8-STEdgeAI](https://github.com/stm32-hotspot/ultralytics/tree/main/examples/YOLOv8-STEdgeAI) provides a collection of pre-trained and quantized yolov8 models. These models are compatible with STM32 platforms, ensuring seamless integration and efficient performance for edge computing applications.
- Offers a set of pre-trained yolov8 models compatible with STM32 platforms and stm32ai-modelzoo.
- Offers a quantization friendly yolov8 pose estimation model.
These models are ready to be deployed and you can go directly to the deployment section.
The other sections below explain how to start from a model trained with Ultralytics scripts and not quantized.

If you just want to deploy pre-trained and quantized pose estimation, you can get them from the STMicroelectronics Ultralytics.
New: if you want to train, you can use now use directly Ultralytics repository at [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).
The latest release of Ultralytics fixed the quantization issue on the pose estimation and is now equivalent to ST fork. 

## Pre-requisite
  
By default, Ultralytics requirements do not install the packages required to export to onnx or tensorflow lite.
When exporting for the first time, it will either use pre-installed packages or do an auto update installing the latest versions which then causes compatibility issues.
To ensure compatibility, you need to install (or downgrade) the versions of tensorflow, onnx and onnxruntime following below requirements:
Validated on Ultralytics version 8.3.88
Use a python 3.9 environment (for the tflite_support package dependency)
Tensorflow version between 2.8.3 and 2.15.1
ONNX version between 1.12.0 and 1.15.0
ONNX runtime version between 1.13 and 1.18.1
```
	pip install tensorflow==2.15.0
	pip install onnx==1.15.0
	pip install onnxruntime==1.18.1
```
Other packages can be installed through the auto update procedure.

## Training a model with Ultralytics scripts

Train the `Yolov8n-pose` model as usual using Ultralytics scripts or start from the pre-trained Yolov8n-pose Pytorch model.
Please refer to [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).


## Exporting a model with Ultralytics scripts

To export the model as int8 tflite, let's take the default example of a model trained on COCO keypoints dataset using Ultralytics CLI:

```
	yolo export model=yolov8n-pose.pt format=tflite imgsz=256  int8=True
```

Where yolov8n-pose.pt is the trained weights, the output format is specified as tflite, int8=True means the model will be quantized using 8-bits signed for the weights and the activations.  
In this case the default COCO keypoint dataset is used by Ultralytics to pre-train the models.  
If no data for calibration is specified, a very small subset of images of COCO 2017 will be downloaded and the 4 images of the validation set will be used to calibrate.
From our experiment, the quantization is nevertheless efficient even more than using the full validation dataset of COCO (that you can test by using "data=coco-pose.yaml").


By default the exported models are:
1. A tensorflow float saved model, the saved model generated differs from exporting directly to saved model as the output is normalized to allow quantization: yolov8n-pose_saved_model directory.
2. An onnx float model: yolov8n-pose.onnx.
3. A quantized model per tensor with input / output in integer int8 format: yolov8n-pose_saved_model/yolov8n-pose_full_integer_quant.tflite.
4. A quantized model per tensor with input / output in float format: yolov8n-pose_saved_model/yolov8n-pose_integer_quant.tflite.

> [!TIP] 
> It is recommended to use per-channel quantization to better maintain the accuracy, so we recommend to use directly tensorflow lite converter to do the quantization.

Start from the generated saved model (1 above) as input for the tensorflow converter. Be sure to used the saved model generated through the export command with int8=True.  
A script is provided to quantize the model, the yaml file provide the quantization information (see below details).

For deployment the model shall be quantized with input as uint8 and output as float.


## Model quantization

You can find the quantization scripts here:  
[https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/tutorials/scripts/yolov8_quantization](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/tutorials/scripts/yolov8_quantization)

Features:
* The script can take in input a saved model (tensorflow or Keras) or a Keras h5
* The script supports per channel or per tensor quantization
* The script can do fake quantization or use a specified calibration dataset (mandatory to have meaningful accuracy)
* The script supports different configurations for input / output, float, int8 or uint8

> [!NOTE]
> For experiment, the same coco8 subset can be used as calibration dataset.  
> Download the coco8 subset from Ultralytics site: [https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip)
> Unzip the coco8.zip in your working directory.

Update the yaml file to generate a model with **uint8 input, float output**:

```yaml
model:
    name: yolov8n_256
    uc: pose_coco
    model_path: ./yolov8n-pose_saved_model
    input_shape: [256, 256, 3]
quantization:
    fake: False
    quantization_type: per_channel
    quantization_input_type: uint8
    quantization_output_type: float
    calib_dataset_path: ./coco8/images/val
    export_path: ./quantized_models
pre_processing:
  	rescaling: {scale : 255, offset : 0}
```

Where:
* model_path: the path to the directory created after the export command, here ./yolov8n-pose_saved_model
* input shapes: the input resolution, here 256x256x3
* quantization_input_type: model input format float, int8 or uint8
* quantization_output_type: model output format float, int8 or uint8
* calib_dataset_path: folder with the calibration images that will be used for quantization, here the coco8 validation subset
* rescaling: the normalization used during training with float value for images
  * if the normalization is between [0,1], the {scale : 255, offset : 0} values shall be used (Ultralytics case)
  * if the normalization is between [-1,1], the {scale : 127.5, offset : -1} values shall be used

Launch the quantization:
```powershell
python tflite_quant.py --config-name user_config.yaml
```

This will generate the quantized model with uint8 input and int8 output:  
quantized_models/yolov8n_256_quant_pc_ui_od_coco.tflite


Optional: update the yaml file to generate a model with **float input, float output**:

```yaml
model:
    name: yolov8n_256
    uc: pose_coco
    model_path: ./yolov8n-pose_saved_model
    input_shape: [256, 256, 3]
quantization:
    fake: False
    quantization_type: per_channel
    quantization_input_type: float
    quantization_output_type: float
    calib_dataset_path: ./coco8/images/val
    export_path: ./quantized_models
pre_processing:
  	rescaling: {scale : 255, offset : 0}
```

Launch the quantization:
```powershell
python tflite_quant.py --config-name user_config.yaml
```

This will generate the quantized model with float input and float output:  
quantized_models/yolov8n_256_quant_pc_ff_pose_coco.tflite

This model is equivalent to the uint8 input and float output model except the type of the input.  
It cannot be used for deployment, but can be used for evaluation with Ultralytics scripts.


## Evaluation of the quantized model with the STM32 model zoo evaluation service

The models with uint8 input and float output can be evaluated using the STM32 model zoo evaluation service described in following [readme](../README_EVALUATION.md).

## Evaluation of the quantized model with Ultralytics scripts

Use the model with float input and output, then use the CLI to evaluate the model onn the COCO validation set:

```
	yolo pose val model=./quantized_models/yolov8n-pose_256_quant_pt_ff_od_coco.tflite data=coco-pose.yaml imgsz=256
```

## Deployment of the quantized model on the STM32N6

The models with uint8 input and float output can be deployed using the STM32 model zoo deployment service [readme](../README_DEPLOYMENT_STM32N6.md).
