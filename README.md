# STMicroelectronics – STM32 model zoo services

Welcome to STM32 model zoo services!

**🎉 We are excited to announce that the STM32 AI model zoo now includes comprehensive PyTorch support, joining TensorFlow and ONNX.
It now features a vast library of PyTorch models, all seamlessly integrated with our end-to-end workflows. Whether you want to train, evaluate, quantize, benchmark, or deploy, you’ll find everything you need – plus the flexibility to choose between PyTorch, TensorFlow, and ONNX. Dive into the expanded <a href="https://github.com/STMicroelectronics/stm32ai-modelzoo/">STM32 model zoo</a> and take your AI projects further than ever on STM32 devices.**

---


The STM32 AI model zoo is a set of services and scripts used to ease end to end AI models integration on ST devices. This can be used in conjunction with the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/), which contains a collection of reference machine learning models optimized to run on STM32 microcontrollers.
Available on GitHub, it is a valuable resource for anyone looking to add AI capabilities to their STM32-based projects.

- Scripts to easily retrain or fine-tune any model from user datasets (BYOD and BYOM)
- A set of services and chained services to quantize, benchmark, predict, and evaluate any model (BYOM)
- Application code examples automatically generated from user AI models

These models can be useful for quick deployment if you are interested in the categories they were trained on. We also provide training scripts to perform transfer learning or to train your own model from scratch on your custom dataset.

The performance on reference STM32 MCUs and MPUs is provided for both float and quantized models.
This project is organized by application. For each application, you will have a step-by-step guide indicating how to train and deploy the models.

To clone the repository please use:

```bash
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git --depth 1
```
## What's new in releases :

<details open><summary><b>4.1:</b></summary>

* Support of **STEdgeAI Core v4.0.0**.
* Updated **Audio Event Detection (AED)** to support deployment on **STM32U3**. 
* Added support for the **YOLO26** model.
* Multiple bug fixes and overall quality improvements.

</details>

<details><summary><b>4.0:</b></summary>

* Major PyTorch support for Image Classification (IC) and Object Detection (OD)
* Support of **STEdgeAI Core from v3.0.0 to latest**
* New training and evaluation scripts for PyTorch models
* Expanded model selection and improved documentation
* Unified workflow for TensorFlow and PyTorch
* Performance and usability improvements
* New use cases: **Face Detection (FD)**, **Arc Fault Detection (AFD)**, **Re-Identification (ReID)**
* New mixed precision models (Weights 4-bits, Activations 8-bits) for IC and OD use cases
* Support for Keras 3.8.0, TensorFlow 2.18.0, PyTorch 2.7.1, and ONNX 1.16.1
* Python software architecture rework
* Docker-based setup available, with a ready-to-use image including the full software stack.

</details>

<details><summary><b>3.2:</b></summary>

* Support of **STEdgeAI Core v2.2.0**.
* Support of **X-Linux-AI v6.1.0** support for MPU.
* New use cases added: **StyleTransfer** and **FastDepth**.
* New models added: **Face Detection**, available in the Object Detection use case, and **Face Landmarks**, available in the Pose Estimation use case.
* Architecture and codebase clean-up.
</details>

<details><summary><b>3.1:</b></summary>

* Support for **STEdgeAI Core v2.1.0**.
* Application code for STM32N6 board is now directly available in the STM32 model zoo repository, eliminating the need for separate downloads.
* Support of **On device evaluation** and **On device prediction** on the **STM32N6570-DK** boards integrated in evaluation and prediction services.
* More models supported: Yolov11, LSTM model added in Speech Enhancement, ST Yolo X variants.
* ClearML support.
* A few bug fixes and improvements, such as proper imports and OD metrics alignment.


</details>
<details><summary><b>3.0:</b></summary>

* Full support of the new [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk) board.
* Included additional models compatible with the `STM32N6`.
* Included support for **STEdgeAI Core v2.0.0**.
* Split of model zoo and services into two GitHub repositories
* Integrated support for `ONNX model` quantization and evaluation from h5 models.
* Expanded use case support to include **Instance Segmentation** and **Speech Enhancement**.
* Added `Pytorch` support through the speech enhancement Use Case.
* Support of **On device evaluation and prediction** on the **STM32N6570-DK** boards.
* Model Zoo hosted on <a href="#Hugging Face">Hugging Face</a>

</details>
<details><summary><b>2.1:</b></summary>

* Included additional models compatible with the [STM32MP257F-EV1](https://www.st.com/en/evaluation-tools/stm32mp257f-ev1) board.
* Added support for per-tensor quantization.
* Integrated support for `ONNX model` quantization and evaluation.
* Included support for **STEdgeAI Core v1.0.0**.
* Expanded use case support to include **Pose Estimation** and **Semantic Segmentation**.
* Standardized logging information for a unified experience.
</details>
<details><summary><b>2.0:</b></summary>

* An aligned and **uniform architecture** for all the use cases
* A modular design to run different operation modes (training, benchmarking, evaluation, deployment, quantization) independently or with an option of chaining multiple modes in a single launch.
* A simple and `single entry point` to the code : a .yaml configuration file to configure all the needed services.
* Support of the `Bring Your Own Model (BYOM)` feature to allow the user (re-)training his own model. Example is provided [here](./image_classification/docs/README_TRAINING.md#51-training-your-own-model), chapter 5.1.
* Support of the `Bring Your Own Data (BYOD)` feature to allow the user finetuning some pretrained models with his own datasets. Example is provided [here](./image_classification/docs/README_TRAINING.md#23-dataset-specification), chapter 2.3.
</details>

<div align="center" style="margin-top: 80px; padding: 20px 0;">
    <p align="center">
      <a href="https://www.python.org/downloads/" target="_blank"><img src="https://img.shields.io/badge/python-3.12.9-blue" /></a>
      <a href="https://www.tensorflow.org/install/pip" target="_blank"><img src="https://img.shields.io/badge/TensorFlow-2.18.0-FF6F00?style=flat&logo=tensorflow&logoColor=white&link=https://www.tensorflow.org/install/pip"/></a>
      <br/>
      <a href="https://onnx.ai/" target="_blank"><img src="https://img.shields.io/badge/ONNX-1.16.1-0094C4?style=flat&logo=onnx&logoColor=white&link=https://onnx.ai/"/></a>
      <a href="https://pytorch.org/" target="_blank"><img src="https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?style=flat&logo=pytorch&logoColor=white&link=https://pytorch.org/"/></a>
      <a href="https://stedgeai-dc.st.com/home"><img src="https://img.shields.io/badge/STEdgeAI-Developer%20Cloud-FFD700?style=flat&logo=stmicroelectronics&logoColor=white"/></a>
    </p>
</div>

## Available use-cases
The ST model zoo provides a collection of independent `services` and pre-built `chained services` that can be used to perform various functions related to machine learning. The individual services include tasks such as training or quantization of a model, while the chained services combine multiple services to perform more complex functions, such as training the model, quantizing it, and evaluating the quantized model successively before benchmarking it on a HW of your choice.

**All trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. This is a very good baseline to start with!**

>[!TIP]
> All services are available for following use cases with quick and easy examples that are provided and can be executed for a fast ramp up (click on use cases links below).
* <a href="#IC">Image Classification</a>
* <a href="#OD">Object Detection</a>
* <a href="#PE">Pose Estimation</a>
* <a href="#FD">Face Detection</a>
* <a href="#SemSeg">Semantic Segmentation</a>
* <a href="#InstSeg">Instance Segmentation</a>
* <a href="#DE">Depth Estimation</a>
* <a href="#NST">Neural Style Transfer</a>
* <a href="#REID">Re-Identification</a>
* <a href="#AED">Audio Event Detection</a>
* <a href="#SE">Speech Enhancement</a>
* <a href="#HAR">Human Activity Recognition</a>
* <a href="#HPR">Hand Posture Recognition</a>
* <a href="#AFD">Arc Fault Detection</a>

## <a id="IC">Image Classification</a>
Image classification is used to classify the content of an image within a predefined set of classes. Only one class is predicted from an input image.

<div align="center" style="width:100%; margin: auto;">

![plot](./image_classification/docs/img/output_application.JPG)
</div>

<details open><summary><b>Image classification (IC) models</b></summary>

| Suitable Targets for Deployment | Models |
|---------------------------------|--------|
| STM32H747I-DISCO | [MobileNet v1 0.25](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv1/README.md), [MobileNet v1 0.5](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv1/README.md), [MobileNet v2 0.35](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv2/README.md), [ResNet8 v1](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet/README.md), [ST ResNet8](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet/README.md), [ResNet32 v1](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet/README.md), [SqueezeNet v1.1](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/squeezenetv11/README.md), [FD MobileNet 0.25](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/fdmobilenet/README.md), [ST FD MobileNet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/fdmobilenet/README.md), [ST EfficientNet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/efficientnet/README.md), [Mnist](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/st_mnist/README.md) |
| NUCLEO-H743ZI2 | [MobileNet v1 0.25](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv1/README.md), [MobileNet v1 0.5](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv1/README.md), [MobileNet v2 0.35](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv2/README.md), [ResNet8 v1](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet/README.md), [ST ResNet8](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet/README.md), [ResNet32 v1](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet/README.md), [SqueezeNet v1.1](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/squeezenetv11/README.md), [FD MobileNet 0.25](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/fdmobilenet/README.md), [ST FD MobileNet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/fdmobilenet/README.md), [ST EfficientNet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/efficientnet/README.md), [Mnist](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/st_mnist/README.md) |
| STM32MP257F-EV1 | [MobileNet v1 1.0](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv1/README.md), [MobileNet v2 1.0](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv2/README.md), [MobileNet v2 1.4](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv2/README.md), [ResNet50 v2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet50v2/README.md), [EfficientNet v2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/efficientnetv2/README.md) |
| STM32N6570-DK | [MobileNet v1 1.0](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv1/README.md), [MobileNet v2 1.0](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv2/README.md), [MobileNet v2 1.4](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv2/README.md), [ResNet50 v2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet50v2/README.md), [EfficientNet v2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/efficientnetv2/README.md), [DarkNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/darknet_pt/README.md), [Dla_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/dla_pt/README.md), [FdMobileNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/fdmobilenet_pt/README.md), [HardNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/hardnet_pt/README.md), [MnasNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mnasnet_pt/README.md), [MobileNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenet_pt/README.md), [MobileNetv2_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv2_pt/README.md), [MobileNetv4_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv4_pt/README.md), [PeleeNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/peleenet_pt/README.md), [PreresNet18_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/preresnet18_pt/README.md), [ProxylessNas_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/proxylessnas_pt/README.md), [RegNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/regnet_pt/README.md), [SemnasNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/semnasnet_pt/README.md), [ShuffleNetv2_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/shufflenetv2_pt/README.md), [Sqnxt_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/sqnxt_pt/README.md), [SqueezeNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/squeezenet_pt/README.md), [St_ResNet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/st_resnet_pt/README.md) |


</details>

Selecting a model for a specific task or a specific device is not always an easy task, and relying on metrics like the inference time and the accuracy, as in the example figure on food-101 classification below, can help you make the right choice before fine-tuning your model.

<div align="center" style="width:100%; margin: auto;">

![plot](./image_classification/docs/img/ic_food101_bubble.JPG)
</div>

Please find below some tutorials for a quick ramp up!
* [How can I define and train my own model?](./image_classification/docs/tuto/how_to_define_and_train_my_own_model.md)
* [How can I fine tune a pretrained model on my own dataset?](./image_classification/docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I check the accuracy after quantization of my model?](./image_classification/docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./image_classification/docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I evaluate my model on STM32N6 target?](./image_classification/docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Image Classification top readme **[here](./image_classification/README.md)**

## <a id="OD">Object Detection</a>
Object detection is used to detect, locate and estimate the occurrences probability of predefined objects from input images.

<div align="center" style="width:80%; margin: auto;">

![plot](./object_detection/docs/img/output_application.JPG)
</div>

<details open><summary><b>Object Detection (OD) Models</b></summary>

| Suitable Targets for Deployment | Models |
|---------------------------------|--------|
| STM32H747I-DISCO | [ST Yolo LC v1](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/st_yololcv1/README.md) |
| STM32N6570-DK | [Tiny Yolo v2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/yolov2t/README.md), [ST Yolo X](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/st_yoloxn/README.md), [Yolo v8](https://github.com/stm32-hotspot/ultralytics/tree/main/examples/YOLOv8-STEdgeAI/stedgeai_models/object_detection), [Yolo v11](https://github.com/stm32-hotspot/ultralytics/tree/main/examples/YOLOv8-STEdgeAI/stedgeai_models/object_detection), [Yolo 26](https://github.com/stm32-hotspot/ultralytics/tree/main/examples/YOLOv8-STEdgeAI/stedgeai_models/object_detection),  [Blazeface front](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/facedetect_front/README.md), [SSD_MobileNetV1_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/ssd_mobilenetv1_pt/README.md), [SSD_MobileNetV2_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/ssd_mobilenetv2_pt/README.md), [SSDLite_MobileNetV1_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/ssdlite_mobilenetv1_pt/README.md), [SSDLite_MobileNetV2_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/ssdlite_mobilenetv2_pt/README.md), [SSDLite_MobileNetV3Large_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/ssdlite_mobilenetv3large_pt/README.md), [SSDLite_MobileNetV3Small_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/ssdlite_mobilenetv3small_pt/README.md), [ST_YoloDv2Milli_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/st_yolodv2milli_pt/README.md), [ST_YoloDv2Tiny_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/object_detection/st_yolodv2tiny_pt/README.md) |


</details>
Relying on metrics like the inference time and the mean Average Precision (mAP) as in example figure on people detection below can help making the right choice before fine tuning your model, as well as checking HW capabilities for OD task.

<div align="center" style="width:80%; margin: auto;">

![plot](./object_detection/docs/img/od_coco_2017_person_bubble.JPG)
</div>

Please find below some tutorials for a quick ramp up!
* [How can I use my own dataset?](./object_detection/docs/tuto/how_to_use_my_own_object_detection_dataset.md)
* [How can I fine tune a pretrained model on my own dataset?](./object_detection/docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I check the accuracy after quantization of my model?](./object_detection/docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./object_detection/docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I quantize, evaluate and deploy an Ultralytics Yolov8 model?](./object_detection/docs/tuto/How_to_deploy_yolov8_yolov5_object_detection.md)
* [How can I evaluate my model on STM32N6 target?](./object_detection/docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Object Detection top readme **[here](./object_detection/README.md)**




## <a id="FD">Face Detection</a>
Face detection is used to detect, locate and estimate the occurrences probability of faces from input images.


<div align="center" style="width:80%; margin: auto;">

![plot](./face_detection/docs/img/output_application.png)
</div>

<details open><summary><b>Face Detection (FD) Models</b></summary>

| Models             | Input Resolutions | Supported Services    | Targets for deployment |
|--------------------|------------------|-----------------------|-------------------|
| [Blazeface front](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/face_detection/facedetect_front/README.md)  |  128x128x3<br>   | Benchmarking / Prediction / Deployment/ Evaluation   | [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html)<br>     |
| [Yunet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/face_detection/yunet/README.md)                        |  3x320x320<br>   | Benchmarking / Prediction / Deployment/ Evaluation   | [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html)<br>     |


[Full FD Services](face_detection/README.md) : evaluation, quantization, benchmarking, prediction, deployment

</details>

Please find below some tutorials for a quick ramp up!
* [How can I use my own dataset?](./face_detection/docs/tuto/how_to_use_my_own_object_detection_dataset.md)
* [How can I check the accuracy after quantization of my model?](./face_detection/docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./face_detection/docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I evaluate my model on STM32N6 target?](./face_detection/docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Face Detection top readme **[here](./face_detection/README.md)**


## <a id="PE">Pose Estimation</a>
Pose estimation allows to detect key points on some specific objects (people, hand, face, ...). It can be single pose where key points can be extracted from a single object, or multi pose where location of key points are estimated on all detected objects from the input images.

<div align="center" style="width:80%; margin: auto;">

![plot](./pose_estimation/docs/img/output_mpu_application.JPG)
</div>

<details open><summary><b>Pose Estimation (PE) Models</b></summary>

| Models             | Input Resolutions | Supported Services    | Targets for deployment |
|--------------------|------------------|-----------------------|-------------------|
| [Yolo v8n pose](https://github.com/stm32-hotspot/ultralytics/tree/master/examples/YOLOv8-STEdgeAI/stedgeai_models/pose_estimation)   |  192x192x3<br> 256x256x3<br> 320x320x3<br>  | Evaluation / Benchmarking / Prediction / Deployment      | [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html)<br> |
| [Yolo v11n pose](https://github.com/stm32-hotspot/ultralytics/tree/master/examples/YOLOv8-STEdgeAI/stedgeai_models/pose_estimation/yolo11)   |  256x256x3<br> 320x320x3<br>  | Benchmarking / Prediction / Deployment      | [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html)<br> |
| [ST MoveNet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/movenet/README.md)   |  192x192x3<br> 224x224x3<br> 256x256x3<br>  | All services      | [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html)<br> [STM32MP257F-EV1](./application_code/pose_estimation/STM32MP-LINUX/STM32MP2/README.md) <br>|
| [MoveNet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/movenet/README.md)   |  192x192x3<br> 256x256x3<br>   | Evaluation / Quantization / Benchmarking / Prediction      | [STM32MP257F-EV1](./application_code/pose_estimation/STM32MP-LINUX/STM32MP2/README.md) <br>  |
| [Face landmarks](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/headlandmarks)  |  192x192x3<br>  | Benchmarking / Prediction      | [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html)<br> |
| [Hand landmarks](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/handlandmarks)  |  224x224x3<br>  | Benchmarking / Prediction      | [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html)<br> |


[Full PE Services](pose_estimation/README.md) : training, evaluation, quantization, benchmarking, prediction, deployment
</details>
Various metrics can be used to estimate quality of a single or multiple pose estimation use case. Metrics like the inference time and the Object Key point Similarity (OKS) as in example figure on single pose estimation below can help making the right choice before fine tuning your model, as well as checking HW capabilities for PE task.

<div align="center" style="width:80%; margin: auto;">

![plot](./pose_estimation/docs/img/spe_coco_2017_bubble.JPG)
</div>

Please find below some tutorials for a quick ramp up!
* [How can I use my own dataset?](./pose_estimation/docs/tuto/how_to_use_my_own_dataset.md)
* [How to define and train my own model?](./pose_estimation/docs/tuto/how_to_define_and_train_my_own_model.md)
* [How can I fine tune a pretrained model on my own dataset?](./pose_estimation/docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I check the accuracy after quantization of my model?](./pose_estimation/docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./pose_estimation/docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I deploy an Ultralytics Yolov8 pose estimation model?](./pose_estimation/docs/tuto/How_to_deploy_yolov8_pose_estimation.md)
* [How can I evaluate my model on STM32N6 target?](./pose_estimation/docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Pose Estimation top readme **[here](./pose_estimation/README.md)**

## <a id="SemSeg">Semantic Segmentation</a>
Semantic segmentation is an algorithm that associates a label to every pixel in an image. It is used to recognize a collection of pixels that form distinct categories. It doesn't differentiate instances of the same category, which is the main difference between instance and semantic segmentation.

<div align="center" style="width:80%; margin: auto;">

![plot](./semantic_segmentation/docs/img/output_mpu_application.JPG)
</div>

<details open><summary><b>Semantic Segmentation (SemSeg) Models</b></summary>

| Models             | Input Resolutions | Supported Services    | Targets for deployment |
|--------------------|------------------|-----------------------|-------------------|
| [DeepLab v3](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/semantic_segmentation/deeplab/README.md)   | 256x256x3<br> 320x320x3<br> 416x416x3<br> 512x512x3<br>  | Full Seg Services     | [STM32MP257F-EV1](./application_code/semantic_segmentation/STM32MP-LINUX/STM32MP2/README.md) <br> [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html) <br> |

[Full Seg Services](./semantic_segmentation/README.md) : training, evaluation, quantization, benchmarking, prediction, deployment

</details>

Various metrics can be used to estimate the quality of a segmentation use case. Metrics like the inference time and IoU, as in the example figure on person segmentation below, can help you make the right choice before fine-tuning your model, as well as checking HW capabilities for the segmentation task.

<div align="center" style="width:80%; margin: auto;">

![plot](./semantic_segmentation/docs/img/semseg_person_coco_2017_pascal_voc_2012_bubble.JPG)
</div>

Please find below some tutorials for a quick ramp up!
* [How to define and train my own model?](./semantic_segmentation/docs/tuto/how_to_define_and_train_my_own_model.md)
* [How can I fine tune a pretrained model on my own dataset?](./semantic_segmentation/docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I check the accuracy after quantization of my model?](./semantic_segmentation/docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./semantic_segmentation/docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I evaluate my model on STM32N6 target?](./semantic_segmentation/docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Semantic Segmentation top readme **[here](./semantic_segmentation/README.md)**


## <a id="InstSeg">Instance Segmentation</a>
Instance segmentation is an algorithm that associates a label to every pixel in an image. It also outputs bounding boxes on detected class objects. It is used to recognize a collection of pixels that form distinct categories and instances of each category. It differentiates instances of the same category, which is the main difference between instance and semantic segmentation.

<div align="center" style="width:80%; margin: auto;">

![plot](./instance_segmentation/docs/img/output_application_instseg.JPG)
</div>

<details open><summary><b>Instance Segmentation (InstSeg) Models</b></summary>

| Models             | Input Resolutions | Supported Services    | Targets for deployment |
|--------------------|------------------|-----------------------|-------------------|
| [yolov8n_seg](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/instance_segmentation/yolov8n_seg/README.md)   | 256x256x3<br> 320x320x3<br> | Prediction, Benchmark, Deployment     |  [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html) <br> |
| [yolov11n_seg](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/instance_segmentation/yolov11n_seg/README.md)   | 256x256x3<br> 320x320x3<br> | Prediction, Benchmark, Deployment     |  [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html) <br> |


</details>

Please find below some tutorials for a quick ramp up!
* [How can I deploy an Ultralytics Yolov8 instance segmentation model?](./instance_segmentation/docs/tuto/How_to_deploy_yolov8_instance_segmentation.md)

Instance Segmentation top readme **[here](./instance_segmentation/README.md)**

## <a id="DE">Depth Estimation</a>

</div>

This allows to predict the distance to objects from an image as a pixel-wise depth map.


<div align="center" style="width:80%; margin: auto;">

![plot](./depth_estimation/docs/img/output_application_de.JPG)
</div>
<details open><summary><b>Depth Estimation (DE) Models</b></summary>

| Models             | Input Resolutions | Supported Services    |
|--------------------|------------------|-----------------------|
| [fast_depth](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/depth_estimation/fastdepth/README.md)   | 224x224x3<br>  256x256x3<br>   320x320x3<br>  | benchmarking / prediction  |


</details>

Depth Estimation top readme **[here](./depth_estimation/README.md)**.

## <a id="NST">Neural Style Transfer</a>
Neural Style Transfer is a deep learning technique that applies the artistic style of one image to the content of another image by optimizing a new image to simultaneously match the content features of the original and the style features of the reference image.

<div align="center" style="width:80%; margin: auto;">

![plot](./neural_style_transfer/docs/img/output_application_nst.JPG)
</div>

<details open><summary><b>Neural style transfer (NST) Models</b></summary>

| Models             | Input Resolutions | Supported Services    |
|--------------------|------------------|-----------------------|
| [Xinet_picasso_muse](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/neural_style_transfer/xinet_picasso_muse/README.md)   | 160x160x3<br> | Prediction, Benchmark     |

</details>

Neural style transfer top readme **[here](./neural_style_transfer/README.md)**

## <a id="REID">Re-Identification (ReID)</a>
Re-Identification is used to recognize a specific object (person, vehicle, ...) from a set of images.
<div align="center" style="width:80%; margin: auto;">

![plot](./re_identification/docs/img/output_application_reid.png)
</div>

<details open><summary><b>Re-Identification (ReID) models</b></summary>

| Models             | Input Resolutions | Supported Services    | Suitable Targets for deployment |
|--------------------|------------------|-----------------------|-------------------|
| [MobileNet v2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/re_identification/mobilenetv2/README.md)   | 256x128x3   | Full IC Services      | [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html) <br>   |
| [OSNet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/re_identification/osnet/README.md)   | 256x128x3     | Full IC Services      | [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html) <br>   |

[Full IC Services](re_identification/README.md) : training, evaluation, quantization, benchmarking, prediction, deployment
</details>

Re-Identification top readme **[here](./re_identification/README.md)**


## <a id="AED">Audio Event Detection</a>
This is used to detect a set of pre-defined audio events.

<div align="center" style="width:80%; margin: auto;">

![plot](./audio_event_detection/docs/img/output_application.JPG)
</div>

<details open><summary><b>Audio Event Detection (AED) Models</b></summary>

[Audio Event Detection use case](audio_event_detection)
| Models             | Input Resolutions | Supported Services    | Targets for deployment |
|--------------------|------------------|-----------------------|-------------------|
| [miniresnet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/audio_event_detection/miniresnetv1/README.md)   |  64x50x1<br>  | Full AED Services      | [B-U585I-IOT02A](application_code)<br>    |
| [miniresnet v2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/audio_event_detection/miniresnetv2/README.md)   |  64x50x1<br>  | Full AED Services      | [B-U585I-IOT02A](application_code) <br>    |
| [yamnet 256/1024](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/audio_event_detection/yamnet/README.md)   |  64x96x1<br>  | Full AED Services      | [B-U585I-IOT02A](application_code) <br> [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html)   |

[Full AED Services](audio_event_detection/README.md) : training, evaluation, quantization, benchmarking, prediction, deployment

</details>

Various metrics can be used to estimate quality of an audio event detection UC. The main ones are the inference time and the accuracy (percentage of good detections) on esc-10 dataset as in example figure below. This may help making the right choice before fine tuning your model, as well as checking HW capabilities for such AED task.

<div align="center" style="width:80%; margin: auto;">

![plot](./audio_event_detection/docs/img/aed_esc10_bubble.JPG)
</div>

Please find below some tutorials for a quick ramp up!
* [How to define and train my own model?](./audio_event_detection/docs/tuto/how_to_define_and_train_my_own_model.md)
* [How to fine tune a model on my own dataset?](./audio_event_detection/docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I evaluate my model before and after quantization?](./audio_event_detection/docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./audio_event_detection/docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I deploy my model?](./audio_event_detection/docs/tuto/how_to_deploy_a_model_on_a_target.md)
* [How can I evaluate my model on STM32N6 target?](./audio_event_detection/docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Audio Event Detection top readme **[here](./audio_event_detection/README.md)**


## <a id="SE">Speech Enhancement</a>
Speech Enhancement is an algorithm that enhances audio perception in a noisy environment.

<div align="center" style="width:80%; margin: auto;">

![plot](./speech_enhancement/docs/img/output_application_se.JPG)
</div>

<details open><summary><b>Speech Enhancement (SE) Models</b></summary>

| Models             | Input Resolutions | Supported Services    | Targets for deployment |
|--------------------|------------------|-----------------------|-------------------|
| [stft_tcnn](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/speech_enhancement/stft_tcnn/README.md)   | 257x40 <br>  | Full SE Services     |  [STM32N6570-DK](https://www.st.com/en/development-tools/stm32n6-ai.html) <br> |


[Full SE Services](./speech_enhancement/README.md) : training, evaluation, quantization, benchmarking, deployment

</details>

Speech Enhancement top readme **[here](./speech_enhancement/README.md)**

## <a id="HAR">Human Activity Recognition</a>
This allows to recognize various activities like walking, running, ...

<div align="center" style="width:80%; margin: auto;">

![plot](./human_activity_recognition/docs/img/output_application_har.JPG)
</div>

<details open><summary><b>Human Activity Recognition (HAR) Models</b></summary>

[Human Activity Recognition use case](human_activity_recognition)
| Models             | Input Resolutions | Supported Services    | Targets for deployment |
|--------------------|------------------|-----------------------|-------------------|
| [gmp](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/human_activity_recognition/st_gmp/README.md)   |  24x3x1<br> 48x3x1<br>  | training / Evaluation / Benchmarking / Deployment      | [B-U585I-IOT02A](./application_code/sensing/STM32U5/) <br>    |
| [ign](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/human_activity_recognition/st_ign/README.md)   |  24x3x1<br> 48x3x1<br>  | training / Evaluation / Benchmarking / Deployment      | [B-U585I-IOT02A](./application_code/sensing/STM32U5/) <br>    |

</details>

Please find below some tutorials for a quick ramp up!
* [How to define and train my own model?](./human_activity_recognition/docs/tuto/how_to_define_and_train_my_own_model.md)
* [How to fine tune a model on my own dataset?](./human_activity_recognition/docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I quickly check the performance of my model using the dev cloud?](./human_activity_recognition/docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I deploy my model?](./human_activity_recognition/docs/tuto/how_to_deploy_a_model_on_a_target.md)

Human Activity Recognition top readme **[here](./human_activity_recognition/README.md)**

## <a id="HPR">Hand Posture Recognition</a>
This allows to recognize a set of hand postures using Time of Flight (ToF) sensor.

<div align="center" style="width:80%; margin: auto;">

![plot](./hand_posture/docs/img/output_application.JPG)
</div>

<details open><summary><b>Hand Posture Recognition (HPR) Models</b></summary>

[Hand Posture Recognition use case](hand_posture)
| Models             | Input Resolutions | Supported Services    | Targets for deployment |
|--------------------|------------------|-----------------------|-------------------|
| [ST CNN 2D Hand Posture](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/hand_posture/st_cnn2d_handposture/README.md)   |  64x50x1<br>  | training / Evaluation / Benchmarking / Deployment       | [NUCLEO-F401RE](application_code/hand_posture/STM32F4) with X-NUCLEO-53LxA1 Time-of-Flight Nucleo expansion board<br>    |

</details>

Hand Posture Recognition top readme **[here](./hand_posture/README.md)**


## <a id="AFD">Arc Fault Detection</a>
Arc fault detection is used to classify electrical signals as normal or arc fault conditions.

<div align="center" style="width:80%; margin: auto;">

![plot](./arc_fault_detection/docs/img/output_application.JPG)
</div>

<details open><summary><b>Arc Fault Detection (AFD) Models</b></summary>

[Arc Fault Detection use case](arc_fault_detection)
| Models             | Input Resolutions | Supported Services    |
|--------------------|------------------|-----------------------|
| [st_conv](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/arc_fault_detection/st_conv/README.md)   |  4x512x1<br> 1x512x1 | training, evaluation, quantization, benchmarking, prediction      |
| [st_dense](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/arc_fault_detection/st_dense/README.md)   |  8x512x1<br> 1x512x1 |training, evaluation, quantization, benchmarking, prediction    |


</details>

Please find below some tutorials for a quick ramp up!
* [How can I quickly benchmark a model using ST Model Zoo?](./arc_fault_detection/docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I define and train my own model with ST Model Zoo?](./arc_fault_detection/docs/tuto/how_to_define_and_train_my_own_model.md)

Arc Fault Detection top readme **[here](./arc_fault_detection/README.md)**



## <a id="STM32 Docker Image">STM32 model zoo Docker Image</a>

A Docker-based setup is available for the STM32AI Model Zoo, including a ready-to-use image that captures the full software stack (tools, dependencies, and configuration) in a single, consistent environment. This Docker configuration reduces host-specific installation and compatibility issues, and offers a straightforward way to run the project on different platforms with identical behavior. It also makes it easier to share and reproduce workflows, whether training, evaluating, or running experiments, by keeping the runtime environment standardized across machines.

## <a id="Hugging Face">Hugging Face host</a>
The Model Zoo Dashboard is hosted in a Docker environment under the [STMicroelectronics Organization](https://huggingface.co/STMicroelectronics). This dashboard is developed using Dash Plotly and Flask, and it operates within a Docker container.
It can also run locally if Docker is installed on your system. The dashboard provides the following features:

•	Training: Train machine learning models.
•	Evaluation: Evaluate the performance of models.
•	Benchmarking: Benchmark your model using ST Edge AI Developer Cloud
•	Visualization: Visualize model performance and metrics.
•	User Configuration Update: Update and modify user configurations directly from the dashboard.
•	Output Download: Download model results and outputs.

You can also find our models on Hugging Face under the [STMicroelectronics Organization](https://huggingface.co/STMicroelectronics). Each model from the STM32AI Model Zoo is represented by a model card on Hugging Face, providing all the necessary information about the model and linking to dedicated scripts.


## Before you start
For a detailed guide on installing and setting up the model zoo and its requirements, especially when operating behind a proxy in a corporate environment, refer to the wiki article [How to install STM32 model zoo](https://wiki.st.com/stm32mcu/index.php?title=AI:How_to_install_STM32_model_zoo).

* Create an account on myST and sign in to [STEdgeAI Developer Cloud](https://stedgeai-dc.st.com/home) to access the service.
* Alternatively, install [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally and obtain the path to the `stm32ai` executable.
* If using a GPU, install the appropriate GPU driver. For NVIDIA GPUs, refer to the [CUDA and cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html). On Windows, for optimal GPU training performance, avoid using WSL. If using conda, see below for installation.
  to https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html to install CUDA and CUDNN. On Windows, it is
  not recommended to use WSL to get the best GPU training acceleration. If using conda, see below for installation.
* For Docker-based execution of the Model Zoo, see [README.md](./docker/README.md).
* Python **3.12.9** is required. Download it from [python.org](https://www.python.org/downloads/).
    * On Windows, ensure the **Add python.exe to PATH** option is selected during installation.
    * On Windows, if you plan to use the `pesq` library (for speech quality evaluation), you must have Visual Studio with C++ build tools installed. Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/visual-cpp-build-tools/).


Clone this repository:

```
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git --depth 1
cd stm32ai-modelzoo-services
```

Create a Python environment using either venv or conda:

- With venv:
  ```
  python -m venv st_zoo
  ```
- With conda:
  ```
  conda create -n st_zoo python=3.12.9
  ```

Activate your environment:

- venv (Windows):
  ```
  st_zoo\Scripts\activate.bat
  ```
- venv (Unix/Mac):
  ```
  source st_zoo/bin/activate
  ```
- conda:
  ```
  conda activate st_zoo
  ```

If using an NVIDIA GPU with conda, install CUDA libraries and set the path:
```
conda install -c conda-forge cudatoolkit=11.8 cudnn
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Then install all required Python packages:
```
pip install -r requirements.txt
```

## Initialize Git Submodules

Some application code in this repository is provided as git submodules. These submodules contain essential code for specific use cases and are not included in the main repository by default. To ensure all features and application examples work correctly, you need to initialize and update the submodules after cloning the repository:

```bash
git submodule update --init --recursive
```

This command will download all necessary submodules content, it is only needed if you plan to use deployment features.
## Practical Notes

> [!CAUTION]
> If there are any white spaces in the paths (for Python, STM32CubeIDE, or STEdgeAI Core local installation), this can result in errors. Avoid having paths with white spaces.

> [!TIP]
> In this project, we are using the ClearML library to log the results of different runs.

### ClearML Setup

1. **Sign Up**: Sign up for free to the [ClearML Hosted Service](https://app.clear.ml). Alternatively, you can set up your own server as described [here](https://clear.ml/docs/latest/docs/deploying_clearml/).

2. **Create Credentials**: Go to your ClearML workspace and create new credentials.

3. **Configure ClearML**: Create a `clearml.conf` file and paste the credentials into it. If you are behind a proxy or using SSL portals, add `verify_certificate = False` to the configuration to make it work. Here is an example of what your `clearml.conf` file might look like:

    ```ini
    api {
        web_server: https://app.clear.ml
        api_server: https://api.clear.ml
        files_server: https://files.clear.ml
        # Add this line if you are behind a proxy or using SSL portals
        verify_certificate = False
        credentials {
            "access_key" = "YOUR_ACCESS_KEY"
            "secret_key" = "YOUR_SECRET_KEY"
        }
    }

    ```

Once configured, your experiments will be logged directly and shown in the project section under the name of your project.

### MLflow Setup

In this project, we are also using the MLflow library to log the results of different runs.

#### Windows Path Length Limitation

Depending on which version of Windows OS you are using or where you place the project, the output log files might have a very long path, which might result in an error at the time of logging the results. By default, Windows uses a path length limitation (MAX_PATH) of 256 characters. To avoid this potential error, follow these steps:

1. **Enable Long Paths**: Create (or edit) a variable named `LongPathsEnabled` in the Registry Editor under `Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem` and assign it a value of `1`. This will change the maximum length allowed for the file path on Windows machines and will avoid any errors resulting due to this. For more details, refer to [Naming Files, Paths, and Namespaces](https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file).

2. **GIT Configuration**: If you are using Git, the line below may help solve the long path issue:

    ```bash
    git config --system core.longpaths true
    ```
