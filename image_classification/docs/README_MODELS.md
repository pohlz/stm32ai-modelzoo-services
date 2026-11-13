# Overview of image classification STM32 model zoo

The STM32 model zoo includes several models for image classification use cases pre-trained on custom and public datasets. Under each model directory, you can find the following model categories:

- `Public_pretrainedmodel_public_dataset` contains public image classification models trained on public datasets.
- `ST_pretrainedmodel_custom_dataset` contains different image classification models trained on ST custom datasets using our [training scripts](./README_TRAINING.md). 
- `ST_pretrainedmodel_public_dataset` contains different image classification models trained on various public datasets following the [training section](./README_TRAINING.md) in STM32 model zoo.

**Feel free to explore the model zoo and get pre-trained models [here](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/master/image_classification/).**

## Model Families

You can get comprehensive footprints and performance information for each model family following the links below: 

### Mobile-Optimized Architectures
You can get footprints and performance information for each model following links below:
- [mobilenetv1](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv1/README.md)  – Efficient depthwise separable convolutions (alpha variants: 0.25, 0.50, 1.0)
- [mobilenetv2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv2/README.md) – Inverted residual blocks with linear bottlenecks (alpha variants: 0.35, 1.0, 1.4)
- [fdmobilenet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/fdmobilenet/README.md)– Fast downsampling variants for reduced latency
(two variants: 0.25 and in-house designed model)

### STMicroelectronics In-house Model
- [efficientnet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/efficientnet/README.md) - A ST customization of first published version of EfficientNet tailored for STM32 platforms, and modified to be quantization-friendly 
- [st_mnistv1](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/st_mnistv1/README.md) - A ST customized topology leveraging the benefits of depthwise separable convolutions and well suited for MNIST-like datasets. 

### Standard Architectures
- [efficientnetv2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/efficientnetv2/README.md) - One of the best topology for image classification (several variants: B0 (224x224), B1 (240x240), B2 (260x260), B3 (300x300), S (384x384))
- [squeezenetv11](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/squeezenetv11/README.md) – Fire modules with squeeze and expand
- [resnet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet/README.md) - ResNetv1-8 model trained on CIFAR-10 and CIFAR-100 datasets
- [resnet50v2](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet50v2/README.md) - Well known residual connection architecture to target more challenging image datasets


To get started, update the [user_config.yaml](../user_config.yaml) file, which specifies the parameters and configuration options for the services you want to use. The  `model` section of this yaml specifically relates to the model definition. Some topologies are already registered and can be accessed by the `model_name` attribute. The exhaustive list of possible values is provided hereafter: 

- 'custom_model'
- 'st_efficientnetlcv1'
- 'st_fdmobilenetv1'
- 'st_mnistv1'
- 'efficientnetv2b0'
- 'efficientnetv2b1'
- 'efficientnetv2b2'
- 'efficientnetv2b3'
- 'efficientnetv2s'
- 'fdmobilenet_a025'
- 'fdmobilenet_a050'
- 'fdmobilenet_a075'
- 'fdmobilenet_a100'
- 'mobilenetv1_a025'
- 'mobilenetv1_a050'
- 'mobilenetv1_a075'
- 'mobilenetv1_a100'
- 'mobilenetv2_a035'
- 'mobilenetv2_a050'
- 'mobilenetv2_a075'
- 'mobilenetv2_a100'
- 'mobilenetv2_a130'
- 'mobilenetv2_a140'
- 'resnet50v2'
- 'resnet8'
- 'resnet20'
- 'resnet32'
- 'squeezenetv11'

## Quick Selection Guide

### By Inference Speed (on STM32N6570-DK, dataset food101, input size: 224x224)
- **Ultra-Fast (<5ms)**: fdmobilenet_a025 (1.29ms), st_fdmobilenetv1 (1.67ms), mobilenetv1_a025 (2.37ms)   
- **Fast (5-10ms)**: mobilenetv1_a050 (5.38ms), mobilenetv2_a035 (5.43ms), squeezenetv11 (7.97ms)
- **Moderate (10-20ms)**: mobilenetv1_a100 (16.36ms), mobilenetv2_a100 (16.43ms), st_efficientnetlcv1 (17.31ms)
- **Balanced (20-40ms)**: 
- **Large (>40ms)**: efficientnetv2b0 (57.05ms), efficientnetv2b1 (80.50ms), efficientnetv2b2 (140.38ms), resnet50v2 (238.49ms)

### By Model Size (weights)
- **Tiny (<500KB)**: fdmobilenet_a025 (148KB), st_fdmobilenetv1 (167KB), mobilenetv1_a025 (241KB), mobilenetv2_a035 (423KB)
- **Small (500KB-1.5MB)**: squeezenetv11 (753KB), mobilenetv1_a050 (865KB)
- **Medium (1.5-3MB)**: mobilenetv2_a100 (2336KB)
- **Large (>3MB)**: mobilenetv1_a100 (3348KB), efficientnetv2b0/b1/b2 (4237KB to 6885KB), resnet50v2 (13268KB), efficientnetv2S (14837KB)

### By RAM Requirements (on STM32N6570-DK, dataset food101, input size: 224x224)
- **Internal RAM Only (<1MB)**: Most MobileNet v1 and v2, st_efficientnetlcv1, st_fdmobilenetv1...
- **Requires External RAM**: efficientnetv2 b2/b3/s (528KB to ~3500KB), resnet50v2 (2352KB)

## Platform Support

- **Primary Target**: STM32N6570-DK with NPU acceleration. The smallest can also be considered for STM32H7.
- **Framework**: some models are avaliable in tflite full Int8, others are available in ONNX QDQ full Int8 or mixed-precision (Int8/Int4). Activations are always in Int8.
- **STEdgeAI Core**: Version 3.0.0+

## Performance Notes
- All inference times measured on STM32N6570-DK with NPU/MCU execution
- Models using only internal RAM are preferred for simpler deployment
- External RAM models (some efficientnetv2 and resnet50v2) offer different accuracy/performance tradeoffs

**Feel free to explore the model zoo and get pre-trained models [here](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/master/image_classification/).**

For training and deployment guidance, refer to the STM32 AI model zoo documentation.