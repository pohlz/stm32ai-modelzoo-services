# Overview of PyTorch STM32 Model Zoo for Image Classification

The STM32 model zoo includes several PyTorch-based models for image classification use cases, converted to ONNX format and optimized for STM32N6 NPU deployment. All models are pre-trained on ImageNet and and converted to ONNX format with QDQ quantization.

## Model Categories

- `Public_pretrainedmodel_public_dataset` contains public PyTorch image classification models trained on public datasets and converted to ONNX format with QDQ quantization.
- `ST_pretrainedmodel_public_dataset` contains public PyTorch image classification models trained on public dataset by ST. 


**Explore the complete PyTorch model zoo with pre-trained models optimized for STM32N6.**

## Model Families

You can get comprehensive footprints and performance information for each model family following the links below:

### Mobile-Optimized Architectures
- [mobilenet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenet_pt/README.md) – Efficient depthwise separable convolutions (alpha variants: 0.25, 0.50, 0.75)
- [mobilenetv2_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv2_pt/README.md) – Inverted residual blocks with linear bottlenecks (alpha/width variants)
- [mobilenetv4_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mobilenetv4_pt/README.md) – Latest MobileNet iteration with enhanced efficiency
- [fdmobilenet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/fdmobilenet_pt/README.md) – Fast downsampling variants for reduced latency

### Lightweight Efficient Models
- [shufflenetv2_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/shufflenetv2_pt/README.md) – Channel shuffle operations (0.5x, 1.0x width multipliers)
- [mnasnet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/mnasnet_pt/README.md) – Mobile neural architecture search optimized models
- [regnet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/regnet_pt/README.md) – Network design space exploration

### STMicroelectronics In-house Model
- [st_resnet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/st_resnet_pt/README.md) – Custom ResNet variants optimized for STM32 (pico, nano, micro, milli, tiny)

### Standard Architectures
- [resnet18_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/resnet18_pt/README.md) – Classic residual networks (width variant 0.25)
- [preresnet18_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/preresnet18_pt/README.md) – Pre-activation ResNet variant
- [dla_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/dla_pt/README.md) – Deep Layer Aggregation with hierarchical features
- [squeezenet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/squeezenet_pt/README.md) – Fire modules with squeeze and expand
- [sqnxt_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/sqnxt_pt/README.md) – More efficient SqueezeNet variants

### Specialized Architectures
- [hardnet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/hardnet_pt/README.md) – Harmonic DenseNet design
- [peleenet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/peleenet_pt/README.md) – Efficient DenseNet variant
- [proxylessnas_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/proxylessnas_pt/README.md) – Direct NAS on target hardware
- [semnasnet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/semnasnet_pt/README.md) – Squeeze-and-Excitation MnasNet
- [darknet_pt](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/darknet_pt/README.md) – Lightweight feature extraction backbone


## Quick Selection Guide

### By Inference Speed (on STM32N6570-DK)
- **Ultra-Fast (<5ms)**: fdmobilenet_a025 (1.88ms), mobilenet_a025 (2.98ms)
- **Fast (5-10ms)**: fdmobilenet_a050 (4.07ms), mobilenet_a050 (6.55ms), shufflenetv2_x050 (8.35ms)
- **Moderate (10-20ms)**: mobilenetv2_a050 (10.08ms), mobilenetv4small (13.74ms), st_resnetmicro (13.83ms)
- **Balanced (20-40ms)**: mobilenetv2_a100 (20.35ms), st_resnettiny (24.10ms), proxylessnas (27.65ms)

### By Model Size
- **Tiny (<500KB)**: fdmobilenet_a025 (377KB), mobilenet_a025 (469KB), st_resnetpico (607KB)
- **Small (500KB-1.5MB)**: fdmobilenet_a050 (973KB), mobilenet_a050 (1.3MB), shufflenetv2_x050 (1.4MB)
- **Medium (1.5-3MB)**: mobilenetv2_a050 (1.97MB), mnasnet_d050 (2.3MB), peleenet (2.75MB)
- **Large (>3MB)**: mobilenetv4small (3.76MB), mobilenetv2_a100 (3.81MB), st_resnettiny (4.06MB)

### By RAM Requirements
- **Internal RAM Only (<1MB)**: Most MobileNet, ShuffleNet, and ST ResNet variants
- **Requires External RAM**: DLA family (~6MB), SqueezeNext family (3-9MB), SqueezeNetv10 (~6.7MB)

## Platform Support
All PyTorch models are specifically optimized for:
- **Primary Target**: STM32N6570-DK with NPU acceleration
- **Framework**: ONNX with QDQ INT8 quantization
- **STEdgeAI Core**: Version 3.0.0+

## Performance Notes
- All inference times measured on STM32N6570-DK with NPU/MCU execution
- Internal RAM usage varies from 294 KiB (fdmobilenet_a025) to 11,350 KiB (sqnxt23_x200)
- Models using only internal RAM are preferred for simpler deployment
- External RAM models (DLA, SqueezeNext) offer different accuracy/performance tradeoffs

**Feel free to explore the model zoo and get pre-trained models [here](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/master/image_classification/).**

For training and deployment guidance, refer to the STM32 AI model zoo documentation.
