# Depth Estimation STM32 model zoo

**Depth estimation** is a computer vision task that involves predicting the distance between each pixel in an image and the camera. It is a key component in applications such as robotics, autonomous navigation, and augmented reality, where understanding the 3D structure of a scene is essential.


## Overview

This use case provides tools for running **prediction** and **benchmark** services on depth estimation models using the STM32AI Model Zoo framework.

A minimal set of example configuration files is available in the [`config_file_examples/`](../config_file_examples/) folder to help you get started with supported services. Additionally, the [STM32AI Model Zoo repository](https://github.com/STMicroelectronics/stm32ai-modelzoo/) offers ready-to-use models and their configuration `.yaml` files.


## How to Use

To run the services, use the main launcher script [`stm32ai_main.py`](../stm32ai_main.py) with a configuration file (e.g. [`user_config.yaml`](../user_config.yaml)).

For general instructions on writing and customizing your YAML files, please refer to the [README_BENCHMARKING.md](./README_BENCHMARKING.md) and [README_PREDICTION.md](./README_PREDICTION.md).



## Supported Services

| Service      | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `predict`    | Runs inference on a single image or a folder of images using a depth model. |
| `benchmark`  | Evaluates model performance (e.g. speed, memory, and optionally accuracy).  |

Other services like training, quantization, or evaluation are not currently available for this use case.
