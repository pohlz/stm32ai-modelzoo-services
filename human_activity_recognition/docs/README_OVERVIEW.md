# Human Activity Recognition (HAR) STM32 model zoo

This directory contains scripts and tools for training, benchmarking, evaluating, and deploying HAR models using **TensorFlow** & **STEdgeAI Core**.

Remember that minimalistic yaml files are available [here](../config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml files that are used to generate them. These are very good starting points to start playing with!

## Training
In [README_TRAINING.md](./README_TRAINING.md), you can find a step by step guide plus the necessary scripts and tools to train and evaluate the HAR models on custom and public datasets.

## Evaluate
In [README_EVALUATION](./README_EVALUATION.md), you can find a step by step guide plus the necessary scripts and tools for evaluating your model performances if datasets are provided.

## Benchmarking
In [README_BENCHMARKING](./README_BENCHMARKING.md), you can find a step by step guide plus the necessary scripts and tools to benchmark your model using STM32Cube.AI through our STM32Cube.AI Developer Cloud Services or from the local download.


## Deployment
In [README_DEPLOYMENT](./README_DEPLOYMENT.md), you can find a step by step guide plus the necessary scripts and tools to deploy your own pre-trained HAR model on your STM32 board using STM32Cube.AI.

You can also use a pretrained model from our `Human Activity Recognition STM32 model zoo`. Check out the available models in the [human_activity_recognition/pretrained_models](./README_MODELS.md) directory, or on the [model zoo on GitHub](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/master/human_activity_recognition/).
