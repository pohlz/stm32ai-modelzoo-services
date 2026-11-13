# Face Detection STM32 Model Zoo

## Directory Components:
* [datasets](./docs/README_DATASETS.md) placeholder for the Face Detection datasets.
* [docs](./docs/) contains all readmes and documentation specific to the Face Detection use case.
* [src](./docs/README_OVERVIEW.md) contains tools to evaluate, benchmark, quantize and deploy your model on your STM32 target.

## Quick & easy examples:
The `operation_mode` top-level attribute specifies the operations or the service you want to execute. This may be a single operation or a set of chained operations.

You can refer to the README links below that provide typical examples of operation modes and tutorials on specific services:
- [quantization, chain_eqe, chain_qb](./docs/README_QUANTIZATION.md)
- [evaluation, chain_eqeb](./docs/README_EVALUATION.md)
- [benchmarking](./docs/README_BENCHMARKING.md)
- [prediction](./docs/README_PREDICTION.md)
- deployment, [STM32N6](./docs/README_DEPLOYMENT_STM32N6.md)

All `.yaml` configuration examples are located in the [config_file_examples](./config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmarking, and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations |
|:---------------------------|:-----------|
| `evaluation` | Evaluate the accuracy of a float or quantized model on a test or validation dataset|
| `quantization` | Quantize a float model |
| `prediction`   | Predict the classes some images belong to using a float or quantized model |
| `benchmarking` | Benchmark a float or quantized model on an STM32 board |
| `deployment`   | Deploy a model on an STM32 board |
| `chain_eqe`    | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model |
| `chain_qb`     | Sequentially: quantization of a float model, benchmarking of quantized model |
| `chain_eqeb`   | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model |
| `chain_qd`     | Sequentially: quantization of a float model, deployment of quantized model |


The `model_type` attributes currently supported for the object detection are:
- `facedetect_front` : BlazeFace Front 128x128 is a lightweight and efficient Face Detection model optimized for real-time applications on embedded devices. It is a variant of the BlazeFace architecture, designed specifically for detecting frontal faces at a resolution of 128x128 pixels.
- `yunet` : YuNet is a high-performance Face Detection model developed by MediaTek. It is designed to efficiently detect faces in images and videos, making it suitable for real-time applications on embedded devices. YuNet utilizes a deep learning architecture that combines convolutional neural networks (CNNs) with advanced techniques to achieve accurate and fast Face Detection.

## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up : 
* [How can I use my own dataset?](./docs/tuto/how_to_use_my_own_object_detection_dataset.md)
* [How can I check the accuracy after quantization of my model?](./docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I evaluate my model on STM32N6 target?](./docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Remember that minimalistic yaml files are available [here](./config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!

