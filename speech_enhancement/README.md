# Speech enhancement STM32 model zoo

## Directory Components:
* [docs](./docs/) contains all readmes and documentation specific to the speech enhancement use case.
* [pt/src](./pt/src) contains code to train, evaluate, benchmark, quantize and deploy speech enhancement models.
* [datasets/](./datasets) is a placeholder folder for speech enhancement datasets. Usage of this folder is optional, you can store your datasets wherever you please.
* [config_file_examples](./config_file_examples/) contains examples for various configuration files used to run the model zoo. It's a great place to get started.


## Quick & easy examples:
The `operation_mode` top-level parameter in the [configuration file](./user_config.yaml) specifies the operations or the service you want to execute. This may be a single operation or a set of chained operations.

You can refer to the README links below that provide typical examples of operation modes, and tutorials on specific services:
   - [training](./docs/README_TRAINING.md)
   - [quantization](./docs/README_QUANTIZATION.md)
   - [evaluation](./docs/README_EVALUATION.md)
   - [deployment](./docs/README_DEPLOYMENT.md)

To help you get started, configuration file examples are located in the [config_file_examples](./config_file_examples/) folder.

The different values of the `operation_mode` parameter and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmark and 'd' for deployment on an STM32 board.

| operation_mode | Operations |
|:---------------------------|:-----------|
| `training`| Train a speech enhancement model, using an architecture provided in the zoo, or your own custom architecture.|
| `evaluation` | Evaluate a float or a quantized speech enhancement model|
| `quantization` | Quantize a speech enhancement model |
| `benchmarking` | Benchmark a float or quantized model on an STM32 board |
| `deployment`   | Deploy a speech enhancement model on an STM32 board |
| `chain_tqeb`  | Sequentially: training, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tqe`    | Sequentially: training, quantization of trained model, evaluation of quantized model |
| `chain_eqe`    | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model |
| `chain_qb`     | Sequentially: quantization of a float model, benchmarking of quantized model |
| `chain_eqeb`   | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model |
| `chain_qd`     | Sequentially: quantization of a float model, deployment of quantized model |

## You don't know where to start? You feel lost?
Remember that minimalistic yaml files are available [here](./config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with !


## Changes in update 4.0

### Interface changes

- The `model_type` attribute in the `model` section has been renamed to `model_name`. It functions the same as before the update.

- The `onnx_path` attribute in the `model` section has been renamed to `model_path`. It functions the same as before the update.
