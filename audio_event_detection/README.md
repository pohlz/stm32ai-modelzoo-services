# Audio event detection (AED) STM32 model zoo

For the Google `kws_streaming` TensorFlow custom model and train/quantize/evaluate configuration, see [BC-ResNet TensorFlow](docs/README_BCRESNET_TF.md).

For the original Qualcomm PyTorch custom KWS model, see [BC-ResNet PyTorch setup and configuration](docs/README_BCRESNET_PT.md).

## Directory components:
* [docs](docs/) contains all readmes and documentation specific to the audio event detection use case.
* [tf/src](docs/README_OVERVIEW.md) contains tools to train, evaluate, benchmark, quantize and deploy your model on your STM32 target.
* [config_files_example](./config_file_examples/) contains YAML configuration file examples. It's a good place to get started if you are lost.
* [datasets](./datasets/) is a placeholder folder for AED datasets. You do not need to use it, it is there for convenience.

## Tutorials and documentation: 
* [Complete AED model zoo and configuration file documentation](docs/README_OVERVIEW.md)
* [A short tutorial on training a model using the model zoo](docs/README_TRAINING.md)
* [A short tutorial on quantizing a model using the model zoo](docs/README_QUANTIZATION.md)
* [A short tutorial on deploying a model on an STM32 board](docs/README_DEPLOYMENT.md)


The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmark and 'd' for deployment on an STM32 board.

| operation_mode | Operations |
|:-------------------------|:-----------|
| `training`               | Train a model  |
| `evaluation`             | Evaluate the accuracy of a float or quantized model on a test or validation dataset|
| `quantization`           | Quantize a float model |
| `prediction`             | Predict the classes some audio events belong to using a float or quantized model |
| `benchmarking`           | Benchmark a float or quantized model on an STM32 board |
| `deployment`             | Deploy a model on an STM32 board |
| `chain_tqeb`             | Sequentially: training, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tqe`              | Sequentially: training, quantization of trained model, evaluation of quantized model |
| `chain_eqe`              | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model |
| `chain_qb`               | Sequentially: quantization of a float model, benchmarking of quantized model |
| `chain_eqeb`             | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model |
| `chain_qd`               | Sequentially: quantization of a float model, deployment of quantized model |


## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up : 
* [How to define and train my own model?](./docs/tuto/how_to_define_and_train_my_own_model.md)
* [How to fine tune a model on my own dataset?](./docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I evaluate my model before and after quantization?](./docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I deploy my model?](./docs/tuto/how_to_deploy_a_model_on_a_target.md)
* [How can I evaluate my model on STM32N6 target?](./docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Remember that minimalistic yaml files are available [here](./config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are a great way to get started.

## Changes in version 4.0

### Interface changes

- In the YAML config files, the `model` section of has been moved out of the `training` section and given its own standalone section.
- The `model_path` attribute, previously under the `general` section has moved to this new section.
- The `name` attribute has been changed to `model_name`.
- The model names of each model have been slightly changed.
- For more details on these changes, how the new `model` section works, and the `model_name` associated with each model, consult sections 3.5 of the [main README](docs/README_OVERVIEW.md), and appendix A of the same README.
