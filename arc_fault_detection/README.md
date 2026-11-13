# Arc fault detection (AFD) STM32 model zoo

## Directory components:
* [datasets](docs/README_DATASETS.md) placeholder for the arc fault detection datasets.
* [docs](docs/) contains all readmes and documentation specific to the arc fault detection use case.
* [config_file_examples](./config_file_examples/) contains YAML configuration file examples. It's a good place to get started if you are lost.
* [tf/src](tf/src/) contains tools to train, evaluate and quantize your model on your STM32 target.

## Tutorials and documentation: 
* [Complete AFD model zoo and configuration file documentation](docs/README_OVERVIEW.md)
* [Training service](docs/README_TRAINING.md)
* [Evaluation service](docs/README_EVALUATION.md)
* [Quantization service](docs/README_QUANTIZATION.md)
* [Benchmarking service](docs/README_BENCHMARKING.md)
* [Prediction service](docs/README_PREDICTION.md)
* [Chained modes](docs/README_CHAINED_MODES.md)
* [Datasets](docs/README_DATASETS.md)
* [Learning rate schedule](docs/README_LR_SCHEDULE.md)

All .yaml configuration examples are located in [config_file_examples](./config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, and 'b' for benchmark on an STM32 board.

| operation_mode attribute | Operations |
|:-------------------------|:-----------|
| `training`               | Train a model  |
| `evaluation`             | Evaluate the accuracy of a float or quantized model on a test or validation dataset |
| `quantization`           | Quantize a float model |
| `prediction`             | Predict the classes some arc fault events belong to using a float or quantized model |
| `benchmarking`           | Benchmark a float or quantized model on an STM32 board |
| `chain_tbqeb`            | Sequentially: training, benchmarking, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tb`               | Sequentially: training and benchmarking of trained model |
| `chain_tqe`              | Sequentially: training, quantization of trained model, evaluation of quantized model |
| `chain_eqe`              | Sequentially: evaluation of a float model, quantization, evaluation of the quantized model |
| `chain_qb`               | Sequentially: quantization of a float model, benchmarking of quantized model |
| `chain_eqeb`             | Sequentially: evaluation of a float model, quantization, evaluation of quantized model, benchmarking of quantized model |

## You don't know where to start? You feel lost?
Use the minimalistic yaml files in [config_file_examples](./config_file_examples/) and the guidance in [docs/README_OVERVIEW.md](docs/README_OVERVIEW.md) to get started quickly. You can also follow:
* [How can I quickly benchmark a model using ST Model Zoo?](./docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I define and train my own model with ST Model Zoo?](./docs/tuto/how_to_define_and_train_my_own_model.md)

