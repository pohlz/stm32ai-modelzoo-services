# Semantic Segmentation STM32 Model Zoo

## Directory Structure & Main Components

- **datasets/** — Contains supported datasets and step-by-step tutorials for creating, preparing, and customizing datasets for semantic segmentation:
    - `pascal_voc_2012/`, `coco_2017_pascal_voc_2012/`, `n_class_coco_2017_pascal_voc_2012/`
    - Guidance and scripts for dataset preparation
    - Tutorials for dataset creation and augmentation
- **tf** — Main Python package for all services (training, quantization, evaluation, prediction, benchmarking, deployment, etc.)
- **config_file_examples** — Ready-to-use YAML config files for all services and chains (training, quantization, evaluation, prediction, benchmarking, deployment).
- **outputs** — Stores experiment results, logs, and model artifacts (organized by date).
- **docs** — All documentation and tutorials:
    - Service-specific READMEs (training, quantization, evaluation, prediction, benchmarking, deployment, augmentation, datasets, models, overview, etc.)
    - Tutorials in `docs/tuto/`
    - Images in `docs/img/`
- **stm32ai_main.py** — Main entry point for running services and chains.
- **user_config.yaml** — User-editable config file for custom runs.

---

## Quick Start & Examples

To get started, set the `operation_mode` in your config YAML to select a service or chain. See the following documentation for details and examples:

- [Training, chain_tqe, chain_tqeb](./docs/README_TRAINING.md)
- [Quantization, chain_eqe, chain_qb](./docs/README_QUANTIZATION.md)
- [Evaluation, chain_eqeb](./docs/README_EVALUATION.md)
- [Benchmarking](./docs/README_BENCHMARKING.md)
- [Prediction](./docs/README_PREDICTION.md)
- [Deployment: STM32N6](./docs/README_DEPLOYMENT_STM32N6.md), [STM32MPU](./docs/README_DEPLOYMENT_MPU.md)

All configuration examples are in [config_file_examples/](./config_file_examples/).

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmark and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations                                                                                                                                           |
|:---------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `training`| Train a model from the variety of segmentation models in the model zoo **(BYOD)** or your own model **(BYOM)**                                       |
| `evaluation` | Evaluate the accuracy of a float or quantized model on a test or validation dataset                                                                  |
| `quantization` | Quantize a float model                                                                                                                               |
| `prediction`   | Predict the classes some images belong to using a float or quantized model                                                                           |
| `benchmarking` | Benchmark a float or quantized model on an STM32 board                                                                                               |
| `deployment`   | Deploy a model on an STM32 board                                                                                                                     |
| `chain_tqeb`  | Sequentially: training, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tqe`    | Sequentially: training, quantization of trained model, evaluation of quantized model                                                                 |
| `chain_eqe`    | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model                                                          |
| `chain_qb`     | Sequentially: quantization of a float model, benchmarking of quantized model                                                                         |
| `chain_eqeb`   | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model                             |
| `chain_qd`     | Sequentially: quantization of a float model, deployment of quantized model                                                                           |




## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up : 
- [How to define and train my own model?](./docs/tuto/how_to_define_and_train_my_own_model.md)
- [How to fine-tune a pretrained model on my own dataset?](./docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
- [How to check accuracy after quantization?](./docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
- [How to quickly benchmark model performance?](./docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
- [How to evaluate my model on STM32N6 target?](./docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Minimalistic YAML templates are available in [config_file_examples/](./config_file_examples/). All pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration YAML files.

For more details, see the [README_OVERVIEW.md](./docs/README_OVERVIEW.md).

