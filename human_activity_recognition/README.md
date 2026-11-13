# Human Activity Recognition STM32 Model Zoo


## Directory components:
* [config_file_examples](./config_file_examples/) includes the sample minimalistic configuration files to run different operation modes, such as [training](./config_file_examples/training_config.yaml), [deployment](./config_file_examples/deployment_config.yaml), [evaluation](./config_file_examples/evaluation_config.yaml) and [benchmarking](./config_file_examples/benchmarking_config.yaml). These files can be used directly to launch one of the operation modes with little to no editions.
* [datasets](./datasets/) placeholder for the human activity recognition datasets.
* [docs](./docs/) contains all readmes and documentation specific to the human activity recognition use case.
* [tf](./tf/) contains all the source code to build, train, evaluate, benchmark, quantize and deploy your human_activity_recognition model on your STM32 target.

## Quick & easy examples:
The `operation_mode` top-level attribute specifies the operations or the service you want to execute. This may be single operation or a set of chained operations.

You can refer to following readme tutorials that provide typical examples of operation modes:

   - [training, chain_tb](./docs/README_TRAINING.md)
   - [evaluation](./docs/README_EVALUATION.md)
   - [benchmarking](./docs/README_BENCHMARKING.md)
   - [deployment](./docs/README_DEPLOYMENT.md)

All .yaml configuration examples can be found in the [config_file_examples](./config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, and 'b' for benchmark on an STM32 board.

| operation_mode attribute | Operations |
|:---------------------------|:-----------|
| `training`| Train an HAR model |
| `evaluation` | Evaluate the accuracy of a pretrained float model on a test or validation dataset |
| `benchmarking` | Benchmark a pretrained float model on an STM32 board |
| `deployment`   | Deploy a pretrained float model on an STM32 board |
| `chain_tb`  | Sequentially: train, and then benchmark an HAR model |


## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up : 
* [How to define and train my own model?](./docs/tuto/how_to_define_and_train_my_own_model.md)
* [How to fine tune a model on my own dataset?](./docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I quickly check the performance of my model using the dev cloud?](./docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I deploy my model?](./docs/tuto/how_to_deploy_a_model_on_a_target.md)

Remember that minimalistic yaml files are available [here](./config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/human_activity_recognition/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!

