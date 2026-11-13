# Neural style transfer STM32 model zoo

## Directory Components:
* [datasets](docs/README_DATASETS.md) placeholder for the neural style transfer datasets.
* [docs](docs/) contains all readmes and documentation specific to the neural style transfer use case.
* [tf/src](tf/src/) contains tools to do predictions or benchmark your model on your STM32 target.

## Quick & easy examples:
The `operation_mode` top-level attribute specifies the operations or the service you want to execute.

You can refer to readme links below that provide typical examples of operation modes, and tutorials on specific services:

   - [benchmarking](./docs/README_BENCHMARKING.md)
   - [prediction](./docs/README_PREDICTION.md)

Deployment for this use-case is not supported currently. It will be added in future releases.
All .yaml configuration examples are located in [config_file_examples](./config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below:

| operation_mode attribute | Operations |
|:---------------------------|:-----------|
| `prediction`   | Predict and generate the image blended with the style pattern using a float or quantized model |
| `benchmarking` | Benchmark a float or quantized model on an STM32 board |

The `model_type` attributes currently supported for the neural style transfer are:
- `xinet_picasso_muse` : is a lightweight Neural Style Transfer approach based on [XiNets](https://openaccess.thecvf.com/content/ICCV2023/papers/Ancilotto_XiNet_Efficient_Neural_Networks_for_tinyML_ICCV_2023_paper.pdf), neural networks especially developed for microcontrollers and embedded applications. It has been trained using the painting *La Muse* of **Pablo Picasso**. This model achieves an extremely lightweight transfer style mechanism and high-quality stylized outputs, significantly reducing computational complexity.


## You don't know where to start? You feel lost?

Remember that minimalistic yaml files are available [here](./config_file_examples/) to get a hands-on experience with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start getting familiar with the model zoo!