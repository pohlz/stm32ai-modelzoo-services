# Benchmarking of Human Activity Recognition (HAR) models

The benchmarking functionality of the Human Activity Recognition (HAR) models enables users to evaluate the performance of their pretrained Keras (.h5, and .keras) models. With benchmarking service, users can easily configure the settings of STEdge AI Core to benchmark the keras models and generate various metrics, including memory (RAM and FLASH) and computational (MACs and inference time) footprints. The provided scripts can perform benchmarking by either utilizing the [STEdge AI Developer Cloud](https://stedgeai-dc.st.com/home) to benchmark on different STM32 target boards, or by using [STEdge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) to estimate (only) the memory footprints locally.

## <a id="">Table of contents</a>

<details open><summary><a href="#1"><b>1. Configure the yaml file</b></a></summary><a id="1"></a> 

To use this service and achieve your goals, you can use either the [user_config.yaml](../user_config.yaml) or directly update and use the [benchmarking_config.yaml](../config_file_examples/benchmarking_config.yaml) file. This document provides an example of how to configure the benchmarking service to meet your specific needs.

<ul><details open><summary><a href="#1-1">1.1 Setting the model and the operation mode</a></summary><a id="1-1"></a>

As mentioned previously, users can either use the minimalistic example [configuration file for the benchmarking](../config_file_examples/benchmarking_config.yaml) file or alternatively follow the steps below to modify all the sections of the [user_config.yaml](../user_config.yaml) main YAML file. 

The first thing to be configured is the operation_mode in the configuration file. Set it to `benchmarking`.

```yaml
general:
  project_name: human_activity_recognition # optional, if not provided <human_activity_recognition> is used 
operation_mode: benchmarking

```

</details></ul>
<ul><details open><summary><a href="#1-2">1.2 Set model path for the benchmarking</a></summary><a id="1-2"></a>
The model that is to be benchmarked is provided through setting the model path in the `model.model_path` parameter in the configuration file, as shown below:

```yaml
model:
  model_path: ../../stm32ai-modelzoo/human_activity_recognition/st_ign/ST_pretrainedmodel_custom_dataset/mobility_v1/st_ign_wl_24/st_ign_wl_24.keras # mandatory
```
</details></ul>
<ul><details open><summary><a href="#1-3">1.3 Set benchmarking tools and parameters</a></summary><a id="1-3"></a>

The [STM32Cube.AI Developer Cloud](https://stedgeai-dc.st.com/home) allows you to benchmark your model and estimate its footprints and inference time for different STM32 target boards. To do this, the user will need an internet connection and a free account on [st.com](https://www.st.com). Also, the user needs to set the `on_cloud` attribute to `True`. Alternatively, you can use a locally installed CLI of [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) to benchmark your model and estimate its footprints for STM32 target devices locally (no inference time with this option). To do this, make sure to provide the path to the `stedgeai` executable under the `path_to_stedgeai` attribute and set the `on_cloud` attribute to `False`.

The `optimization` defines the optimization used to generate the C-model. Available choices are: 
- balanced (default option, uses a balanced approach for optimizing the RAM and inference time)
- time (optimizes for the best inference time and can result in a bigger RAM consumption)
- ram (optimizes for the best RAM size and can result in a longer inference time)

The `board` attribute is used to provide the name of the STM32 board to benchmark the model. Various choices are available. 
After the configuration of these parameters, the sections of the configuration file should look like below:
```yaml
tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
                     # replace the paths with your path of STM32Cube.AI and STM32CubeIDE
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

benchmarking:
   board: B-U585I-IOT02A     # Name of the STM32 board to benchmark the model on
          # available choices
          # [STM32H747I-DISCO, STM32H7B3I-DK, STM32F469I-DISCO, B-U585I-IOT02A,
          # STM32L4R9I-DISCO, NUCLEO-H743ZI2, STM32H747I-DISCO, STM32H735G-DK,
          # STM32F769I-DISCO, NUCLEO-G474RE, NUCLEO-F401RE, STM32F746G-DISCO]

```
</details></ul>
</details>
<details open><summary><a href="#2"><b>2. Benchmark your model</b></a></summary><a id="2"></a>

If you chose to modify the [user_config.yaml](../user_config.yaml), you can benchmark the model by running the following command from the UC folder after the file is modified:

```bash
python stm32ai_main.py
```
If you chose to update the [benchmarking_config.yaml](../config_file_examples/benchmarking_config.yaml) and use it, then run the following command from the UC folder: 

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name benchmarking_config.yaml
```
Note that you can also overwrite the parameters directly in the CLI by using a provided YAML file. An example of overwriting the `operation_mode` and `model_path` is given below:

```bash
python stm32ai_main.py operation_mode='benchmarking' model.model_path='../pretrained_models/ign/ST_pretrainedmodel_custom_dataset/mobility_v1/ign_wl_24/ign_wl_24.h5'
```

</details>
<details open><summary><a href="#3"><b>3. Visualizing the Benchmarking Results</b></a></summary><a id="3"></a>

The results of the benchmark are printed in the terminal. However, you can also access the results later for the previously run benchmarks either by manually viewing them or by using `mlflow`. To view the detailed benchmarking results, you can access the log file `stm32ai_main.log` located in the directory `experiments_outputs/<launch-date-and-time>`. Additionally, you can navigate to the `experiments_outputs` directory and use the MLflow Webapp to view the metrics saved for each trial or launch. To access the MLflow Webapp, run the following command:

```bash
mlflow ui
``` 

This will open a browser window where you can view the metrics and results of your different experiments run before.

</details>
