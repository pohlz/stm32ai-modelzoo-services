# Benchmarking of Hand Posture model

The Hand Posture Model Benchmarking service is a powerful tool that enables users to evaluate the performance of their Hand Posture models built with Keras (.keras and .h5). With this service, users can easily upload their model and configure the settings to benchmark it and generate various metrics, including memory footprints and inference time. This can be achieved by utilizing the [STEdge AI Developer Cloud](https://stedgeai-dc.st.com/home) to benchmark on different STM32 target devices or by using [STEdge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) to estimate the memory footprints.

<details open><summary><a href="#1"><b>1. Configure the YAML file</b></a></summary><a id="1"></a>

To use this service and achieve your goals, you can use the [user_config.yaml](../user_config.yaml) or directly update the [benchmarking_config.yaml](../config_file_examples/benchmarking_config.yaml) file and use it. This file provides an example of how to configure the benchmarking service to meet your specific needs.

Alternatively, you can follow the tutorial below, which shows how to benchmark your pre-trained Hand Posture model using our evaluation service.

<ul><details open><summary><a href="#1-1">1.1 Setting the model and the operation mode</a></summary><a id="1-1"></a>

The first step is to set the operation_mode to `benchmarking` and specify the model path as in the following example: 

```yaml
operation_mode: benchmarking

model:
   model_path: ../../stm32ai-modelzoo/hand_posture/st_cnn2d_handposture/ST_pretrainedmodel_custom_dataset/ST_VL53L8CX_handposture_dataset/st_cnn2d_handposture_8classes/st_cnn2d_handposture_8classes.keras

```

In this example, the path to the st_cnn2d_handposture_8classes model (for VL53L8CX sensor) is provided in the `model.model_path` parameter.

</details></ul>
<ul><details open><summary><a href="#1-2">1.2 Set benchmarking tools and parameters</a></summary><a id="1-2"></a>

The [STEdge AI Developer Cloud](https://stedgeai-dc.st.com/home) allows you to benchmark your model and estimate its footprints and inference time for different STM32 target devices. To use this feature, set the `on_cloud` attribute to _True_. Alternatively, you can use [STEdge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) to benchmark your model and estimate its memory footprints for STM32 target devices locally. To do this, make sure to add the path to the `stedgeai` executable under the `path_to_stedgeai` attribute and set the `on_cloud` attribute to False.

The `optimization` defines the optimization used to generate the C model for benchmarking, options are: "balanced", "time", "ram".

The `board` attribute is used to provide the name of the STM32 board to benchmark the model on. The available board choices are 'STM32H747I-DISCO', 'STM32H7B3I-DK', 'STM32F469I-DISCO', 'B-U585I-IOT02A', 'STM32L4R9I-DISCO', 'NUCLEO-H743ZI2', 'STM32H747I-DISCO', 'STM32H735G-DK', 'STM32F769I-DISCO', 'NUCLEO-G474RE', 'NUCLEO-F401RE', and 'STM32F746G-DISCO'.

```yaml
tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

benchmarking:
   board: NUCLEO-F401RE     # Name of the STM32 board to benchmark the model on
```

</details></ul>
</details>
<details open><summary><a href="#2"><b>2. Benchmark your model</b></a></summary><a id="2"></a>

If you chose to modify the [user_config.yaml](../user_config.yaml), you can evaluate the model by running the following command from the UC folder:

```bash
python stm32ai_main.py
```
If you chose to update the [benchmarking_config.yaml](../config_file_examples/benchmarking_config.yaml) and use it, then run the following command from the UC folder: 

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name benchmarking_config.yaml
```
Note that you can provide YAML attributes as arguments in the command, as shown below:

```bash
python stm32ai_main.py operation_mode='benchmarking'
```

</details>
<details open><summary><a href="#3"><b>3. Visualizing the Benchmarking Results</b></a></summary><a id="3"></a>

To view the detailed benchmarking results, you can access the log file `stm32ai_main.log` located in the directory `./tf/src/experiments_outputs/<date-and-time>`. Additionally, you can navigate to the `./tf/src/experiments_outputs` directory and use the MLflow Webapp to view the metrics saved for each trial or launch. To access the MLflow Webapp, run the following command:

```bash
mlflow ui
``` 

This will open a browser window where you can view the metrics and results of your benchmarking trials.

</details>
