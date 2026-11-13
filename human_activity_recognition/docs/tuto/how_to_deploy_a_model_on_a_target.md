# How to deploy a model on a B-U585I-IOT02A board?

This tutorial explains how to deploy a model from the ST public model zoo directly on your STM32 target board. In this version deployment on the B-U585I-IOT02A is supported.


## Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Deployment: 
    - To simply deploy a quantized model on the B-U585I-IOT02A board

For any details regarding the parameters of the config file, you can look to the [Deployment documentation](../README_DEPLOYMENT.md)


## Deployment yaml configuration example:

Below is an example of configuration file used to deploy a pretrained and quantized yamnet from ST Model Zoo on a B-U585I-IOT02A.
You can find other example of user_config.yaml [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/human_activity_recognition/config_file_examples)

The most important parts here are to define:
- The model path of the model to be deployed. 
- The operation mode to deployment
- The name of the classes
- Copy the preprocessing used during the training of the model 
- Select the example code deployed (sensing_free_rtos or sensing_thread_x)
- Set the target to B-U585I-IOT02A
- Add your local path of STM32CubeIDE


```yaml
# user_config.yaml

model:
   model_path: ../../stm32ai-modelzoo/human_activity_recognition/ign/ST_pretrainedmodel_custom_dataset/mobility_v1/ign_wl_24/ign_wl_24.h5     # Path to the model file to deploy

operation_mode: deployment

dataset:
  name: mobility_v1 # mandatory
  class_names: [Stationary,Walking,Jogging,Biking] # optional

preprocessing: # Mandatory
  gravity_rot_sup: true # mandatory
  normalization: false # mandatory

tools:
  stedgeai:
    optimization: balanced
    on_cloud: True # True for online, False for local
    # path to st edge AI, see below
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
  # path to STM32CubeIDE
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

deployment:
  c_project_path: ../../application_code/sensing_thread_x/STM32U5/
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32U5
    # board for deployment
    board: B-U585I-IOT02A

mlflow:
  uri: ./experiments_outputs/mlruns

hydra:
  run:
    dir: ./experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

```

You can look at user_config.yaml examples for any operation mode [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/human_activity_recognition/config_file_examples)

## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the folder human_activitiy_recognition/). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```

## Local deployment:

You can use the [STM32 developer cloud](https://stedgeai-dc.st.com/home) to access the STEdge AI functionalities without installing the software. This requires internet connection and making a free account. Or, alternatively, you can install [STEdge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally. In addition to this you will also need to install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) for building the embedded project.
 
For local installation :
 
- Download and install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html).
- If opting for using [STEdge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally, download it then extract both `'.zip'` and `'.pack'` files.
The detailed instructions on installation are available in this [wiki article](https://wiki.st.com/stm32mcu/index.php?title=AI:How_to_install_STM32_model_zoo).

## Application on the board

Once flashed the board can be connected through a serial terminal and the output of the inference can be seen in the serial terminal. 
Find more information [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/human_activity_recognition/docs/README_DEPLOYMENT.md)


