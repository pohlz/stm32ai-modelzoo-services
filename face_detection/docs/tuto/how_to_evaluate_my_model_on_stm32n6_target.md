# How to evaluate my model on STM32N6 target?

The evaluation of a model consists in running several inferences on a representative test set in order to get a quality metric of the model, like the accuracy, the mAP, the OKS or any other depending on the UC. This evaluation can be done on :
   - "host" using TensorFlow or ONNX Run Times and executed on the host machine.
   - "stedgeai_host" using a DLL containing emulated STM32 kernels implementation and executed on the host machine
   - "stedgeai_n6" using a generic test application and executed on the STM32N6 target


## Environment setup:
The evaluation on the target requires installation and configuration of STEdgeAI Core you can find here :
- [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html)
- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)

A few configurations are required, please find below an example following a standard installation of STEdgeAI_Core v4.x.

- The 'C:/ST/STEdgeAI_Core/4.x/scripts/N6_scripts/config_n6l.json' file should be updated to configure the N6 loader.
```json
{
	// The 2lines below are _only used if you call n6_loader.py ALONE (memdump is optional and will be the parent dir of network.c by default)
	"network.c": "C:/ST/STEdgeAI_Core/4.0/script/N6_scripts/st_ai_output/network.c",
	//"memdump_path": "C:/Users/foobar/CODE/stm.ai/stm32ai_output",
	// Location of the "validation" project  + build config name to be built (if applicable)
	// If using the provided project, valid build_conf names are "N6-DK" (CR5 boards), "N6-DK-legacy" (older-than-CR5-boards); "N6-Nucleo" can also be used for IAR project.
	"project_path": "C:/ST/STEdgeAI_Core/4.0/Projects/STM32N6570-DK/Applications/NPU_Validation",
	"project_build_conf": "N6-DK",
	// Skip programming weights to earn time (but lose accuracy) -- useful for performance tests
	"skip_external_flash_programming": false,
	"skip_ram_data_programming": false
}
```
- The 'C:/ST/STEdgeAI_Core/4.x/scripts/N6_scripts/config.json' file should be updated to indicate the paths to find the external tools.
```json
{
	// Set Compiler_type to either gcc or iar
	"compiler_type": "gcc",
	"cubeide_path":"C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE"
}
```
Please refer to [stedge ai core getting started on how to evaluate a model on STM32N6 board](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_getting_started.html#ref_tools_config_n6l_json) for more information on how it works and on the setup.


## Before launching the stm32ai_eval_on_target.py script:
The script to be used for the evaluation on target is taking as parameter a configuration file. The one to use and to adapt is [evaluation_n6_config.yaml](../../config_file_examples/evaluation_n6_yunet_config.yaml) in config_file_examples folder.
It is using a standard configuration file used for evaluation, with a few more parameters to define.
Below are the main parameters to set.
* The `model_path` in the general section : path to the model you want to evaluate.
* The `operation_mode` must be set to `evaluation`

In the evaluation section:
* `profile` : This relates to the user_neuralart.json file that contains various profiles the memory mapping and the compiler options. This is for advanced users and `profile_O3` is very good to start with. More information in [this article](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_neural_art_compiler.html#ref_built_in_tool_profiles).
* `input_type` : This is the input type provided by the pipeline, before entering the model. In this use case, this can be set to uint8 as this is the expected camera pipeline output format.
* `output_type` : This is the output type expected for the post processing, after model execution. This can be set to int8 here.
* `input_chpos` : This refers to the input data layout (NHWC vs NCHW). As the camera pipeline present data channel last, for standard .tflite and .onnx models (no transpose at the start of the model), this should be set to chlast.
* `output_chpos` : This refers to the output data layout (NHWC vs NCHW). Here as well, for consistency this should be set to chlast.
* `target` : This is to set the type of evaluation (host, STM32 emulated on host, or on STM32N6 HW)

In the dataset section:
* The `test_path` must be set to the dataset folder used for the evaluation

The `preprocessing` and `postprocessing` sections must be set as usual to align the preprocessing with the same configuration used during the model training.

Please refer to the online documentation on the [I/O data type or layout changes](https://stedgeai-dc.st.com/assets/embedded-docs/how_to_change_io_data_type_format.html) for more information.

In the Tools section:
* `path_to_stedgeai` : This the path of the stedgeai core executable

```yaml
general:
  display_figures: False

model:
  model_path: ../../stm32ai-modelzoo/face_detection/yunet/Public_pretrainedmodel_public_dataset/widerface/yunetn_320/yunetn_320_qdq_int8.onnx
  model_type: yunet

operation_mode: evaluation

evaluation:
  profile: profile_O3
  input_type: uint8     # int8 / uint8 / float32
  output_type: int8     # int8 / uint8 / float32
  input_chpos: chlast   # chlast / chfirst
  output_chpos: chlast  # chlast / chfirst
  target: stedgeai_n6 # host, stedgeai_host, stedgeai_n6

dataset:
  class_names: [person]
  test_path: ./datasets/test

preprocessing:
  rescaling: { scale: 1, offset: 0 }
  resizing:
    aspect_ratio: fit
    interpolation: bilinear
  color_mode: bgr

postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: True   # Plot precision versus recall curves. Default is False.
  max_detection_boxes: 100
  #crop_stretch_percents: [0,40,0,40] #List of 4 floats (stretch_xmin%, stretch_ymin%, stretch_xmax%, stretch_ymax%) representing the stretch percentage for each coordinate.

tools:
   stedgeai:
      path_to_stedgeai: C:/ST/STEdgeAI/4.0/Utilities/windows/stedgeai.exe

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```


## Run the script:
Open a new terminal (Git bash or WSL on Windows) (make sure to be in the application folder containing the stm32ai_main.py script), and run the command:

```powershell
python stm32ai_main.py --config-path ./config_file_examples --config-name evaluation_n6_config.yaml
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```

## Script outcome:
Close to the end of the log, you will see mAP results for the evaluation of your model, based on the dataset you used.
Please remember that evaluation on target may take time mainly because of data transfers between the host and the target, so a appropriate number of samples should be used for this task. Not less than 100 for a representative statistics and no more than 1000 for a reasonable evaluation time sounds a good tradeoff.

