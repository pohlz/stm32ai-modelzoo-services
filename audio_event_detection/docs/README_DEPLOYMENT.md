# Audio Event Detection STM32 model deployment

This tutorial shows how to deploy your pre-trained keras, ONNX or tflite models on an STM32 board using *STEdgeAI*.

In addition, this tutorial will also explain how to deploy a model from the **[ST public model zoo](./README_MODELS.md)** directly on your *STM32 target board*. In this version deployment on the [B-U585I-IOT02A](https://www.st.com/en/evaluation-tools/b-u585i-iot02a.html), [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html) and [NUCLEO-U3C5ZI-Q](https://www.st.com/en/evaluation-tools/nucleo-u3c5zi-q.html) are supported.
If running on [NUCLEO-U3C5ZI-Q](https://www.st.com/en/evaluation-tools/nucleo-u3c5zi-q.html), you will additionally require the [X-NUCLEO-IKS02A1](https://www.st.com/en/evaluation-tools/x-nucleo-iks02a1.html) expansion board, as the U3C5 does not come with a microphone.

Deployment on [NUCLEO-U3C5ZI-Q](https://www.st.com/en/evaluation-tools/nucleo-u3c5zi-q.html) supports use of the board's HSP feature : it is enabled by default both to accelerate the model and the preprocessing. See section <a href="#2-7">2.7 Configuring the deployment section</a> for more details.

We strongly recommend following the [training tutorial](./README_TRAINING.md) first.

## <a id="">Table of contents</a>

<details open><summary><a href="#1"><b>1. Before you start</b></a></summary><a id="1"></a>

Please check out [STM32 model zoo](./README_MODELS.md) for audio event detection (AED) results.

<ul><details open><summary><a href="#1-1">1.1 Hardware Setup</a></summary><a id="1-1"></a>

The [STM32 C application](../../application_code/sensing/STM32U5/README.md) is running on an STMicroelectronics evaluation kit board called [B-U585I-IOT02A](https://www.st.com/en/evaluation-tools/b-u585i-iot02a.html), on the [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html), or on the [NUCLEO-U3C5ZI-Q
](https://www.st.com/en/evaluation-tools/nucleo-u3c5zi-q.html). The current version of the application code only supports usage of the digital microphone on the first two boards, and usage of the [X-NUCLEO-IKS02A1](https://www.st.com/en/evaluation-tools/x-nucleo-iks02a1.html) expansion board on the latter.

</details></ul>
<ul><details open><summary><a href="#1-2">1.2 Software requirements</a></summary><a id="1-2"></a>

You can use the [STEdgeAI developer cloud](https://stedgeai-dc.st.com/home) to access the STEdgeAI functionalities without installing the software. This requires internet connection and making a free account. Or, alternatively, you can install [STEdgeAI](https://www.st.com/en/development-tools/stedgeai-core.html) locally. In addition to this you will also need to install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) for building the embedded project.

For local installation :

- Download and install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) __v1.17.0__.
- Download and install [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html).

</details></ul>
<ul><details open><summary><a href="#1-3">1.3 Specifications</a></summary><a id="1-3"></a>

- `serie` : STM32U5 or STM32N6 or STM32U3
- `board` : B-U585I-IOT02A or STM32N6570-DK or NUCLEO-U3C5ZI-Q
- `hsp` : 4096 if using NUCLEO-U3C5ZI-Q, None otherwise.
- `IDE` : GCC
- `quantization_input_type` : int8
- `quantization_output_type` : float

</details></ul>
</details>
<details open><summary><a href="#2"><b>2. YAML file configuration</b></a></summary><a id="2"></a>

The deployment of the model is driven by a configuration file written in the YAML language. This configuration file is called [user_config.yaml](../user_config.yaml) and is located in the Use Case directory.

This tutorial only describes enough settings for you to be able to deploy a pretrained model from the model zoo. Please refer  to the [main README](./README_OVERVIEW.md) for more information on the configuration file.

In this tutorial, we will be deploying a pretrained model from the STM32 model zoo.
Pretrained models informations can be found in this [readme](./README_MODELS.md) folder. Each model has its own subfolder. Each of these subfolders has a copy of the configuration file used to train the model. You can copy the `preprocessing` and `feature_extraction` sections to your own configuration file, to ensure you have the correct preprocessing parameters.

In this tutorial, we will deploy a quantized [Yamnet-256](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/audio_event_detection/yamnet/ST_pretrainedmodel_public_dataset/esc10/yamnet_256_64x96_tl/yamnet_256_64x96_tl_int8.tflite) that has been trained on ESC-10 using transfer learning.

<ul><details open><summary><a href="#2-1">2.1 Operation mode</a></summary><a id="2-1"></a>

The `operation_mode` attribute of the configuration file lets you choose which service of the model zoo you want to use (training, evaluation, quantization, deployment, or benchmarking). You can even chain these services together ! Refer to section 3.2 of the [main README](./README_OVERVIEW.md)

For this tutorial, you just need to set `operation_mode` to `"deployment"`, like so : 

```yaml
operation_mode: deployment
```

</details></ul>
<ul><details open><summary><a href="#2-2">2.2 General and model settings</a></summary><a id="2-2"></a>

The first section of the configuration file is the `general` section that provides information about your project.

Critically, you must set the `model_path` attribute to the path of the model you wish to deploy, in the `model`section like in this example : 

```yaml
general:
  project_name: myproject           # Project name. Optional, defaults to "<unnamed>".
  logs_dir: logs                    # Name of the directory where log files are saved. Optional, defaults to "logs".
  saved_models_dir: saved_models    # Name of the directory where model files are saved. Optional, defaults to "saved_models".    
  global_seed: 120                  # Seed used to seed random generators (an integer). Optional, defaults to 120.
  deterministic_ops: False          # Enable/disable deterministic operations (a boolean). Optional, defaults to False.
  display_figures: True             # Enable/disable the display of figures (training learning curves and confusion matrices).
                                     # Optional, defaults to True.
  gpu_memory_limit: 5              # Maximum amount of GPU memory in GBytes that TensorFlow may use (an integer).

model:
  model_path: <path_to_your_tflite_or_onnx_file>
```

</details></ul>
<ul><details open><summary><a href="#2-3">2.3 Dataset configuration</a></summary><a id="2-3"></a>

Information about the dataset you want to use is provided in the `dataset` section of the configuration file, as shown in the YAML code below.

```yaml
dataset:
  dataset_name: esc10 # Name of the dataset. Use 'esc10' for ESC-10, 'fsd50k' for FSD50K and 'custom' for any other dataset
  class_names: ['dog', 'chainsaw', 'crackling_fire', 'helicopter', 'rain', 'crying_baby', 'clock_tick', 'sneezing', 'rooster', 'sea_waves'] # Names of the classes your model was trained on
  # Must be included when deploying as well.
  file_extension: '.wav' # File extension of the audio files
  training_audio_path: ..\datasets\ESC-50\audio # Not mandatory for deployment
  training_csv_path:   ..\datasets\ESC-50\meta\esc50.csv # Not mandatory for deployment

  validation_audio_path: # Optional
  validation_csv_path: # Optional
  validation_split: 0.2  # Optional, default value is 0.2

  quantization_audio_path: # Optional
  quantization_csv_path: # Optional
  quantization_split: 0.1 # Optional

  test_audio_path: # Optional
  test_csv_path: # Optional

  multi_label: False # Set to True if dataset is multi-label
  use_garbage_class: False # See the "Handling out of distribution data" section
  n_samples_per_garbage_class: 2 # See the "Handling out of distribution data" section
  expand_last_dim: True # Set to True to make patches of shape (n_mels, n_frames, 1) instead of (n_mels, n_frames)
  seed: 120 # Optional, there is a default seed
  to_cache: True # Cache the resulting Tensorflow datasets
  shuffle: True # Shuffle the resulting Tensorflow datasets. Won't shuffle validation & test sets.
```

When deploying, you don't need to provide paths to any dataset. However, you do need to provide the `class_names` used during training. Beware, the class names are sorted alphabetically before being passed to the board. If you deploy a model trained outside the zoo, this may cause the displayed classes on the board to be wrong.

**IMPORTANT** : If you set `use_garbage_class` to True when training your model, you MUST also set it to True when deploying it.

For more details on this section, please consult section 3.5 and section 6 of the [main README](./README_OVERVIEW.md)

</details></ul>
<ul><details open><summary><a href="#2-4">2.4 Audio temporal domain preprocessing</a></summary><a id="2-4"></a>

When performing AED, it is customary to perform some preprocessing directly on the input waveform in the temporal domain before doing any feature extraction in the frequency domain, such as converting the waveform into a spectrogram.

When a model is deployed, audio is acquired directly from the microphone, and processed into spectrogram patches on the fly. 
The temporal domain preprocessing that happens on the board happens at sensor level, and can't be piloted from the model zoo. The only temporal domain parameter from the zoo that has any impact on-board is the `target_rate` parameter, which lets you set the sampling rate (in Hz) of the on-board microphone. The only valid values are `16000` and `48000`.

An example of the `preprocessing` section from the YAML file is shown below : 

```yaml
preprocessing:
  min_length: 1 # Not used for deployment
  max_length : 10 # Not used for deployment
  target_rate: 16000 # Must be either 16000 or 48000 if deploying on a STM32 board
  top_db: 60 # Not used for deployment
  frame_length: 3200 # Not used for deployment
  hop_length: 3200 # Not used for deployment
  trim_last_second: False # Not used for deployment
  lengthen : 'after' # Not used for deployment
```

For more details on what each parameter does, please refer to section 3.7 of the [main README](./README_OVERVIEW.md)

Different models are trained using different set of preprocessing parameters, and using different ones may lead to poor performance. Please refer to [section 2](#2) of this README for instructions on how to retrieve the configuration files used to train the different pretrained models provided in the zoo.

</details></ul>
<ul><details open><summary><a href="#2-5">2.5 Audio feature extraction (frequency domain preprocessing)</a></summary><a id="2-5"></a>

In a typical AED pipeline, once the temporal domain preprocessing has been performed on the input waveform, it is usually converted to a frequency-domain representation, such as a mel-spectrogram, or array of MFCC coefficients, and the model is trained on this representation.

At inference time on the board, we convert audio to spectrogram patches. Each patch is then fed as input to the model in real-time.

The C library that handles the conversion of audio PCM samples to spectrogram patches on the board requires several parameters and look-up tables. These parameters are read from the configuration file, used to compute the necessary look-up tables, and then copied to the C application that is deployed on the board.

Therefore, unlike the previous section, nearly all parameters in this section impact how audio is preprocessed on the board. It is critical to model performance that these parameters are the same ones that were used for training the model you're trying to deploy.

An example of the `feature_extraction` section from the YAML file is shown below : 

```yaml
feature_extraction:
  patch_length: 96
  n_mels: 64
  overlap: 0.25 # Not used during deployment
  n_fft: 512 # Must be a power of 2 if deploying on a STM32 board
  hop_length: 160
  window_length: 400
  window: hann
  center: False
  pad_mode: constant
  power: 1.0
  fmin: 125
  fmax: 7500
  norm: None
  htk : True
  to_db : False
```
For more details on what each parameter does, please refer to section 3.8 of the [main README](./README_OVERVIEW.md)

Different models are trained using different set of feature extraction parameters, and using different ones may lead to poor performance. Please refer to [section 2](#2)  of this README for instructions on how to retrieve the configuration files used to train the different pretrained models provided in the zoo.

</details></ul>
<ul><details open><summary><a href="#2-6">2.6 Configuring the tool section</a></summary><a id="2-6"></a>

Next, you'll want to configure the `tools` section in your configuration file. This section
This section covers the usage of the STEdgeAI tool, which benchmarks .tflite, .keras and .h5 models, and converts them to C code.

To convert your model to C code, you can either use the [STM32 developer cloud](https://stedgeai-dc.st.com/home) (requires making an account), or use the local versions of STEdgeAI Core and CubeIDE you installed earlier in the tutorial.

If you wish to use the [STEdgeAI developer cloud](https://stedgeai-dc.st.com/home), simply set the `on_cloud` attribute to True, like in the example below. If using the developer cloud, you do not need to specify paths to STEdgeAI Core or CubeIDE.

```yaml
tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe
```

For more details on what each parameter does, please refer to section 3.14 of the [main README](./README_OVERVIEW.md)

</details></ul>
<ul><details open><summary><a href="#2-7">2.7 Configuring the deployment section</a></summary><a id="2-7"></a>

Finally, you need to configure the `deployment` section of your configuration file, like in the example below.

For deployment on B-U585I-IOT02A : 

```yaml
deployment:
  c_project_path: ../../application_code/sensing/STM32U5
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32U5
    board: B-U585I-IOT02A
  unknown_class_threshold: 0.0 # Threshold used for OOD detection. Mutually exclusive with use_garbage_class
                               # Set to 0 to disable. To enable, set to any float between 0 and 1.
  build_conf: Debug
```


For deployment on STM32N6570-DK : 

```yaml
deployment:
  c_project_path:  ../application_code/audio/STM32N6
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32N6
    board: STM32N6570-DK
  build_conf : "BM" # this is default configuration
  # build_conf : "FREERTOS"
  # build_conf : "BM_LP"
  # build_conf : "FREERTOS_LP"
  unknown_class_threshold: 0.5 # Threshold used for OOD detection. Mutually exclusive with use_garbage_class
                               # Set to 0 to disable. To enable, set to any float between 0 and 1.

```

For deployment on NUCLEO-U3C5ZI-Q : 

```yaml
tools:
  stedgeai:
    optimization: balanced
    on_cloud: False
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
    hsp: 4096 # Set to None to disable HSP for the model. Only 4096 or None are supported for now.
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe
  
deployment:
  c_project_path: ../application_code/audio/STM32U3
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32U3
    board: NUCLEO-U3C5ZI-Q
  unknown_class_threshold: 0.0 # Threshold used for OOD detection. Mutually exclusive with use_garbage_class
                               # Set to 0 to disable. To enable, set to any float between 0 and 1.
  build_conf: Debug # Use "Debug" to enable HSP for preprocessing, and use "Debug_No_HSP" to disable HSP for preprocessing.
```

**NOTE : ** The C application on NUCLEO-U3C5ZI-Q uses the board's HSP feature. By default, this feature is used to accelerate both the model AND the preprocessing. You can change this behaviour by editing your config file. Setting `hsp` under `tools.stedgeai` to `None` disables HSP usage for the model, and setting `build_conf` to `Debug_No_HSP` disables HSP usage for the preprocessing.


There is one C application available for AED when deploying on STM32U5, found under [application_code/sensing/STM32U5/](../../application_code/sensing/STM32U5/README.md). This the same application code usable for deploying Human Activity Recognition models.

There is one C application available for AED when deploying on STM32N6, found under [application_code/audio/STM32N6/](../../application_code/audio/STM32N6/).

Lastly, there is one C application available for AED when deploying on STM32U3, found under [application_code/audio/STM32U3/](../../application_code/audio/STM32U3/).

**Make sure you choose the correct application code path for the board you want to deploy on**

You only need to specify the path to one of these applications in `c_project_path`.
Currently, the C application only supports the `B-U585I-IOT02A`,  `STM32N6570-DK` and `NUCLEO-U3C5ZI-Q`  boards.

`unknown_class_threshold` is a parameter used for OOD (out of distribution) sample detection during runtime. It is mutually exclusive with `dataset.use_garbage_class`. For more details on how OOD detection is handled, please consult section <a href="#3"> 3. Out of distribution sample detection in the model zoo </a>, and section 6 of the [main README](./README_OVERVIEW.md).

</details></ul>
<ul><details open><summary><a href="#2-8">2.8 Configuring the mlflow section</a></summary><a id="2-8"></a>

The model zoo uses MLFlow to record logs when running. You'll want to configure the `mlflow` section of your configuration file like in the example below

```yaml
mlflow:
  uri: ./tf/src/experiments_outputs/mlruns
```
You'll then be able to access the logs by going to `./tf/src/experiments_outputs` in your favourite shell, using the command `mlflow ui`, and accessing the provided IP address in your browser.

</details></ul>
</details>
<details open><summary><a href="#3"><b>3. Out of distribution sample detection in the model zoo</b></a></summary><a id="3"></a>

A common issue in audio event detection applications is being able to reject samples which do not come from one of the classes the model is trained on.
The model zoo provides several baseline options for doing this.

The first option consists of thresholding the network output probabilities at runtime. **This is a naïve baseline which does not yield great results**, but is a good starting point.

You can set the threshold in the config file, by tuning the `deployment.unknown_class_threshold` parameters (see section <a href="#2-7">2.7 Configuring the deployment section</a>).
This parameter is a float between 0 and 1.

Setting it to 0 disables the thresholding, while setting it above 0 will display the sample as belonging to the "unknown" class, if the maximum probability output by the network for this sample is lower than the threshold, and will display the class predicted by the network if the probability is above the threshold.

The second option for OOD detection consists of adding an additional "Other" class to your model at training time, using samples from the dataset which do not belong to any of the classes specified in `dataset.class_names`.

**IMPORTANT NOTE** : This feature is **experimental**. It will yield **very poor results** if the dataset provided does not have a very large number of unused classes to lump into the "Other" class. Do not expect great results from this, unless you are using a dataset with a LOT of different classes, and you are using a small subset of them (for example, using 5 classes out of 200 with FSD50K.)

To enable this option, set the `dataset.use_garbage_class` parameter to `True` (see section 3.5 of the [main README](./README_OVERVIEW.md)).

Additionally, you can use the `dataset.n_samples_per_garbage_class`(see section 3.5 of the [main README](./README_OVERVIEW.md) ) parameter to control how many samples from each unused class are sent to the "Other" class. This can help keeping the resulting dataset balanced.

**IMPORTANT NOTE : To use this feature in deployment, it must have been enabled when training your model.**

**IMPORTANT NOTE** These two methods are **NOT COMPATIBLE**, and cannot be used together. You must enable one or the other, or none at all.
This means that if `dataset.use_garbage_class` is set to `True`, then `deployment.unknown_class_threshold` must be set to 0, and that if `deployment.unknown_class_threshold` is set to a float > 0, then `dataset.use_garbage_class` must be set to `False`

</details>
<details open><summary><a href="#4"><b>4. Running deployment</b></a></summary><a id="4"></a>
<ul><details open><summary><a href="#4-1">4.1 Attach the board</a></summary><a id="4-1"></a>

To run build the project and flash the target board, connect a B-U585I-IOT02A to your computer using the microUSB port on the board.

</details></ul>
<ul><details open><summary><a href="#4-2">4.2 Run stm32ai_main.py</a></summary><a id="4-2"></a>

Then, once your configuration file is properly configured run the following command from the UC :
Make sure you properly set `operation_mode` to `"deployment"`.

```bash
python stm32ai_main.py
```

This will generate the C code, copy the model files in the stm32ai application C project, build the C project, and flash the board.

</details></ul>
</details>
<details open><summary><a href="#5"><b>5. Results on the board</b></a></summary><a id="5"></a>

Once flashed the board can be connected through a serial terminal and the output of the inference can be seen in the serial terminal. 
To connect the serial port please follow the steps shown in the figure below:
![plot](./img/tera_term_connection.png)

**If deploying on STM32U5 or STM32U3 using the application loacated in `application_code/audio/STM32U5/` or in  `application_code/audio/STM32U3/` , the expected baud rate is 115200**

**If deploying on STM32N6 using the application loacated in `application_code/audio/STM32N6/`, the expected baud rate is 14400**

After successful connection perform a reset using [RESET] button on the board. This will reset the board and start running the inference of the AI model on the board using real-time data from the digital microphone. Following figure shows a screenshot of [stm32ai application code](../../application_code/sensing/STM32U5/README.md) project running the inference running on the board:


![plot](./img/getting_started_running.png)


Please note that the output display may look slightly different depending on the C application used.

**If nothing is displayed on the serial output after flashing and resetting the board, make sure you set the correct baud rate in your serial connection as described above.**

Each of the line in the Tera Term terminal shows the output of one inference from the live data.
Inference is run once each second.
The labels `"signal"` shows the signal index or number, the `"class"` has the label of the class detected and `"dist"` shows the probability distribution of the confidence of the signal to belong to any given class, with classes sorted in alphabetical order.

</details>
<details open><summary><a href="#6"><b>6. Restrictions</b></a></summary><a id="6"></a>

- In this version, application code for deployment is only supported on the [B-U585I-IOT02A](https://www.st.com/en/evaluation-tools/steval-stwinkt1b.html), [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html) and [NUCLEO-U3C5ZI-Q](https://www.st.com/en/evaluation-tools/nucleo-u3c5zi-q.html) boards.
- Only the *int8* type input is supported for the quantization operation.

</details>
