# Speech enhancement STM32 model deployment

This tutorial will show you how to deploy a quantized ONNX speech enhancement model to an STM32N6 board, in addition, this tutorial will also explain how to deploy a model from the **[ST public model zoo](./README_MODELS.md)** directly on your *STM32N6 target board*. 

In this version only deployment on the [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html) is supported.

We strongly recommend following the [training tutorial](./README_TRAINING.md), and [quantization tutorial](./README_QUANTIZATION.md) first.

Once your model is deployed, it will directly denoise audio acquired from the on-board microphone. You will also be able to enable and disable the denoiser to compare the denoised and the raw audio.

**IMPORTANT: You will require headphones, preferably closed-back, with a 3.5mm jack connector to listen to the output of the model.**


<details open><summary><a href="#1"><b>1. Configuration</b></a></summary><a id="1"></a>
<ul><details open><summary><a href="#1-1">1.1 Hardware Setup</a></summary><a id="1-1"></a>

The [stm32 C application](../../application_code/audio/STM32N6/README.md) is running on an STMicroelectronics evaluation kit board called [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html). The current version of the application code only supports this board and usage of the on-board digital microphone.

</details></ul>
<ul><details open><summary><a href="#1-2">1.2 Software requirements</a></summary><a id="1-2"></a>

You can use the [STEdgeAI developer cloud](https://stedgeai-dc.st.com/home) to access the STEdgeAI Core functionalities without installing the software. This requires an internet connection and making a free account. Alternatively, you can install [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally. In addition to this, you will also need to install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) for building the embedded project.

For local installation:

- Download and install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html).
- If opting to use [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) locally, download it then run the installer.
The detailed instructions on installation are available in this [wiki article](https://wiki.st.com/stm32mcu/index.php?title=AI:How_to_install_STM32_model_zoo).

</details></ul>
<ul><details open><summary><a href="#1-3">1.3 Specifications</a></summary><a id="1-3"></a>

- `serie`: STM32N6
- `board`: STM32N6570-DK
- `IDE`: GCC

</details></ul>
</details>
<details open><summary><a href="#2"><b>2. YAML file configuration</b></a></summary><a id="2"></a>

The deployment of the model is driven by a configuration file written in the YAML language. This configuration file is called [user_config.yaml](../user_config.yaml) and is located in the [the root folder of this UC](../) directory.

This tutorial only describes enough settings for you to be able to deploy a pretrained model from the model zoo. Please refer to the [main README](./README_OVERVIEW.md) for more information on the configuration file.

In this tutorial, we will be deploying a pretrained model from the STM32AI model zoo.

You can copy the `preprocessing` section from the configuration file present in the model zoo to your own configuration file, to ensure you have the correct audio preprocessing parameters.

In this tutorial, we will deploy a quantized [STFT-TCNN](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/main/speech_enhancement/stft_tcnn) that has been trained on the CSTR VCTK + DEMAND, colloquially referred to as the Valentini dataset.

<ul><details open><summary><a href="#2-1">2.1 Operation mode</a></summary><a id="2-1"></a>

The `operation_mode` attribute of the configuration file lets you choose which service of the model zoo you want to use (training, evaluation, quantization, deployment, or benchmarking). You can even chain these services together! Refer to section 3.2 of the [main README](./README_OVERVIEW.md).

For this tutorial, you just need to set `operation_mode` to `"deployment"`, like so : 

```yaml
operation_mode: deployment
```

</details></ul>
<ul><details open><summary><a href="#2-2">2.2 General settings</a></summary><a id="2-2"></a>

The first section of the configuration file is the `general` section that provides information about your project.

```yaml
general:
  project_name: speech_enhancement_project
  logs_dir: logs # Name of the directory where logs are saved
  saved_models_dir: saved_models # Name of the directory where models are saved
  gpu_memory_limit: 0.5 # Fraction of GPU's memory to use.
  display_figures: True # Set to True to display figures. Figures are saved even if set to False.
```

Here, since no model is saved, the GPU is unused, and no figure is created, only `project_name` and `logs_dir` are used.
</details></ul>
<ul><details open><summary><a href="#2-3">2.3 Select your model</a></summary><a id="2-3"></a>

Select the model you would like to deploy by filling the `model` section of the configuration file like so : 

```yaml
model:
  model_path: path/to/your/model.onnx # For quantization, evaluation, benchmarking and deployment only
```

**VERY IMPORTANT : MAKE SURE YOUR MODEL HAS A STATIC INPUT SHAPE !**
**STEdgeAI does not handle models with a dynamic input shape.**

See section <a href="#2-6"> 2.6 </a> of this README for more details on the input shape requirements.

</details></ul>

<ul><details open><summary><a href="#2-4">2.4 Audio preprocessing</a></summary><a id="2-4"></a>

The general flow of inference is the following: A complex spectrogram of the noisy audio is computed by performing a Short-time Fourier Transform, and the corresponding magnitude spectrogram is given as input to the model. 

The model outputs a mask of the same shape as its input, and this mask is applied to the complex spectrogram. The masked complex spectrogram is then transformed back to the time domain by inverse STFT. This gives us the denoised time-domain signal.

You can additionally choose whether or not to peak normalize your audio data, and which preprocessing pipeline to use.

This preprocessing pipeline is replicated in C code on the STM32 target, but inference works a little differently : Because we need to run the model in real time, we do not process the entire signal at once like we might do in Python : instead, we process chunks of contiguous spectrogram frames, called "patches" with some overlap between patches to allow the model to "look ahead" a few frames, and smooth over some of the boundary artifacts of a convolutional model.

More details on these differences are explained in section <a href="#2-6">2.6</a> of this README.

The 'preprocessing' section handles this part of the deployment pipeline, and an example is shown below.

```yaml
preprocessing:
  pipeline_type: LibrosaSpecPipeline # Do not change 
  peak_normalize: False
  sample_rate: 16000
  n_fft: 512
  hop_length: 160
  win_length: 400
  window: hann
  center: True
  power: 1
```

**IMPORTANT NOTE :** Currently, only the `LibrosaSpecPipeline` pipeline type is supported. Other pipelines are present in [preprocessing/freq_pipeline.py](../pt/src/preprocessing/freq_pipeline.py) but are not supported in the on-board application code.


For more details on what each parameter does, please refer to section 3.6 of the [main README](./README_OVERVIEW.md)

Different models are trained using different set of preprocessing parameters, and using different ones may lead to poor performance. Please refer to section 2.2 of the [training README](./README_TRAINING.md) for instructions on how to retrieve the configuration files used to train the different pretrained models provided in the zoo.

</details></ul>
<ul><details open><summary><a href="#2-5">2.5 Configuring the tool section</a></summary><a id="2-5"></a>

Next, you'll want to configure the `tools` section in your configuration file. This section covers the usage of the STEdgeAI tool, which benchmarks Keras, Tflite and ONNX models and converts them to C code.

To convert your model to C code, you can either use the [STEdgeAI developer cloud](https://stedgeai-dc.st.com/home) (requires making an account), or use the local versions of STEdgeAI and CubeIDE you installed earlier in the tutorial.

If you wish to use the [STEdgeAI developer cloud](https://stedgeai-dc.st.com/home), simply set the `on_cloud` attribute to True, like in the example below. If using the developer cloud, you do not need to specify paths to STEdgeAI or CubeIDE.

```yaml

tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

```

For more details on what each parameter does, please refer to section 3.10 of the [main README](./README_OVERVIEW.md).

</details></ul>
<ul><details open><summary><a href="#2-6">2.6 Configuring the deployment section</a></summary><a id="2-6"></a>

Finally, you need to configure the `deployment` section of your configuration file, like in the example below.

```yaml
deployment:
  frames_per_patch: 30 # Number of frames per inference

  # Number of lookahead and lookback frames.
  # MAKE SURE YOUR MODEL HAS INPUT SHAPE (1, n_fft // 2 + 1, frames_per_patch + 2*lookahead_frames) BEFORE DEPLOYING !!!!!!
  lookahead_frames: 5 

  # After denoising is applied, any patches with power less than
  # output_silence_threshold dB get silenced completely.
  # This also applies when disabling the denoiser using the button on the board.
  # In dB. A higher number means a stricter filtering. -100 disables the filtering.
  output_silence_threshold: -50
  # Change this to the path on your machine.
   
  c_project_path:  ../application_code/audio/STM32N6/
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32N6
    board: STM32N6570-DK
  build_conf : "BM" # this is the default configuration
  # build_conf : "BM_LP"

```

A few comments:

- The `frames_per_patch` attribute corresponds to how many spectrogram frames get denoised per inference on the board. How much this represents in terms of time depends on your preprocessing parameters. For example, with a sampling rate of 16000 and a hop length of 160, one inference is done every 300 ms. In turn, this induces a 300 ms delay between audio acquisition and denoised audio playback.

- The model is allowed to look ahead a few frames into the future, to attenuate patch boundary artifacts on its output. It is also allowed to look the same number of frames into the past. This number of lookahead frames is controlled by the `lookahead_frames` attribute. If you reduce this attribute too much, or if your model's receptive field increases, you may see the patch boundary artifacts appear again.

- Note that the total length of a patch fed to the model during inference is equal to `frames_per_patch + 2*lookahead_frames`. Using the values in the above example, this means each patch contains 40 frames (30+5*2). One inference is still done every 300 ms, but the induced delay is 400 ms.

**VERY IMPORTANT : MAKE SURE YOUR MODEL'S INPUT SHAPE IS COMPATIBLE WITH THESE PARAMETERS**

**Your model's input shape should be (1, n_fft // 2 + 1, frames_per_patch + 2 * lookahead_frames)**
If you quantize your model using the model zoo, simply set the `static_sequence_length` attribute to `frames_per_patch` + 2 * `lookahead_frames`. See the [Quantization README](./README_QUANTIZATION.md) for additional details.

If this is not respected, glitches WILL occur, as patch overlap will not happen properly.


- Before audio is output by the board, a threshold (in dB) is applied. If the mean power of a patch is below this threshold, then it is silenced (e.g. the whole output buffer is set to 0). The attribute `output_silence_threshold` controls this threshold.
A higher number increases the strictness of this filter. 0 filters everything, and -100 disables the filtering. Note that this filter remains on even when the denoising model is disabled.

</details>
<details open><summary><a href="#3"><b>3. Running deployment</b></a></summary><a id="3"></a>
<ul><details open><summary><a href="#3-1">3.1 Attach the board</a></summary><a id="3-1"></a>

To build the project and flash the target board, connect an STM32N6570-DK to your computer using the USB-C port on the board.

**There are two ports on the board. Use the STLINK port.**

**IMPORTANT: Before flashing your board, you must set the two jumpers at the top right of the board TO THE RIGHT, and then power cycle your board (disconnect/reconnect from the computer).**

**MAKE SURE YOU DO THE ABOVE**

</details></ul>
<ul><details open><summary><a href="#3-2">3.2 Run stm32ai_main.py</a></summary><a id="3-2"></a>

Then, once your configuration file is properly configured, run the following command from [the root folder of this UC](../):

```bash
python stm32ai_main.py
```

This will generate the C code, copy the model files in the stm32ai application C project, build the C project, and flash the board.

**VERY IMPORTANT : Once your board is flashed, you will need to switch the jumpers in the top-right of the board TO THE LEFT, and power cycle your board (disconnect and reconnect to your computer).**

**MAKE SURE YOU DO THE ABOVE. If done correctly and the board was properly flashed, the LED1 in the bottom left of the board should flash green.**

</details></ul>
</details>

<details open><summary><a href="#4"><b>4. Results on the board</b></a></summary><a id="4"></a>

You can then plug your headphones into the 3.5 mm jack on the board to listen to the results.

**You can press the USER1 button (second blue button from the top, right of the board) to disable/enable the denoiser.**

We recommend you play around with enabling/disabling the denoiser to hear the difference. The raw output from the microphone is quite noisy, even in silence!

</details>
<details open><summary><a href="#5"><b>5. Limitations</b></a></summary><a id="5"></a>

Some speech enhancement models, including those in the zoo, are not trained to evaluate the level of noise of their input. 

Therefore, they only perform well at SNRs that they have seen during training. If you trained a model in the zoo using the Valentini dataset, your model will not have seen SNRs below 0 dB during training, and so will perform poorly in these cases. 

You can see this phenomenon for yourself by playing some loud noise, and then moving your voice closer and farther from the board: once you get close enough, the model will denoise well, but once you move away sufficiently, it will get much worse.

<details open><summary><a href="#6"><b>6. Restrictions</b></a></summary><a id="6"></a>

- In this version, application code for deployment is only supported on the [STM32N6570-DK].
- Only models quantized with int8 weights and activations are supported.
</details>
