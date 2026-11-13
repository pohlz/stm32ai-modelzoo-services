# Speech enhancement model quantization

This tutorial will show you how to quantize a speech enhancement model using the STM32 model zoo. 

For this tutorial, we will use the CSTR VCTK + DEMAND, colloquially and henceforth referred to as Valentini dataset. The dataset can be downloaded here : https://datashare.ed.ac.uk/handle/10283/2791 .

The model zoo uses the ONNX quantizer from onnxruntime to perform post-training quantization, and outputs quantized model in the ONNX QDQ format.
**Currently, the model zoo only supports post-training quantization !**

**IMPORTANT NOTE :** For this use case, we have chosen to support and provide models that operate in the frequency domain, as time-domain models do not perform well when quantized to 8-bit integer precision.

The general flow of inference is the following : A complex spectrogram of the noisy audio is computed and the corresponding magnitude spectrogram is given as input to the model. 

The model outputs a mask of the same shape as its input, and this mask is applied to the complex spectrogram. The masked complex spectrogram is then transformed back to the time domain by inverse STFT. This gives us the denoised time-domain signal.

Therefore, we expect all models to take magnitude spectrograms as input, i.e. tensors of the shape (batch, n_fft // 2 + 1, sequence_length), and output tensors of the same shape, corresponding to the mask.

**Float models provided for quantization must have input shape (batch, n_fft // 2 + 1, seq_len) where the batch and seq_len axis have dynamic length.**


## <a id="">Table of contents</a>

<details open><summary><a href="#1"><b>1. Download and extract the dataset</b></a></summary><a id="1"></a>

Download the Valentini dataset from https://datashare.ed.ac.uk/handle/10283/2791

Then, extract the archive to a folder of your choice on your machine.

By default, the model zoo expects that datasets are given in a similar format to the Valentini dataset. 
This means clean speech files and noisy speech files must be in separate folders, both with the same number of files. Corresponding files should be present in the same order in both folders, and ideally have the same filename.

For this tutorial, we will only need a training set, as the quantization dataset is sampled from the training set.

This means that your dataset must be comprised of :
- A folder containing the clean training audio files. All audio files must share the same format. No mixing .wav and .flac in the same folder, for example.
- A folder containing the noisy training audio files. All audio files must share the same format. This folder must have the same number of files as the above folder.

You can optionally provide test audio files, but they will be ignored for quantization, as the quantization dataset is sampled from the training set.

If you're using the Valentini dataset, then all these conditions are already satisfied, and you don't need to worry about anything.

</details>
<details open><summary><a href="#2"><b>2. Create your configuration file</b></a></summary><a id="2"></a>
<ul><details open><summary><a href="#2-1">2.1 Overview</a></summary><a id="2-1"></a>

The training, evaluation, quantization and benchmarking of the model are driven by a configuration file written in the YAML language. This configuration file is called [user_config.yaml](../user_config.yaml) and is located in the [the root folder of this UC](../) directory.

A configuration file includes the following sections:

- `general`, describes your project, including project name, directory where to save models, etc.
- `operation_mode`, a string describing the operation mode of the model zoo. You'll want to set it to "training" for this tutorial.
- `model`, describes the model you want to use and lets you eventually load a Pytorch state dict
- `model_specific`, contains parameters specific to the model architecture you want to use, such as the number of blocks, number of layers per blocks, intermediate dimensions, etc.
- `dataset`, describes the dataset you are using, including directory paths, file extension, how many samples to use, etc.
- `preprocessing`, parameters used to perform both time and frequency-domain preprocessing on the audio data

- `quantization`, contains parameters related to quantization, such as number of quantization samples, quantizer options, etc.
- `evaluation` contains parameters related to model evaluation
- `stedgeai`, specifies the STEdgeAI configuration to benchmark your model on a board, including memory footprints, inference time, etc.
- `tools`, specifies paths and options to the ST tools used for benchmarking and deployment
- `deployment`, contains parameters used for deployment on STM32N6 target.
- `mlflow`, specifies the folder to save MLFlow logs.
- `hydra`, specifies the folder to save Hydra logs.

This tutorial only describes enough settings for you to be able to run the quantization service.  Please refer to the [main README](./README_OVERVIEW.md) for more information. The model zoo offers many more features than those described in this short tutorial.
</details></ul>

<ul><details open><summary><a href="#2-2">2.2 Operation mode</a></summary><a id="2-2"></a>

The `operation_mode` attribute of the configuration file lets you choose which service of the model zoo you want to use (training, evaluation, quantization, deployment, or benchmarking). You can even chain these services together ! Refer to the [main README](./README_OVERVIEW.md) for more details.

For this tutorial, you just need to set `operation_mode` to `quantization`, like so : 

```yaml
operation_mode: quantization
```

</details></ul>
<ul><details open><summary><a href="#2-3">2.3 General settings</a></summary><a id="2-3"></a>

The first section of the configuration file is the `general` section that provides information about your project.

```yaml
general:
  project_name: speech_enhancement_project
  logs_dir: logs # Name of the directory where logs are saved
  saved_models_dir: saved_models # Name of the directory where models are saved
  gpu_memory_limit: 0.5 # Fraction of GPU's memory to use.
  display_figures: True # Set to True to display figures. Figures are saved even if set to False.
  
```

These `logs_dir` and `saved_models_dir` directories are located under the top level <hydra> directory.

For more details on the structure of the output directory, please consult section 1.2 of the [main README](./README_OVERVIEW.md)

Since we are running quantization, the GPU is unused, and no figures are displayed.

</details></ul>

<ul><details open><summary><a href="#2-4">2.4 Model settings</a></summary><a id="2-4"></a>

Information about the model you wish to quantize is provided in the `model`section of the configuration file, as show in the YAML code below : 

```yaml
model:
  model_path: path/to/your/model.onnx # For quantization, evaluation, benchmarking and deployment only
```

You can only quantize float ONNX models using the model zoo. When training a model using the zoo, ONNX model files are given as output. If you wish to use your own model, you must first export it to ONNX.

Provide the path to your float model using the `model_path` attribute.

**NOTE WHEN USING CUSTOM MODELS : float models provided for quantization must have input shape (batch, n_fft // 2 + 1, seq_len) where the batch and seq_len axis have dynamic length.** 

The general flow of inference is the following : A complex spectrogram of the noisy audio is computed and the corresponding magnitude spectrogram is given as input to the model. 

The model outputs a mask of the same shape as its input, and this mask is applied to the complex spectrogram. The masked complex spectrogram is then transformed back to the time domain by inverse STFT. This gives us the denoised time-domain signal.

Therefore, we expect all models to take magnitude spectrograms as input, i.e. tensors of the shape (batch, n_fft // 2 + 1, sequence_length), and output tensors of the same shape, corresponding to the mask.


</details></ul>
<ul><details open><summary><a href="#2-5">2.5 Dataset specification</a></summary><a id="2-5"></a>

Information about the dataset you want to use is provided in the `dataset` section of the configuration file, as shown in the YAML code below.

```yaml
dataset:
  dataset_name: valentini # Or "custom"
  root_folder: /local/datasets/Valentini # Root folder of dataset
  n_speakers: 56 # For Valentini, 28 or 56 speaker dataset. Does nothing if name is "custom"
  file_extension: '.wav' # Extension of audio files. Valentini dataset uses .wav

  noisy_train_files_path:
```

If you wish to use a quantization set separate from the training set, you will need to provide the path to your quantization files in the `quantization` section (see section 2.7 of this README).

For more details on this section, please consult section 3.5 of the [main README](./README_OVERVIEW.md).

</details></ul>
<ul><details open><summary><a href="#2-6">2.6 Audio preprocessing</a></summary><a id="2-6"></a>

The general flow of inference is the following : A complex spectrogram of the noisy audio is computed by peforming a Short-time Fourier Transform, and the corresponding magnitude spectrogram is given as input to the model. 

The model outputs a mask of the same shape as its input, and this mask is applied to the complex spectrogram. The masked complex spectrogram is then transformed back to the time domain by inverse STFT. This gives us the denoised time-domain signal.

You can additionally choose whether or not to peak normalize your audio data, and which preprocessing pipeline to use.

The 'preprocessing' section handles this part of the pipeline, and an example is shown below.

```yaml
preprocessing:
  pipeline_type: LibrosaSpecPipeline # Do not change if unsure.
  peak_normalize: False
  sample_rate: 16000
  n_fft: 512
  hop_length: 160
  win_length: 400
  window: hann
  center: True
  power: 1
```

**IMPORTANT NOTE :** Currently, only the `LibrosaSpecPipeline` pipeline type is supported. Other pipelines are present in [preprocessing/freq_pipeline.py](../pt/src/preprocessing/freq_pipeline.py) but are not supported by the rest of the code in the model zoo.


For more details on what each parameter does, please refer to section 3.6 of the [main README](../README.md).

</details></ul>
<ul><details open><summary><a href="#2-7">2.7 Quantization setup</a></summary><a id="2-7"></a>
 
The quantization setup is described in the `quantization` section of the configuration file, as illustrated in the example below.

```yaml
quantization:
  # N° of samples or fraction of training set to include in quantization set.
  # Leave empty to use the whole training set.
  num_quantization_samples: 100
  # Random seed used for sampling
  random_seed:

  # Use the following parameters if using a quantization set different from the training set.
  # If left empty, the noisy files from the training dataset specified in the dataset section will be used.
  noisy_quantization_files_path:

  # STEDGEAI only accepts models with static input shapes. 
  # The model zoo will output two models : one with dynamic and one with static input shape.
  # E.g, if the dynamic input shape model has input shape[?, 257, ?] 
  # The static shape model will have input shape [1, 257, 40] 
  # if static_sequence_length is 40. 
  static_sequence_length: 40 
  static_axis_name: "seq_len"

  # The following parameters are passed to ONNXruntime's quantize_static function
  granularity : per_channel 
  reduce_range: False
  onnx_quant_parameters:  
    op_types_to_quantize: ["Add", "Conv", "Relu"] # Op types to quantize. Leave empty to quantize all defaults for the ONNX quantizer.
    calibrate_method: "MinMax" # Calibration method of ONNX quantizer
  onnx_extra_options: 
    CalibMovingAverage: True # Add extra quantizer options in this dict.

```

Some comments : 

- We recommend you only use a small portion of the training set to act as quantization dataset. You can do this using the `num_quantization_samples` attribute.
- If using a custom dataset, pass the path to the NOISY files using the `noisy_quantization_files_path` attribute. If this attribute is empty, files will be taken from the `noisy_training_files_path` attribute under the `dataset` section.

- **IMPORTANT :** Only models with a static input shape can be deployed on STM32 boards.
However, float models provided for quantization must have input shape (batch, n_fft // 2 + 1, seq_len) where the batch and seq_len axis have dynamic length.
**When quantizing a model, the model zoo will output TWO models : one with the dynamic axes preserved, and one with fixed input shape.**
**The model with fixed input shape will have shape (1, n_fft // 2 + 1, `static_sequence_length`).**
**This second model with static input shape is for use in deployment on STM32N6 boards.**

- `static_axis_name` is the name of the axis corresponding to sequence length, which should be the last axis. If using the model zoo, keep it to `seq_len`.

- The parameters following this are directly passed to `onnxruntime`'s `quantize_static` function. See [here](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py#L461) for more details.

</details></ul>
</details>
<details open><summary><a href="#3"><b>3. Quantize your model</b></a></summary><a id="3"></a>

Run the following command, from the [the root folder of this UC](../) directory :

```bash
python stm32ai_main.py
```
</details>