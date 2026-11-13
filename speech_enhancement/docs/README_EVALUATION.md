# Speech enhancement model evaluation

This tutorial will show you how to evaluate a speech enhancement model using the STM32 model zoo. 

For this tutorial, we will use the CSTR VCTK + DEMAND, colloquially and henceforth referred to as Valentini dataset. The dataset can be downloaded here : https://datashare.ed.ac.uk/handle/10283/2791

The model zoo allows you to evaluate Torch models, ONNX models, and quantized ONNX models. We'll be walking you through all three cases.


**IMPORTANT NOTE :** For this use case, we have chosen to support and provide models that operate in the frequency domain, as time-domain models do not perform well when quantized to 8-bit integer precision.

The general flow of inference is the following : A complex spectrogram of the noisy audio is computed and the corresponding magnitude spectrogram is given as input to the model. 

The model outputs a mask of the same shape as its input, and this mask is applied to the complex spectrogram. The masked complex spectrogram is then transformed back to the time domain by inverse STFT. This gives us the denoised time-domain signal.

Therefore, we expect all models to take magnitude spectrograms as input, i.e. tensors of the shape (batch, n_fft // 2 + 1, sequence_length), and output tensors of the same shape, corresponding to the mask.




## <a id="">Table of contents</a>

<details open><summary><a href="#1"><b>1. Download and extract the dataset</b></a></summary><a id="1"></a>

Download the Valentini dataset from https://datashare.ed.ac.uk/handle/10283/2791

Then, extract the archive to a folder of your choice on your machine.

By default, the model zoo expects that datasets are given in a similar format to the Valentini dataset. 
This means clean speech files and noisy speech files must be in separate folders, both with the same number of files. Corresponding files should be present in the same order in both folders, and ideally have the same filenames.

For this tutorial, we will only need a test set.

This means that your dataset must be comprised of :
- A folder containing the clean test audio files. All audio files must share the same format. No mixing .wav and .flac in the same folder, for example.
- A folder containing the noisy test audio files. All audio files must share the same format. This folder must have the same number of files as the above folder.

You can optionally provide training audio files, but they will be ignored for evaluation.

If you're using the Valentini dataset, then all these conditions are already satisfied, and you don't need to worry about anything.

**NOTE :** If using the Valentini dataset, the noisy audio clips in the test set have slightly higher SNR than in the training set. Therefore, you will see better test metrics than validation metrics. This is expected.

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
- `training`, specifies your training setup, including batch size, number of epochs, optimizer, etc.
- `quantization`, contains parameters related to quantization, such as number of quantization samples, quantizer options, etc.
- `evaluation` contains parameters related to model evaluation
- `tools`, specifies paths and options to the ST tools used for benchmarking and deployment
- `deployment`, contains parameters used for deployment on STM32N6 target.
- `mlflow`, specifies the folder to save MLFlow logs.
- `hydra`, specifies the folder to save Hydra logs.

This tutorial only describes enough settings for you to be able to run the evaluation service. Please refer to the [main README](./README_OVERVIEW.md) for more information. The model zoo offers many more features than those described in this short tutorial.
</details></ul>

<ul><details open><summary><a href="#2-2">2.2 Operation mode</a></summary><a id="2-2"></a>

The `operation_mode` attribute of the configuration file lets you choose which service of the model zoo you want to use (training, evaluation, quantization, deployment, or benchmarking). You can even chain these services together ! Refer to the [main README](./README_OVERVIEW.md) for more details

For this tutorial, you just need to set `operation_mode` to `evaluation`, like so : 

```yaml
operation_mode: evaluation
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

Since we are running an evaluation, no model is saved, and the GPU is only used if evaluating a Torch model. ONNX model evaluation only runs on CPU.

</details></ul>

<ul><details open><summary><a href="#2-4">2.4 Model settings</a></summary><a id="2-4"></a>

Information about the model you wish to evaluate is provided in the `model` and `model_specific` sections of the configuration file, as shown in the YAML code below : 

```yaml
model:
  model_name: STFTTCNN # For training
  state_dict_path: path/to/your/state_dict.pt # For training and evaluating torch models
  model_path: path/to/your/model.onnx # For quantization, evaluation, benchmarking and deployment only

model_specific:
  # Parameters specific to your model type, e.g. n_blocks, tcn_latent_dim for STFT-TCNN
  n_blocks: 2
  num_layers: 3
  in_channels: 257
  tcn_latent_dim: 512
  init_dilation: 2
  mask_activation: "tanh"
```

You can evaluate both Torch and ONNX models (both float and quantized) using the model zoo.

If you wish to evaluate an ONNX model, provide the path to the model using the `model_path` attribute. This works for both quantized and float models. When evaluating an ONNX model, the other attributes in this section (`model_name` and `state_dict_path`) are ignored.

If you instead wish to evaluate a Torch model, leave the `model_path` attribute empty, and use the `model_name` and `state_dict_path` attributes. 

A model of the class `model_name` will be initialized, and the weights in the state dict given in `state_dict_path` will be loaded into the model before evaluation.

If both `model_path` and `state_dict_path` are provided, the ONNX model is evaluated, and `state_dict_path` and `model_name` are ignored.

The `model_name` attribute designates the architecture of the model you want to evaluate. For now, only the STFTCNN architecture is available. The STFTTCNN is an adaptation of the TCNN model in the frequency domain. See the original paper [here](https://ieeexplore.ieee.org/document/8683634).

The `model_specific` block lets you modify parameters of the specific model_name you chose. It will contain different attributes for different models. For details on what each attribute does, refer to the [main README](./README_OVERVIEW.md), or to the docstring of the appropriate model class found in [pt/src/models/](../pt/src/models/) folder

**NOTE WHEN USING CUSTOM MODELS : Currently, the model zoo expects models to accept tensors of shape (batch, n_fft // 2  + 1, sequence_length) as input, corresponding to magnitude spectrograms. Make sure this is the case for your custom model.** 

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

  # For the following parameters, leave empty to include all samples.
  # You can set them to a specific n° of samples (integer), 
  # or a fraction (float) of their respective sets.

  num_test_samples:  # N° of samples or fraction of test set to include in test set
  random_seed: 42 # Random seed used for sampling. If left empty, sampling is not seeded.

  # The following parameters are to be used for custom datasets. 
  # You can leave them empty if "name" is "valentini", and the default paths will be used.
  clean_test_files_path:
  noisy_test_files_path:
```

If using a custom dataset, you only need to provide `clean_test_files_path` and `noisy_test_files_path`.

You can choose to only run evaluation on a select number of samples, or a fraction of the test set by using the `num_test_samples` attribute.

For more details on this section, please consult section 3.5 of the [main README](./README_OVERVIEW.md)

</details></ul>
<ul><details open><summary><a href="#2-6">2.6 Audio preprocessing</a></summary><a id="2-6"></a>

The general flow of inference is the following : A complex spectrogram of the noisy audio is computed by peforming a Short-Time Fourier Transform, and the corresponding magnitude spectrogram is given as input to the model. 

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

**IMPORTANT NOTE :** Currently, only the `LibrosaSpecPipeline` pipeline type is supported. Other pipelines are present in [pt/src/preprocessing/freq_pipeline.py](../pt/src/preprocessing/freq_pipeline.py) but are not supported by the rest of the code in the zoo.


For more details on what each parameter does, please refer to section 3.6 of the [main README](./README_OVERVIEW.md)

Different models are trained using different set of preprocessing parameters, and using different ones may lead to poor performance. Please refer to section 2.2 of the [training README](./README_TRAINING.md) for instructions on how to retrieve the configuration files used to train the different pretrained models provided in the zoo.

</details></ul>
<ul><details open><summary><a href="#2-7">2.7 Evaluation setup</a></summary><a id="2-7"></a>
 
The evaluation setup is described in the `evaluation` section of the configuration file, as illustrated in the example below.

```yaml
evaluation:
  logs_path: eval_logs/ # Path to evaluation logs, appended to general.logs_dir
  device: "cuda:0" # Only used when evaluating Torch models.
  # If evaluating models with a fixed sequence length axis length, set the following parameter
  # to the length of the axis. E.g. if input shape is [1, 257, 20], set fixed_sequence_length to 20.
  # If evaluating models with a dynamic sequence length axis, leave empty.
  fixed_sequence_length: 
```

Some comments : 

`device` is only used when evaluating a float Torch model, as ONNX inference only happens on CPU.

`fixed_sequence_length` is used when evaluating an ONNX model with a static sequence length axis (the last axis, as we expect tensors to be of the shape (batch, n_fft // 2 +1, sequence_length)). If you train a model using the model zoo, the output will be models with a DYNAMIC sequence length axis (e.g. it will accept tensors of any sequence length as input.).

When quantizing a model using the model zoo, you will receive both a quantized model with DYNAMIC sequence length axis, and one with STATIC sequence length axis as output.

**We recommend you evaluate the model with a DYNAMIC sequence length axis, and use only the model with a STATIC sequence length axis for deployment.**

To evaluate a model with a dynamic sequence length axis, simply leave `static_sequence_length` empty.

**IMPORTANT NOTE : If you evaluate a model with a STATIC sequence length axis, clips will be trimmed or padded to that sequence length. This can heavily skew evaluation results, meaning the results may not be representative.**


</details></ul>
</details>
<details open><summary><a href="#3"><b>3. Evaluate your model</b></a></summary><a id="3"></a>

Run the following command, from the [the root folder of this UC](../) directory:

```bash
python stm32ai_main.py
```

</details>
<details open><summary><a href="#4"><b>4. Evaluation metrics</b></a></summary><a id="4"></a>

After evaluation, five metrics will be reported : 
- [PESQ (Perceptual Evaluation of Speech Quality)](https://en.wikipedia.org/wiki/Perceptual_Evaluation_of_Speech_Quality)
- [STOI (Short-Time Objective Intelligibility)](https://ieeexplore.ieee.org/document/5495701)
- MSE between the clean and denoised waveforms
- [SNR (Signal-to-Noise Ratio)](https://en.wikipedia.org/wiki/Signal-to-noise_ratio) between the clean and denoised waveforms
- Scale-invariant SNR

</details>
<details open><summary><a href="#5"><b>5. Visualize evaluation results</b></a></summary><a id="5"></a>

All evaluation metrics and results are saved under the output directory specified in the config file, like so : 

```yaml
hydra:
  run:
    dir: ./experiment_outputs/${now:%Y_%m_%d_%H_%M_%S}
```
By default, the output directory is `./pt/src/experiment_outputs/<date_time_of_your_run>/` folder. Note that this directory will NOT exist before you run the model zoo at least once.

This directory contains the following files : 
- The .hydra folder contains Hydra logs
- The eval_logs or eval_logs_quantized directory, depending on whether a quantized or float model was evaluated
    - In case both models were evaluated (i.e. running a chain), both folders will be present
    - This folder contains metrics_dict.json, which contains the average evaluation metrics on the test set, and detailed_metrics.csv, containing the evaluation metrics on each individual clip, which you can use to compute further statistics.
- stm32ai_main.log is a text log of the events that happened during this run of the model zoo. 

For more details on the list of outputs, and the structure of the output directory, please consult section 1.2 of the [main README](./README_OVERVIEW.md)

</details>
<details open><summary><a href="#6"><b>6. Run MLFlow</b></a></summary><a id="6"></a>

MLflow is an API for logging parameters, code versions, metrics, and artifacts while running machine learning code and for visualizing results.
To view and examine the results of multiple trainings, you can simply access the MLFlow Webapp by running the following command in the `./pt/src/experiments_outputs/` directory :
```bash
mlflow ui
```
And open the given IP adress in your browser.

</details>
