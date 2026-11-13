# Speech enhancement model training

This tutorial will show you how to train a speech enhancement model using the STM32 model zoo. 

For this tutorial, we will use the CSTR VCTK + DEMAND, colloquially and henceforth referred to as Valentini dataset. The dataset can be downloaded here : https://datashare.ed.ac.uk/handle/10283/2791

**IMPORTANT NOTE :** For this use case, we have chosen to support and provide models that operate in the frequency domain, as time-domain models do not perform well when quantized to 8-bit integer precision.

The general flow of inference is the following : A complex spectrogram of the noisy audio is computed and the corresponding magnitude spectrogram is given as input to the model. 

The model outputs a mask of the same shape as its input, and this mask is applied to the complex spectrogram. The masked complex spectrogram is then transformed back to the time domain by inverse STFT. This gives us the denoised time-domain signal.

Therefore, we expect all models to take magnitude spectrograms as input, i.e. tensors of the shape (batch, n_fft // 2 + 1, sequence_length), and output tensors of the same shape, corresponding to the mask.



## <a id="">Table of contents</a>

<details open><summary><a href="#1"><b>1. Download and extract the dataset</b></a></summary><a id="1"></a>

Download the Valentini dataset from https://datashare.ed.ac.uk/handle/10283/2791 .

Then, extract the archive to a folder of your choice on your machine.

By default, the model zoo expects that datasets are given in a similar format to the Valentini dataset. 
This means clean speech files and noisy speech files must be in separate folders, both with the same number of files. Corresponding files should be present in the same order in both folders, and ideally have the same filename.

Training and test sets must also be in different folders.

This means that your dataset must be comprised of :
- A folder containing the clean training audio files. All audio files must share the same format. No mixing .wav and .flac in the same folder, for example.
- A folder containing the noisy training audio files. All audio files must share the same format. This folder must have the same number of files as the above folder
- A folder containing the clean test audio files. All audio files must share the same format. No mixing .wav and .flac in the same folder, for example.
- A folder containing the noisy test audio files. All audio files must share the same format. This folder must have the same number of files as the above folder.

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

This tutorial only describes enough settings for you to be able to run the training service. Please refer  to the [main README](./README_OVERVIEW.md) for more information. The model zoo offers many more features than those described in this short tutorial.

</details></ul>
<ul><details open><summary><a href="#2-2">2.2 Using a premade configuration file</a></summary><a id="2-2"></a>

The [pretrained_models on GH](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/main/speech_enhancement/) directory contains several subfolders, one for each model architecture.
Some of these models need quite different pre-processing, feature extraction and training parameters, and using different ones could lead to wildly varying performance.

**Each of these subdirectories contains the config.yaml file that was used to train the model**.
To use these in training, copy them over to the [the root folder of this UC](../) folder, and rename them to `user_config.yaml`.

If using one of these configuration files, you will need to change the `operation_mode` parameter to `training`. See the next section for more information.

**If you want to reproduce the listed performance, we recommend you use these available .yaml files.**

**Performance may be quite different if you use different parameters.**

</details></ul>
<ul><details open><summary><a href="#2-3">2.3 Operation mode</a></summary><a id="2-3"></a>

The `operation_mode` attribute of the configuration file lets you choose which service of the model zoo you want to use (training, evaluation, quantization, deployment, or benchmarking). You can even chain these services together ! Refer to the [main README](./README_OVERVIEW.md) for more details.

For this tutorial, you just need to set `operation_mode` to `"training"`, like so : 

```yaml
operation_mode: training
```

</details></ul>
<ul><details open><summary><a href="#2-4">2.4 General settings</a></summary><a id="2-4"></a>

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

</details></ul>

<ul><details open><summary><a href="#2-5">2.5 Model settings</a></summary><a id="2-5"></a>

Information about the model you wish to train is provided in the `model` and `model_specific` sections of the configuration file, as show in the YAML code below : 

```yaml
model:
  model_name: STFTTCNN # For training
  state_dict_path: # For training
  model_path: # For quantization, evaluation, benchmarking and deployment only

model_specific:
  # Parameters specific to your model type, e.g. n_blocks, tcn_latent_dim for STFT-TCNN
  n_blocks: 2
  num_layers: 3
  in_channels: 257
  tcn_latent_dim: 512
  init_dilation: 2
  mask_activation: "tanh"
```

The `model_name` attribute designates the architecture of the model you want to train. For now, only the STFTCNN architecture is available. The STFTTCNN is an adaptation of the TCNN model in the frequency domain. See the original paper here : (https://ieeexplore.ieee.org/document/8683634). This attribute is ONLY used for training, and the evaluation of float Torch models (but NOT ONNX models).

Additionally, you can train a custom model by setting `model_name` to `Custom`, and defining your architecture in [this python file](../pt/src/models/custom.py).

The `state_dict_path` attribute lets you provide a Pytorch state dict that is loaded into your model before training starts. This can be useful to start training from specific model weights.

The `model_path` attribute is unused for training.

The `model_specific` block lets you modify parameters of the specific model_name you chose. It will contain different attributes for different models. For details on what each attribute does, refer to the [main README](./README_OVERVIEW.md), or to the docstring of the appropriate model class (found in [models/](../pt/src/models/)).

**NOTE WHEN USING CUSTOM MODELS : Currently, the model zoo expects models to accept tensors of shape (batch, n_fft // 2  + 1, sequence_length) as input, corresponding to magnitude spectrograms. Make sure this is the case for your custom model.** 

The general flow of inference is the following : A complex spectrogram of the noisy audio is computed and the corresponding magnitude spectrogram is given as input to the model. 

The model outputs a mask of the same shape as its input, and this mask is applied to the complex spectrogram. The masked complex spectrogram is then transformed back to the time domain by inverse STFT. This gives us the denoised time-domain signal.

Therefore, we expect all models to take magnitude spectrograms as input, i.e. tensors of the shape (batch, n_fft // 2 + 1, sequence_length), and output tensors of the same shape, corresponding to the mask.


</details></ul>
<ul><details open><summary><a href="#2-6">2.6 Dataset specification</a></summary><a id="2-6"></a>

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

  num_training_samples:  # N° of samples or fraction of training set to include in training set
  num_validation_samples: 100 # N° of samples or fraction of training set to include in validation set. 
  num_test_samples:  # N° of samples or fraction of test set to include in test set
  shuffle: True # If True, training dataset is shuffled each epoch
  random_seed: 42 # Random seed used for sampling. If left empty, sampling is not seeded.

  # The following parameters are to be used for custom datasets. 
  # You can leave them empty if "name" is "valentini", and the default paths will be used.
  clean_train_files_path:
  clean_test_files_path:
  noisy_train_files_path:
  noisy_test_files_path:
```

For more details on this section, please consult section 3.6 of the [main README](./README_OVERVIEW.md).

</details></ul>
<ul><details open><summary><a href="#2-7">2.7 Audio preprocessing</a></summary><a id="2-7"></a>

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

**IMPORTANT NOTE :** Currently, only the `LibrosaSpecPipeline` pipeline type is supported. Other pipelines are present in [preprocessing/freq_pipeline.py](../pt/src/preprocessing/freq_pipeline.py) but are unsupported by the rest of the code in the model zoo.


For more details on what each parameter does, please refer to section 3.7 of the [main README](./README_OVERVIEW.md).

Different models are trained using different set of preprocessing parameters, and using different ones may lead to poor performance. Please refer to section [2.2](#2-2) of this README for instructions on how to retrieve the configuration files used to train the different pretrained models provided in the zoo.

</details></ul>
<ul><details open><summary><a href="#2-8">2.8 Training setup</a></summary><a id="2-8"></a>
 
The training setup is described in the `training` section of the configuration file, as illustrated in the example below.

```yaml
training:
 device: cuda:0 # Device on which to run training
 epochs: 100 # Number of epochs

 # Optimizer parameters
 optimizer: Adam # Can use any torch optimizer
 # Add additional arguments to be passed to the optimizer to this dict
 optimizer_arguments: {lr: 0.001}

 # Training loop parameters
 loss: spec_mse # one of ["wave_mse", "wave_snr", "wave_sisnr", "spec_mse"]
 batching_strategy: pad # one of ["trim", "pad"]

 # Dataloader parameters
 num_dataloader_workers: 4 # Should divide batch_size. Set to 0 if on Windows and having issues.
 batch_size: 4 # Recommend keeping it low if using "trim" batching strategy

 # Reference metric used for early stopping, and saving the best model produced during training
 reference_metric: pesq # One of 'train_loss', 'wave_mse', 'stoi', 'pesq', 'snr', 'si-snr'
 
 # Early stopping parameters
 early_stopping: True # True/False to enable/disable
 early_stopping_patience: 50 # Number of epochs with no improvement in reference_metric before training stops.


#  Regularization parameters
#  Comment the following block to remove all regularization during training.
 regularization:
  weight_clipping_max: 1.0 # Leave empty to disable weight clipping
  activation_regularization: 1e-4 # Leave empty to disable activation regularization
  act_reg_layer_types: [Conv1d, DepthwiseSeparableConv] # Type of layers to regularize
  act_reg_threshold: 50.0 # Will not penalize activations below threshold. Leave empty to penalize all activations.
  penalty_type: l2 # "l1" or "l2"

 # Checkpointing and snapshotting parameters
 save_every: 2 # Saves a checkpoint and snapshot every n epochs
 snapshot_path:  # Set this to a previously saved training snapshot to resume training.
 ckpt_path: ckpts/ # Path to checkpoints, appended to general.saved_model_dir
 logs_filename: training_logs.csv # appended to general.logs_dir

 # ONNX exporting parameters
 opset_version: 17
```

Some comments : 

- Use the `device` attribute to choose where to run your training. `cpu` runs on CPU, `cuda:0` runs on the first CUDA-enabled GPU, `cuda:1` on the second, etc.
- `optimizer` accepts any optimizer found in the [torch.optim](https://pytorch.org/docs/stable/optim.html) module.
- `optimizer_arguments` is a dict passed directly to your chosen Torch optimizer. You can give a lot more arguments than just learning rate (e.g. momentum).
- `loss` can be one of four different losses : 

    -`spec_mse` (MSE between the clean and denoised complex spectrograms), 

    -`wave_mse` (MSE between the clean and denoised waveforms), 

    -`wave_snr` (SNR between the clean and denoised waveforms, in dB),

    -`wave_sisnr`(Scale-invariant SNR between the clean and denoised waveforms, in dB).

    These different losses have different scales, so make sure to change your learning rate accordingly.

- `batching_strategy`: Because different audio clips in a batch have different lengths, we must either pad them to the length of the longest one in a batch, or trim them to the length of the shortest one in a batch. In the case where `pad` is selected, a loss mask is applied so that the loss on pad frames is not computed.

- `reference_metric` is the metric used to determine early stopping, and which model to save when saving the "best model" at the end of training.

- The `regularization` attributes let you tune how much, if any regularization is applied to your model during training. 

    Regularization is important, because unregularized models can sometimes have a small amount of extremely large coefficients in some activation or weight tensors, which leads to extreme performance loss after quantization. 

- The `save_every` parameter controls how often training checkpoints and snapshots are saved. A checkpoint is the state_dict of the model during training, and a snapshot is comprised of the model state, optimizer state, training metrics and best model state at the time of saving.

- **IMPORTANT** You can restart a training (after a crash, for example) by providing the path to a training snapshot in the `snapshot_path` attribute !

- After training, models are saved as .onnx files. The `onnx_version` attribute determines which opset is used for the main domain, e.g. the ai.onnx domain.

The best model obtained at the end of the training is saved in the `./pt/src/experiments_outputs/\<date-and-time\>/saved_models` directory and is called 'best_trained_model.onnx' (see section 1.2 of the [main README](./README_OVERVIEW.md)).
For more details on what each parameter does, please refer to the [main README](./README_OVERVIEW.md)

</details></ul>
</details>
<details open><summary><a href="#3"><b>3. Train your model</b></a></summary><a id="3"></a>

Run the following command, from the [the root folder of this UC](../) directory:

```bash
python stm32ai_main.py
```

</details>
<details open><summary><a href="#4"><b>4. Model validation</b></a></summary><a id="4"></a>

After each epoch, your model will be evaluated on the validation set, which will be a subset of the test set.

Five metrics will be reported : 
- [PESQ (Perceptual Evaluation of Speech Quality)](https://en.wikipedia.org/wiki/Perceptual_Evaluation_of_Speech_Quality)
- [STOI (Short-Time Objective Intelligibility)](https://ieeexplore.ieee.org/document/5495701)
- MSE between the clean and denoised waveforms
- [SNR (Signal-to-Noise Ratio)](https://en.wikipedia.org/wiki/Signal-to-noise_ratio) between the clean and denoised waveforms
- Scale-invariant SNR

</details>
<details open><summary><a href="#5"><b>5. Visualize training results</b></a></summary><a id="5"></a>

All training artifacts, figures, and models are saved under the output directory specified in the config file, like so : 

```yaml
hydra:
  run:
    dir: ./experiment_outputs/${now:%Y_%m_%d_%H_%M_%S}
```
By default, the output directory is `./pt/src/experiment_outputs/<date_time_of_your_run>/` folder. Note that this directory will NOT exist before you run the model zoo at least once.

This directory contains the following files : 
- The .hydra folder contains Hydra logs
- The ckpts folder contains model checkpoints
- The training_logs folder contains the training snapshot, a training_logs.csv file containing the training & validation metrics, and a training_metrics.png figure plotting the aformentioned metrics.
- The saved_models directory contains the output float models (if there are any)
  - best_trained_model.onnx is the float model that obtained the best `reference_metric`
  - trained_model.onnx is the float model obtained at the end of training.
- stm32ai_main.log is a text log of the events that happened during this run of the model zoo. 

For more details on the list of outputs, and the structure of the output directory, please consult section 1.2 of the [main README](./README_OVERVIEW.md).

</details>
<details open><summary><a href="#6"><b>6. Run MLFlow</b></a></summary><a id="6"></a>

MLflow is an API for logging parameters, code versions, metrics, and artifacts while running machine learning code and for visualizing results.
To view and examine the results of multiple trainings, you can simply access the MLFlow Webapp by running the following command in the `./pt/src/experiments_outputs/` directory:
```bash
mlflow ui
```
And open the given IP adress in your browser.

</details>
