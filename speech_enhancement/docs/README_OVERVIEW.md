# Speech enhancement STM32 model zoo

Remember that minimalistic yaml files are available [here](../config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/main/speech_enhancement/) are provided with their configuration .yaml file used to generate them. These are very good starting points to play with !

## <a id="">Table of contents</a>
<details open><summary><a href="#0"><b>0. Before you start</b></a></summary><a id="0"></a>

This README can be a bit detailed and overwhelming, so if you're new to this part of the model zoo, we suggest you check out some of the detailed tutorials. They will get you familiar with the different modes of operation of the model zoo.

- [Training tutorial](./README_TRAINING.md)
- [Evaluation tutorial](./README_EVALUATION.md)
- [Quantization tutorial](./README_QUANTIZATION.md)
- [Deployment tutorial](./README_DEPLOYMENT.md)


</details>
<details open><summary><a href="#1"><b>1. Model Zoo Overview</b></a></summary><a id="1"></a>
<ul><details open><summary><a href="#1-1">1.1 YAML configuration file</a></summary><a id="1-1"></a>

The model zoo is piloted solely from the [user_config.yaml](../user_config.yaml) located in the [pt/src/](../pt/src/) directory (where this README is located.)

This README explains the structure and syntax used in this file, what each parameter does, and how to edit the config file to use all of the functionalities offered by the model zoo.

</details></ul>
<ul><details open><summary><a href="#1-2">1.2 Output directory structure</a></summary><a id="1-2"></a>

When you run the Model Zoo, the files that get created are located in the `./pt/src/experiment_outputs/` by default. This behaviour can be changed. Note that this folder will not be present until you have run the model zoo at least once.

This directory contains the following files : 
- The .hydra folder contains Hydra logs
- The ckpts folder contains model checkpoints
- The training_logs folder contains the training snapshot, a training_logs.csv file containing the training & validation metrics, and a training_metrics.png figure plotting the aformentioned metrics.
- The saved_models directory contains the output float and quantized models (if there are any)
  - best_trained_model.onnx is the float model that obtained the best reference metric (see the [training README](./README_TRAINING.md) for more details)
  - trained_model.onnx is the float model obtained at the end of training.
  - preprocessed_model.onnx is the preprocessed float model used for quantization. It can safely be ignored
  - quantized_model_int8.onnx is the quantized model with dynamic input shape
  - quantized_model_int8_static.onnx is the quantized model with static input shape
- The eval_logs and eval_logs_quantized folders contain evaluation metrics for the float & quantized model respectively. These folders contain metrics_dict.json, which contains the average metrics on the test set, and detailed_metrics.csv which contains the metrics on each individual audio clip in the test set, for you to compute further statistics.
- stm32ai_main.log is a text log of the events that happened during this run of the model zoo. 
</details></ul>
</details>
<details open><summary><a href="#2"><b>2. Supported dataset format</b></a></summary><a id="2"></a>

By default, the model zoo expects that datasets are given in a similar format to the [ CSTR VCTK + DEMAND, a.k.a. Valentini](https://datashare.ed.ac.uk/handle/10283/2791) dataset. 
This means clean speech files and noisy speech files must be in separate folders, both with the same number of files. Corresponding files should be present in the same order in both folders, and ideally have the same filename.

Training and test sets must also be in different folders.

This means that your dataset must be comprised of 
- A folder containing the clean training audio files. All audio files must share the same format. No mixing .wav and .flac in the same folder, for example.
- A folder containing the noisy training audio files. All audio files must share the same format. This folder must have the same number of files as the above folder
- A folder containing the clean test audio files. All audio files must share the same format. No mixing .wav and .flac in the same folder, for example.
- A folder containing the noisy test audio files. All audio files must share the same format. This folder must have the same number of files as the above folder

If you're using the Valentini dataset, then all these conditions are already satisfied, and you don't need to worry about anything.

**NOTE :** If using the Valentini dataset, the noisy audio clips in the test set have slightly higher SNR than in the training set. Therefore, you will see better test metrics than validation metrics. This is normal.

</details>
<details open><summary><a href="#3"><b>3. Configuration file</b></a></summary><a id="3"></a>
<ul><details open><summary><a href="#3-1">3.1 YAML syntax extensions</a></summary><a id="3-1"></a>

A description of the YAML language can be found at https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started (many other sources are available on the Internet). We only cover here the extensions that have been made in the Model Zoo. 

We use "attribute-value" instead of "key-value" as in the YAML terminology, the term "attribute" begin more relevant to our application. We may use the term "attribute" or "section" for nested attribute-value pairs constructs as the one shown below. In the example YAML code below, we may indifferently refer to "model" as an attribute or as a section.

```yaml
model:
  model_name: STFTTCNN # For training
  state_dict_path: # For training and evaluating torch models
  model_path: # For quantization, evaluation, benchmarking and deployment only

```

The YAML code below shows the syntax extensions that have been made available with the Model Zoo.

```yaml
# Equivalent syntaxes for attributes with no value
attribute_1:
attribute_2: null
attribute_2: None

# Equivalent syntaxes for boolean values
attribute_1: true
attribute_2: false
attribute_3: True
attribute_4: False

# Syntax for environment variables
model_path: ${PROJECT_ROOT}/models/mymodel.h5
```

Attributes with no values can be useful to list in the configuration file all the attributes that are available in a given section and explicitly show which ones were not used.

Environment variables can be used to avoid hardcoding in the configuration file the paths to directories and files. If directories or files are moved around, you only need to change the value of the environment variables and your configuration file will keep working with no edits.

</details></ul>
<ul><details open><summary><a href="#3-2">3.2 Operation mode</a></summary><a id="3-2"></a>

The `operation_mode` top-level attribute specifies the operations you want to execute. This may be single operation or a set of chained operations.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmark and 'd' for deployment.

| operation_mode attribute | Operations |
|:---------------------------|:-----------|
| `training`| Train a speech enhancement model, using an architecture provided in the zoo, or your own custom architecture.|
| `evaluation` | Evaluate a float or a quantized speech enhancement model|
| `quantization` | Quantize a speech enhancement model |
| `benchmarking` | Benchmark a float or quantized model on an STM32 board |
| `deployment`   | Deploy a speech enhancement model on an STM32 board |
| `chain_tqeb`  | Sequentially: training, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tqe`    | Sequentially: training, quantization of trained model, evaluation of quantized model |
| `chain_eqe`    | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model |
| `chain_qb`     | Sequentially: quantization of a float model, benchmarking of quantized model |
| `chain_eqeb`   | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model |
| `chain_qd`     | Sequentially: quantization of a float model, deployment of quantized model |

</details></ul>
<ul><details open><summary><a href="#3-3">3.3 Top-level sections</a></summary><a id="3-3"></a>

The top-level sections of a configuration file are listed in the table below. They will be described in detail in the following sections.

- `general`, describes your project, including project name, directory where to save models, etc.
- `operation_mode`, a string describing the operation mode of the model zoo. You'll want to set it to "training" for this tutorial.
- `model`, describes the model you want to use and lets you eventually load a Pytorch state dict
- `model_specific`, contains parameters specific to the model architecture you want to use, such as the number of blocks, n° of layers per blocks, intermediate dimensions, etc.
- `dataset`, describes the dataset you are using, including directory paths, file extension, how many samples to use, etc.
- `preprocessing`, parameters used to perform both time and frequency-domain preprocessing on the audio data
- `training`, specifies your training setup, including batch size, number of epochs, optimizer, etc.
- `quantization`, contains parameters related to quantization, such as n° of quantization samples, quantizer options, etc.
- `evaluation` contains parameters related to model evaluation
- `tools`, specifies paths and options to the ST tools used for benchmarking and deployment
- `benchmarking` specifies the board to benchmark on.
- `deployment`, contains parameters used for deployment on STM32N6 target.
- `mlflow`, specifies the folder to save MLFlow logs.
- `hydra`, specifies the folder to save Hydra logs.

</details></ul>
<ul><details open><summary><a href="#3-4">3.4 Global settings</a></summary><a id="3-4"></a>

The `general` section and its attributes are shown below.

```yaml
general:
  project_name: speech_enhancement_project
  logs_dir: logs # Name of the directory where logs are saved
  saved_models_dir: saved_models # Name of the directory where models are saved
  gpu_memory_limit: 0.5 # Fraction of GPU's memory to use.
  display_figures: True # Set to True to display figures. Figures are saved even if set to False.
```

The `gpu_memory_limit` attribute is used to tell Pytorch how much of the GPU's memory is to be used. It must be a float in (0, 1). The GPU is used for training models, and for evaluating Pytorch models. It is not used when evaluating ONNX models, quantizing models, benchmarking or deploying models.

You can also not use a GPU at all by setting `device`  to `cpu` in the appropriate sections.

</details></ul>


<ul><details open><summary><a href="#3-5">3.5 Model settings</a></summary><a id="3-5"></a>

Information about the model you wish to train is provided in the `model` and `model_specific` sections of the configuration file, as show in the YAML code below : 

```yaml
model:
  model_name: STFTTCNN # For training
  state_dict_path: # For training or evaluation. Used to start training/evaluating from specific parameters.
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

You can also train a recurrent model composed of a 1D pointwise convolutional encoder, an LSTM stack and a 1D pointwise convolutional decoder by setting `model_name` to `ConvLSTMDenoiser`. **NOTE THAT DEPLOYMENT IS NOT SUPPORTED YET FOR  THIS MODEL**. You can still train and evaluate tthis model.

Additionally, you can train a custom model by setting `model_name` to `Custom`, and defining your architecture in [this python file](../pt/src/models/custom.py) See section <a href="#4-2"> 4.2 </a> of this README for more details.

The `state_dict_path` attribute lets you provide a Pytorch state dict that is loaded into your model before training starts. This can be useful to start training from specific model weights. 

It also lets you evaluate a Torch model, by providing both `model_name` and `state_dict_path`. Note that you can evaluate both Torch and ONNX models, and when that you do not need to provide a `model_name` or `state_dict_path` when evaluating an ONNX model. See the [Evaluation tutorial README](./README_EVALUATION.md) for more details.

The `model_path` attribute is used for evaluation, quantization, benchmarking and deployment.

The `model_specific` block lets you modify parameters of the specific `model_name` you chose. It will contain different attributes for different models. For details on what each attribute does, refer to the rest of this section, or to the docstring of the appropriate model class (found in [models/](../pt/src/models/)).

**NOTE WHEN USING CUSTOM MODELS : Currently, the model zoo expects models to accept tensors of shape (batch, n_fft // 2  + 1, sequence_length) as input, corresponding to magnitude spectrograms. Make sure this is the case for your custom model.** 

The general flow of inference is the following : A complex spectrogram of the noisy audio is computed and the corresponding magnitude spectrogram is given as input to the model. 

The model outputs a mask of the same shape as its input, and this mask is applied to the complex spectrogram. The masked complex spectrogram is then transformed back to the time domain by inverse STFT. This gives us the denoised time-domain signal.

Therefore, we expect all models to take magnitude spectrograms as input, i.e. tensors of the shape (batch, n_fft // 2 + 1, sequence_length), and output tensors of the same shape, corresponding to the mask.

- `model_name` : Model class to use. Currently, must be one of `STFTTCNN`, `ConvLSTMDenoiser` or `Custom` (the uppercase is important) (Note that `ConvLSTMDenoiser` models cannot be deployed yet).
- `state_dict_path` : Path to a Pytorch state dict. Optional. Must be compatible with the specified `model_name` and `model_specific` arguments provided. Used for training and evaluation.
If provided, training starts from the weights in the state dict, and evaluation uses the weights in the state dict. If not provided, training starts from random weights.
- `model_path` : Path to an ONNX model. Used for evaluation, quantization, benchmarking and deployment.
**If running evaluation, `model_path` takes precedence over `state_dict_path`, meaning that if both are provided, the ONNX model will be evaluated.**

- `model_specific` attributes for the `STFTTCNN` class. For details on the model, see <a href="#A-1"> Appending A.1 </a>
    - `n_blocks` : Number of TCN blocks 
    - `num_layers` : Number of residual blocks per TCN block. Max dilation factor is equal to (`init_dilation`^`num_layers` - 1)
    - `in_channels` :  Number of input channels. Should be `n_fft` // 2 + 1
    - `tcn_latent_dim` : Number of channels in intermediary Conv1D layers in TCN blocks.
    - `init_dilation` : Initial dilation factor. Dilation factor in each residual block is `init_dilation` ^(i-1). Max dilation factor is equal to (`init_dilation`^`num_layers` - 1)
    - `mask_activation` : Must be `tanh` or `sigmoid`. Activation function for the output of the model. Sigmoid activation tends to provide models that remove more noise, but degrade speech more.

- `model_specific` attributes for the `ConvLSTMDenoiser` class. For details on the model, see <a href="#A-2"> Appending A.2 </a>
    - `lstm_hidden_size` : Hidden dimension of each LSTM layer
    - `num_lstm_layers` : Number of LSTM layers
    - `in_channels` :  Number of input channels. Should be `n_fft` // 2 + 1
    - `out_channels` : Number of output channels. Should also be `n_fft`// 2 + 1
      mask_activation: "sigmoid"
    - `mask_activation` : Must be `tanh` or `sigmoid`. Activation function for the output of the model. Sigmoid activation tends to provide models that remove more noise, but degrade speech more.
</details></ul>

<ul><details open><summary><a href="#3-6">3.6 Dataset</a></summary><a id="3-6"></a>

The `dataset` section and its attributes are shown in the YAML code below.
Detailed explanations of each parameter are provided at the end of this section.

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

When a training is run, the training set is split in two to create a validation dataset.
When evaluation is run, the test set is used. Evaluation will not run if no test set is provided. The Valentini dataset includes a test set.

The `num_validation_samples` attribute specifies the training/validation set size to use when splitting the training set to create a validation set. You can provide either a fraction, or a specific number of samples.

The default value is 100, meaning that 100 samples are used to create the validation set. The `random_seed` attribute specifies the seed value to use for randomly shuffling the dataset file before splitting it. (default 42)

The `num_training_samples` and `num_test_samples` attributes allow you to train and evaluate models, respectively, on a portion of your training / test set.

The use of quantization datasets is covered in the "Quantization" section of the documentation.

- `name` : *string*, name of the dataset. Must be one of `valentini` or `custom`.
- `root_folder` : *string* : Only used if `name` is `valentini`. Path to the root folder of the Valentini dataset.
- `n_speakers` : *int* : Only used if `name` is  `valentini`. Must be 28 or 56. Used to specify whether the 28 speaker or the 56 speaker set should be used.
- `file_extension` : *string*, Extension of the audio files.
- `num_training_samples` : *int* or *float* : Number of training samples to use. If left empty, uses the whole training set. If a float is given, uses a fraction of the training set.
- `num_validation_samples` : *int* or *float* : Number of validation samples to take from the training set. If a float is given, uses a fraction of the training set.
- `num_test_samples` : *int* or *float* : Number of test samples to use. If left empty, uses the whole test set. If a float is given, uses a fraction of the test set.
- `shuffle` : *bool* If True, the training set is shuffled between epochs.
- `random_seed` : *int* Seed used to split the training set into training/validation, and to subsample the training/test set if needed.

**The following parameters are only used for custom dataset, i.e. if `name` is `custom`. They are ignored if `name` is `valentini`. Leave empty if using the Valentini dataset.**

- `clean_train_files_path` : *string* Path to the folder containing clean training audio samples
- `clean_test_files_path`: *string* Path to the folder containing clean test audio samples
- `noisy_train_files_path`: *string* Path to the folder containing noisy training audio samples
- `noisy_test_files_path`: *string* Path to the folder containing noisy test audio samples

</details></ul>
<ul><details open><summary><a href="#3-7">3.7 Audio preprocessing</a></summary><a id="3-7"></a>

The general flow of inference is the following : A complex spectrogram of the noisy audio is computed by peforming a Short-time Fourier Transform, and the corresponding magnitude spectrogram is given as input to the model. 

The model outputs a mask of the same shape as its input, and this mask is applied to the complex spectrogram. The masked complex spectrogram is then transformed back to the time domain by inverse STFT. This gives us the denoised time-domain signal.

You can additionally choose whether or not to peak normalize your audio data, and which preprocessing pipeline to use.

The 'preprocessing' section and its attributes is shown below.

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


**IMPORTANT NOTE :** Currently, only the `LibrosaSpecPipeline` pipeline type is supported, but other pipelines are present in [./pt/src/preprocessing/freq_pipeline.py](../pt/src/preprocessing/freq_pipeline.py). You can experiment with these, but be ready for bugs.

- `pipeline_type` : Type of preprocessing pipeline to use. Currently, should be left to `LibrosaSpecPipeline`.
- `peak_normalize` : *bool*, if True, audio clips are peak normalized before STFT.
- `sample_rate` : *bool*, sampling rate to use. Audio clips will be resampled if necessary to match this sampling rate. **For now, only 16kHz is supported for deployment on STM32N6**
- `n_fft` : *int*, Size of the FFT, in number of samples
- `hop_length` : *int*, Hop length (i.e. number of successive samples) between different frames, in number of samples.
- `win_length` : *int*, Size of the signal window. Set equal to `n_fft` if you want to avoid window padding
- `window` : *string*, Window type. Passed to librosa.filters.get_window().
- `center` : *boolean*, If True, frames are centered, i.e. frame `n` is centered around sample number `n * hop_length`. If False, frames begin at sample number `n * hop_length`.

**Important :** If training a model with a loss that requires an ISTFT (for example, SNR or SI-SNR loss), keep `center` to `True`, as the `torch.istft` function we use to have a differentiable inverse STFT implementation does NOT accept `center` = `False`.

- `power` : *float*, Exponent for the magnitude spectrogram. Set to 1 for energy spectrogram, and 2 for power spectrogram.

Different models are trained using different set of preprocessing parameters, and using different ones may lead to poor performance. Please refer to section <a href="#1-1"> 1.1 </a> of this README for instructions on how to retrieve the configuration files used to train the different pretrained models provided in the zoo.

</details></ul>
<ul><details open><summary><a href="#3-8">3.8 Training</a></summary><a id="3-8"></a>

A 'training' section is required in all the operation modes that include a training, namely 'training', 'tqeb' and 'tqe'.

The YAML code below is a typical example of 'training' section.
A detailed explanation of every parameter is provided at the end of this section.

```yaml
 device: cuda:0
 epochs: 100

 # Optimizer parameters
 optimizer: Adam # Can use any torch optimizer
 # Add additional arguments to be passed to the optimizer to this dict
 optimizer_arguments: {lr: 0.001}

 # Training loop parameters
 loss: spec_mse # one of ["wave_mse", "wave_snr", "wave_sisnr", "spec_mse"]
 batching_strategy: pad # one of ["trim", "pad"]

 # Dataloader parameters
 num_dataloader_workers: 4 # Should divide batch_size Set to 0 if on Windows and having issues.
 batch_size: 16 # Recommend keeping it low if using `trim` batching strategy.

 # Reference metric used for early stopping, and saving the best model produced during training
 reference_metric: pesq # One of 'train_loss', 'wave_mse', 'stoi', 'pesq', 'snr', 'si-snr'
 
 # Early stopping parameter
 early_stopping: True # True/False to enable/disable
 early_stopping_patience: 50 # Number of epochs with no improvement in reference_metric before training stops.


#  Regularization parameters
#  Comment the following block to remove all regularization during training.
 regularization:
  weight_clipping_max:  # Leave empty to disable weight clipping
  activation_regularization:  # Leave empty to disable activation regularization
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
- `optimizer_arguments` is a dict passed directly to your chosen Torch optimizer. You can give a lot more arguments than just learning rate (e.g. momentum)
- `loss` can be one of four different losses : 

    -`spec_mse` (MSE between the clean and denoised complex spectrograms), 

    -`wave_mse` (MSE between the clean and denoised waveforms), 

    -`wave_snr` (SNR between the clean and denoised waveforms, in dB),

    -`wave_sisnr`(Scale-invariant SNR between the clean and denoised waveforms, in dB)

    These different losses have different scales, so make sure to change your learning rate accordingly.

- `batching_strategy`: Because different audio clips in a batch have different lengths, we must either pad them to the length of the longest one in a batch, or trim them to the length of the shortest one in a batch. In the case where `pad` is selected, a loss mask is applied so that the loss on pad frames is not computed. If using `trim`, keep the batch size low.

- `reference_metric` is the metric used to determine early stopping, and which model to save when saving the "best model" at the end of training.

- The `regularization` attributes let you tune how much, if any regularization is applied to your model during training. 

    Regularization is important, because unregularized models can sometimes have a small amount of extremely large coefficients in some activation or weight tensors, which leads to extreme performance loss after quantization. 

- The `save_every` parameter controls how often training checkpoints and snapshots are saved. A checkpoint is the state_dict of the model during training, and a snapshot is comprised of the model state, optimizer state, training metrics and best model state at the time of saving.

- **IMPORTANT** You can restart a training (after a crash, for example) by providing the path to a training snapshot in the `snapshot_path` attribute !

- After training, models are saved as .onnx files. The `onnx_version` attribute determines which opset is used for the main domain, e.g. the ai.onnx domain.

- **IMPORTANT : If on Windows and encountering crashes during training, try setting `num_dataloader_workers` to 0. Pytorch multiprocessing has been known to cause issues on Windows.**

The best model obtained at the end of the training is saved in the 'experiments_outputs/\<date-and-time\>/saved_models' directory and is called 'best_trained_model.onnx' (see section <a href="#1-2"> 1.2 </a> of this README.)

- `device` : *string* A valid `torch.device` identifier. Use `cpu` to run your training on CPU, `cuda` to run on GPU, and `cuda:n` to run on a specific GPU
- `epochs` : *int*, number of epochs to train for
- `optimizer` : *string*, Type of optimizer to use. Must be an optimizer found in the [torch.optim](https://pytorch.org/docs/stable/optim.html) module.
- `optimizer_arguments` : *dict* : dict passed directly to your chosen Torch optimizer. You can give a lot more arguments than just learning rate (e.g. momentum)

- `loss` : *string*, one of `spec_mse`, `wave_mse`, `wave_snr` or `wave_sisnr`. Beware that the latter three require `processing.center = True`, because the Torch ISTFT function used during training requires it.
- `batching_strategy` : *string*, one of `trim` or `pad`. See above for a detailed explanation. In general, you should use `pad`. If using `trim` keep the batch size low.
- `num_dataloader_workers` : *int*, Number of CPU workers allocated to dataloaders. Should divide `batch_size` **IMPORTANT : If on Windows, and encountering crashes during training, set to 0**
- `batch_size` : *int*, Batch size used to train model. Keep low if using `trim` batching strategy.

- `reference_metric` : *string*, One of `train_loss`, `wave_mse`, `stoi`, `pesq`, `snr`, `si-snr`. This is the metric used for early stopping, and to pick the "best model" saved at the end of training. All metrics other than `train_loss` are computed on the validation set.
- `early_stopping` : *bool*, set to True to enable early stopping
- `early_stopping_patience` : *int*, If early stopping is enabled, training stops after there has been no improvement to `reference_metric` for `early_stopping_patience` epochs.

- `regularization` contains various regularization parameters. Some trained models can quantize poorly, and regularizing them during training may help.
  - `weight_clipping_max`: *float*, If set to x, weights are clamped between -x and x at the end of each batch. Leave empty to disable weight clipping
  - `activation_regularization`: *float*, Scale factor to use for activation regularization. Recommend small values, e.g. 10e-4. Leave empty to disable activation regularization
  - `act_reg_layer_types`: *list of str* : If activation regularization is enabled, these types of layers will get regularized. Leave empty to regularize all layers in the model.
  - `act_reg_threshold` : *float* : Will not penalize activations below threshold. Leave empty to penalize all activations.
  - `penalty_type`: Activation regularization penalty type. Must be "l1" or "l2".

- `save_every`: *int*, Saves a checkpoint and snapshot every n epochs
- `snapshot_path`: *string*, Path to where snapshots are saved. Set this to a previously saved training snapshot to resume training.
- `ckpt_path`: *string*, Path to checkpoints, appended to general.saved_model_dir
- `logs_filename`: *string*, Filename for training logs csv file.

- `opset_version`: *int* : Version of the main ONNX opset to use when exporting trained models.

</details></ul>
<ul><details open><summary><a href="#3-9">3.9 Quantization</a></summary><a id="3-9"></a>

This section is used to configure the quantization process, which optimizes the model for efficient deployment on embedded devices by reducing its memory usage (Flash/RAM) and accelerating its inference time, with minimal degradation in model accuracy.

**The model zoo only supports post-training quantization.**

If you run one of the operation modes that includes a model quantization, you need to include a "quantization" section in your configuration file, as shown in the YAML code below. You still need a dataset, mode and preprocessing section, but they are omitted here for the sake of readability.

Consult the [quantization tutorial README](./README_QUANTIZATION.md) for a detailed walkthrough.

```yaml
model:
  model_path: path/to/your/float/model.onnx

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

- The model zoo only supports post-training quantization.

- We recommend you only use a small portion of the training set to act as the quantization dataset. You can do this using the `num_quantization_samples` attribute.
- If using a custom dataset, pass the path to the NOISY files using the `noisy_quantization_files_path` attribute. If this attribute is empty, files will be taken from the `noisy_training_files_path` attribute under the `dataset` section.

- **IMPORTANT :** Only models with a static input shape can be deployed on STM32 boards.
However, float models provided for quantization must have input shape (batch, n_fft // 2 + 1, seq_len) where the batch and seq_len axis have dynamic length.
**When quantizing a model, the model zoo will output TWO models : one with the dynamic axes preserved, and one with fixed input shape.**
**The model with fixed input shape will have shape (1, n_fft // 2 + 1, `static_sequence_length`)**
**This second model with static input shape is for use in deployment on STM32N6 boards**

- `static_axis_name` is the name of the axis corresponding to sequence length, which should be the last axis. If using the model zoo, keep it to `seq_len`

- The parameters following this are directly passed to `onnxruntime`'s `quantize_static` function. See [here](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py#L461) for more details


- `num_quantization_samples` : *int or float* Number of samples or fraction of training set to use as quantization set.
- `random seed` : *int*, Seed used to subsample quantization samples from training set. Default is 42.
- `noisy_quantization_file_path` : *string* If using a custom dataset, pass the path to the NOISY files using this attribute. If this attribute is empty, files will be taken from the `noisy_training_files_path` attribute under the `dataset` section.
- `static_sequence_length` : *int* Length of sequence length axis when converting model to static input shape. See above for an explanation
- `static_axis_name` : *str* Name of sequence length axis. See above for an explanation.

- **The following parameters are directly passed to onnxruntime's quantize_static() function**

- `granularity` : *string*, can be per_channel or per_tensor
- `onnx_quant_parameters` : *Field* of parameters as below:
  - `calibrate_method` : *str*, calibration method used to determine scale/offset of QDQ layers.
  - `op_types_to_quantize`: *list of str*, ONNX op types to quantize. Leave empty to quantize all ops. **WARNING** : Some ONNX ops will not be quantized by default unless specified here (e.g. LSTM). This behaviour is caused by the onnxruntime package. If using an STFTTCNN, leaving this field empty will quantize every op.
- `reduce_range` : *bool*, if True, weights are quantized to 7 bit instead of 8 bit.
- `onnx_extra_options` : extra options to pass to `quantize_static()`. See [here](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py#L461) for more details. In particular, we recommend keeping `CalibMovingAverage : True`



</details></ul>
<ul><details open><summary><a href="#3-10">3.10 Model evaluation</a></summary><a id="3-10"></a>

The YAML code below shows how you can evaluate a speech enhancement model. You still need a dataset, model and preprocessing section, but they are omitted here for the sake of readability.

For more details, consult the [Evaluation tutorial README](./README_EVALUATION.md)

```yaml
model:
  model_path: path/to/your/model.onnx
evaluation:
  logs_path: eval_logs/ # Path to evaluation logs, appended to general.logs_dir
  device: "cuda:0" # Only used when evaluating torch models.
  # If evaluating models with a fixed sequence length axis length, set the following parameter
  # to the length of the axis. E.g. if input shape is [1, 257, 20], set fixed_sequence_length to 20.
  # If evaluating models with a dynamic sequence length axis, leave empty.
  fixed_sequence_length: 
```

Some comments : 

- You can evaluate both Pytorch models and ONNX models. To evaluate a Pytorch model, use the `model_name` and `state_dict_path` attributes in the `model` section. To evaluate an ONNX model, use the `onnx_model_path` attribute in the `model` section.

- You can evaluate both quantized and float ONNX models.

- `device` is only used when evaluating a float Torch model, as ONNX inference only happens on CPU.

- `fixed_sequence_length` is used when evaluating an ONNX model with a static sequence length axis (the last axis, as we expect tensors to be of the shape (batch, n_fft // 2 +1, sequence_length)). If you train a model using the model zoo, the output will be models with a DYNAMIC sequence length axis (e.g. it will accept tensors of any sequence length as input.).

When quantizing a model using the model zoo, you will recieve both a quantized model with DYNAMIC sequence length axis, and one with STATIC sequence length axis as output.

**We recommend you evaluate the model with a DYNAMIC sequence length axis, and use only the model with a STATIC sequence length axis for deployment**

To evaluate a model with a dynamic sequence length axis, simply leave `static_sequence_length` empty.

**IMPORTANT NOTE : If you evaluate a model with a STATIC sequence length axis, clips will be trimmed or padded to that sequence length. This can heavily skew evaluation results, meaning the results may not be representative**

- `device` : *string*, Pytorch device to use for evaluating a Torch model. Is ignored if evaluating an ONNX model
- `logs_path` : *string* : Path under which to save the evaluation logs
- `fixed_sequence_length` : *int*, If evaluating a static input shape model, length of the sequence length axis. See above for a detailed explanation. We heavily recomment evaluating dynamic input shape models instead of static input shape models, which should only be used for deployment.

</details></ul>
<ul><details open><summary><a href="#3-11">3.11 STM32 tools</a></summary><a id="3-11"></a>

This section covers the usage of the STEdgeAI tool, which benchmarks ONNX models, and converts them to C code

The `tools` section in the config file looks like this : 

```yaml
tools:
  stedgeai:

tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/Users/<>/stedgeai_10_RC1/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe
```
where : 
- `optimization` - *String*, define the optimization used to generate the C model, options: "*balanced*", "*time*", "*ram*".
- `on_cloud` : Set to True to use the STEdgeAI developer cloud to benchmark and convert models. You will need to make an account at [https://stedgeai-dc.st.com/home](https://stedgeai-dc.st.com/home) and will be prompted for your credentials at runtime. If you use the developer cloud, you do not need to set the next two parameters.
- `path_to_stedgeai` - *Path* to stedgeai executable file. Is only used if `on_cloud` is set to False
- `path_to_cubeIDE` - *Path* to CubeIDE executable file. Is only used if `on_cloud` is set to False

</details></ul>
<ul><details open><summary><a href="#3-12">3.12 Benchmarking</a></summary><a id="3-15"></a>

The YAML code below shows how to benchmark a model on an STM32 board. You can not benchmark Pytorch models directly, they must be exported to ONNX first.

```yaml

model:
  model_path: path/to/your/model.onnx

operation_mode: benchmarking

benchmarking:
   board: STM32N6570-DK     # Name of the STM32 board to benchmark the model on
```

The model file must be:
- an ONNX model file (quantized model) with an '.onnx' filename extension.
 
The `board` attribute is used to provide the name of the STM32 board to benchmark the model on. The available boards are 'STM32H747I-DISCO', 'STM32H7B3I-DK', 'STM32F469I-DISCO', 'B-U585I-IOT02A', 'STM32L4R9I-DISCO', 'NUCLEO-H743ZI2', 'STM32H747I-DISCO', 'STM32H735G-DK', 'STM32F769I-DISCO', 'NUCLEO-G474RE', 'NUCLEO-F401RE' and 'STM32F746G-DISCO', 'STM32N6570-DK'

For speech enhancement, the only board available for deployment is the STM32N6570-DK, and so we recommend setting `board` to 'STM32N6570-DK'.

</details></ul>
<ul><details open><summary><a href="#3-13">3.13 Deployment</a></summary><a id="3-13"></a>

The YAML code below shows how to deploy a model on an STM32 board.
Note that you need a preprocessing section, even though no data is being preprocessed. 
This is because the parameters in these sections are being used to create look-up tables that are used by the C application to preprocess data on the board in realtime, and to parametrize the preprocessing library used in the C application.

**IMPORTANT : Float model cannot be deployed on STM32N6. Your model needs to be int8 quantized for deployment.**

**VERY IMPORTANT : MAKE SURE YOUR MODEL HAS A STATIC INPUT SHAPE !**
**STEdgeAI does not handle models with a dynamic input shape.**

**For more details on deployment, see the [deployment tutorial README](./README_DEPLOYMENT.md)**

```yaml
model:
  model_path: path/to/your/quantized/model.onnx
  
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

tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

deployment:
  frames_per_patch: 30 # Number of frames per inference

  # Number of lookahead and lookback frames.
  # MAKE SURE YOUR MODEL HAS INPUT SHAPE (1, n_fft // 2 + 1, frames_per_path + 2*lookahead_frames)
  # BEFORE DEPLOYING !!!!!!
  lookahead_frames: 5 

  # After denoising is applied, any patches with power less than
  # output_silence_threshold dB get silenced completely.
  # This also applies when disabling the denoiser using the button on the board.
  # In dB. A higher number means a stricter filtering. -100 disables the filtering.
  output_silence_threshold: -50
  # Change this to the path on your machine.
   
  c_project_path:  ../../application_code/audio/STM32N6/
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32N6
    board: STM32N6570-DK
  build_conf : "BM" # this is the default configuration
  # build_conf : "BM_LP"

```

A few comments : 

- The `frames_per_patch` attributes corresponds to how many spectrogram frames get denoised per inference on the board. How much this represents in terms of time depends on your preprocessing parameters. For example, with a sampling rate of 16000 and a hop length of 160 (corresponding to 10 ms), and a `frames_per_patch` value of 30, one inference is done every 300 ms. In turn, this induces a 300 ms delay between audio acquisition and denoised audio playback.

- The model is allowed to look ahead a few frames into the future, to attenuate patch boundary artifacts on its output. It is also allowed to look the same number of frames into the past. This number of lookahead frames is controlled by the `lookahead_frames` attribute. If you reduce this attribute too much, or if your model's receptive field increases, you may see the patch boundary artifacts appear again.

- Note that the total length of a patch fed to the model during inference is equal to `frames_per_patch + 2*lookahead_frames`. Using the values in the above example, this means each patch contains 40 frames (30+5*2). One inference is still done every 300 ms, but the induced delay is 400 ms.

**VERY IMPORTANT : MAKE SURE YOUR MODEL'S INPUT SHAPE IS COMPATIBLE WITH THESE PARAMETERS !**

**Your model's input shape should be (1, n_fft // 2 + 1, frames_per_patch + 2 * lookahead_frames)**
If you quantize your model using the model zoo, simply set the `static_sequence_length` attribute to `frames_per_patch` + 2 * `lookahead_frames`. See the [Quantization README](./README_QUANTIZATION.md) for additional details.

If this is not respected, glitches WILL occur, as patch overlap will not happen properly.


- Before audio is output by the board, a threshold (in dB) is applied. If the mean power of a patch is below this threshold, then it is silenced (e.g. the whole output buffer is set to 0). The attribute `output_silence_threshold` controls this threshold.
A higher number increases the strictness of this filter. 0 filters everything, and -100 disables the filtering. Note that this filter remains on even when the denoising model is disabled.

- `frames_per_patch` : *int*, number of STFT frames processed per model inference. See above.
- `lookahaead_frames` : *int*, number of look-ahead and look-back frames given to the model per inference. See above for a detailed explanation.
- `output_silence_threshold` : *negative int* : Silence threshold applied to board audio output, in dB. 0 filters everything, and -100 disables filtering.
Note that this filter is present even when the denoiser is deactivated using the button on the board.
- `c_project_path` : *Path*, Path to the C application. Should point to [../../application_code/audio/STM32N6](../../application_code/audio/STM32N6). 
- `IDE` : Toolchain to use for compiling the C application. Should be `GCC`
- `verbosity` : Verbosity of the compiler
- `hardware_setup` : Series and board on which to dpeloy the model. Currently, only the STM32N6570-DK board is supported
- `build_conf` : Build configuration. We recommend using the default `N6 Audio Bare Metal`
</details></ul>
</details>


<details open><summary><a href="#4"><b>4. Training tips</b></a></summary><a id="4"></a>
<ul><details open><summary><a href="#4-1">4.1 Training your own model</a></summary><a id="4-1"></a>

You may want to train your own model rather than a model or architecture from the Model Zoo.

This can be done by defining your model architecture in [models/custom.py](../pt/src/models/custom.py). Define your model under the `Custom` class provided in the file, and set the `model_name` argument in your config file to `Custom`, like so : 

```yaml
  model:
   model_name: Custom
  
  model_specific:
   # Your custom model params here
```
You can also set parameters for your custom model directly from the config file by putting them in the model_specific section. These parameters must be valid input for the __init__() method of your model.
</details></ul>

<ul><details open><summary><a href="#4-2">4.2 Resuming a training</a></summary><a id="4-2"></a>

You may want to resume a training that you interrupted or that crashed.

When training a model, a snapshot is periodically saved in the current experiment output directory tree. By default, The snapshot is in the `./pt/src/experiments_outputs/<date-and-time\>/training_logs` directory and is named 'training_snapshot.pth'

To resume a training from a snapshot, simply provide the path to this snapshot in the `snapshot_path` attribute of the training section, like so : 


```yaml 
training:
  snapshot_path: path/to/your/snapshot.pth
```

It is recommended that you restart from the same configuration file that you used for the training you are resuming.

</details></ul>
</details>

<details open><summary><a href="#A"><b>Appendix A: Available Model Zoo models</b></a></summary><a id="A"></a>

The models that are available with the Model Zoo and their parameters are described below.

<ul><details open><summary><a href="#A-1">A-1 STFT-TCNN</a></summary><a id="A-1"></a>


The TCNN is a time-domain speech enhancement temporal convolutional model proposed in 2019 by Pandey and Wang in the paper [TCNN: TEMPORAL CONVOLUTIONAL NEURAL NETWORK FOR REAL-TIME SPEECH ENHANCEMENT IN THE TIME DOMAIN](https://ieeexplore.ieee.org/document/8683634).


Unfortunately, time-domain models do not perform well when quantized to 8-bit integer precision.
Therefore, we made several modifications to the model, in order to make it work in the frequency domain.
Mainly, we removed the convolutional encoder and decoder described in the paper, keeping only the main TCN part of the model, and instead substituted the encoder/decoder pair with STFT pre-processing, and inverse STFT post-processing.


This means that the model takes as input magnitude spectrogram frames, and outputs a mask of the same dimension.
Inference is then performed by applying this output mask to the complex spectrogram corresponding to the input, and performing inverse STFT on the masked complex spectrogram to retrieve the corresponding time domain denoised signal.

The parameters related to the STFT-TCNN are the following : 

- `n_blocks` : Number of TCN blocks 
- `num_layers` : Number of residual blocks per TCN block. Max dilation factor is equal to (`init_dilation`^`num_layers` - 1)
- `in_channels` :  Number of input channels. Should be `n_fft` // 2 + 1
- `tcn_latent_dim` : Number of channels in intermediary Conv1D layers in TCN blocks.
- `init_dilation` : Initial dilation factor. Dilation factor in each residual block is `init_dilation` ^(i-1). Max dilation factor is equal to (`init_dilation`^`num_layers` - 1)
- `mask_activation` : Must be `tanh` or `sigmoid`. Activation function for the output of the model. Sigmoid activation tends to provide models that remove more noise, but degrade speech more.

These parameters should be placed in the `model_specific` section of the config file. See section <a href = "#3-5"> 3.5 Model settings </a>.

</details>

<ul><details open><summary><a href="#A-2">A-2 ConvLSTMDenoiser</a></summary><a id="A-2"></a>


The ConvLSTMDenoiser is a very simple recurrent architecture consisting of a single-layer 1D pointwise convolutional encoder, followed by a stack of LSTM layers, ending with a single-layer 1D pointwise convolutional decoder.

Much like the STFT-TCNN, the ConvLSTMDenoiser operates in the frequency domain, takes magnitude spectrogram frames as input and outputs a mask that is then applied to the magnitude spectrogram.

**NOTE THAT THIS MODEL CANNOT CURRENTLY BE DEPLOYED OR BENCHMARKED**

The parameters related to this model are the following : 

- `lstm_hidden_size` : Hidden dimension of each LSTM layer
    - `num_lstm_layers` : Number of LSTM layers
    - `in_channels` :  Number of input channels. Should be `n_fft` // 2 + 1
    - `out_channels` : Number of output channels. Should also be `n_fft`// 2 + 1
      mask_activation: "sigmoid"
    - `mask_activation` : Must be `tanh` or `sigmoid`. Activation function for the output of the model. Sigmoid activation tends to provide models that remove more noise, but degrade speech more.

These parameters should be placed in the `model_specific` section of the config file. See section <a href = "#3-5"> 3.5 Model settings </a>.

</details>
