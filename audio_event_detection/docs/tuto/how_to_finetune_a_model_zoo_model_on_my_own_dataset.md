# How can I finetune a ST Model Zoo model on my own dataset ?

With ST Model Zoo, you can easily pick an already pretrained available model and finetune it on your own dataset.

## Pick a pretrained model

A choice of model architectures pretrained on multiple datasets can be found [here](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/main/audio_event_detection).

## Operation modes

Depending on what you want to do, you can use the operation modes below:
- Training:
    - To simply train the model on my data and get as output the trained tensorflow model (.h5).
- Chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.h5 and .tflite)
- Chain_tbqeb:
    - To train, benchmark, quantize, evaluate and benchmark again the model in one go.

For any details regarding the parameters of the config file, you can look here:
- [Training documentation](../README_TRAINING.md)
- [Quantization documentation](../README_QUANTIZATION.md)
- [Benchmark documentation](../README_OVERVIEW.md)
- [Evaluation documentation](../README_OVERVIEW.md)


## Finetune the model on my dataset

As usual, to retrain the model we edit the user_config.yaml and run the stm32ai_main.py python script.

The standard format supported by training scripts for audio event detection is the ESC-50 format. 
However, it is also possible to use the [FSD50K dataset](https://zenodo.org/records/4060432) as-is in the model zoo, without converting it to the ESC format yourself.

Find below two examples of training configuration files, one for a custom dataset using the ESC format (we'll use [ESC-50](https://github.com/karolpiczak/ESC-50) here), and one for the FSD50K dataset.
More information [here]()

The most important parts here are to define:
- The operation mode to training
- The data paths for training, validation and test
- Which and how many classes you want to detect
- Choose a model, its pretrained weights and input size
- The other training parameters

```yaml
# user_config.yaml ESC-50 dataset Format

general:
  project_name: aed_project
  logs_dir: logs
  saved_models_dir: saved_models
  global_seed: 120
  gpu_memory_limit: 5
  display_figures: True 


operation_mode: training

model: 
  model_name: yamnet_e256
  input_shape: (64, 96, 1)
  pretrained: True # Set to True if you want to use pretrained weights provided in the model zoo
                           # Yamnet-256 can only be used with pretrained weights.
dataset:
  dataset_name: custom # ESC-50 format dataset
  class_names: ['dog', 'chainsaw', 'crackling_fire', 'helicopter', 'rain', 'crying_baby', 'clock_tick', 'sneezing', 'rooster', 'sea_waves'] # your classes
  file_extension: '.wav'
  training_audio_path: ./datasets/ESC-50/audio # Mandatory
  training_csv_path:   ./datasets/ESC-50/meta/esc50.csv # Mandatory

  validation_audio_path: # Optional
  validation_csv_path: # Optional
  validation_split: 0.2  # Optional, default value is 0.2

  test_audio_path: # Optional
  test_csv_path: # Optional

  multi_label: False 
  use_garbage_class: False 
  n_samples_per_garbage_class: 2
  expand_last_dim: True
  seed: 120 # Optional, there is a default seed
  to_cache: True
  shuffle: True

preprocessing:
  min_length: 1
  max_length : 10
  target_rate: 16000
  top_db: 60
  frame_length: 3200
  hop_length: 3200
  trim_last_second: False
  lengthen : 'after'

feature_extraction:
  patch_length: 96
  n_mels: 64
  overlap: 0.25
  n_fft: 512
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
  include_last_patch: False

# Optional
data_augmentation:
  GaussianNoise: 
    enable: True
    scale : 0.1
  VolumeAugment:
    enable: True
    min_scale: 0.8
    max_scale: 1.2
  SpecAug: 
    enable : False
    freq_mask_param: 1
    time_mask_param: 1
    n_freq_mask: 3
    n_time_mask: 4
    mask_value : 0

training:
  fine_tune: False # Set to True if you want to fine-tune a pretrained model from the zoo
  dropout: 0
  batch_size: 16
  epochs: 50 
  resume_training: # Optional, use to resume a training from a previous experiment.
  optimizer:
    Adam:
      learning_rate: 0.001
  callbacks:          # Optional section
    ReduceLROnPlateau:
      monitor: val_accuracy
      mode: max
      factor: 0.5
      patience: 100
      min_lr: 1.0e-05
    EarlyStopping:
      monitor: val_accuracy
      mode: max
      restore_best_weights: true
      patience: 60

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/yamnet_256_esc_10_second_run
  
```

Below is an example using the FSD50K dataset. Only the dataset part changes a little bit, the rest can stay the same:

```yaml
# user_config.yaml with FSD50K format
general:
  project_name: aed_project
  logs_dir: logs
  saved_models_dir: saved_models
  global_seed: 120
  gpu_memory_limit: 5
  display_figures: False 
  deterministic_ops: False

operation_mode: training 

model: 
  model_name: yamnet_e256
  input_shape: (64, 96, 1)
  pretrained: True # Set to True if you want to use pretrained weights provided in the model zoo
                           # Yamnet-256 can only be used with pretrained weights.

dataset:
  dataset_name: fsd50k # FSD50K format dataset
  class_names: ['Speech', 'Gunshot_and_gunfire', 'Crying_and_sobbing', 'Knock', 'Glass'] # your classes
  file_extension: '.wav'
  training_audio_path: # overwritten by the dataset_specific path below
  training_csv_path: # overwritten by the dataset_specific path below

  validation_audio_path: # Optional
  validation_csv_path: # Optional
  validation_split: 0.2  # Optional, default value is 0.2

  test_audio_path: # Optional
  test_csv_path: # Optional

  multi_label: False 
  use_garbage_class: True 
  n_samples_per_garbage_class: 2
  expand_last_dim: True
  seed: 120 # Optional, there is a default seed
  to_cache: True
  shuffle: True

dataset_specific:
  # Contains dataset-specific parameters.
  # Currently only supports fsd50k.
  # These parameters only need to be filled out IF the dataset name is set to 'fsd50K'
  fsd50k:
    csv_folder: ./datasets/FSD50K/FSD50K.ground_truth
    dev_audio_folder: ./datasets/FSD50K/FSD50K.dev_audio
    eval_audio_folder: ./datasets/FSD50K/FSD50K.eval_audio
    # Change this next line to the ontology path on your machine. 
    # Download the ontology at https://github.com/audioset/ontology
    audioset_ontology_path: src/preprocessing/dataset_utils/fsd50k/audioset_ontology.json 
    only_keep_monolabel: True

preprocessing:
  min_length: 1
  max_length : 10
  target_rate: 16000 # Must be either 16000 or 48000 if deploying on a STM32 board
  top_db: 60
  frame_length: 3200
  hop_length: 3200
  trim_last_second: False
  lengthen : 'after'

feature_extraction:
  patch_length: 96
  n_mels: 64
  overlap: 0.25
  n_fft: 512 # Must be a power of 2 if deploying on an STM32 board
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
  include_last_patch: False

# Optional
data_augmentation:
  GaussianNoise: 
    enable: True
    scale : 0.2
  VolumeAugment:
    enable: True
    min_scale: 0.8
    max_scale: 1.2
  SpecAug: 
    enable : False
    freq_mask_param: 1
    time_mask_param: 1
    n_freq_mask: 3
    n_time_mask: 4
    mask_value : 0

training:
  fine_tune: False # Set to True if you want to fine-tune a pretrained model from the zoo
  dropout: 0
  batch_size: 32
  epochs: 200 
  resume_training: # Optional, use to resume a training from a previous experiment.
                      
  optimizer:
    Adam:
      learning_rate: 0.005
  callbacks:          # Optional section
    ReduceLROnPlateau:
      monitor: val_accuracy
      mode: max
      factor: 0.5
      patience: 100
      min_lr: 1.0e-05
    EarlyStopping:
      monitor: val_accuracy
      mode: max
      restore_best_weights: true
      patience: 60

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/whatever
```

You can look at user_config.yaml examples for any operation mode [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/audio_event_detection/config_file_examples)

You can also look at the configuration files used to obtain the pretrained yamnet available in the ST Model Zoo:
- [FSD50K pretrained yamnet yaml](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/audio_event_detection/yamnet/ST_pretrainedmodel_public_dataset/fsd50k/yamnet_256_64x96_tl/without_unknown_class/yamnet_256_64x96_tl_config.yaml)
- [ESC-10 pretrained yamnet yaml](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/audio_event_detection/yamnet/ST_pretrainedmodel_public_dataset/esc10/yamnet_256_64x96_tl/yamnet_256_64x96_tl_config.yaml)

## Run the script

Edit the user_config.yaml then open a terminal (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```