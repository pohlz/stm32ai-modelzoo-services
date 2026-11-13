# How can I finetune a ST model on my own dataset?

With ST Model Zoo, you can easily pick an already pretrained available model and finetune it on your own dataset.

## Pick a pretrained model

A choice of model architectures pretrained on multiple dataset can be found [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/human_activity_recognition/pretrained_models).

## Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Training:
    - To simply train the model on my data and get as output the trained tensorflow model (.h5).
- Chain_tb:
    - To train and benchmark the model on a real hardware in one go.

For any details regarding the parameters of the config file, you can look here:
- [Training documentation](../README_TRAINING.md)
- [Benchmark documentation](../README_BENCHMARKING.md)


## Finetune the model on my dataset

As usual, to retrain the model we edit the user_config.yaml and run the stm32ai_main.py python script (both found in /src).

In this example, the dataset used is the [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php).


The most important parts here are to define:
- The operation mode to training
- The data paths for the training. Optinally, a validation and a test sets
- Define which and how many classes you want to detect
- Choose a model, its pretrained weights and input size
- The other training parameters

```yaml
# user_config.yaml 
 # user_config.yaml

general:
  project_name: human_activity_recognition
  logs_dir: logs
  saved_models_dir: saved_models
  display_figures: True
  global_seed: 123
  gpu_memory_limit: 4

operation_mode: training

dataset:
  dataset_name: wisdm
  # Define the classes you want to detect
  class_names: [Jogging,Stationary,Stairs,Walking]
  training_path: ../datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt
  validation_split: 0.2
  test_path:
  test_split: 0.25

preprocessing: # mandatory
  gravity_rot_sup: true  # mandatory
  normalization: false # mandatory
model:
  # Name of the model retrieved from model zoo
  model_name: ign
  input_shape: (24, 3, 1) # input shape: window of 24 sample of 3 axis accelerometer

training:
  # all the parameters below are standard in machine learning, you can look for them in google
  # they mostly depends on the topology of your model and will need a lot of testing
  dropout: 0.5
  batch_size: 256
  epochs: 200
  optimizer:
    Adam:
      learning_rate: 0.001
  # all tensorflow callbacks are compatible with ST Model Zoo
  callbacks:
    ReduceLROnPlateau:
      monitor: val_accuracy
      mode: max
      factor: 0.5
      patience: 40
      min_lr: 1.0e-05
    EarlyStopping:
      monitor: val_accuracy
      mode: max
      restore_best_weights: true
      patience: 60

mlflow:
  uri: ./experiments_outputs/mlruns

hydra:
  run:
    dir: ./experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

You can look at user_config.yaml examples for any operation mode [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/human_activity_recognition/config_file_examples)


## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the folder /src). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```