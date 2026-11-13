# How can I define and train my own model with ST Model Zoo?

With ST Model Zoo, you can easily define and train your own TensorFlow neural network model.

## Define my model

First, create your own model in tf/src/models/custom_model.py for it to be automatically used with the model zoo training script.
Open the python file and copy your model topology inside, here is the default example model:

```python
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers

def get_custom_model(num_classes: int = None, input_shape: Tuple[int, int, int] = None,
              dropout: Optional[float] = None, **kwargs) -> tf.keras.Model:
    """
    Creates a custom audio classification model with the given number of classes and input shape.
    Edit this function to define your custom model.
    Args:
        num_classes (int): Number of classes in the classification task.
        input_shape (Tuple[int, int, int]): Shape of the input spectrogram
        dropout (Optional[float]): Dropout rate to be applied to the model.

    Returns:
        keras.Model: Custom classification model.
    """
    # Define the input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Define the feature extraction layers
    x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)

    # Define the classification layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Define and return the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_model")
    return model


```
The model must be created inside the function get_custom_model. The input size and number of classes are then define in the user_config.yaml. See below.

## Training my model

To train the model, edit the user_config.yaml and run the training using the python script stm32ai_main.py.
In this example, the dataset used is the [ESC-10](https://github.com/karolpiczak/ESC-50) and we used only some classes.

### Operation modes:

Depending on what you want to do, you can use the operation modes below:

- Training:
    - To simply train the model and get as output the trained tensorflow model (.h5).
- Chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.h5 and .tflite)
- Chain_tbqeb:
    - To train, benchmark, quantize, evaluate and benchmark again the model in one go.

For any details regarding the parameters of the config file, you can look here:

- [Training documentation](../README_TRAINING.md)
- [Main README](../README_OVERVIEW.md)


### YAML configuration example:

In the example below, we use the training operation mode and our custom model. 
For simpler operation modes, you can delete the unneeded parts if you want. 

The most important parts here are to define:
- The operation mode
- The data paths for the training, validation and test sets
- Define which and how many classes you want to detect (ie the model output size)
- The model name to custom for the model zoo to load the model in custom_model.py
- The input_shape and other training parameters

Look at the documentation linked above for more details.

```yaml
# user_config.yaml
general:
  project_name: aed_project
  logs_dir: logs
  saved_models_dir: saved_models
  global_seed: 120
  gpu_memory_limit: 5
  display_figures: True 

operation_mode: training

model: 
  model_name: custom
  input_shape: (64, 96, 1)
  dropout: 0

dataset:
  dataset_name: esc10
  class_names: ['dog', 'chainsaw', 'crackling_fire', 'helicopter', 'rain', 'crying_baby', 'clock_tick', 'sneezing', 'rooster', 'sea_waves']
  file_extension: '.wav'
  training_audio_path: ./datasets/ESC-50/audio # Mandatory
  training_csv_path:   ./datasets/ESC-50/meta/esc50.csv # Mandatory

  validation_audio_path: # Optional
  validation_csv_path: # Optional
  validation_split: 0.2  # Optional, default value is 0.2

  # Needed if quantization in operation mode
  quantization_audio_path: # Optional
  quantization_csv_path: # Optional
  quantization_split: 0.1 # Optional

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
  batch_size: 16
  epochs: 50 
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
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
  
```

You can look at user_config.yaml examples for any operation mode [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/audio_event_detection/config_file_examples).

## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```