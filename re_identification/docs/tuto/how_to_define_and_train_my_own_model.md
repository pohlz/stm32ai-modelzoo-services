# How can I define and train my own model with ST Model Zoo?

With ST Model Zoo, you can easily define and train your own TensorFlow neural network model.

## Define my model

First, create your own model in /tf/src/models/custom_model.py for it to be automatically used with the model zoo training script.
Open the python file and copy your model topology inside, here is the default example model:

```python
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers

def get_custom_model(num_classes: int = None, input_shape: Tuple[int, int, int] = None,
                     dropout: Optional[float] = None) -> tf.keras.Model:
    """
    Creates a custom image classification model with the given number of classes and input shape.

    Args:
        num_classes (int): Number of classes in the classification task.
        input_shape (Tuple[int, int, int]): Shape of the input image.
        dropout (Optional[float]): Dropout rate to be applied to the model.

    Returns:
        keras.Model: Custom image classification model.
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
    if num_classes > 2:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Define and return the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_model")
    return model
```

The model must be created inside the function get_custom_model. The input size and number of classes are then define in the user_config.yaml as below.


## Training my model

To train the model, we then edit the user_config.yaml and run the training using the python script stm32ai_main.py.

### Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Training:
    - To simply train the model and get as output the trained tensorflow model (.keras).
- Chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.keras and .tflite)
- Chain_tqeb:
    - To train, quantize, evaluate and benchmark the quantized model in one go.

For any details regarding the parameters of the config file or the data augmentation, you can look here:

- [Training documentation](../README_TRAINING.md)
- [Evaluation documentation](../README_EVALUATION.md)
- [Quantization documentation](../README_QUANTIZATION.md)
- [Benchmark documentation](../README_BENCHMARKING.md)


### Benchmarking Configuration example:

The most important parts here are to define:
- The operation mode to training
- The data paths for training, validation and test
- Define which and how many classes you want to detect (ie the model output size)
- The model name to custom for model zoo to load the model in custom_model.py
- The input_shape and other training parameters

```yaml
# user_config.yaml

general:
  project_name: my_project
  logs_dir: logs
  saved_models_dir: saved_models
  display_figures: True

operation_mode: training

model:
  model_name: custom  # Mandatory, must be 'custom' to load the model defined in custom_model.py
  input_shape: [256, 128, 3]  # Mandatory, define the model input size

dataset:
  dataset_name: DeepSportradar
  training_path: ./datasets/DeepSportradar-ReID/reid_training  # Mandatory
  validation_path:        # Optional
  validation_split: 0.2   # Optional, default value is 0.2
  test_query_path:        ./datasets/DeepSportradar-ReID/reid_test/query
  test_gallery_path:      ./datasets/DeepSportradar-ReID/reid_test/gallery
  check_image_files: False  # Optional, set it to True if you want to check that all the image files can be read successfully
  seed: 127               # Optional, there is a default seed

preprocessing:
   rescaling:
      scale: 1/127.5
      offset: -1
   resizing:
      aspect_ratio: fit
      interpolation: nearest
   color_mode: rgb

data_augmentation:
  random_contrast:
    factor: 0.4
  random_brightness:
    factor: 0.2
  random_flip:
    mode: horizontal_and_vertical
  random_translation:
    width_factor: 0.2
    height_factor: 0.2
  random_rotation:
    factor: 0.15
  random_zoom:
    width_factor: 0.25
    height_factor: 0.25

training:
  frozen_layers: None
  dropout: 0.25
  batch_size: 128
  epochs: 200
  optimizer:
    Adam:
      learning_rate: 0.01
  callbacks:
    ReduceLROnPlateau:
      monitor: val_loss
      mode: min
      factor: 0.7
      patience: 40
      min_lr: 1.0e-05
    EarlyStopping:
      monitor: val_loss
      mode: min
      restore_best_weights: true
      patience: 100
  triplet_loss:
    margin: 0.3  # Optional between [0, 1], default is 0.3
    strategy: semi_hard   #choices=['hard', 'semi_hard', 'simple'], default is 'hard'
    distance_metric: cosine  # Optional, choices=['euclidean', 'cosine'], default is 'cosine'

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
  
```
For the Chain_tqe and Chain_tqeb operation modes, you need to edit the config file to add part related to the quantization and benchmark. Look at the documentation linked above for more details.

You can also find examples of user_config.yaml [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/re_identification/config_file_examples)

## Run the script:

Edit the user_config.yaml then open a CMD (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```