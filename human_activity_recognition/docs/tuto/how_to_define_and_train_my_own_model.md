# How can I define and train my own model with ST Model Zoo?

With ST Model Zoo, you can easily define and train your own TensorFlow neural network model.

## Define my model

First, create your own model in ./tf/src/models/custom_model.py for it to be automatically used with the model zoo training script.
Open the python file and copy your model topology inside, here is the default example model:

```python
from tensorflow import keras
from tensorflow.keras import layers

# custom model example : replace the current layers with your own topology


def get_custom_model(input_shape: tuple[int] = (24, 3, 1),
                    num_classes: int = 4,
                    dropout: float = None,
                    **kwargs):

    inputs = keras.Input(shape=input_shape)

    # you can start defining your own model layers here
    # ---------------------------------------------------------------------------------------
    x = layers.Conv2D(16, (5, 1), strides=(
        1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(32, (5, 1), strides=(
        1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (5, 1), strides=(
        1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    # ---------------------------------------------------------------------------------------

    x = keras.layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="custom_model")
    return model


```
The model must be created inside the function `get_custom_model`. The example provided here is just an example and the user is free to change the topology to their own preference. The input size and number of classes are then define in the `user_config.yaml`. 

## Training my model

To train the model, edit the `user_config.yaml` and run the training using the python script `stm32ai_main.py`.
In this example, the dataset used is the [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php).

### Operation modes:

Depending on what you want to do, you can use the operation modes below:

- Training:
    - To simply train the model and get as output the trained tensorflow model (.keras).
- Chain_tb:
    - To train and benchmark the model on a real hardware in one go to have the trained model and the footprints (memory and computational).

For any details regarding the parameters of the config file, you can look here:

- [Training documentation](../README_TRAINING.md)
- [Benchmark documentation](../README_BENCHMARKING.md)


### Yaml Configuration example:

The example below shows how to use the training operation mode with a custom model. 
For simplier operation mode, you can delete the unneeded parts if you want. 

The most important part here are to define:
- The operation mode
- The data paths for the training. Optinally, a validation and a test sets (when using `mobility_v1` dataset)
- Define which and how many classes you want to detect (fixed for WISDM dataset)
- The model name to custom for the model zoo to load the model in custom_model.py and the input_shape
- Preprocessing settings, and
- The training parameters, such as batch_size, epochs, etc.

Look at the Training documentation linked above for more details. We provide sample configurations below.

```yaml
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
    # set custom for model zoo to look for your model define earlier
    model_name: custom_model
    input_shape: (24, 3, 1) # Your input shape: window of 24 sample of 3 axis accelerometer

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
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

```

You can look at user_config.yaml examples for any operation mode [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/human_activity_recognition/config_file_examples)

## Run the script:

After the configurations are selected by editing the user_config.yaml, open a terminal (make sure to be in the folder ./human_activity_recognition/). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```