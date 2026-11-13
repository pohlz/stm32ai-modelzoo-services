# How can I define and train my own model with ST Model Zoo?

With ST Model Zoo, you can easily define and train your own TensorFlow neural network model.

## Define my model

First, create your own model in [custom_model.py](../../tf/src/models/custom_model.py) for it to be automatically used with the model zoo training script.
Open the python file and copy your model topology inside, here is the default example model:

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, UpSampling2D, Activation, Add, BatchNormalization, ReLU, MaxPooling2D
from tensorflow.keras.regularizers import L2
from typing import Any


def get_custom_model(input_shape: tuple, nb_keypoints: int, **kwargs) -> Any:

    inputs = Input(shape=input_shape)

    # Define the feature extraction layers
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(nb_keypoints, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    outputs  = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name= "custom")

    return model
```

The model must be created inside the function `get_custom_model`. The input size and number of classes are then define in the `user_config.yaml`. See below.

The `nb_keypoints` of the pose is defined in the yaml, see below in the example. Be careful about the last convolution layer that uses this parameter.

Also be very careful with the `stride` of your model (for example in this model if __input=[192,192,3]__, then __output=[48,48,17]__ so __stride = 192/48 = `4`__).

If your model has a different stride, replace `4` by your stride number in variable [self.network_stride = 4](../../tf/src/training/pe_trainer.py#L219).


## Training my model

To train the model, we then edit the user_config.yaml and run the training using the python script stm32ai_main.py.

### Dataset

For this example, we used the COCO2017 pose estimation dataset. 

You can use any dataset of the YOLO Darknet format. You can take a look at this [tutorial](./how_to_use_my_own_dataset.md) which explain how to convert a COCO dataset using our script.

### Operation modes:

Depending on what you want to do, you can use the operation modes below:
- `training`:
    - To simply train the model and get as output the trained tensorflow model (.keras).
- `chain_tqe`:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.keras and .tflite)
- `chain_tqeb`:
    - To train, quantize, evaluate and benchmark the quantized model in one go.

For any details regarding the parameters of the config file, you can look here :

- [Training documentation](../README_TRAINING.md)
- [Quantization documentation](../README_QUANTIZATION.md)
- [Benchmark documentation](../README_BENCHMARKING.md)
- [Evaluation documentation](../README_EVALUATION.md)


### Benchmarking Configuration example:

The most important parts here are to define:
- The operation mode to training
- The data paths
- Define the number of keypoints of the pose
- The model name to custom for model zoo to load the model in custom_model.py
- The input_shape and other training parameters

```yaml
# user_config.yaml

general:
  project_name: COCO_2017_pose_Demo
  logs_dir: logs
  saved_models_dir: saved_models
  num_threads_tflite: 8
  gpu_memory_limit: 8
  global_seed: 123

operation_mode: training

model:
  model_type: heatmaps_spe
  model_name: custom
  input_shape: (192, 192, 3)

dataset:
  dataset_name: coco
  keypoints: 17
  class_names: [person]
  training_path: ./datasets/coco_train_single_pose
  # validation_path: ./datasets/coco_val_single_pose
  validation_split: 0.1
  test_path: ./datasets/coco_val_single_pose
  # quantization_path: ../datasets/coco_train_single_pose
  quantization_split: 0.3

preprocessing:
  rescaling: { scale: 1/127.5, offset: -1 }
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

data_augmentation:
  random_rotation:
    factor: (-0.2,0.2) # -+0.1 = -+36 degree angle
    fill_mode: constant # constant, wrap
    fill_value: -1.
  random_periodic_resizing:
    image_sizes: [[192,192],[224,224],[256,256]]
    period: 10
  random_contrast:
    factor: 0.4
  random_brightness:
    factor: 0.3
  random_flip:
    mode: horizontal

training:
  batch_size: 64
  epochs: 1000
  optimizer:
    Adam:
      learning_rate: 0.01
  callbacks:
    ReduceLROnPlateau:
      monitor: val_oks
      mode: max
      factor: 0.5
      min_delta: 0.0001
      patience: 30
    ModelCheckpoint:
      monitor: val_oks
      mode: max
    EarlyStopping:
      monitor: val_oks
      mode: max
      min_delta: 0.0001
      patience: 45

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
  
```

You can also find example of user_config.yaml for any operation mode [here](../../config_file_examples)

## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```