# How can I finetune a ST Model Zoo model on my own dataset?

With ST Model Zoo, you can easily pick an already pretrained model and finetune it on your own dataset.

## Pick a pretrained model

A choice of model architectures pretrained on multiple datasets can be found [here](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/main/re_identification).
Find the model you would like to based on it's input size and performances on various benchmarks. 

## Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Training:
    - To simply train the model on my data and get as output the trained tensorflow model (.keras).
- Chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.keras and .tflite)
- Chain_tqeb:
    - To train, quantize, evaluate and benchmark the quantized model in one go.

For any details regarding the parameters of the config file or the data augmentation, you can look here:
- [Training documentation](../README_TRAINING.md)
- [Evaluation documentation](../README_EVALUATION.md)
- [Quantization documentation](../README_QUANTIZATION.md)
- [Benchmark documentation](../README_BENCHMARKING.md)


## Finetune the model on my dataset

As usual, to retrain the model we edit the user_config.yaml and the stm32ai_main.py python script (both found in /src).
In this example, we retrain the mobilenetv2 model with an input size of (256x128x3) pretrained on a large public dataset imagenet, with our data.
In our case, our dataset contains DeepSportradar ReID images: [Dataset](https://github.com/DeepSportradar/player-reidentification-challenge)

The most important parts here are to define:
- The operation mode to training
- The data paths for training, validation and test
- Choose a model, its pretrained weights and input size
- The other training parameters

```yaml
# user_config.yaml

general:
  project_name: my_project
  logs_dir: logs
  saved_models_dir: saved_models
  display_figures: True
  global_seed: 127
  gpu_memory_limit: 8

model:
  model_name: mobilenetv2_a035 
  input_shape: (256, 128, 3)
  pretrained: True        # Optional, Bool 

operation_mode: training

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

For the Chain_tqe and Chain_tqeb operation modes, you need to edit the config file to add part related to the quantization and benchmark.
Look at the documentation linked above for more details.

You can also find examples of user_config.yaml [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/image_classification/config_file_examples)

## Run the script:

Edit the user_config.yaml then open a CMD (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```