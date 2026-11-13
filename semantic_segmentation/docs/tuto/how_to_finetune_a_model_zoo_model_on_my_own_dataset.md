# How can I finetune a ST Model Zoo model on my own dataset?

With ST Model Zoo, you can easily pick an already pretrained model and finetune it on your own dataset.

## Pick a pretrained model

A choice of model architectures pretrained on multiple dataset can be found [here](../README_MODELS.md).
Find the model you would like to based on it's input size and performance on various benchmarks.

## Finetune the model on my dataset

To retrain the model we edit the user_config.yaml and the stm32ai_main.py python script (both found in /src).
In this example, we retrain the mobilenet_v2 model with an input size of (416x416x3) pretrained on a large public dataset imagenet, with our data.
We used the [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset. 


## Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Training:
    - To simply train the model on my data and get as output the trained tensorflow model (.h5).
- Chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.h5 and .tflite)
- Chain_tqeb:
    - To train, quantize, evaluate and benchmark the quantized model in one go.

For any details regarding the parameters of the config file, you can look here:
- [Training documentation](../README_TRAINING.md)
- [Quantization documentation](../README_QUANTIZATION.md)
- [Benchmark documentation](../README_BENCHMARKING.md)
- [Evaluation documentation](../README_EVALUATION.md)


The most important parts here are to define:
- The operation mode to training
- The data paths
- Define which and how many classes you want to detect
- Choose a model, its pretrained weights and input size
- The other training parameters

```yaml
# user_config.yaml

general:
  project_name: segmentation
  saved_models_dir: saved_models
  gpu_memory_limit: 12
  global_seed: 127
  display_figures: False

operation_mode: training

model:
  model_type: deeplab
  model_name: st_deeplabv3_mnv2_a050_s16_asppv2
  input_shape: (416, 416, 3)
  # model_path: ./path/to/your_model.keras

dataset:
  dataset_name: person_coco_2017_pascal_voc_2012
  class_names: ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
                "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]
  # path for data to be use for the training of the model
  training_path: ./datasets/VOC2012_train_val/JPEGImages
  training_masks_path: ./datasets/VOC2012_train_val/SegmentationClassAug
  training_files_path: ./datasets/VOC2012_train_val/ImageSets/Segmentation/trainaug.txt
  # path for data to be use for the validation of the model
  validation_path: ./datasets/VOC2012_train_val/JPEGImages
  validation_masks_path: ./datasets/VOC2012_train_val/SegmentationClassAug
  validation_files_path: ./datasets/VOC2012_train_val/ImageSets/Segmentation/val.txt
  validation_split: 
  
# preprocessing to rescale and resize the data to the model input size define below
preprocessing:
  rescaling: {scale: 1/127.5, offset: -1}
  resizing:
    aspect_ratio: fit
    interpolation: bilinear 
  color_mode: rgb

# Optional
data_augmentation:   
  random_contrast:
    factor: 0.4
    change_rate: 1.0
  random_posterize:
    bits: (4, 8)
    change_rate: 0.025
  random_brightness:
    factor: 0.05
    change_rate: 1.0

training:
  dropout: 0.6
  batch_size: 16
  epochs: 300
  optimizer:
    Adam:
      learning_rate: 0.005
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

For the Chain_tqe and Chain_tqeb operation modes, you need to edit the config file to add parts related to the quantization and benchmark.
Look at the documentation linked above for more details.

You can also find examples of user_config.yaml for any operation mode [here](../../config_file_examples)

## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```