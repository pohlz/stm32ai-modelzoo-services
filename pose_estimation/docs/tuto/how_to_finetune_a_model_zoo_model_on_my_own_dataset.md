# How can I finetune a ST Model Zoo model on my own dataset?

With ST Model Zoo, you can easily pick an already pretrained available model and finetune them on your own dataset.

## Pick a pretrained model

A choice of model architectures pretrained on multiple dataset can be found [here](../README_MODELS.md).

## Operation modes:

Depending on what you want to do, you can use the operation modes below:
- training:
    - To simply train the model on my data and get as output the trained tensorflow model (.keras).
- chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.keras and .tflite)
- chain_tqeb:
    - To train, quantize, evaluate and benchmark the quantized model in one go.

For any details regarding the parameters of the config file, you can look here:
- [Training documentation](../README_TRAINING.md)
- [Quantization documentation](../README_QUANTIZATION.md)
- [Benchmark documentation](../README_BENCHMARKING.md)
- [Evaluation documentation](../README_EVALUATION.md)


## Finetune the model on my dataset

To retrain the model we edit the user_config.yaml and the stm32ai_main.py python script (both found in the UC folder).

In this example, we retrain our ST MoveNet Lightning heatmap model with an input size of (192x192x3) pretrained on a large public dataset imagenet, with our data.


Here, we used the COCO2017 pose estimation dataset. 

You can use any dataset of the YOLO Darknet format. You can take a look at this [tutorial](./how_to_use_my_own_dataset.md) which explain how to convert a COCO dataset using our script.

The most important parts here are to define:
- The operation mode to training
- The data paths
- Define the number of keypoints of the pose
- Choose a model, its pretrained weights and input size
- The other training parameters

> [!NOTE]
> If your number of `keypoints` is different from `17`, the code will give equal an weight to each keypoint in the [OKS metric](../../tf/src/evaluation/metrics.py#L153).

> [!IMPORTANT]  
> If you want to have the connections between the keypoints for your dataset check out the [dictionnary](../../tf/src/utils/connections.py) to add your own connections as well as their colors.

```yaml
# user_config.yaml

general:
  project_name: Custom_dataset_training
  logs_dir: logs
  saved_models_dir: saved_models
  num_threads_tflite: 8
  gpu_memory_limit: 8
  global_seed: 123

operation_mode: training

model:
   model_path: ../../stm32ai-modelzoo/pose_estimation/movenet/ST_pretrainedmodel_custom_dataset/custom_coco_person_17kpts/st_movenet_lightning_a100_heatmaps_192/st_movenet_lightning_a100_heatmaps_192_int8.tflite
   model_type: heatmaps_spe

dataset:
  dataset_name: coco
  keypoints: 17 # Put here your dataset number of keypoints
  class_names: [person]
  training_path: ./datasets/coco_train_single_pose
  validation_split: 0.1
  test_path: ./datasets/coco_val_single_pose
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
      learning_rate: 5.0e-4
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