# Overview of pose estimation STM32 model zoo


The STM32 model zoo includes several models for object detection use case pre-trained on custom and public datasets.
Under each model directory, you can find the following model categories:

- `Public_pretrainedmodel_custom_dataset` contains public pose estimation models trained on custom datasets.
- `ST_pretrainedmodel_custom_dataset` contains different pose estimation models trained on ST custom datasets.
- `ST_pretrainedmodel_public_dataset` contains pose estimation models trained on public datasets.

**Feel free to explore the model zoo and get pre-trained models [here](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/master/pose_estimation/).**


You can get footprints and performance information for each model following links below:
- [MoveNet](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/movenet/README.md)
- [Hand landmarks](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/handlandmarks/README.md)
- [Head landmarks](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/headlandmarks/README.md)
- [YOLOv8n pose](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/yolov8n_pose/README.md)
- [YOLOv11n pose](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/yolov11n_pose/README.md)


To get started, update the [user_config.yaml](../user_config.yaml) file, which specifies the parameters and configuration options for the services you want to use. The  `model` section of this yaml specifically relates to the model definition. The `model_type` is **mandatory** and must correspond to your model. Some topologies are already registered and can be accessed via the `model_name` attribute.

### `model_name`

 The exhaustive list of possible `model_name` is provided hereafter:
- 'custom_models'
- 'st_movenet_lightning_a100_heatmaps'
- 'st_movenet_lightning_heatmaps'
    - alpha = `value between 0.35 & 1.4`, (*but only [0.35, 0.5, 0.75, 1.0 ,1.4] come with pretrained ImageNet weights*)

> [!IMPORTANT]
> `'st_movenet_lightning_heatmaps'` model has a mandatory `alpha` parameter that controls the width of the network. This is known as the width multiplier in the [MobileNetV2 paper](https://arxiv.org/abs/1801.04381). 
>- alpha < 1.0 : proportionally decreases the number of filters in each layer.
>- alpha > 1.0 : proportionally increases the number of filters in each layer.
>- alpha == 1  : default number of filters from the paper are used at each layer -> `equivalent to 'st_movenet_lightning_a100_heatmaps'`

### `model_type`

The `model_type` attribute specifies the type of the model architecture that you want to use, it represents a specific post-processing / output shape / use-case.

- `heatmaps_spe`: These are single pose estimation models that outputs heatmaps that we must 
post-process in order to get the keypoints positions and confidences.

- `spe`: These are single pose estimation models that output directly the keypoints positions and confidences.

- `yolo_mpe `: These are the YOLO (You Only Look Once) multiple pose estimation models from Ultralytics that outputs the same tensor as in object detection but with the addition of a set of keypoints for each bbox.

- `hand_spe`: These are single hand landmarks estimation models that outputs directly the keypoints positions and confidences of the hand pose.

- `head_spe`: These are single head landmarks estimation models that outputs directly the keypoints positions and confidences of the head pose.


### Summary
Models with their corresponding `model_name`, `model_type`, `keypoints`, `class_names` & supported services:

| model | `model_name` | `model_type` | `keypoints` | `class_names` | Training  | Evaluation  | Prediction | Deployment |
|:---------------|:---------------|:---------------|:---------------|:---------------|:---------------|:---------------|:---------------|:---------------|
| [st_movenet_lightning](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/movenet/ST_pretrainedmodel_custom_dataset) | `st_movenet_lightning_a100_heatmaps`, `st_movenet_lightning_heatmaps` | `heatmaps_spe` | `17` | `[person]` | ✅ | ✅ | ✅ | ✅ |
| Custom model| `custom_models` | `heatmaps_spe` | *customizable* | *customizable* | ✅ | ✅ | ✅ | ✅ |
| [movenet_lightning](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/movenet/Public_pretrainedmodel_custom_dataset/custom_dataset_person_17kpts) | *Not registered** | `spe` | `17` | `[person]` | ❌ | ✅ | ✅ | ✅ |
| [hand_landmarks](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/handlandmarks/Public_pretrainedmodel_custom_dataset/custom_dataset_hands_21kpts)| *Not registered** | `hand_spe` | `21` | `[hand]` | ❌ | ❌ | ✅ | ✅ |
| [head_landmarks](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/headlandmarks/Public_pretrainedmodel_custom_dataset)| *Not registered** | `head_spe` | `468` | `[head]` | ❌ | ❌ | ✅ | ✅ |
| [yolo_v8n](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/yolov8n_pose) | *Not registered** | `yolo_mpe` | `17` | `[person]` | ❌ | ✅ | ✅ | ✅ |
| [yolo_v11n](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/pose_estimation/yolov11n_pose) | *Not registered** | `yolo_mpe` | `17` | `[person]` | ❌ | ✅ | ✅ | ✅ |


\* Use the `model_path` instead