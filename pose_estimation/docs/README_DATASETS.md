# <a>Pose estimation STM32 model zoo</a>

## <a>Directory components</a>

`datasets` folder is a placeholder for pose estimation datasets.
This also includes some useful tools to process those datasets:
- [`dataset_converter`](./README_DATASETS_CONVERTER.md) : This tool converts datasets from COCO format to YOLO Darknet format. YOLO Darknet is the format used in the other tools below as well as in the pose estimation model zoo services.

To get started, update  with the dataset path you want to use.

> [!IMPORTANT]
> In your yaml file, under `dataset`, the `dataset_name` should always be set to 'coco' even if you dont use the COCO dataset