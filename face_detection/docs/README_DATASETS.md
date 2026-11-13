# <a>Face Detection STM32 model zoo</a>

## <a>Directory components</a>

`datasets` folder is a placeholder for Face Detection datasets.
This also includes some useful tools to process those datasets:
- [`dataset_converter`](./README_DATASETS_CONVERTER.md) : This tool converts datasets in COCO or Pascal VOC format to YOLO Darknet format. YOLO Darknet is the format used in the other tools below as well as in the object detection model zoo services.
- [`dataset_analysis`](./README_DATASETS_ANALYSIS.md) : This tools analyzes the distribution of the dataset (classes and labels), and should be used before creating the .tfs files.
- [`dataset_create_tfs`](./README_DATASETS_CREATE_TFS.md) : this tools creates .tfs files from the dataset used in order to have a faster training. It is needed to generate the .tfs before running the training through the training operation mode.
