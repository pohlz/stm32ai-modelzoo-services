# COCO 2017 PASCAL VOC 2012 Dataset Creation Script

This script filters the COCO 2017 dataset to include only images that contain one of the 21 PASCAL VOC classes. It then processes the filtered dataset to make it compatible with the PASCAL VOC 2012 format. Finally, it creates a combined dataset that contains images from both datasets, masks, `trainaug.txt`, and `val.txt`.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Folder structure](#folder-structure)

## Installation

1. **Download the datasets**:
    - Download the COCO 2017 dataset from [COCO Dataset](https://cocodataset.org/#download).
    - Download the PASCAL VOC 2012 dataset and the augmented masks by following the [tutorial](../pascal_voc_2012/README.md).

2. **Organize the datasets**:
    - Place the COCO 2017 dataset under `datasets/COCO2017/`.

```
datasets/
├── COCO2017/
│   ├── annotations/
│   │   └── instances_train2017.json
│   └── train2017/
├── VOC2012_train_val/
│   ├── Annotations/
│   ├── calibration_JPEGImages/
│   ├── ImageSets/
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   ├── SegmentationClassAug/
│   ├── SegmentationObject/
│   └── devkit_doc.pdf
└── coco_2017_pascal_voc_2012/
    ├── dataset_creation.py
    ├── dataset_config.yaml
    └── README.md
```

3. **Install the required libraries**:
    ```bash
    pip install pyyaml tqdm pillow pycocotools
    ```

## Configuration

Update the `dataset_config.yaml` file with the correct paths to the COCO 2017 annotations, COCO 2017 images, PASCAL VOC 2012 images, PASCAL VOC 2012 masks, and output directories for final masks and images.

Example `dataset_config.yaml`:

```yaml
coco:
  annotations_path: '../COCO2017/annotations/instances_train2017.json'
  images_path: '../COCO2017/train2017/'

pascal_voc:
  images_path: '../VOC2012_train_val/JPEGImages/'
  masks_path: '../VOC2012_train_val/SegmentationClassAug/'
  txt_training_file: '../VOC2012_train_val/ImageSets/Segmentation/trainaug.txt'
  txt_val_file: '../VOC2012_train_val/ImageSets/Segmentation/val.txt'

output:
  final_images_path: '../COCO2017_VOC2012/JPEGImages'
  final_masks_path: '../COCO2017_VOC2012/SegmentationClassAug'
  final_txt_training_file: '../COCO2017_VOC2012/trainaug.txt'
  finat_txt_val_file: '../COCO2017_VOC2012/val.txt'

smallest_annotation_area: 1000

voc_categories:
  - airplane
  - bicycle
  - bird
  - boat
  - bottle
  - bus
  - car
  - cat
  - chair
  - cow
  - dining table
  - dog
  - horse
  - motorcycle
  - person
  - potted plant
  - sheep
  - couch
  - train
  - tv
```

The `dataset_config.yaml` file contains the following sections:

- **coco**:
  - `annotations_path`: Path to the COCO 2017 annotations file (e.g., `instances_train2017.json`).
  - `images_path`: Path to the COCO 2017 images directory.

- **pascal_voc**:
  - `images_path`: Path to the PASCAL VOC 2012 images directory.
  - `masks_path`: Path to the PASCAL VOC 2012 masks directory.
  - `txt_val_file`: Path to the validation file containing IDs of validation images.

- **output**:
  - `final_images_path`: Path to the directory where the final combined images will be saved.
  - `final_masks_path`: Path to the directory where the final combined masks will be saved.
  - `final_txt_training_file`: Path to the file where the training IDs will be saved (e.g., `trainaug.txt`).
  - `final_txt_val_file`: Path to the file where the validation IDs will be saved (e.g., `val.txt`).

- **smallest_annotation_area**: Minimum area for valid annotations in the COCO dataset.

- **voc_categories**: List of category names to be included from the COCO dataset.

## Usage

Run the script using the following command:

```bash
python dataset_creation.py --config-path dataset_config.yaml
```

or: 

```bash
python dataset_creation.py 
```

## Folder Structure

Your output folder `dataset` should have the following structure:

```
COCO2017_VOC2012/
├── JPEGImages/
│   └── <Combined COCO 2017 and PASCAL VOC 2012 images>
├── SegmentationClassAug/
│   └── <Combined COCO 2017 and PASCAL VOC 2012 masks>
├── trainaug.txt
└── val.txt
```
