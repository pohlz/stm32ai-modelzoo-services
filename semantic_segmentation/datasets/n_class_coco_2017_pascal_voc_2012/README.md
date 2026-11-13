# n CLASS COCO 2017 PASCAL VOC 2012 Dataset Creation Script

This repository provides a Python script and a YAML configuration file designed to create subsets of the COCO PASCAL VOC 21-class dataset with user-specified classes. The script processes both training and validation sets without leakage, ensuring data integrity. Additionally, it analyzes the dataset and prints relevant statistics.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Folder Structure](#folder-structure)

## Installation

To create variants from the COCO 2017 PASCAL VOC 2012 dataset, you must first create it by following the [tutorial](../coco_2017_pascal_voc_2012/README.md). This will provide you with the necessary images, masks, and ID files

## Configuration

Example `dataset_config.yaml`:

```yaml
image_dir: "../COCO2017_VOC2012/JPEGImages"
mask_dir: "../COCO2017_VOC2012/SegmentationClassAug"
train_ids_file: "../COCO2017_VOC2012/trainaug.txt"
val_ids_file: "../COCO2017_VOC2012/val.txt"
output_base_dir: "../"
dataset_name: "indoor_COCO2017_VOC2012"  # Name the dataset and the folder where the dataset will be saved
interesting_classes: ["background", "bottle", "chair", "dining table", "person", "tv/monitor"]  # List the classes that interest you
pixel_threshold: 800  # Number of pixels threshold for interesting classes
```

In this YAML example, we create a variant of COCO 2017 PASCAL VOC 2012 with the indoor classes. The image_dir is the path to the directory containing the images, mask_dir is the path to the directory containing the original masks, train_ids_file is the path to the file containing the list of training image IDs, and val_ids_file is the path to the file containing the list of validation image IDs.

The output_base_dir is the path to the base directory where the output will be saved, and interesting_classes is a list of classes you are interested in. Modify this list as needed. Finally, the pixel_threshold is the number of pixels threshold for interesting classes. Images where none of the interesting classes (excluding "background") exceed this number of pixels will be filtered out.

**Note:** You can customize the dataset_config.yaml file to create other datasets, such as a person dataset, an outdoor dataset, or any custom dataset you want to create by specifying different classes in the interesting_classes list.

## Usage

Ensure the [dataset_config.yaml](./dataset_config.yaml) file is in the same directory as the script.

Run the script using the following command:

```bash
python dataset_creation.py --config-path ./dataset_config.yaml

```
or: 

```bash
python dataset_creation.py 
```

## Folder structure 

```
indoor_COCO2017_VOC2012/
├── JPEGImages/
├── SegmentationClassAug/
├── trainaug.txt
└── val.txt
```