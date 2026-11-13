# Pascal VOC 2012 Segmentation Augmentation Guide

This guide provides instructions on how to augment the Pascal VOC 2012 dataset for segmentation tasks by adding additional masks. These masks include both the original masks and new masks for images that did not have masks in the original dataset. The masks are provided as integer masks with indices of the classes. This augmentation helps to improve the performance of segmentation models when the original dataset is limited.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Augmentation](#dataset-augmentation)
- [Usage](#usage)

## Introduction

The Pascal VOC 2012 dataset is widely used for segmentation tasks. However, the dataset may not always have enough labeled data for training robust models. To address this, additional masks for images that do not have masks in the original dataset can be used.

## Dataset Augmentation

The original Pascal VOC 2012 dataset contains:
- **1464 training images and masks**
- **1449 validation images and masks**

The augmented dataset increases the number of images and masks to a total of 10583.

The additional masks can be downloaded from the following Dropbox link:

[Download Additional Masks](https://www.dropbox.com/scl/fi/xccys1fus0utdioi7nj4d/SegmentationClassAug.zip?rlkey=0wl8iz6sc40b3qf6nidun4rez&e=3&dl=0)

These masks are integer masks with indices representing different classes. They should be placed in the same directory structure as the original Pascal VOC 2012 dataset.

## Usage

1. **Download the Pascal VOC 2012 dataset**:
   - Follow the instructions on the [Pascal VOC 2012 website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) to download the dataset then place it under `datasets/VOC2012_train_val/`.

2. **Download the additional masks**:
   - Use the provided [Dropbox link](https://www.dropbox.com/path/to/masks) to download the additional masks and place them under  `datasets/VOC2012_train_val/` too.

3. **Run the script to create `trainaug.txt`**:
   - Use the provided Python script to generate `trainaug.txt`, which contains the IDs of the images with the additional masks, excluding the IDs present in `val.txt`.

```sh
python create_trainaug_txt.py --config-path dataset_config.yaml
```

At the end, the VOC2012_train_val has the structure below: 
```
datasets/
├── VOC2012_train_val/
│   ├── Annotations/
│   ├── calibration_JPEGImages/
│   ├── ImageSets/
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   ├── SegmentationClassAug/
│   ├── SegmentationObject/
│   └── devkit_doc.pdf
```