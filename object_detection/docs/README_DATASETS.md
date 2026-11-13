# <a>Object detection STM32 model zoo</a>

Before you start using this project, it's important to understand the supported dataset names and formats. Please note that for all the training, evaluation and quantization services, it is expected to have a dataset in TFS Tensorflow format. For the object detection use case, the `get_dataloaders` API call takes care of the conversion of your dataset automatically depending on the `dataset_name` and `format` attributes.

The `dataset` section and its attributes are shown in the YAML example code below.

```yaml
dataset:
  format: pascal_voc
  dataset_name: pascal_voc                                    # Dataset name. Defaults to "<unnamed>".
  class_names: [ aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor ]  # Names of the classes in the dataset.
  data_dir: <tmp-directory-where-dataset-will-be-downloaded> # Path to the tmp directory before the split.
  train_images_path: <JPEGImages-root-directory>             # Path to the root directory of the img before split.
  train_annotations_path: <path-to-the-Annotations-dir>      # Path to the root directory of the xml annotations
  val_images_path: <JPEGImages-root-directory>               # Path to the root directory of the img (usually the same as train_images_path).
  val_annotations_path: <path-to-the-Annotations-dir>        # Path to the root directory of the xml annotations (usually the same as train_annotations_path)
  test_images_path: <JPEGImages-root-directory>              # Path to the root directory of the img (usually the same as train_images_path).
  test_annotations_path: <path-to-the-Annotations-dir>       # Path to the root directory of the xml annotations (usually the same as train_annotations_path)
  train_split:  <path-to-the-ImageSets/Main/train.txt>       # Path to the .txt file containing the split for training.
  val_split:  <path-to-the-ImageSets/Main/val.txt>           # Path to the .txt file containing the split for training. (Optional)
  test_split:  <path-to-the-ImageSets/Main/test.txt>         # Path to the .txt file containing the split for training. (Optional)
  # validation_split: 0.2                                    # Training/validation sets split ratio. (Optional if val_*_path is defined)
  quantization_path: <quantization-set-root-directory>       # Path to the root directory of the quantization set.
  quantization_split:                                        # Quantization split ratio.
  seed: 123                                                  # Random generator seed used when splitting a dataset.
```

The `dataset_name` attribute is required and serves to specify the dataset you are using. This can be a well-known dataset like coco, pascal_voc, or a custom_dataset if you have your own data and it follows the logic below:

| Dataset Name     | Allowed Formats          | Description                                                                                  |
|------------------|-------------------------|----------------------------------------------------------------------------------------------|
| `coco`           | `coco`, `tfs`           | Native COCO format or TFS TensorFlow format                                                     |
| `pascal_voc`     | `pascal_voc`, `tfs`     | Native Pascal VOC format or TFS TensorFlow format                                               |
| `darknet_yolo`   | `darknet_yolo`, `tfs`   | Native Darknet YOLO format or TFS TensorFlow format                                             |
| `custom_dataset` | `tfs`                   | Only TFS TensorFlow format; in case the dataset is already converted before evaluation                          |

Depending on the `dataset_name`, the dataset loader will check the `format` to determine if it is necessary to convert the dataset to the final **TFS TensorFlow format**. These two parameters are mandatory if the operation mode is **training**, **evaluation** and **quantization**.

The `format` attributes defines the annotation format of your dataset. This must match the format of your dataset annotations. 
It serves to check whether your dataset is in its original format or in TFS TensorFlow format. 
This determines whether it is needed to convert the dataset to the required TFS format or not. It accepts the following values: 

  * `tfs`: If the dataset is a TensorFlow Object Detection API format.
  * `coco`: If the dataset is in COCO dataset format (JSON annotations).
  * `pascal_voc`: If the dataset is in Pascal VOC XML annotation format.
  * `darknet_yolo`: If the dataset is in YOLO Darknet text file annotations.

Depending on the `format` value, some additional attributes should be defined in the dataset section:
- If the `format` is set to **coco**, the following attributes should be set:
  * The `data_dir`: Required if `download_data` is set to True, refers to the temporary path where the dataset will be downloaded.
  * The `train_images_path`: Required, refers to the path of the training subset directory where the images are located.
  * The `train_annotations_path`: Required, refers to the path of the training subset JSON file of the annotations.
  * The `val_images_path`: Optional, refers to the path of the validation subset directory where the images are located.
  * The `val_annotations_path`: Optional, refers to the path of the validation subset JSON file of the annotations.
  * The `test_images_path`: Optional, refers to the path of the test subset (evaluation-only) directory where the images are located.
  * The `test_annotations_path`: Optional, refers to the path of the test subset (evalutation-only) JSON file of the annotations.
  * The `validation_split`: Optional, refers to a % ratio of the training set, should be defined if the `val_images_path` and `val_annotations_path` are missing.

- If the `format` is set to **pascal_voc**, the following attributes should be set:
  * The `data_dir`: Required if `download_data` is set to True, refers to the temporary path where the dataset will be downloaded.
  * The `train_images_path`: Required, refers to the path of the training subset directory where the images are located.
  * The `train_annotations_path`: Required, refers to the path of the training subset directory containing the xml files of the annotations.
  * The `train_split`: Required, refers to the train.txt containing the split IDs for training subsets.
  * The `val_images_path`: Optional, refers to the path of the validation subset directory where the images are located.
  * The `val_annotations_path`: Optional, refers to the path of the training subset directory containing the XML files of the annotations.
  * The `val_split`: Required, refers to the val.txt containing the split IDs for validation subsets.
  * The `test_images_path`: Optional, refers to the path of the validation subset directory where the images are located.
  * The `test_annotations_path`: Optional, refers to the path of the training subset directory containing the XML files of the annotations.
  * The `test_split`: Required, refers to the test.txt containing the split IDs for training subsets.
  * The `validation_split`: Optional, refers to a % ratio of the training set, should be defined if the `val_images_path` and `val_annotations_path` are missing.

- If the `format` is set to **darknet_yolo**, the following attributes should be set:
  * The `train_images_path`: Required, refers to the path of the training subset directory where the images are located.
  * The `train_annotations_path`: Required, refers to the path of the training subset directory containing the xml files of the annotations.
  * The `val_images_path`: Optional, refers to the path of the validation subset directory where the images are located.
  * The `val_annotations_path`: Optional, refers to the path of the validation subset directory containing the TXT files of the annotations.
  * The `test_images_path`: Optional, refers to the path of the test subset (evaluation-only) directory where the images are located.
  * The `test_annotations_path`: Optional, refers to the path of the test subset (evaluation-only) directory containing the TXT files of the annotations. 
  * The `validation_split`: Optional, refers to a % ratio of the training set, should be defined if the `val_images_path` and `val_annotations_path` are missing.


The state machine below describes the process of dataset loading for object detection use case.


```
                                                   dataset_name
                                                         |
                                                         |
        +----------------------------------+--------------------------+-------------------------------+
        |                                  |                          |                               |
        |                                  |                          |                               |
      coco                           pascal_voc              darknet_yolo                "custom_dataset"
        |                                  |                          |                               |
        |                                  |                          |                               |
  +-----+------------+           +-----+-----------+          +-------+-------+               +-------+-------+ 
  |                  |           |                 |          |               |               |               |
supported        unsupported    supported    unsupported   supported     unsupported      supported      unsupported        
 format:           format        format         format      format:        format           format         format
      |                             |                           |                             |
  +---+-----+                   +---+---+                  +----+-----+                       |
  |         |                   |       |                  |          |                       |
 coco      tfs             pascal_voc  tfs            darknet_yolo   tfs                     tfs
  |         |                   |       |                  |          |                (Custom dataset
  |         |                   |       |                  |          |                 should be used
  |         |                   |       |                  |          |                if the conversion
  |   dataset.format=tfs        |  dataset.format=tfs      |    dataset.format=tfs     has already been
  |   (already TFS)             |    (already TFS)         |      (already TFS)        done in a previous
  |         |                   |       |                  |          |                training or eval)
  |         |                   |       |                  |          |                       |
  |   load TFS directly         |   load TFS directly      |      load TFS directly      load TFS directly
  |                             |                          |                                  |
  |                             |                          |                                  |
dataset.format=coco     dataset.format=pascal_voc      dataset.format=darknet_yolo            |
(needs conversion)         (needs conversion)             (needs conversion)                  |
        |                         |                               |                           |
        v                         v                               v                           |
convert coco to tfs      convert pascal_voc to tfs     convert darknet yolo to tfs            |
        |                         |                               |                           |
        +-------------------------+-------------------------------+---------------------------+
                                                |
                                        Dataset in TFS format
                                            (used for)
                          +---------------------+-----------------------+
                          |                     |                       |
                      training             evaluation             quantization

```

## Dataset Configuration

### Details of Required / Optional Attributes per `(dataset_name, format)`

---

### 1. `dataset_name = coco`

**Supported `format` values:**

- `tfs`
- `coco`

#### 1.a `format = tfs`

- Dataset is already in **TFS TensorFlow** format.
- Loader reads TFS files directly.

**Required attributes**

- `train_images_path`  
  → Path to training images directory.
- `train_annotations_path`  
  → Path to the training TFS files directory.

**Optional attributes**

- `val_images_path`  
  → Path to validation images directory.
- `val_annotations_path`  
  → Path to the validation TFS files directory.
- `test_images_path`  
  → Path to test images directory.
- `test_annotations_path`  
  → Path to the test TFS files directory.

---

#### 1.b `format = coco`

- Dataset is in **COCO JSON** annotation format and must be converted to TFS.

**Required attributes**

- `data_dir`  
  → Temporary path where the dataset will be downloaded if `download_data` is set to true.
- `train_images_path`  
  → Path to training images directory.
- `train_annotations_path`  
  → Path to training subset COCO JSON annotations file.

**Optional attributes**

- `val_images_path`  
  → Path to validation images directory.
- `val_annotations_path`  
  → Path to validation subset COCO JSON annotations file.
- `test_images_path`  
  → Path to test images directory.
- `test_annotations_path`  
  → Path to test subset COCO JSON annotations file (usually same as validation).

**Conversion flow**

1. Read images/annotations from `train_*` (and optionally `val_*`).
2. Generate TFS TensorFlow records into `labels_tfs` directory containing the corresponding splits (train, val, test).
3. Load resulting TFS dataset for training / evaluation / quantization with the specified split ratios or the defined paths.

---

### 2. `dataset_name = pascal_voc`

**Supported `format` values:**

- `tfs`
- `pascal_voc`

#### 2.a `format = tfs`

- Dataset is already in **TFS TensorFlow** format.
- Loader reads TFS files directly.

**Required attributes**

- `train_images_path`  
  → Path to training images directory.
- `train_annotations_path`  
  → Path to the training TFS files directory.

**Optional attributes**

- `val_images_path`  
  → Path to validation images directory.
- `val_annotations_path`  
  → Path to the validation TFS files directory.
- `test_images_path`  
  → Path to test images directory.
- `test_annotations_path`  
  → Path to the test TFS files directory.

---

#### 2.b `format = pascal_voc`

- Dataset is in **Pascal VOC XML** annotation format and must be converted.

**Required attributes**

- `data_dir`  
  → Temporary path where the dataset files will be downloaded if `download_data` is set to true.
- `train_images_path`  
  → Path to training images directory.
- `train_annotations_path`  
  → Path to directory containing training XML annotation files.
- `train_split`
  → Path to the val.txt file containing the training split IDs.

**Optional attributes**

- `val_images_path`  
  → Path to validation images directory.
- `val_annotations_path`  
  → Path to directory containing validation XML annotation files.
- `val_split`
  → Path to the val.txt file containing the validation split IDs.
- `test_images_path`  
  → Path to test images directory.
- `test_annotations_path`  
  → Path to directory containing test XML annotation files.
- `train_split`
  → Path to the test.txt file containing the test split IDs.

**Conversion flow**

1. Read images/annotations from `train_*` (and optionally `val_*`).
2. Generate TFS TensorFlow records into `labels_tfs` directory containing the corresponding splits (train, val, test).
3. Load resulting TFS dataset for training / evaluation / quantization with the specified split ratios or the defined paths.

---

### 3. `dataset_name = darknet_yolo`

**Supported `format` values:**

- `tfs`
- `darknet_yolo`

#### 3.a `format = tfs`

- Dataset is already in **TFS TensorFlow** format.
- Loader reads TFS files directly.

**Required attributes**

- `train_images_path`  
  → Path to training images directory.
- `train_annotations_path`  
  → Path to the training TFS files directory.

**Optional attributes**

- `val_images_path`  
  → Path to validation images directory.
- `val_annotations_path`  
  → Path to the validation TFS files directory.
- `test_images_path`  
  → Path to test images directory.
- `test_annotations_path`  
  → Path to the test TFS files directory.

---

#### 3.b `format = darknet_yolo`

- Dataset is in **YOLO Darknet text** annotation format and must be converted.

**Required attributes**

- `train_images_path`  
  → Path to training images directory.
- `train_annotations_path`
  → Path to the directory containing the TXT files for training subset.

**Optional attributes**

- `val_images_path`  
  → Path to validation images directory.
- `val_annotations_path`  
  → Path to the directory containing the TXT files for training subset.
- `test_images_path`  
  → Path to test images directory.
- `test_annotations_path`  
  → Path to the directory containing the TXT files for training subset.

**Conversion flow**

1. Parse YOLO `.txt` annotations and corresponding images in `train_images_path` and `train_annotations_path`.
2. Generate TFS TensorFlow records into `labels_tfs` directory containing the corresponding splits (train, val, test).
3. Load resulting TFS dataset for training / evaluation / quantization with the specified split ratios or the defined paths.

---

### 4. `dataset_name = "custom_dataset"`

**Supported `format` values:**

- `tfs`

This case assumes:

- The user has already produced a **TFS TensorFlow dataset** externally or from a previous operation.
- The loader only reads the TFS dataset (no conversion is performed).

**Required attributes**

- `train_images_path`  
  → Path to training images directory.
- `train_annotations_path`  
  → Path to the training TFS files directory.

**Optional attributes**

- `val_images_path`  
  → Path to validation images directory.
- `val_annotations_path`  
  → Path to the validation TFS files directory.
- `test_images_path`  
  → Path to test images directory.
- `test_annotations_path`  
  → Path to the test TFS files directory.

---

### Operation Modes and Mandatory Parameters

For the following operation modes:

- `training`
- `evaluation`
- `quantization`

The following parameters are **mandatory**:

- `dataset_name`
- `format`