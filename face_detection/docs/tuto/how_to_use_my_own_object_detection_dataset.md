# How to use my own object detection dataset?

To use your own dataset with ST Model Zoo Face Detection scripts, you need to have a dataset in the YOLO Darknet format.

Some scripts are available to help you in this task:
- dataset_converter: This tool converts datasets in COCO or Pascal VOC format to YOLO Darknet format.
- dataset_analysis : This tools analyzes the distribution of the dataset (classes and labels), and should be used before creating the .tfs files
- dataset_create_tfs : this tools creates .tfs files from the dataset used in order to have a faster training. It is needed to generate the .tfs before running the training through the training operation mode.

These tools work the same way as the rest of the ST Model zoo, you edit a yaml configuration file and run a python script. Find the documentation of these tools here:
- [dataset_converter](../README_DATASETS_CONVERTER.md)
- [dataset_analysis](../README_DATASETS_ANALYSIS.md)
- [dataset_create_tfs](../README_DATASETS_CREATE_TFS.md)

## Example:

### Convert the dataset

Let's say, you have a COCO format dataset, you first need to use the dataset_converter in the /datasets folder to convert it to the YOLO Darknet format.
To use it, you need to edge the dataset_config.yaml:

```yaml
dataset:
  format: coco_format
  class_names: [<your-classes>]

coco_format:
  images_path: <path-to-images>
  json_annotations_file_path: <path-to-json-annotations>
  export_dir: <path-to-export dir>
```

Then run the python script to convert your dataset to the YOLO Darknet format:
```yaml
python converter.py dataset_config.yaml
```

If you have multiple folder of data (train, valid and test folders for example), you need to edit the yaml and run the python script multiple times.


Once you have your ouput directories (in YOLO Darknet Format), you are ready to use the main ST Model Zoo script. Note that you can use the two other tools and especially the tls file creator to enable the training scripts.

### Use your dataset with the user_config.yaml

Now that you have a dataset in YOLO Darknet format, you can use the paths to your dataset in the user_config.yaml and start working with ST Model zoo. 
The important part here is the dataset :


```yaml
# user_config.yaml 

# ...

# part of the configuration file related to the dataset
dataset:
  dataset_name: <name-of-your-dataset>                                        # Dataset name. Defaults to "<unnamed>".
  class_names: [<your-classes]                                                # Names of the classes in the dataset.
  training_path: <exported-yolo-darknet-training-set-directory>               # Path to the root directory of the training set.
  validation_path: <exported-yolo-darknet-validation-set-directory>           # Path to the root directory of the validation set.
  validation_split: 0.2                                                       # Training/validation sets split ratio.
  test_path: <exported-yolo-darknet-test-set-directory>                       # Path to the root directory of the test set.
  quantization_path: <exported-yolo-darknet-quantization-set-directory>       # Path to the root directory of the quantization set.
  quantization_split:                                                         # Quantization split ratio.
   
# ...
```

Then edit the rest of the user_config in function of your needs as usual.


