# How to use my own dataset?

To use your own dataset with ST Model Zoo pose estimation scripts, you need to have a dataset in the YOLO Darknet format.

dataset_converter is a tool that converts datasets from COCO format to YOLO Darknet format. It works the same way as the rest of the ST Model zoo. 
You edit a yaml configuration file and run a python script. Find the documentation of these tools here:
- [dataset_converter](../README_DATASETS.md)

## Example:

### Convert the dataset

Let's say the the COCO dataset un raw format is in PATH_TO_COCO.

To convert this dataset to Yolo Darknet TXT format for single pose estimation, run the script with the following command:


```powershell
python converter.py --path PATH_TO_COCO --pose single --keypoints 13 --outputdir ./pose_estimation
```

For multiple pose estimation, run the script with the following command:

```powershell
python converter.py --path PATH_TO_COCO --pose multi --keypoints 17 --outputdir ./pose_estimation
```
The converted datasets will be saved in the ./pose_estimation/single/13kpts/train|val and ./pose_estimation/multi/17kpts/train|val directories.


### Use your dataset with the user_config.yaml

Now that you have a dataset in YOLO Darknet format, you can use the paths to your dataset in the user_config.yaml and start working with ST Model zoo. 
The import part here is the dataset part:

> [!NOTE]
> If your number of `keypoints` is different from `17`, the code will give equal an weight to each keypoint in the [OKS metric](../../tf/src/evaluation/metrics.py#L153).

> [!IMPORTANT]  
> If you want to have the connections between the keypoints for your dataset check out the [dictionnary](../../tf/src/utils/connections.py) to add your own connections as well as their colors.

```yaml
# user_config.yaml 

# ...

# part of the configuration file related to the dataset
dataset:
  dataset_name: coco                                               # Dataset name
  keypoints: <your-nb-of-keypoints>                                # Number of keypoints
  class_names: [<your-class>]                                      # Name of the class
  training_path: <training-set-directory>                          # Path to the root directory of the training set.
  validation_path: <validation-set-directory>                      # Path to the root directory of the validation set.
  validation_split: 0.1                                            # Training/validation sets split ratio.
  test_path: <test-set-directory>                                  # Path to the root directory of the test set.
  quantization_path: <quantization-set-directory>                  # Path to the root directory of the quantization set.
  quantization_split: 0.3                                          # Quantization split ratio.

# ...
```

Then edit the rest of the user_config in function of your needs as usual.


