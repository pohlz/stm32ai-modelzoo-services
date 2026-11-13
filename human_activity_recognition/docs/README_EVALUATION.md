# Evaluation of Human Activity Recognition (HAR) model

This document provides details on how a pretrained HAR model can be evaluated. The evaluation service is a comprehensive tool that enables users to assess the accuracy of their Keras (.keras or .h5) HAR models. The inputs are the pretrained model and the dataset, letting the users quickly and easily evaluate the performance of their model and generate various metrics, such as accuracy and confusion matrix.

The details on how to use this service are provided in the document below.

<details open><summary><a href="#1"><b>1. Configure the yaml file</b></a></summary><a id="1"></a>

To evaluate a pretrained HAR model using the evaluation service, users can edit the parameters provided in the main [user_config.yaml](../user_config.yaml) file, or alternatively directly update a few parameters in the minimalistic configuration file provided for the evaluation service [evaluation_config.yaml](../config_file_examples/evaluation_config.yaml).

To edit the main [user_config.yaml](../user_config.yaml) file, follow the steps below, which show how to evaluate your pretrained HAR model trained on mobility_v1 or WISDM datasets.

<ul><details open><summary><a href="#1-1">1.1 Setting the model and the operation mode</a></summary><a id="1-1"></a>

The first thing to set is the `operation_mode` to use the evaluation service.

```yaml
general:
  project_name: human_activity_recognition # optional, if not provided default name is used for the experiment
operation_mode: evaluation # mandatory
```
</details></ul>
<ul><details open><summary><a href="#1-2">Model path</a></summary><a id="1-2"></a>
In this example, we use an st_ign model trained on the WISDM dataset. To do this set the model_path parameter in `model.model_path`.

```yaml
model:
  model_path: ../../stm32ai-modelzoo/human_activity_recognition/st_ign/ST_pretrainedmodel_public_dataset/WISDM/st_ign_wl_24/st_ign_wl_24.keras # mandatory
```
</details></ul>
<ul><details open><summary><a href="#1-3">1.3 Prepare the dataset</a></summary><a id="1-3"></a>

Next, users need to provide the information about the dataset to be used for the evaluation. All the information regarding this is provided in the `dataset` section of the yaml file.

```yaml
dataset:
  dataset_name: wisdm  # wisdm or mobility_v1
  class_names: [Jogging,Stationary,Stairs,Walking]  # [Stationary,Walking,Jogging,Biking] for mobility_v1
  training_path: ./datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt # need paths to train.pkl for mobility_v1
  validation_split: 0.2
  test_path: # need paths to test.pkl for mobility_v1
  test_split: 0.25
```
When evaluating the pretrained model on the WISDM dataset, the dataset path is provided in the `dataset.training_path` parameter and the `dataset.validation_split` and `dataset.test_split` are used to create the test and validation splits. By default, the values for the `validation_split` and `test_split` are set to 0.2 and 0.25 respectively. First, the dataset is split into test and train, and then the train set is further split into the train and validation splits.

In the case of the `mobility_v1` dataset, the `test.pkl` portion is used for the evaluation. 

</details></ul>
<ul><details open><summary><a href="#1-4">1.4 Apply preprocessing</a></summary><a id="1-4"></a>

The frames from the dataset need to be preprocessed before they are presented to the network for evaluation.

This is illustrated in the YAML code below:

```yaml
preprocessing: # mandatory
  gravity_rot_sup: true # mandatory
  normalization: false # mandatory
```

- `gravity_rot_sup` - *boolean*, the flag to control the application of the gravity rotation and then suppression by applying a high pass filter.
- `normalization` - *boolean*, the flag to enable the standard normalization of the frame (x_i - mu)/std 

The window length for the frame creation is inferred from the input model input_shape.
</details></ul>
</details>
<details open><summary><a href="#2"><b>2. Evaluate your model</b></a></summary><a id="2"></a>

If you chose to modify the [user_config.yaml](../user_config.yaml), you can evaluate the model by running the following command from the UC folder:

```bash
python stm32ai_main.py 
```
If you chose to update the [evaluation_config.yaml](../config_file_examples/evaluation_config.yaml) and use it, then run the following command from the UC folder: 

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name evaluation_config.yaml
```

</details>
<details open><summary><a href="#3"><b>3. Visualize the evaluation results</b></a></summary><a id="3"></a>

Once the dataset is prepared and the model is evaluated, the accuracies are printed in the terminal and the confusion matrix is displayed. However, after you have closed the terminal and the confusion matrix image, you can still retrieve the confusion matrix generated after evaluating the model on the test dataset by navigating to the appropriate directory within **./tf/src/experiments_outputs/\<date-and-time\>**.

<div style="text-align:center;">
  <img src="./img/wisdm_ign_wl_24_confusion_matrix.png"
       alt="plot"
       style="width:60%; max-width:60%;">
</div>

You can also find the evaluation results saved in the log file **stm32ai_main.log** under **./tf/src/experiments_outputs/\<date-and-time\>**.

Finally, you can also visualize the results by using MLflow by issuing the command `mlflow ui` from the folder **./tf/src/experiments_outputs/**. That will open MLflow in the browser.

</details>
