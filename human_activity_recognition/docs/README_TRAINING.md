# <a id="">Human Activity Recognition (HAR) STM32 model training</a>

This readme shows how to train an HAR model from scratch or use transfer learning for training a model using a public (WISDM) or custom (mobility_v1) dataset. The scripts have two different choices for the model architecture based on convolutional neural networks (CNN) namely:
- [Ignatov (st_ign)](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/human_activity_recognition/st_ign/README.md) 
- [Global Max Pooling (st_gmp)](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/human_activity_recognition/st_gmp/README.md)

NOTE: The prefix `st_` denotes that these are the modified architectures that we created from the original GMP and IGN.

In this article, details for training the `st_ign` model using [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) (a public dataset) are provided.

<details open><summary><a href="#1"><b>1. Prepare the dataset</b></a></summary><a id="1"></a>

The WISDM dataset was released by the [Wireless Sensor Data Mining (WISDM) Lab](http://www.cis.fordham.edu/wisdm/).

The data collected corresponds to the data used in the research [Activity Recognition using Cell Phone Accelerometers](http://www.cis.fordham.edu/wisdm/public_files/sensorKDD-2010.pdf), Proceedings of the Fourth International Workshop on Knowledge, Discovery from Sensor Data (at KDD-10), Washington DC. 

The dataset is collected in controlled laboratory conditions using a mobile phone accelerometer. The dataset can be downloaded as a zip file using this link [WISDM_ar_latest.tar.gz](https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz). For this tutorial, users are requested to download and extract the contents of the file. After expanding the contents of the file, the user must have these contents in it:
- readme.txt
- WISDM_ar_v1.1_raw.txt
- WISDM_ar_v1.1_raw_about.txt
- WISDM_ar_v1.1_transformed.arff
- WISDM_ar_v1.1_trans_about.txt

For the purpose of this tutorial, we only need the `WISDM_ar_v1.1_raw.txt` file. The path to this file is to be provided to the variable, `dataset.training_path` in the [user_config.yaml](../user_config.yaml) configuration file (the details will be provided in section 2). The file `WISDM_ar_v1.1_raw.txt` contains data like below:
```bash
33,Jogging,49105962326000,-0.6946377,12.680544,0.50395286;
33,Jogging,49106062271000,5.012288,11.264028,0.95342433;
33,Jogging,49106112167000,4.903325,10.882658,-0.08172209;
33,Jogging,49106222305000,-0.61291564,18.496431,3.0237172;
33,Jogging,49106332290000,-1.1849703,12.108489,7.205164;
33,Jogging,49106442306000,1.3756552,-2.4925237,-6.510526;
33,Jogging,49106542312000,-0.61291564,10.56939,5.706926;
33,Jogging,49106652389000,-0.50395286,13.947236,7.0553403;
33,Jogging,49106762313000,-8.430995,11.413852,5.134871;
33,Jogging,49106872299000,0.95342433,1.3756552,1.6480621;
33,Jogging,49106982315000,-8.19945,19.57244,2.7240696;
33,Jogging,49107092330000,1.4165162,5.7886477,2.982856;
33,Jogging,49107202316000,-1.879608,-2.982856,-0.29964766;
```
The data columns of the dataset contain data for: `['user_id', 'Activity_Label', 'time_stamp', 'acc_x', 'acc_y', 'acc_z']`. 
The 3-D accelerometer data is captured at a rate of 20 Hz or samples/sec. 

In the WISDM dataset, there are six different activities performed namely: [`Downstairs`, `Jogging`, `Sitting`, `Standing`, `Upstairs`, `Walking`]. The sample distribution for these activities performed is presented in the figure below.
![plot](./img/wisdm_dataset_distribution.png)

As it is apparent, the activities performed are heavily unbalanced in the dataset, that is why we created two super-classes in an attempt to balance the dataset and simplify the problem.
```tex
- Stairs = Upstairs + Downstairs
- Stationary = Sitting + Standing
```
Finally, the four activities are [`Jogging`, `Stationary`, `Stairs`, `Walking`].

We will use only the columns of `['user_id', 'Activity_Label', 'acc_x', 'acc_y', 'acc_z']`. The data is then framed to create frames of `(wl, 3, 1)` shape. Here, `wl` represents the window length. We provide pretrained models for two different choices of wl, [24, 48].

After the frames or segments of `(wl, 3, 1)` shape are created, the `test_split` and `validation_split` values are used to first split the frames into `Train|Test` distribution and then `Train` is furthermore split into `Train|Validation`. 

NOTE: **In the case of the WISDM dataset, the only way to provide the dataset path is to provide it in the `dataset.training_path` variable, in all the operation modes.**

In the case of the custom dataset `mobility_v1` from STMicroelectronics, the data is already split into train and test distributions. A small subset of the data is provided in [human_activity_recognition/datasets/mobility_v1](../datasets/mobility_v1/). The data is in the form of `.pkl` files containing the list of the 3D accelerometer recordings for five different activities. 

> [!NOTE]  
> `mobility_v1` dataset is a proprietary dataset, and the subset provided here is just for demonstration purposes. The trained model with this small dataset might show bad results on the test dataset as it is only for testing that the scripts work.

**For the `mobility_v1` dataset, the paths to the train and test pkl files are to be provided in `dataset.training_path`, `dataset.test_path`. The script handles the windowing of the accelerometer data and splitting training data into train and validation.**

> [!WARNING]  
> **Currently, no other dataset formats are supported.**

</details>
<details open><summary><a href="#2"><b>2. Create your training configuration file</b></a></summary><a id="2"></a>
<ul><details open><summary><a href="#2-1">2.1 Overview</a></summary><a id="2-1"></a>

All the proposed services like the `training`, `evaluation` etc. of the model are driven by a [user_config.yaml](../user_config.yaml) configuration file written in the YAML language. 

For training, the configuration file should include at least the following sections:

- `general`, describes your project, a flag to control if the plotted figures are to be displayed, max gpu memory allocation etc.
- `operation_mode`, describes the service (like "training", "evaluation", "benchmarking", etc.) or chained services to be used
- `dataset`, describes the dataset you are using, including dataset name, paths, class names, etc.
- `preprocessing`, specifies the preprocessing. 
- `model`, specifies the model to be used, including model name, input shape, etc.
- `training`, specifies your training setup, including batch size, number of epochs, optimizer, and callbacks, etc.
- `mlflow`, specifies the folder to save MLFlow logs.
- `hydra`, specifies the folder to save Hydra logs.

This tutorial only describes the basic settings needed to train a model. In the first part, we describe basic settings.
At the end of this readme, you can also find more advanced settings and callbacks supported.

</details></ul>
<ul><details open><summary><a href="#2-2">2.2 General settings</a></summary><a id="2-2"></a>

The first section of the configuration file is the `general` section that provides information about your project.

```yaml
general:
  project_name: human_activity_recognition # optional <default: human_activity_recognition>
  logs_dir: logs 
  saved_models_dir: saved_models
  display_figures: True
  global_seed: 123
  gpu_memory_limit: 24
  deterministic_ops: True 
```
- `project_name`: is used for creating a project and storing the experiments in the `mlflow` a tool to track the results of different experiments.
- `logs_dir`: is the directory of the run that will hold all the logs created during the experiment.
- `saved_models_dir`: holds the trained model
- `display_figures`: a boolean flag to choose if at the end of the experiment the figures generated are opened or not. This will include the figures of the training_history, confusion_matrix etc.
- `global_seed`: to control the determinism of the random number generation and seeding different functions.
- `gpu_memory_limit`: integer value to allocate the maximum amount of GPU memory (in GigaBytes).
- `deterministic_ops`: a boolean, set to True, if you want your experiments to be fully reproducible.

Enabling the `deterministic_ops` attribute will restrict TensorFlow to use only deterministic operations on the device, but it may lead to a drop in training performance. It should be noted that not all operations in the used version of TensorFlow can be computed deterministically. 

If your case involves any such operation, a warning message will be displayed and the attribute will be ignored.

The `logs_dir` attribute is the name of the directory where the MLFlow and TensorBoard files are saved. The `saved_models_dir` attribute is the name of the directory where models are saved, which includes the trained model. This directory is located under the top level <.hydra> directory.

</details></ul>
<ul><details open><summary><a href="#2-3">2.3 Dataset specification</a></summary><a id="2-3"></a>

Information about the dataset that you want to use, (path to it, the classes available, the train_test_split etc.) is provided in the `dataset` section of the [user_config.yaml](../user_config.yaml) file, as shown in the YAML code below.

```yaml
dataset:
  dataset_name: wisdm  # wisdm or mobility_v1
  class_names: [Jogging,Stationary,Stairs,Walking]  # [Stationary,Walking,Jogging,Biking] for mobility_v1
  training_path: ./datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt # need paths to train.pkl for mobility_v1
  validation_split: 0.2
  test_path: # need paths to test.pkl for mobility_v1
  test_split: 0.25
```
- `dataset_name`: name of the dataset, `wisdm`
- `class_names`: labels of the classes [Jogging, Stationary, Stairs, Walking] in the case of `wisdm`
- `training_path`: path to `WISDM_ar_v1.1_raw.txt` file for `wisdm` and to train.pkl for `mobility_v1`
- `validation_split`: (float between 0-1), portion of training data to be used for the validation
- `test_split`: (float between 0-1) portion of the total data to be used for the test


In this example, we are using the `WISDM` dataset so only the `training_path` has to be provided, as it is a single text file. The values of the validation_split and test_split no validation set path is provided, so the available data under the *training_path* directory is split in two to create a training set and a validation set. 
By default, 25% of the data is used for the test set and 75% data is further split into train_validation (80-20%).
If you want to use a different split ratio, you need to specify in `test_split`, and `validation_split` the portion to be used for the train, test, and validation set.

</details></ul>
<ul><details open><summary><a href="#2-4">2.4 Dataset preprocessing</a></summary><a id="2-4"></a>

The acceleration data is divided into frames by using a given window length value. This is retrieved from `training.model.input_shape[0]`. The frames from the dataset then need to be preprocessed before they are presented to the network. We provide two possible preprocessing algorithms to preprocess the frames.

These include "gravity rotation and suppression" and "normalization", as illustrated in the YAML code below.

```yaml
preprocessing:
  gravity_rot_sup: True
  normalization: False
```
- `gravity_sup_rot` - *boolean*, to choose if gravity is to be rotated in the z-axis and then suppressed using a high-pass filter.
- `normalization` - *boolean*, the flag to choose if standard normalization is to be applied to the frame. $(x-\mu/\sigma)$

The mentioned settings above are used for all the pretrained models in the model zoo of Human activity recognition in the [pretrained_models](./README_MODELS.md).

</details></ul>
<ul><details open><summary><a href="#2-5">2.5 Model parameters</a></summary><a id="2-5"></a>
Information about the model you want to train is provided in the `model` section. That include the name, the input shape and other configurations related to the model.

```yaml
model:
  model_name: st_ign  # choices=['st_ign', 'st_gmp', 'custom_model']
  input_shape: (24, 3, 1)
```
</details></ul>

<ul><details open><summary><a href="#2-6">2.6 Training parameters</a></summary><a id="2-6"></a>

The training section contains various training configurations which includes callbacks, optimizers, etc. as illustrated below:

```yaml
training:
  dropout: 0.5  # Optional, only used for `st_ign` or `st_gmp` model use it if you want a dropout layer to be included in the model
  batch_size: 256
  epochs: 200
  optimizer:
    Adam:
      learning_rate: 0.001
  callbacks:          # Optional section
    ReduceLROnPlateau:
      monitor: val_accuracy
      mode: max
      factor: 0.5
      patience: 40
      min_lr: 1.0e-05
    EarlyStopping:
      monitor: val_accuracy
      mode: max
      restore_best_weights: true
      patience: 60
```

The `batch_size`, `epochs`, and `optimizer` attributes are mandatory. All the others are optional.

The `dropout` attribute only makes sense if your model includes a dropout layer. This will control the dropout ratio for all the available dropout layers in the model.

All the TensorFlow optimizers can be used in the `optimizer` subsection. All the TensorFlow callbacks can be used in the `callbacks` subsection, except the ModelCheckpoint and TensorBoard callbacks that are built-in and can't be redefined.

A variety of learning rate schedulers are provided with the Model Zoo. If you want to use one of them, just include it in the `callbacks` subsection. Refer to [the learning rate schedulers README](../../common/training/lr_schedulers_README.md) for a description of the available callbacks and learning rate plotting utility.
</details></ul>

<ul><details open><summary><a href="#2-7">2.7 Hydra and MLflow settings</a></summary><a id="2-7"></a>

The `mlflow` and `hydra` sections must always be present in the YAML configuration file. The `hydra` section can be used to specify the name of the directory where experiment directories are saved and/or the pattern used to name experiment directories. With the YAML code below, every time you run the Model Zoo, an experiment directory is created that contains all the directories and files created during the run. The names of experiment directories are all unique as they are based on the date and time of the run.

```yaml
hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

The `mlflow` section is used to specify the location and name of the directory where MLflow files are saved, as shown below:

```yaml
mlflow:
  uri: ./tf/src/experiments_outputs/mlruns
```
</details></ul>
</details>
<details open><summary><a href="#3"><b>3. Train your model</b></a></summary><a id="3"></a>

To launch your model training using a real dataset, run the following command from the UC folder:

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config.yaml
```
The trained h5 model can be found in the [model zoo here](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/master/human_activity_recognition/) or ./tf/src/experiments_outputs/{run_time} folders.

</details>
<details open><summary><a href="#4"><b>4. Visualize your results</b></a></summary><a id="4"></a>
<ul><details open><summary><a href="#4-1">4.1 Saved results</a></summary><a id="4-1"></a>

All training and evaluation artifacts are saved under the current output simulation directory **"./tf/src/experiments_outputs/{run_time}"**.

For example, you can retrieve the plots of the accuracy/loss curves as below:

![plot](./img/Training_curves.png)

</details></ul>
<ul><details open><summary><a href="#4-2">4.2 Run TensorBoard</a></summary><a id="4-2"></a>

To visualize the training curves logged by TensorBoard, go to **"./tf/src/experiments_outputs/{run_time}"** and run the following command:

```bash
tensorboard --logdir logs
```

And open the URL `http://localhost:6006` in your browser.

</details></ul>
<ul><details open><summary><a href="#4-3">4.3 Run ClearML</a></summary><a id="4-3"></a>

ClearML is an open-source tool used for logging and tracking machine learning experiments. It allows you to record metrics, parameters, and results, making it easier to monitor and compare diffrent runs.

Follow these steps to configurate ClearML for logging your results. This setup only needs to be done once. if you haven't set it up yet, complete the steps below. if you've already configured ClearML, your results should be automatically logged and available in your session.

- Sign up for free to the [ClearML Hosted Service](https://app.clear.ml), then go to your ClearML workspace and create new credentials.

- Create a `clearml.conf` file and paste the credentials into it. If you are behind a proxy or using SSL portals, add `verify_certificate = False` to the configuration to make it work. Here is an example of what your `clearml.conf` file might look like:

    ```ini
    api {
        web_server: https://app.clear.ml
        api_server: https://api.clear.ml
        files_server: https://files.clear.ml
        # Add this line if you are behind a proxy or using SSL portals
        verify_certificate = False
        credentials {
            "access_key" = "YOUR_ACCESS_KEY"
            "secret_key" = "YOUR_SECRET_KEY"
        }
    }
  
    ```

Once configured, your experiments will be logged directly and shown in the project section under the name of your project.

</details></ul>
<ul><details open><summary><a href="#4-4">4.4 Run MLFlow</a></summary><a id="4-4"></a>

MLflow is an API for logging parameters, code versions, metrics, and artifacts while running machine learning code and for visualizing results.
To view and examine the results of multiple trainings, you can simply access the MLFlow Webapp by running the following command from the `./tf/src/experiments_outputs/:
```bash
mlflow ui
```
And open the given IP address in your browser.

</details></ul>
</details>
<details open><summary><a href="#5"><b>5. Advanced settings</b></a></summary><a id="5"></a>
<ul><details open><summary><a href="#5-1">5.1 Continue Training your own model and Transfer Learning</a></summary><a id="5-1"></a>

If you wish to further train an existing model on a new dataset, instead of starting from scratch, you can do so by specifying the `model_path` attribute in the `model:` section of your configuration file. This allows you to load a pre-trained model for continued training or to resume training after an interruption or crash.

To facilitate resuming training after an interruption or crash, the model is saved at the end of each epoch in the current experiment's output directory. These saved model files can be found under `tf/src/experiments_outputs/<date-and-time>/saved_models`.


```yaml
operation_mode: training

model:
   model_path: <path-to-a-Keras-model-file>    # Path to the model file to use for training

training:
   batch_size: 32
   epochs: 150
   dropout: 0.5
   frozen_layers:
   optimizer:
      Adam:                               
         learning_rate: 0.001
   callbacks:                    
      ReduceLROnPlateau:
         monitor: val_loss
         factor: 0.1
         patience: 10
```

The model file must be a Keras model file with a '.h5' or '.keras' filename extension.

The `model:` subsection of the `training:` section is not present as we are not training a model from the Model Zoo. An error will be thrown if it is present when `model_path` is set.

About the model loaded from the file:
- if some layers are frozen in the model, they will be reset to trainable before training. You can use the `frozen_layers` attribute if you want to freeze these layers (or different ones).
- If you set the `dropout` attribute but the model does not include a dropout layer, an error will be thrown. Reciprocally, an error will also occur if the model includes a dropout layer but the `dropout` attribute is not set.
- If the model was trained before, the state of the optimizer won't be preserved as the model is compiled before training.

</details></ul>
<ul><details open><summary><a href="#5-2">5.2 Freezing layers</a></summary><a id="5-2"></a>

Once the model has been loaded, some layers are often frozen, that is made non-trainable, before training the model. A commonly used approach is to freeze all the layers but the last one, which is the classifier.

By default, all the layers are trainable. If you want to freeze some layers, then you need to add the optional `frozen_layers` attribute to the `training:` section of your configuration file. The indices of the layers to freeze are specified using the Python syntax for indexing into lists and arrays. Below are some examples.


```yaml
training:
   frozen_layers: (0:-1)    # Freeze all the layers but the last one
   
training:
   frozen_layers: (0:2)   # Freeze layers with indices from 0 to 2

```
Note that if you want to make it explicit that all the layers are trainable, you may add the `frozen_layers` attribute and leave it unset or set to *None*.

</details></ul>
<ul><details open><summary><a href="#5-4">5.4 Using chain_tb as operation mode (training and benchmarking)</a></summary><a id="5-4"></a>

You can also use a special operation mode called `chain_tb` which will run a chain of operations, first training the model and then benchmarking the model either locally using the STM32Cube.AI CLI application (will provide memory footprints both RAM and FLASH) or using the STM32Cube.AI dev cloud (RAM, FLASH, Inference Time[ms]). To do this, users have to provide additional parameters as illustrated below:
```yaml
tools:
  stedgeai:
    optimization: balanced
    on_cloud: True # True for using cloud or set to False for using local installation
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

benchmarking:
  board: B-U585I-IOT02A
```

</details></ul>
</details>
