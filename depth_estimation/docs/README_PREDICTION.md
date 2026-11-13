# Depth Estimation Prediction

<details open><summary><a href="#1"><b>1. Depth estimation prediction tutorial</b></a></summary><a id="1"></a>

This tutorial demonstrates how to use the `prediction` service to use the `Fast Depth` to generate some predictions.

To get started, you will need to update the [user_config.yaml](../user_config.yaml) file, which specifies the parameters and configuration options for the services that you want to use. Each section of the [user_config.yaml](../user_config.yaml) file is explained in detail in the following sections.

<details open><summary><a href="#2"><b>2. Choose the operation mode</b></a></summary><a id="2"></a>

The `operation_mode` top-level attribute specifies the operations or the service you want to execute. 

In this tutorial, the `operation_mode` used is the `prediction`.

```yaml
operation_mode: prediction
```

</details></ul>
<details open><summary><a href="#3"><b>3. Global settings</b></a></summary><a id="3"></a>

The `general` section and its attributes are shown below.

```yaml
general:
  project_name: depth_estimation_fast_depth
  gpu_memory_limit: 16  # Maximum amount of GPU memory in GBytes that TensorFlow may use (an integer).

model: 
  model_path: ../../stm32ai-modelzoo/depth_estimation/fastdepth/Public_pretrainedmodel_public_dataset/nyu_depthv2/fastdepth_320/fastdepth_320_int8.tflite
  model_type: fast_depth
  
```

The `model_path` attribute is used to provide the path to the model file you want to use to run the operation mode you selected.

The `gpu_memory_limit` attribute sets an upper limit in GBytes on the amount of GPU memory TensorFlow may use. This is an optional attribute with no default value. If it is not present, memory usage is unlimited. If you have several GPUs, be aware that the limit is only set on logical gpu[0].

</details></ul>
<details open><summary><a href="#5"><b>5. Apply image preprocessing</b></a></summary><a id="5"></a>

depth estimation requires images to be preprocessed by rescaling and resizing them before they can be used. This is specified in the 'preprocessing' section, which is mandatory in most operation modes. The 'preprocessing' section for this tutorial is shown below.

```yaml
preprocessing:
   rescaling:
      scale: 1/127.5
      offset: -1
   resizing:
      interpolation: bilinear
      aspect_ratio: fit
   color_mode: rgb
```

Images are rescaled using the formula "Out = scale\*In + offset". Pixel values of input images usually are integers in the interval [0, 255]. If you set *scale* to 1/255 and offset to 0, pixel values are rescaled to the interval [0.0, 1.0]. If you set *scale* to 1/127.5 and *offset* to -1, they are rescaled to the interval [-1.0, 1.0].

The resizing interpolation methods that are supported include 'bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', 'lanczos5', 'gaussian', and 'mitchellcubic'. Refer to the TensorFlow documentation of the tf.image.resize function for more detail.

Please note that the 'fit' option is the only supported option for the `aspect_ratio` attribute. When using this option, the images will be resized to fit the target size. It is important to note that input images may be smaller or larger than the target size and will be distorted to some extent if their original aspect ratio is not the same as the resizing aspect ratio.

The `color_mode` attribute can be set to either *"grayscale"*, *"rgb"*, or *"rgba"*.

</details></ul>
<details open><summary><a href="#6"><b>6. Specify the Path to the Images to Predict</b></a></summary><a id="6"></a>

In the 'dataset' section, users must provide the path to the directory containing the images to predict using the `prediction_path` attribute as well as the name of the dataset as shown bellow: 

```yaml
dataset: 
  dataset_name: nyu_depthv2
  prediction_path : ./datasets/nyu_depthv2/pred
```

</details></ul>
<details open><summary><a href="#8"><b>8. Hydra and MLflow settings</b></a></summary><a id="8"></a>

The `mlflow` and `hydra` sections must always be present in the YAML configuration file. The `hydra` section can be used to specify the name of the directory where experiment directories are saved and/or the pattern used to name experiment directories. In the YAML code below, it is set to save the outputs as explained in the section <a id="4">visualize the chained services results</a>:

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

<details open><summary><a href="#9"><b>9. Visualize the Results</b></a></summary><a id="9"></a>

Every time you run the Model Zoo, an experiment directory is created that contains all the directories and files created during the run. The names of experiment directories are all unique as they are based on the date and time of the run.

Experiment directories are managed using the Hydra Python package. Refer to [Hydra Home](https://hydra.cc/) for more information about this package.

By default, all the experiment directories are under the <MODEL-ZOO-ROOT>/object_detection/src/experiments_outputs directory and their names follow the "%Y_%m_%d_%H_%M_%S" pattern.

This is illustrated in the figure below.

```
                                  experiments_outputs
                                          |
                                          |
      +--------------+--------------------+--------------------+
      |              |                    |                    |
      |              |                    |                    |
    mlruns    <date-and-time>        <date-and-time>      <date-and-time> 
      |                                   |              
  MLflow files                             +--- stm32ai_main.log                      
                                          |
                +-------------------------+
                |                         |                                           
                |                         |                                
           predictions                 .hydra
                                          |                               
                                     Hydra files
                                        
```

</details></ul>
<details open><summary><a href="#10"><b>10. Run MLflow</b></a></summary><a id="10"></a>

MLflow is an API that allows you to log parameters, code versions, metrics, and artifacts while running machine learning code, and provides a way to visualize the results.

To view and examine the results of multiple trainings, you can navigate to the **experiments_outputs** directory and access the MLflow Webapp by running the following command:

```bash
mlflow ui
```

This will start a server and its address will be displayed. Use this address in a web browser to connect to the server. Then, using the web browser, you will be able to navigate the different experiment directories and look at the metrics they collected. Refer to [MLflow Home](https://mlflow.org/) for more information about MLflow.

</details></ul>
</details>
