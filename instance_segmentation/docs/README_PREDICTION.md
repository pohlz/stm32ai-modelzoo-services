# Instance Segmentation Prediction

<details open><summary><a href="#1"><b>1. Instance segmentation prediction tutorial</b></a></summary><a id="1"></a>

This tutorial demonstrates how to use the `prediction` service to use the `Yolov8n seg` to generate some predictions.

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
  project_name: coco_instance_seg          # Project name. Optional, defaults to "<unnamed>".
  gpu_memory_limit: 16                     # Maximum amount of GPU memory in GBytes that TensorFlow may use (an integer).

model: 
  model_path: https://github.com/stm32-hotspot/ultralytics/raw/refs/heads/main/examples/YOLOv8-STEdgeAI/stedgeai_models/segmentation/yolov8n_256_quant_pc_ii_seg_coco-st.tflite

```

The `model_path` attribute under the model section is used to provide the path to the model file you want to use to run the operation mode you selected.

The `gpu_memory_limit` attribute sets an upper limit in GBytes on the amount of GPU memory TensorFlow may use. This is an optional attribute with no default value. If it is not present, memory usage is unlimited. If you have several GPUs, be aware that the limit is only set on logical gpu[0].

</details></ul>
<details open><summary><a href="#4"><b>4. Dataset specification</b></a></summary><a id="4"></a>

The `dataset` section and its attributes are shown in the YAML code below.

```yaml
dataset:
  dataset_name: coco_is                   # Dataset name. Optional, defaults to "<unnamed>".
  # One of the following parameters should be provided:
  classes_file_path: ./dataset/coco_classes.txt
  class_names: [person, bicycle, car, motorcycle, airplane, bus, train, ...] # Names of the classes in the dataset.
  prediction_path: ./dataset/coco/images
```

The `dataset_name` attribute is optional and can be used to specify the name of your dataset.

The `classes_file_path` attribute specifies the path to a file that contains the names of the classes in the dataset. Each class name should be listed on a new line in the file. This attribute is useful when you have a large number of classes and prefer to manage them in a separate file.

The `class_names` attribute specifies the classes in the dataset. This information must be provided in the YAML file. If the `class_names` attribute is absent, the `classes_name_file` argument can be used as an alternative, pointing to a text file containing the class names.

The `prediction_path` specifies the folder containing the images for prediction.

</details></ul>
<details open><summary><a href="#5"><b>5. Apply image preprocessing</b></a></summary><a id="5"></a>

Instance segmentation requires images to be preprocessed by rescaling and resizing them before they can be used. This is specified in the 'preprocessing' section, which is mandatory in most operation modes. The 'preprocessing' section for this tutorial is shown below.

```yaml
preprocessing:
  rescaling:
    scale: 1/255.0
    offset: 0
  resizing:
    interpolation: bilinear
    aspect_ratio: fit
  color_mode: rgb
```

Images are rescaled using the formula "Out = scale\*In + offset". Pixel values of input images usually are integers in the interval [0, 255]. If you set *scale* to 1/255 and offset to 0, pixel values are rescaled to the interval [0.0, 1.0]. If you set *scale* to 1/127.5 and *offset* to -1, they are rescaled to the interval [-1.0, 1.0].

The resizing interpolation methods that are supported include 'bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', 'lanczos5', 'gaussian', and 'mitchellcubic'. Refer to the TensorFlow documentation of the tf.image.resize function for more detail.

Please note that the 'fit' option is the only supported option for the `aspect_ratio` attribute. When using this option, the images will be resized to fit the target size. It is important to note that input images may be smaller or larger than the target size and will be distorted to some extent if their original aspect ratio is not the same as the resizing aspect ratio.

The `color_mode` attribute can be set to either *"grayscale"*, *"rgb"*, or *"rgba"*.


<details open><summary><a href="#6"><b>6. Set the postprocessing parameters</b></a></summary><a id="6"></a>

A 'postprocessing' section is required in all operation modes for instance segmentation models. This section includes parameters such as NMS threshold, confidence threshold, IoU evaluation threshold, and maximum detection boxes. These parameters are necessary for proper post-processing of instance segmentation results.

```yaml
postprocessing:
  confidence_thresh: 0.5
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
```

The `confidence_thresh` parameter controls the minimum confidence score required for a detection to be considered valid. A higher confidence threshold will result in fewer detections, while a lower threshold will result in more detections.

The `IoU_eval_thresh` parameter controls the minimum overlap required between two bounding boxes for them to be considered as the same object. A higher IoU threshold will result in fewer detections, while a lower threshold will result in more detections.

Overall, improving instance segmentation requires careful tuning of these post-processing parameters based on your specific use case. Experimenting with different values and evaluating the results can help you find the optimal values for your instance segmentation model.

</details></ul>
<details open><summary><a href="#7"><b>7. Hydra and MLflow settings</b></a></summary><a id="7"></a>

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

<details open><summary><a href="#8"><b>8. Visualize the Results</b></a></summary><a id="8"></a>

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
<details open><summary><a href="#9"><b>9. Run MLflow</b></a></summary><a id="9"></a>

MLflow is an API that allows you to log parameters, code versions, metrics, and artifacts while running machine learning code, and provides a way to visualize the results.

To view and examine the results of multiple trainings, you can navigate to the **experiments_outputs** directory and access the MLflow Webapp by running the following command:

```bash
mlflow ui
```

This will start a server and its address will be displayed. Use this address in a web browser to connect to the server. Then, using the web browser, you will be able to navigate the different experiment directories and look at the metrics they collected. Refer to [MLflow Home](https://mlflow.org/) for more information about MLflow.

</details></ul>
</details>
