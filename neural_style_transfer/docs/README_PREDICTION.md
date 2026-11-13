# Neural style transfer Prediction

<details open><summary><a href="#1"><b>1. Neural style transfer prediction tutorial</b></a></summary><a id="1"></a>

This tutorial demonstrates how to use the `prediction` service to use the `Xinet_picasso_muse_a75_160` model to generate images with picasso's "La muse" artistic style .

To get started, you will need to update the [user_config.yaml](../user_config.yaml) file, which specifies the parameters and configuration options for the services that you want to use. Each section of the [user_config.yaml](../user_config.yaml) file is explained in detail in the following sections.

<details open><summary><a href="#2"><b>2. Choose the operation mode</b></a></summary><a id="2"></a>

The `operation_mode` top-level attribute specifies the operations or the service you want to execute.

In this tutorial, the `operation_mode` used is the `prediction`.

```yaml
operation_mode: prediction
```

</details></ul>
<details open><summary><a href="#3"><b>3. Model settings</b></a></summary><a id="3"></a>

The `model` section and its attributes are shown below.

```yaml
model:
  model_type: xinet_picasso_muse # xinet_picasso_muse
  # path to a `.tflite` or `.onnx` file.
  model_path:  ../../stm32ai-modelzoo/neural_style_transfer/xinet_picasso_muse/Public_pretrainedmodel_public_dataset/coco_2017_80_classes_picasso/xinet_a75_picasso_muse_160/xinet_a75_picasso_muse_160_nomp.tflite
```

The `model_path` attribute is used to provide the path to the model file you want to use to run the operation mode you selected.

The `gpu_memory_limit` attribute sets an upper limit in GBytes on the amount of GPU memory TensorFlow may use. This is an optional attribute with no default value. If it is not present, memory usage is unlimited. If you have several GPUs, be aware that the limit is only set on logical gpu[0].

</details></ul>
<details open><summary><a href="#4"><b>4. Dataset specification</b></a></summary><a id="4"></a>

The `dataset` section and its attributes are shown in the YAML code below.

```yaml
dataset:
  dataset_name: coco                    # Dataset name. required, supports only coco value.
  prediction_path: ../datasets/images/coco
```

The `name` attribute is optional and can be used to specify the name of your dataset. For this use case, the `classes_file_path` and the `class_names` attributes are to be unset.

</details></ul>
<details open><summary><a href="#5"><b>5. Apply image preprocessing</b></a></summary><a id="5"></a>

Neural style transfer requires images to be preprocessed by rescaling and resizing them before they can be used. This is specified in the 'preprocessing' section, which is mandatory in most operation modes. The 'preprocessing' section for this tutorial is shown below.

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

</details></ul>
<details open><summary><a href="#6"><b>6. Specify the Path to the Images to Predict</b></a></summary><a id="6"></a>

In the 'prediction' section, users must provide the path to the directory containing the images to predict using the `test_files_path` attribute.

<details open><summary><a href="#7"><b>7. Set the postprocessing parameters</b></a></summary><a id="7"></a>

A 'postprocessing' section is required in all operation modes for neural style transfer models. This section includes parameters such as bilateral filter diameter, sigma color, sigma space, alpha and beta factors for used to adjust the contrast and brightness of the image and the saturation scale. These parameters are necessary for proper post-processing to generate a visually acceptable image from neural style transfer results.

```yaml
postprocessing:
  bilateral_diameter: 15
  sigma_color: 20
  sigma_space: 10
  scaling_alpha: 1.25
  scaling_beta: -10
  saturation_scale: 1.2
```

The `bilateral_diameter` parameter specifies the diameter of the pixel neighborhood used in the bilateral filter. A larger diameter means the filter considers a wider area around each pixel, which can smooth the image while preserving edges.

The `sigma_color` parameter controls the intensity of color smoothing in the bilateral filter. Higher values allow more significant differences in color to be smoothed together, reducing noise while maintaining edge sharpness.

The `sigma_space` parameter defines the spatial extent of the bilateral filter. Larger values mean that pixels farther apart in the image can influence each other, which can result in a more global smoothing effect.

The `alpha` parameter is used to adjust the contrast of the image. Higher values increase the contrast, making the image appear sharper and more vivid.

The `beta` parameter is added to each pixel in the image to adjust its brightness. Positive values brighten the image, while negative values darken it.

The `saturation_scale` is used to scale the saturation of the image. Increasing this value makes colors more vibrant, while decreasing it desaturates the image, making it appear more muted.

Overall, improving neural style transfer requires careful tuning of these post-processing parameters based on your specific use case. Experimenting with different values and evaluating the results can help you find the optimal values for your neural style transfer model.

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
