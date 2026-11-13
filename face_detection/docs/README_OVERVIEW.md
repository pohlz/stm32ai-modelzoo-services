# Face Detection STM32 model zoo

Remember that minimalistic yaml files are available [here](../config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!

## <a id="">Table of contents</a>

1. [Face Detection Model Zoo introduction](#1)
2. [Face Detection tutorial](#2)
   - [2.1 Choose the operation mode](#2-1)
   - [2.2 Global settings](#2-2)
   - [2.3 Dataset specification](#2-3)
   - [2.4 Apply image preprocessing](#2-4)
   - [2.5 Set the postprocessing parameters](#2-5)
   - [2.6 Model quantization](#2-6)
   - [2.7 Benchmark the model](#2-7)
   - [2.8 Deploy the model](#2-8)
   - [2.9 Hydra and MLflow settings](#2-9)
3. [Run the Face Detection chained service](#3)



<details open><summary><a href="#1"><b>1. Face Detection Model Zoo introduction</b></a></summary><a id="1"></a>

The Face Detection model zoo provides a collection of independent services and pre-built chained services that can be
used to perform various functions related to machine learning for Face Detection. The individual services include
tasks such as quantizing the model, while the chained services combine multiple services to
perform more complex functions, such as evaluating the model, quantizing it, and evaluating the quantized model
successively.

To use the services in the Face Detection model zoo, you can utilize the model zoo [stm32ai_main.py](../stm32ai_main.py) along with the [user_config.yaml](../user_config.yaml) file as input. The yaml file specifies the service or the chained services and a set of configuration parameters such as the model (either from the model zoo or your own custom model), the dataset, the number of epochs, and the preprocessing parameters, among others.

More information about the different services and their configuration options can be found in the <a href="#2">next
section</a>.

The Face Detection datasets are expected to be in [TFS format](README_DATASETS_CREATE_TFS.md).

An example of this structure is shown below:

```yaml
<dataset-root-directory>
train/:
  train_image_1.jpg
  train_image_1.tfs
  train_image_2.jpg
  train_image_2.tfs
val/:
  val_image_1.jpg
  val_image_1.txt
  val_image_2.jpg
  val_image_2.txt
```

</details>
<details open><summary><a href="#2"><b>2. Face Detection tutorial</b></a></summary><a id="2"></a>

This tutorial demonstrates how to use the `chain_eqeb` services to evaluation, benchmark, quantize, evaluate, and benchmark
the model.

To get started, you will need to update the [user_config.yaml](../user_config.yaml) file, which specifies the parameters and configuration options for the services that you want to use. Each section of the [user_config.yaml](../user_config.yaml) file is explained in detail in the following sections.

<ul><details open><summary><a href="#2-1">2.1 Choose the operation mode</a></summary><a id="2-1"></a>

The `operation_mode` top-level attribute specifies the operations or the service you want to execute. This may be a single operation or a set of chained operations.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 'e' for evaluation, 'q' for quantization, 'b' for benchmark, and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations                                                                                                                                          |
|:-------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| `evaluation`             | Evaluate the accuracy of a float or quantized model on a test or validation dataset                                                                 |
| `quantization`           | Quantize a float model                                                                                                                              |
| `prediction`             | Predict the classes and bounding boxes of some images using a float or quantized model.                                                             |
| `benchmarking`           | Benchmark a float or quantized model on an STM32 board                                                                                              |
| `deployment`             | Deploy a model on an STM32 board                                                                                                                    |
| `chain_eqe`              | Sequentially: evaluation of a float model, quantization, evaluation of the quantized model                                                          |
| `chain_qb`               | Sequentially: quantization of a float model, benchmarking of quantized model                                                                        |
| `chain_eqeb`             | Sequentially: evaluation of a float model, quantization, evaluation of quantized model, benchmarking of quantized model                             |
| `chain_qd`               | Sequentially: quantization of a float model, deployment of quantized model                                                                          |

You can refer to readme links below that provide typical examples of operation modes, and tutorials on specific services:

- [quantization, chain_eqe, chain_qb](./README_QUANTIZATION.md)
- [evaluation, chain_eqeb](./README_EVALUATION.md)
- [benchmarking](./README_BENCHMARKING.md)
- deployment, chain_qd ([STM32N6](./README_DEPLOYMENT_STM32N6.md))

In this tutorial, the `operation_mode` used is the `chain_eqeb` as shown below to train a model, quantize, evaluate it to be later deployed in the STM32 boards.

```yaml
operation_mode: chain_eqeb
```

</details></ul>
<ul><details open><summary><a href="#2-2">2.2 Global settings</a></summary><a id="2-2"></a>

The `general` section and its attributes are shown below.

```yaml
general:
  gpu_memory_limit: 16              # Maximum amount of GPU memory in GBytes that TensorFlow may use (an integer).
```

The `gpu_memory_limit` attribute sets an upper limit in GBytes on the amount of GPU memory Tensorflow may use. This is
an optional attribute with no default value. If it is not present, memory usage is unlimited. If you have several GPUs,
be aware that the limit is only set on logical gpu[0].



```yaml
model:
  model_path: ../../stm32ai-modelzoo/face_detection/yunet/Public_pretrainedmodel_public_dataset/widerface/yunetn_320/yunetn_320.onnx
  model_type: yunet            
```
The `model_path` attribute is utilized to indicate the path to the model file that you wish to use for the selected
operation mode. The accepted formats for `model_path` are listed in the table below:

The `model_type` attribute specifies the type of the model architecture that you want to train. It is important to note that only certain models are supported. These models include:

- `yunet`: Yunet is a lightweight and efficient Face Detection model optimized for real-time applications on embedded devices. Yunet designed specifically for detecting faces and 5 keypoints (2x eyes, 2x mouth, nose).

- `facedetect_front`: BlazeFace Front 128x128 is a lightweight and efficient Face Detection model optimized for real-time applications on embedded devices. It is a variant of the BlazeFace architecture, designed specifically for detecting frontal faces and 6 keypoints (2x eyes, 2x ears, nose, mouth) at a resolution of 128x128 pixels.


It is important to note that each model type has specific requirements in terms of input image size, output size of the head and/or backbone, and other parameters. Therefore, it is important to choose the appropriate model type for your specific use case, and to configure the training process accordingly.

| Operation mode | `model_path` |
|:---------------|:-------------|
| 'evaluation'   | Onnx or TF-Lite model file |
| 'quantization', 'chain_eqe', 'chain_eqeb', 'chain_qb', 'chain_qd' | Onnx model file |
| 'prediction'   | Onnx or TF-Lite model file |
| 'benchmarking' | TF-Lite or ONNX model file |
| 'deployment'   | Onnx or TF-Lite model file |


</details></ul>
<ul><details open><summary><a href="#2-3">2.3 Dataset specification</a></summary><a id="2-3"></a>

Before you start using this project It's important to convert your dataset to the `tfs format`, to do so you can use our [tfs converter](./README_DATASETS_CONVERTER.md). Please note that the converter expects as input the [yolo darknet txt format](https://roboflow.com/formats/yolo-darknet-txt).

The `dataset` section and its attributes are shown in the YAML code below.

```yaml
dataset:
  dataset_name: wider_face                                   # Dataset name. Optional, defaults to "<unnamed>".
  class_names: [Face] # Names of the classes in the dataset.
  test_path: <test-set-root-directory>                       # Path to the root directory of the test set.
  quantization_path: <quantization-set-root-directory>       # Path to the root directory of the quantization set.
```

The `dataset_name` attribute is optional and can be used to specify the name of your dataset.

The `class_names` attribute specifies the classes in the dataset. This information must be provided in the YAML file. 

The `quantization_path` attribute is used to specify a dataset for the quantization process. You can set
the `quantization_split` attribute to use only a portion of the dataset for quantization.

</details></ul>
<ul><details open><summary><a href="#2-4">2.4 Apply image preprocessing</a></summary><a id="2-4"></a>

Face Detection requires images to be preprocessed by rescaling and resizing them before they can be used. This is
specified in the 'preprocessing' section, which is mandatory in all operation modes. Additionally, bounding boxes should
be processed along with the images to accurately detect objects in the images.
This is specified in the 'preprocessing' section that is required in all the operation modes.

The 'preprocessing' section for this tutorial is shown below.

```yaml
preprocessing:
  rescaling: { scale: 1, offset: 0 }
  resizing:
    aspect_ratio: fit
    interpolation: bilinear
  color_mode: bgr
```

Images are rescaled using the formula "Out = scale\*In + offset". Pixel values of input images usually are integers in
the interval [0, 255]. If you set *scale* to 1./255 and offset to 0, pixel values are rescaled to the
interval [0.0, 1.0]. If you set *scale* to 1/127.5 and *offset* to -1, they are rescaled to the interval [-1.0, 1.0].

The resizing interpolation methods that are supported include 'bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', '
lanczos5', 'gaussian' and 'mitchellcubic'. Refer to the Tensorflow documentation of the tf.image.resize function for
more detail.

Please note that the 'fit' option is the only supported option for the `aspect_ratio` attribute. When using this option,
the images will be resized to fit the target size. It is important to note that input images may be smaller or larger
than the target size, and will be distorted to some extent if their original aspect ratio is not the same as the
resizing aspect ratio. Additionally, bounding boxes should be adjusted to maintain their relative positions and sizes in
the resized images.

The `color_mode` attribute can be set to either *"grayscale"*, *"rgb"* or *"rgba"*.

</details></ul>

<ul><details open><summary><a href="#2-5">2.5 Set the postprocessing parameters</a></summary><a id="2-5"></a>

A 'postprocessing' section is required in all operation modes for object detection models. This section includes
parameters such as NMS threshold, confidence threshold, IoU evaluation threshold, and maximum detection boxes. These
parameters are necessary for proper post-processing of object detection results.

```yaml
postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: False   # Plot precision versus recall curves. Default is False.
  max_detection_boxes: 100
```

- **NMS_thresh (Non-Maximum SuppressionThreshold)**: This parameter controls the overlapping bounding boxes that are considered as separate detections. A higher NMS threshold will result in fewer detections, while a lower threshold will result in more detections. To improve object detection, you can experiment with different NMS thresholds to find the optimal value for your specific use case.

- **confidence_thresh**: This parameter controls the minimum confidence score required for a detection to be considered valid. A higher confidence threshold will result in fewer detections, while a lower threshold will result in more detections.

- **IoU_eval_thresh**: This parameter controls the minimum overlap required between two bounding boxes for them to be considered as the same object. A higher IoU threshold will result in fewer detections, while a lower threshold will result in more detections.

- **max_detection_boxes**: This parameter controls the maximum number of detections that can be output by the object detection model. A higher maximum detection boxes value will result in more detections, while a lower value will result in fewer detections.

- **plot_metrics**: This parameter is an optional parameter in the object detection model that controls whether or not to plot the precision versus recall curves. By default, this parameter is set to False, which means that the precision versus recall curves will not be plotted. If you set this parameter to True, the object detection model will generate and display the precision versus recall curves, which can be helpful for evaluating the performance of the model.

Overall, improving object detection requires careful tuning of these post-processing parameters based on your specific use case. Experimenting with different values and evaluating the results can help you find the optimal values for your object detection model.

</details></ul>
<ul><details open><summary><a href="#2-6">2.6 model quantization </a></summary><a id="2-6"></a>

The `quantization` section is required in all the operation modes that include a quantization, namely `quantization`, `chain_eqe`, `chain_eqeb`, `chain_qb`, and `chain_qd`.

The `quantization` section for this tutorial is shown below.

```yaml
quantization:
  quantizer: TFlite_converter
  quantization_type: PTQ
  quantization_input_type: float
  quantization_output_type: uint8
  granularity: per_channel           # Optional, defaults to "per_channel".
  optimize: False                     # Optional, defaults to False.
  export_dir: quantized_models       # Optional, defaults to "quantized_models".
```

This section is used to configure the quantization process, which optimizes the model for efficient deployment on
embedded devices by reducing its memory usage (Flash/RAM) and accelerating its inference time, with minimal degradation
in model accuracy. The `quantizer` attribute expects the value "TFlite_converter" or "onnx_quantizer", which is used to convert the trained
model weights from float to integer values and transfer the model to a TensorFlow Lite format or an onnx QDQ format.

The `quantization_type` attribute only allows the value "PTQ," which stands for Post Training Quantization. To specify
the quantization type for the model input and output, use the `quantization_input_type` and `quantization_output_type`
attributes, respectively.

The `quantization_input_type` attribute is a string that can be set to "int8", "uint8," or "float" to represent the
quantization type for the model input. Similarly, the `quantization_output_type` attribute is a string that can be set
to "int8", "uint8," or "float" to represent the quantization type for the model output. With ONNX QDQ quantization input and output
will always be float whatever the parameters settings.

The quantization `granularity` is either "per_channel" or "per_tensor". If the parameter is not set, it will default to 
"per_channel". 'per channel' means all weights contributing to a given layer output channel are quantized with one 
unique (scale, offset) couple. The alternative is 'per tensor' quantization which means that the full weight tensor of 
a given layer is quantized with one unique (scale, offset) couple. 
It is obviously more challenging to preserve original float model accuracy using 'per tensor' quantization. But this 
method is particularly well suited to fully exploit STM32MP2 platforms HW design.

Some topologies can be slightly optimized to become "per_tensor" quantization friendly. Therefore, we propose to 
optimize the model to improve the "per-tensor" quantization. This is controlled by the `optimize` parameter. By default, 
it is False and no optimization is applied. When set to True, some modifications are applied on original network. 
Please note that these optimizations only apply when granularity is "per_tensor". To finish, some topologies cannot be 
optimized. So even if `optimize` is set to True, there is no guarantee that "per_tensor" quantization will preserve the 
float model accuracy for all the topologies.

By default, the quantized model is saved in the 'quantized_models' directory under the 'experiments_outputs' directory.
You may use the optional `export_dir` attribute to change the name of this directory.

</details></ul>

<ul><details open><summary><a href="#2-7">2.7 Benchmark the model</a></summary><a id="2-7"></a>

The [STEdgeAI Developer Cloud](https://stedgeai-dc.st.com/home) allows you to benchmark your model and estimate its
footprints and inference time for different STM32 target devices. To use this feature, set the `on_cloud` attribute to
True. Alternatively, you can use [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) to benchmark
your model and estimate its footprints for STM32 target devices locally. To do this, make sure to add the path to
the `stedgeai` executable under the `path_to_stedgeai` attribute and set the `on_cloud` attribute to False.

The `optimization` defines the optimization used to generate the C model, options: "balanced", "time", "ram".

The `board` attribute is used to provide the name of the STM32 board to benchmark the model on. The available boards are 'STM32H747I-DISCO', 'STM32H7B3I-DK', 'STM32F469I-DISCO', 'B-U585I-IOT02A', 'STM32L4R9I-DISCO', 'NUCLEO-H743ZI2', 'STM32H747I-DISCO', 'STM32H735G-DK', 'STM32F769I-DISCO', 'NUCLEO-G474RE', 'NUCLEO-F401RE' and 'STM32F746G-DISCO'.

```yaml
tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

benchmarking:
  board: STM32H747I-DISCO     # Name of the STM32 board to benchmark the model on
```

The `path_to_cubeIDE` attribute is for the deployment service which is not part of the
`chain_tqeb` used in this tutorial.

</details></ul>
<ul><details open><summary><a href="#2-8">2.8 Deploy the model</a></summary><a id="2-8"></a>

In this tutorial, we are using the `chain_eqeb` toolchain, which does not include the deployment service. However, if
you want to deploy the model after running the chain, you can do so by referring to
the deployment README and modifying the `deployment_config.yaml` file or by setting the `operation_mode`
to `deploy` and modifying the `user_config.yaml` file as described below:

```yaml
operation_mode: deployment

model:
  model_path: ../../stm32ai-modelzoo/face_detection/yunet/Public_pretrainedmodel_public_dataset/widerface/yunetn_320/yunetn_320_qdq_int8.onnx
  model_type: yunet

dataset:
   class_names: [person]

preprocessing:
  resizing:
    aspect_ratio: fit
    interpolation: bilinear
  color_mode: rgb

postprocessing:
  confidence_thresh: 0.5
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  max_detection_boxes: 10

tools:
   stedgeai:
      optimization: balanced
      on_cloud: False
      path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
   path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

deployment:
  c_project_path: ../application_code/face_detection/STM32N6/
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32N6
    board: STM32N6570-DK

mlflow:
   uri: experiments_outputs/mlruns

hydra:
   run:
      dir: experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```


The `dataset` section requires users to provide the names of the classes using the `class_names` attribute.

The `postprocessing` section requires users to provide the values for the post-processing parameters. These parameters
include the `NMS_thresh`, `confidence_thresh`, `IoU_eval_thresh`, and `max_detection_boxes`. By providing
these values in the postprocessing section, the object detection model can properly post-process the results and
generate accurate detections. It is important to carefully tune these parameters based on your specific use case to
achieve optimal performance.

The `tools` section includes information about the **stedgeai** toolchain, such as the version, optimization level, and path
to the `stedgeai.exe` file.

Finally, in the `deployment` section, users must provide information about the hardware setup, such as the series and
board of the STM32 device, as well as the input and output interfaces. Once all of these sections have been filled in,
users can run the deployment service to deploy their model to the STM32 device.

Please refer to readme below for a complete deployment tutorial:
- on N6-NPU : [README_STM32N6.md](./README_DEPLOYMENT_STM32N6.md)

</details></ul>
<ul><details open><summary><a href="#2-9">2.9 Hydra and MLflow settings</a></summary><a id="2-9"></a>

The `mlflow` and `hydra` sections must always be present in the YAML configuration file. The `hydra` section can be used
to specify the name of the directory where experiment directories are saved and/or the pattern used to name experiment
directories. In the YAML code below, it is set to save the outputs as explained in the section <a id="4">visualize the
chained services results</a>:

```yaml
hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

The `mlflow` section is used to specify the location and name of the directory where MLflow files are saved, as shown
below:

```yaml
mlflow:
  uri: ./tf/src/experiments_outputs/mlruns
```

</details></ul>
</details>
<details open><summary><a href="#3"><b>3. Run the Face Detection chained service</b></a></summary><a id="3"></a>

After updating the [user_config.yaml](../user_config.yaml) file, please run the following command:

```bash
python stm32ai_main.py
```