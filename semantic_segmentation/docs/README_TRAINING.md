# <a id="">Semantic Segmentation STM32 model training</a>

This readme shows how to train from scratch or apply transfer learning on a semantic segmentation model.
As an example we will demonstrate the workflow on the [COCO 2017 Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) segmentation dataset.


<details open><summary><a href="#1"><b>1. Prepare the dataset</b></a></summary><a id="1"></a>

The dataset should be structured to include directories for the images and their corresponding segmentation masks, as well as lists of filenames for training and validation. The directory tree for the dataset is outlined below:

```bash
dataset_root_directory/
   Images/
      image_1.jpg
      image_2.jpg
      ...
   Segmentation_masks/
      mask_1.png
      mask_2.png
      ...
   Image_sets/
      train.txt
      val.txt
```

A directory contains all the images used for training, validation, and testing, and another one holds the segmentation masks corresponding to the images and the last one is for text files like `train.txt` and `val.txt` which list the filenames of images that are included in the training and validation sets, respectively.

**Please ensure that the segmentation masks are formatted as images with pixel values as integers. Each integer should correspond to a different class label, effectively segmenting the image into regions based on the class they belong to.**
</details>

<details open><summary><a href="#2"><b>2. Create your training configuration file</b></a></summary><a id="2"></a>
<ul><details open><summary><a href="#2.1">2.1 Overview</a></summary><a id="2.1"></a>

All the proposed services like the training of the model are driven by a configuration file written in the YAML language.

For training, the configuration file should include at least the following sections:

- `general`, describes your project, including your project name, etc.
- `operation_mode`, describes the service or chained services to be used.
- `model`, defines the model architecture or the external model file.
- `dataset`, describes the dataset you are using, including directory paths, class names, etc.
- `preprocessing`, specifies the methods you want to use for rescaling and resizing the images. 
- `data_augmentation`, lists augmentation operations.
- `training`, specifies your training setup, including batch size, number of epochs, optimizer, callbacks, etc.
- `mlflow`, specifies the folder to save MLFlow logs.
- `hydra`, specifies the folder to save Hydra logs.

This tutorial only describes the settings needed to train a model. In the first part, we describe basic settings.
At the end of this readme, you can also find more advanced settings and callbacks supported.
</details></ul>

<ul><details open><summary><a href="#2.2">2.2 General settings</a></summary><a id="2.2"></a>

The first section of the configuration file is the `general` section that provides information about your project.

```yaml
general:
  project_name: segmentation 
  logs_dir: logs
  saved_models_dir: saved_models
  gpu_memory_limit: 12
  global_seed: 127
  display_figures: False
```

The `logs_dir` attribute is the name of the directory where the MLFlow and TensorBoard files are saved. The `saved_models_dir` attribute is the name of the directory where trained models are saved. These two directories are located under the top level "hydra" directory (please see [chapter 2.8](#2-8) for hydra informations).

The `gpu_memory_limit` attribute sets an upper limit in GBytes on the amount of GPU memory Tensorflow may use. This is an optional attribute with no default value. If it is not present, memory usage is unlimited. If you have several GPUs, be aware that the limit is only set on logical gpu[0].

The `global_seed` attribute specifies the value of the seed to use to seed the Python, numpy and Tensorflow random generators at the beginning of the main script. This is an optional attribute, the default value being 120. If you don't want random generators to be seeded, then set `global_seed` to 'None' (not recommended as this would make training results less reproducible).

</details></ul>

<ul><details open><summary><a href="#2.3">2.3 Model section</a></summary><a id="2.3"></a>

The `model` section defines all parameters related to the model architecture and selection for training.

**Parameters:**
- `model_type`: The only supported value for now is `deeplab`.
- `model_name`: The name of the registered model you want to use. Available options are:
  - `st_deeplabv3_mnv2_a050_s16_asppv2`
  - `st_deeplabv3_rn50v1_s16_asppv2`
- `input_shape`: The expected input dimensions for the model, typically in the format `(height, width, channels)`.
- `model_path`: Use this to provide a path to a pre-trained or custom model file if you want to train from your own checkpoint, do transfer learning, or fine-tune a model. This is not for selecting a built-in model, but for starting from a specific saved model.

**Example:**
```yaml
model:
  model_type: deeplab
  model_name: st_deeplabv3_mnv2_a050_s16_asppv2
  input_shape: (416, 416, 3)
  # model_path: ./path/to/your_model.keras
```

**Registered Models:**
- The following models are currently registered and available for direct selection:
  - `st_deeplabv3_mnv2_a050_s16_asppv2`
  - `st_deeplabv3_rn50v1_s16_asppv2`
- You can also register your own custom models by following the instructions in the wrapper code (`tf/wrappers/models/custom_models/models.py`).

**Registered Model Name Format Explanation:**

Registered model names encode key architectural parameters. For example:

- `st_deeplabv3_rn50v1_s16_asppv2`
  - `st_deeplabv3`: Model architecture (DeepLabV3, ST variant)
  - `rn50v1`: Backbone (ResNet-50, version 1)
  - `s16`: Output stride (16)
  - `asppv2`: ASPP version 2

Each part of the name helps you identify the model’s backbone, configuration, and special features.
</details></ul>

<ul><details open><summary><a href="#2.4">2.4 Dataset specification</a></summary><a id="2.4"></a>

Information about the dataset you want use is provided in the `dataset` section of the configuration file, as shown in the YAML code below.
State machine below describes the rules to follow when handling dataset path for the training.
<div align="center" style="width:50%; margin: auto;">

![plot](../../common/doc/img/state_machine_training.JPG)
</div>

```yaml
dataset:
  dataset_name: person_coco_2017_pascal_voc_2012
  class_names: ["background", "person"]

  training_path: ./datasets/person_COCO2017_VOC2012/JPEGImages
  training_masks_path: ./datasets/person_COCO2017_VOC2012/SegmentationClassAug
  training_files_path: ./datasets/person_COCO2017_VOC2012/ImageSets/Segmentation/trainaug.txt
  validation_path: ./datasets/person_COCO2017_VOC2012/JPEGImages
  validation_masks_path: ./datasets/person_COCO2017_VOC2012/SegmentationClassAug
  validation_files_path: ./datasets/person_COCO2017_VOC2012/ImageSets/Segmentation/val.txt

  test_path:                                 # Path to test JPEG images
  test_masks_path:                         # Path to test masks files
  test_files_path:             # Path to file listing the 

```

The `dataset_name` holds the identifier for the dataset, which in this case is pascal_voc. The `class_names` attribute specifies the classes in the dataset. This information must be provided in the YAML file. If the `class_names` attribute is absent, the `classes_name_file` argument can be used as an alternative, pointing to a text file containing the class names.

The `training_path` specifies the directory path to the training images, the `training_masks_path` points to the location of the segmentation masks corresponding to the training images and `training_files_path` indicates the file that contains the list of image filenames used for training.

The `validation_path` is designated for the directory containing the validation images, the `validation_masks_path` directs to the segmentation masks associated with these validation images, while the `validation_files_path` provides the location of the file listing the image filenames to be used for validation purposes.

By default, when the validation_path is not provided, 80% of the data is used for the training set and the remaining 20% is used for the validation set.
If you want to use a different split ratio, you need to specify in `validation_split` the ratio to be used for the validation set (value between 0 and 1).

For testing, the `test_path`, `test_masks_path`, and `test_files_path` keys are present but not populated with paths. These would typically specify the directory for test images, the directory for the corresponding segmentation masks, and the file with the list of test image filenames, respectively. The absence of values suggests that the validation set is used as the test set.

</details></ul>

<ul><details open><summary><a href="#2.5">2.5 Dataset preprocessing</a></summary><a id="2.5"></a>

The images from the dataset need to be preprocessed before they are presented to the network. This includes rescaling and resizing, as illustrated in the YAML code below.

```yaml
preprocessing:
   rescaling: {scale : 1/127.5, offset : -1}
   resizing: {interpolation: bilinear, 
               aspect_ratio: "fit"}
   color_mode: rgb
```

The pixels of the input images are in the interval [0, 255], that is UINT8. If you set `scale` to 1./255 and `offset` to 0, they will be rescaled to the interval [0.0, 1.0]. If you set *scale* to 1/127.5 and *offset* to -1, they will be rescaled to the interval [-1.0, 1.0].

The `resizing` attribute specifies the image resizing methods you want to use:
- The value of `interpolation` must be one of *{"bilinear", "nearest", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"}*.
- The value of `aspect_ratio` must be either *"fit"* or *"crop"*. If you set it to *"fit"*, the resized images will be distorted if original aspect ratio is not the same as in the resizing size. If you set it to *"crop"*, images will be cropped as necessary to preserve the aspect ratio.

The `color_mode` attribute must be one of "*grayscale*", "*rgb*" or "*rgba*".
</details></ul>

<ul><details open><summary><a href="#2.6">2.6 Data augmentation</a></summary><a id="2.6"></a>

Data augmentation is an effective technique to reduce the overfit of a model when the dataset is too small or the semantic segmentation problem to solve is too easy for the model.

The data augmentation functions to apply to the input images are specified in the `data_augmentation` section of the configuration file file as illustrated in the YAML code below.

```yaml
data_augmentation:   
  random_contrast:
    factor: 0.4
    change_rate: 1.0
  random_posterize:
    bits: (4, 8)
    change_rate: 0.025
  random_brightness:
    factor: 0.05
    change_rate: 1.0
```

The data augmentation functions with their parameter settings are applied to the input images in their order of appearance in the configuration file. 
Refer to the data augmentation documentation **[README.md](./README_DATA_AUGMENTATION.md)** for more information about the available functions and their arguments.
</details></ul>

<ul><details open><summary><a href="#2.7">2.7 Training section</a></summary><a id="2.7"></a>

The training setup is described in the `training` section of the configuration file, as illustrated in the example below.

```yaml
training:
  dropout: 0.6
  batch_size: 16
  epochs: 1
  optimizer:
    Adam:
      learning_rate: 0.005
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

The `batch_size`, `epochs` and `optimizer` attributes are mandatory. All the others are optional.

The `dropout` attribute only makes sense if your model includes a dropout layer. 

All the Tensorflow optimizers can be used in the `optimizer` subsection. All the Tensorflow callbacks can be used in the `callbacks` subsection, except the ModelCheckpoint and TensorBoard callbacks that are built-in and can't be redefined.

A number of learning rate schedulers are provided with the Model Zoo as custom callbacks. The YAML code below shows how to use the LRCosineDecay scheduler that implements a cosine decay function.

```yaml
training:
   batch_size: 64
   epochs: 400
   optimizer: Adam
   callbacks:
      LRCosineDecay:
         initial_learning_rate: 0.01
         decay_steps: 170
         alpha: 0.001
```
Refer to [Appendix A: Learning rate schedulers](#A) for a list of the available learning rate schedulers.
</details></ul>

<ul><details open><summary><a href="#2.8">2.8 Hydra and MLflow settings</a></summary><a id="2.8"></a>

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

To launch your model training using a real dataset, run the following command from UC folder:
```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config.yaml
```
Trained h5 model can be found in corresponding **experiments_outputs/** folder.
</details>

<details open><summary><a href="#4"><b>4. Visualize training results</b></a></summary><a id="4"></a>
<ul><details open><summary><a href="#4-1">4.1  Saved results</a></summary><a id="4-1"></a>

All training and evaluation artifacts are saved under the current output simulation directory **"outputs/{run_time}"**.

</details></ul>
<ul><details open><summary><a href="#4-2">4.2  Run tensorboard</a></summary><a id="4-2"></a>

To visualize the training curves logged by tensorboard, go to **"outputs/{run_time}"** and run the following command:

```bash
tensorboard --logdir logs
```

Then open the URL `http://localhost:6006` in your browser.
</details></ul>
<ul><details open><summary><a href="#4-3">4.3  Run MLFlow</a></summary><a id="4-3"></a>

MLflow is an API for logging parameters, code versions, metrics, and artifacts while running machine learning code and for visualizing results.
To view and examine the results of multiple trainings, you can simply access the MLFlow Webapp by running the following command:
```bash
mlflow ui
```
Then open the given IP address in your browser.
</details></ul>
</details>

<details open><summary><a href="#5"><b>5. Advanced settings</b></a></summary><a id="5"></a>

<ul><details open><summary><a href="#5.1">5.1 Training your own model</a></summary><a id="5.1"></a>

You may want to train your own model rather than a model from the Model Zoo.

This can be done using the `model_path` attribute in the top-level `model` section to provide the path to the model file to use, as illustrated in the example below:

```yaml
model:
  model_path: <path-to-a-Keras-model-file>    # Path to the model file to use for training

operation_mode: training

dataset:
  dataset_name: <your_dataset_name>
  class_names: [<list_of_class_names>]
  training_path: <path_to_training_images>
  training_masks_path: <path_to_training_masks>
  training_files_path: <path_to_training_filelist>
  validation_split: 0.2

training:
  batch_size: 64
  epochs: 150
  dropout: 0.3
  frozen_layers: (0, -1)
  optimizer:
    Adam:
      learning_rate: 0.001
  callbacks:
    ReduceLROnPlateau:
      monitor: val_loss
      factor: 0.1
      patience: 10
```


The model file must be in Keras format with a '.keras' filename extension.

About the model loaded from the file:
- Any frozen layers in the loaded model will be reset to trainable before training. To freeze specific layers, use the `frozen_layers` attribute.
- An error will occur if you set the `dropout` attribute but the model does not include a dropout layer, or if the model includes a dropout layer but the `dropout` attribute is not set.
- The optimizer state will not be preserved, as the model is recompiled before training.
- Several learning rate schedulers are provided with the Model Zoo. If you want to use one of them, just include it in the `callbacks` subsection. See [the learning rate schedulers README](../../common/training/lr_schedulers_README.md) for a description of the available callbacks and learning rate plotting utility.



<ul><details open><summary><a href="#5.2">5.2 Transfer learning</a></summary><a id="5.2"></a>

Transfer learning is a popular training methodology that is used to take advantage of models trained on large datasets, such as ImageNet. The Model Zoo features that are available to implement transfer training are presented in the next sections.

Transfer learning is a popular training methodology that is used to take advantage of models trained on large datasets, such as ImageNet.
The Model Zoo features that are available to implement transfer training are presented in the next sections.
<ul><details open><summary><a href="#5.2.1">5.2.1 Using ImageNet pretrained weights</a></summary><a id="5.2.1"></a>

Weights pretrained on the ImageNet dataset are available for the MobileNetV2 backbone. To use them, set `pretrained_weights: True` in the top-level `model` section. This boolean automatically means ImageNet weights for MobileNetV2.

```yaml
model:
  model_name: deeplabv3_mnv2_a050_s16_asppv1
  input_shape: (height, width, channels)
  pretrained_weights: True
```


By default, no pretrained weights are loaded. If you want to make it explicit that you are not using the ImageNet weights, you may add the `pretrained_weights` attribute and leave it unset or set to *null*.
</details></ul>

<ul><details open><summary><a href="#5.2.2">5.2.2 Using weights from another model</a></summary><a id="5.2.2"></a>

To use weights from another model, set the `model_path` in the top-level `model` section to the path of your saved model file. The specified model will be loaded for training or fine-tuning.

```yaml
model:
  model_path: <path-to-a-Keras-model-file>
```

You may want to resume a training that you interrupted or that crashed. To resume a training, you first need to choose the experiment you want to restart from. Then, set the `model_path` attribute of the 'model' section to the path to the 'last_model.keras' file of the experiment. 
</details></ul>

<ul><details open><summary><a href="#5.2.3">5.2.3 Freezing layers</a></summary><a id="5.2.3"></a>

By default, all layers are trainable. If you want to freeze some layers, add the optional `frozen_layers` attribute to the `training:` section of your configuration file. The indices of the layers to freeze are specified using Python list/array indexing syntax. Below are some examples.


```yaml
training:
   frozen_layers: (0:-1)    # Freeze all the layers but the last one
   
training:
   frozen_layers: (10:120)   # Freeze layers with indices from 10 to 119

training:
   frozen_layers: (150:)     # Freeze layers from index 150 to the last layer

training:
   frozen_layers: (8, 110:121, -1)  # Freeze layers with index 8, 110 to 120, and the last layer
```

To explicitly indicate that all layers are trainable, add the `frozen_layers` attribute and leave it unset or set to *None*.
</details></ul>

<ul><details open><summary><a href="#5.2.4">5.2.4 Multi-step training</a></summary><a id="5.2.4"></a>

In some cases, better results may be obtained using multiple training steps.

The first training step is generally done with only a few trainable layers, typically the head only. Then, more layers are made trainable in subsequent steps. Other parameters, such as the learning rate, may also be adjusted from one step to another. Therefore, a different configuration file is needed at each step.


The `model_path` attribute in the top-level `model` section should be used for multi-step training. At each step, set `model_path` to the model file produced by the previous step. The newly trained model will be saved automatically according to your configuration.

Assume, for example, that you are doing a 3-step training. Your 3 configurations would look as shown below.

**Training step #1 configuration file (initial training):**

```yaml
model:
  model_type: deeplab
  model_name: st_deeplabv3_mnv2_a050_s16_asppv2
  input_shape: (128, 128, 3)
  pretrained: True
training:
  frozen_layers: (0:-1)
```

**Training step #2 configuration file:**

```yaml
model:
  model_path: ${MODELS_DIR}/step_1.keras
training:
  frozen_layers: (50:)
```

**Training step #3 configuration file:**

```yaml
model:
  model_path: ${MODELS_DIR}/step_2.keras
training:
  frozen_layers: None
```
</details></ul>

<ul><details open><summary><a href="#5.3">5.3 Creating your own custom model</a></summary><a id="5.3"></a>

You can create your own custom model and use it like any built-in Model Zoo model. To do this, you need to modify several Python source code files located under the *<MODEL-ZOO-ROOT>/semantic_segmentation/src* directory root.

An example custom model is given in **models/custom_model.py** located in *<MODEL-ZOO-ROOT>/semantic_segmentation/src/models/*. The model is constructed in the body of the *get_custom_model()* function, which returns the model. Modify this function to implement your own model.

In the provided example, the *get_custom_model()* function takes the following arguments:
- `num_classes`: the number of classes.
- `input_shape`: the input shape of the model.
- `dropout`: the dropout rate if a dropout layer must be included in the model.

When adding new arguments to the *get_custom_model()* function (e.g., `alpha` as a float), ensure you:
- Add them to the function’s signature,
- Update the `prepare_kwargs_for_model` utility to handle the new arguments,
- Include the parameters in your YAML configuration file.

This ensures that your custom arguments are correctly passed from the configuration file to your model implementation.

Then, your custom model can be used as any other Model Zoo model using the configuration file as shown in the YAML code below:
```yaml

model:
   model_name: custom_model
   alpha: 0.5       # The argument you added to get_custom_model().
   input_shape: (128, 128, 3)
training: 
   dropout: 0.2
```
</details></ul>

<ul><details open><summary><a href="#5.4">5.4 Train, quantize, benchmark and evaluate your model</a></summary><a id="5.4"></a>


If you want to train and quantize a model, you can either launch the training operation mode followed by the quantization operation on the trained model (see **[README.md](./README_QUANTIZATION.md)** for details on quantization), or you can use chained services such as [chain_tqe](../config_file_examples/chain_tqe_config.yaml) with the command below:
```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_tqe_config.yaml
```
This example trains a MobileNetV2 model with ImageNet pre-trained weights, fine-tunes it by retraining the last seven layers except the fifth one (as an example), and quantizes it to 8 bits using a quantization_split (30% in this example) of the training dataset for calibration before evaluating the quantized model.

If you also want to execute a benchmark in addition to training and quantization, it is recommended to launch the chain service called [chain_tqeb](../config_file_examples/chain_tqeb_config.yaml), which stands for train, quantize, evaluate, benchmark, using the command below:
```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_tqeb_config.yaml
```

</details></ul>
</details>

