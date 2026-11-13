# <a id="">Pose estimation STM32 model training</a>

This readme shows how to train from scratch or apply transfer learning on a single pose estimation model using a custom
dataset.
As an example, we will demonstrate the workflow on the COCO 2017 single pose dataset, which can be downloaded
on the [COCO 2017](https://cocodataset.org/#download) website.

## <a id="">Table of contents</a>

<details open><summary><a href="#1"><b>1. Prepare the dataset</b></a></summary><a id="1"></a>

First download the [COCO2017 images training dataset](http://images.cocodataset.org/zips/train2017.zip), the [COCO2017 images validation dataset](http://images.cocodataset.org/zips/val2017.zip) and the [COCO2017 annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (be carefull, the size of this dataset is approximatly 20GB total and you will need at least 10GB more to create the formated pose dataset, so 30GB total).

Unzip all of them in the same folder, you should have : train, val and annotations subfolders now.

Finally format this raw dataset using this [tutorial](./README_DATASETS_CONVERTER.md).


After all of this, the dataset directory tree should look like below (the folders will exist or not depending on the parameters you putted in the dataset converter script):

```yaml
<dataset-root-directory>/
  single/
    17kpts/
      train/
        train_image_1.jpg
        train_image_1.txt
        train_image_2.jpg
        train_image_2.txt
      val/
        val_image_1.jpg
        val_image_1.txt
        val_image_2.jpg
        val_image_2.txt
```

Please note that training and evaluation are only possible with datasets containing images (jpgs or PNGs) + txt files in the YOLO Darknet format, other formats are not
compatible.

</details>
<details open><summary><a href="#2"><b>2. Create your training configuration file</b></a></summary><a id="2"></a>
<ul><details open><summary><a href="#2-1">2.1 Overview</a></summary><a id="2-1"></a>

All the proposed services like the training of the model are driven by a configuration file written in the YAML
language.

For training, the configuration file should include at least the following sections:

- `general`, describes your project, including project name, directory where to save models, etc.
- `model`, describes the model to use, model path, model type, etc.
- `operation_mode`, describes the service or chained services to be used
- `dataset`, describes the dataset you are using, including directory paths, class names, etc.
- `preprocessing`, specifies the methods you want to use for rescaling and resizing the images.
- `training`, specifies your training setup, including batch size, number of epochs, optimizer, callbacks, etc.
- `mlflow`, specifies the folder to save MLFlow logs.
- `hydra`, specifies the folder to save Hydra logs.

This tutorial only describes the settings needed to train a model. In the first part, we describe basic settings.
At the end of this readme, you can also find more advanced settings and callbacks supported.

</details></ul>
<ul><details open><summary><a href="#2-2">2.2 General settings</a></summary><a id="2-2"></a>

The first section of the configuration file is the `general` section that provides information about your project.

```yaml
general:
  project_name: COCO_2017_pose_Demo # Project name. Optional, defaults to "<unnamed>".
  logs_dir: logs                    # Name of the directory where log files are saved. Optional, defaults to "logs".
  saved_models_dir: saved_models    # Name of the directory where model files are saved. Optional, defaults to "saved_models".
  deterministic_ops: False          # Enable/disable deterministic operations (a boolean). Optional, defaults to False.
  num_threads_tflite: 8             # Number of threads used on the CPU to run the TFLITE models
  gpu_memory_limit: 8               # Maximum amount of GPU memory in GBytes that TensorFlow may use (an integer).
  global_seed: 123                  # Seed used to seed random generators (an integer). Optional, defaults to 123.
```

If you want your experiments to be fully reproducible, you need to activate the `deterministic_ops` attribute and set it
to True.
Enabling the `deterministic_ops` attribute will restrict TensorFlow to use only deterministic operations on the device,
but it may lead to a drop in training performance. It should be noted that not all operations in the used version of
TensorFlow can be computed deterministically.
If your case involves any such operation, a warning message will be displayed and the attribute will be ignored.

The `logs_dir` attribute is the name of the directory where the MLFlow and TensorBoard files are saved.

The `saved_models_dir` attribute is the name of the directory where models are saved, which includes the trained model
and the quantized model. These two directories are located under the top level <hydra> directory.

The `global_seed` attribute specifies the value of the seed to use to seed the Python, numpy and Tensorflow random
generators at the beginning of the main script. This is an optional attribute, the default value being 123. If you don't
want random generators to be seeded, then set `global_seed` to 'None' (not recommended as this would make training
results less reproducible).

The `num_threads_tflite` parameter is only used as an input parameter for the tflite interpreter. Therefore, it has no effect on .keras or .onnx models. 
This parameter may accelerate the tflite model evaluation in the following operation modes:  `evaluation` (if a .tflite is specified in `model_path`), 
`chain_eqe`, and `chain_eqeb` (if the quantizer is the TFlite_converter). 
However, the acceleration depends on your system resources.

The `gpu_memory_limit` attribute sets an upper limit in GBytes on the amount of GPU memory Tensorflow may use. This is
an optional attribute with no default value. If it is not present, memory usage is unlimited. If you have several GPUs,
be aware that the limit is only set on logical gpu[0].

</details></ul>
<ul><details open><summary><a href="#2-3">2.3 Model specification</a></summary><a id="2-3"></a>

```yaml
model:
   model_name: st_movenet_lightning_a100_heatmaps  # use 'st_movenet_lightning_heatmaps' if you want to specify a different 'alpha' parameter
   # alpha : 0.35                      # use only if model_name = 'st_movenet_lightning_heatmaps' to set alpha parameter
   # model_path: <file-path>           # Path to a .keras file to use an already trained model.
   model_type: heatmaps_spe            # only heatmaps_spe is supported for training
   input_shape: (192, 192, 3)          # recommended : (192,192,3), (224,224,3), (256,256,3)
   pretrained: True                    # True : backbone weights will be initialized with pretrained ones | False : random initialized backbone weights
   final_activation: softmax
```

The `model_name` attribute is used to load a known architecture, the possible .

The `alpha` attribute is a parameter of mobilenetv2 backbone (use only if `model_name : st_movenet_lightning_heatmaps`)

The `model_path` attribute is used to specify the path to the model file that you want to fine-tune or resume training
from. This parameter is essential for loading the pre-trained weights of the model and continuing the training process.
By providing the path to the model file, you can easily fine-tune the model on a new dataset or resume training from a
previous checkpoint. This allows you to leverage the knowledge learned by the model on a previous task and apply it to a
new problem, saving time and resources in the process.

The `model_type` attribute specifies the type of the model architecture that you want to train. The only types accepted for training are the following :

- `heatmaps_spe`: These are single pose estimation models that outputs heatmaps that we must post-process in order to get the keypoints positions and confidences.

It is important to note that each model type has specific requirements in terms of input image size, output size of the
head and/or backbone, and other parameters. Therefore, it is important to choose the appropriate model type for your
specific use case, and to configure the training process accordingly.


The `pretrained` attribute is set to `True`, which indicates that the model will load weights pretrained on the ImageNet dataset for the backbone only. This is a common practice in transfer learning, where the model is initialized with weights learned from a large dataset and then fine-tuned on a smaller, task-specific dataset. Note that the weights form the head of the model (all layers added on top of the backbone) will be initialized randomly.

If `pretrained` was set to `False`, no backbone pretrained weights would be loaded in the model and the training would
start *from scratch*, i.e. from randomly initialized weights.

The `final_activation` attribute specifies the type of final actication that you want use at the output of the model. The only types accepted for training are the following :
- `softmax`: ST custom 2D softmax (yields good metrics)
- `sigmoid`: classical activation (yields average metrics)
- `empty`: can be usefull if your loss function already contains an activation

</details></ul>
<ul><details open><summary><a href="#2-4">2.4 Dataset specification</a></summary><a id="2-4"></a>

Information about the dataset you want use is provided in the `dataset` section of the configuration file, as shown in
the YAML code below.

```yaml
dataset:
  dataset_name: coco                                         # Dataset name/type
  keypoints: 17                                              # Number of keypoints per poses
  training_path: <training-set-root-directory>               # Path to the root directory of the training set.
  validation_path: <validation-set-root-directory>           # Path to the root directory of the validation set.
  validation_split: 0.2                                      # Training/validation sets split ratio.
  test_path: <test-set-root-directory>                       # Path to the root directory of the test set.
  quantization_path: <quantization-set-root-directory>       # Path to the root directory of the quantization set.
  quantization_split:                                        # Quantization split ratio.
  seed: 123                                                  # Random generator seed used when splitting a dataset.
```

> [!IMPORTANT]
> The `dataset_name` is mandatory and should always be set to 'coco' even if you dont use the COCO dataset

The `keypoints` attribute is mandatory to correctly parse the dataset.

When `training_path` is set, the training set is splited in two to create a validation dataset if `validation_path` is not
provided. When a model accuracy evaluation is run, the `test_path` is used if there is one, otherwise the `validation_path` is
used (either provided or generated by splitting the training set).

The `validation_split` attribute specifies the training/validation set size ratio to use when splitting the training set
to create a validation set. The default value is 0.2, meaning that 20% of the training set is used to create the
validation set. The `seed` attribute specifies the seed value to use for randomly shuffling the dataset file before
splitting it (default value is 123).

The `quantization_path` attribute is used to specify a dataset for the quantization process. If this attribute is not
provided and a training set is available, the training set is used for the quantization. However, training sets can be
quite large and the quantization process can take a long time to run. To avoid this issue, you can set
the `quantization_split` attribute to use only a portion of the dataset for quantization.

</details></ul>
<ul><details open><summary><a href="#2-5">2.5 Dataset preprocessing</a></summary><a id="2-5"></a>

The images from the dataset need to be preprocessed before they are presented to the network. This includes rescaling
and resizing, as illustrated in the YAML code below.

```yaml
preprocessing:
  rescaling:
    # Image rescaling parameters
    scale: 1/127.5
    offset: -1
  resizing:
    # Image resizing parameters
    interpolation: nearest
    aspect_ratio: fit
  color_mode: rgb
```

Images are rescaled using the formula "Out = scale\*In + offset". Pixel values of input images usually are integers in
the interval [0, 255]. If you set *scale* to 1./255 and offset to 0, pixel values are rescaled to the
interval [0.0, 1.0]. If you set *scale* to 1/127.5 and *offset* to -1, they are rescaled to the interval [-1.0, 1.0].

The `resizing` attribute specifies the image resizing methods you want to use:

- The value of `interpolation` must be one of *{"bilinear", "nearest", "bicubic", "area", "lanczos3", "lanczos5", "
  gaussian", "mitchellcubic"}*.
- The value of `aspect_ratio` must be *"fit"* or *"padding"*. When using this option,
the images will be resized to fit the target size. It is important to note that input images may be smaller or larger
than the target size, and will be distorted (*"fit"*) or paddded (*"padding"*) to some extent if their original aspect ratio is not the same as the
resizing aspect ratio. Additionally, bounding boxes and keypoints should be adjusted to maintain their relative positions and sizes in
the resized images.

The `color_mode` attribute can be set to either *"grayscale"*, *"rgb"* or *"rgba"*.

</details></ul>
<ul><details open><summary><a href="#2-6">2.6 Data augmentation</a></summary><a id="2-6"></a>

Data augmentation is a crucial technique for improving the performance of pose estimation models, especially when the
dataset is too small. The data_augmentation section of the configuration file specifies the data augmentation functions
to be applied to the input images, such as rotation, flipping, blurring, ... . Each function can be
customized with specific parameter settings to control the degree and type of augmentation applied to the input images.

The data augmentation functions to apply to the input images are specified in the `data_augmentation` section of the
configuration file as illustrated in the YAML code below.

```yaml
data_augmentation:
  random_rotation:
    factor: (-0.2,0.2) # -+0.1 = -+36 degree angle
    fill_mode: constant # constant, wrap
    fill_value: -1.
  random_periodic_resizing:
    image_sizes: [[192,192],[224,224],[256,256]]
    period: 10
  random_contrast:
    factor: 0.4
  random_brightness:
    factor: 0.3
  random_flip:
    mode: horizontal
```

It is important to note that these augmentations are done order, which means that if you put in the config file 'rotation' before 'brigthness', parts of the image that have been filled with pure grey during the 'rotation' phase will be modified by 'brigthness' augmentation.

This would not have been the case if the 'rotation' happend after the 'brigthness'.

</details></ul>
<ul><details open><summary><a href="#2-7">2.7 Training setup</a></summary><a id="2-7"></a>

The training setup is described in the `training` section of the configuration file, as illustrated in the example
below.

```yaml
training:
  bach_size: 64
  epochs: 1000
  frozen_layers: (0:154)   # Make layers until the 154th non-trainable, if you replace 154 by -1 -> only the last layer is trainable
  optimizer:
    # Use Keras Adam optimizer with initial LR set to 0.001
    Adam:
      learning_rate: 0.01
  callbacks:
    # Use Keras ReduceLROnPlateau learning rate scheduler
    ReduceLROnPlateau:
      monitor: val_oks
      mode: max
      min_delta: 0.0001
      patience: 10
    # Use Keras EarlyStopping to stop training and not overfit
    EarlyStopping:
      monitor: val_oks
      mode: max
      min_delta: 0.0001
      patience: 20
    ModelCheckpoint:
      monitor: val_oks
      mode: max
```

The `batch_size`, `epochs` and `optimizer` attributes are mandatory. All the others are optional.

All the Tensorflow optimizers can be used in the `optimizer` subsection. All the Tensorflow callbacks can be used in
the `callbacks` subsection, except the ModelCheckpoint and TensorBoard callbacks that are built-in and can't be
redefined.

A variety of learning rate schedulers are provided with the Model Zoo. If you want to use one of them, just include it in the `callbacks` subsection. Refer to [the learning rate schedulers README](../../common/training/lr_schedulers_README.md) for a description of the available callbacks and learning rate plotting utility.

</details></ul>
<ul><details open><summary><a href="#2-8">2.8 Hydra and MLflow settings</a></summary><a id="2-8"></a>

The `mlflow` and `hydra` sections must always be present in the YAML configuration file. The `hydra` section can be used to specify the name of the directory where experiment directories are saved and/or the pattern used to name experiment directories. With the YAML code below, every time you run the Model Zoo, an experiment directory is created that contains all the directories and files created during the run. The names of experiment directories are all unique as they are based on the date and time of the run.

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
<details open><summary><a href="#3"><b>3. Train your model</b></a></summary><a id="3"></a>

To launch your model training using a real dataset, run the following command from UC folder:

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config.yaml
```

Trained keras model can be found in corresponding **experiments_outputs/** folder.

</details>
<details open><summary><a href="#4"><b>4. Visualise your results</b></a></summary><a id="4"></a>
<ul><details open><summary><a href="#4-1">4.1 Saved results</a></summary><a id="4-1"></a>

All training and evaluation artifacts are saved under the current output simulation directory **"outputs/{run_time}"**.

</details></ul>
<ul><details open><summary><a href="#4-2">4.2 Run tensorboard</a></summary><a id="4-2"></a>

To visualize the training curves logged by tensorboard, go to **"outputs/{run_time}"** and run the following command:

```bash
tensorboard --logdir logs
```

And open the URL `http://localhost:6006` in your browser.

</details></ul>
<ul><details open><summary><a href="#4-3">4.3 Run MLFlow</a></summary><a id="4-3"></a>

MLflow is an API for logging parameters, code versions, metrics, and artifacts while running machine learning code and for visualizing results.
To view and examine the results of multiple trainings, you can simply access the MLFlow Webapp by running the following command:

```bash
mlflow ui
```

And open the given IP adress in your browser.

</details></ul>
</details>
<details open><summary><a href="#5"><b>5. Advanced settings</b></a></summary><a id="5"></a>
<ul><details open><summary><a href="#5-1">5.1 Train, quantize, benchmark and evaluate your models</a></summary><a id="5-1"></a>

In case you want to train and quantize a model, you can either launch the training operation mode followed by the quantization operation on the trained model (please refer to quantization **[README.md](./README_QUANTIZATION.md)** that describes in details the quantization part) or you can use chained services like launching chain_tqe service.

This specific example trains a mobilenet v2 model with imagenet pre-trained weights, fine tunes it by retraining latest seven layers but the fifth one (this only as an example), and quantizes it 8-bits using quantization_split (30% in this example) of the train dataset for calibration before evaluating the quantized model.

In case you also want to execute a benchmark on top of training and quantize services, it is recommended to launch the chain service called chain_tqeb that stands for train, quantize, evaluate, benchmark.
Else simple training could be executed like in example below:

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config.yaml
```

This specific example uses the "Bring Your Own Model" feature using `model_path`, then fine tunes the initial model by retraining all the layers but the twenty first (as an example), benchmarks the float model STM32H747I-DISCO board using the STM32Cube.AI developer cloud, quantizes it 8-bits using quantization_split (30% in this example) of the train dataset for calibration before evaluating the quantized model and benchmarking it.

</details></ul>
</details>

