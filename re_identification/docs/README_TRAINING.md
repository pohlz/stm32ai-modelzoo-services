# Re-Identification STM32 Model Training

This README shows how to train from scratch or apply transfer learning on an image classification model using a custom dataset. As an example, we will demonstrate the workflow on the [DeepSportradar](https://github.com/DeepSportradar/player-reidentification-challenge) reid dataset.

A ReID model is typically based on a classification architecture during training, where a classification head is added on top of a backbone network to identify different classes (identities). However, once the model is trained, the classification head is removed, and the model is used solely as a feature extractor. This allows the ReID model to generate discriminative feature embeddings that represent individual identities, enabling comparison and matching of identities across different images without relying on fixed class labels.

<details open><summary><a href="#1"><b>1. Prepare the dataset</b></a></summary><a id="1"></a>

After downloading and extracting the dataset files, the dataset directory tree should look as below:

```bash
dataset_root_dir/
    id1_1.jpg
    id1_2.jpg
    id2_1.jpg
    id2_2.jpg
```
The names of the image files before the first underscore indicate the identity of the person.

Other dataset formats are not supported.

</details>

<details open>
<summary><a href="#2"><b>2. Create your training configuration file</b></a></summary><a id="2"></a>
<ul>
<details open><summary><a href="#2-1">2.1 Overview</a></summary><a id="2-1"></a>

All the proposed services like the training of the model are driven by a configuration file written in the YAML language.

For training, the configuration file should include at least the following sections:

- `general`, describes your project, including project name, directory where to save models, etc.
- `model`, describes the model to train, including model name, input shape, whether to use pretrained weights, etc.
- `operation_mode`, describes the service or chained services to be used.
- `dataset`, describes the dataset you are using, including directory paths, etc.
- `preprocessing`, specifies the methods you want to use for rescaling and resizing the images.
- `data_augmentation`, specifies the data augmentation functions you want to apply to the input images during training.
- `training`, specifies your training setup, including batch size, number of epochs, optimizer, callbacks, etc.
- `mlflow`, specifies the folder to save MLFlow logs.
- `hydra`, specifies the folder to save Hydra logs.

This tutorial only describes the settings needed to train a model. In the first part, we describe basic settings. At the end of this README, you can also find more advanced settings and callbacks supported.

</details>

<details open><summary><a href="#2-2">2.2 General settings</a></summary><a id="2-2"></a>

The first section of the configuration file is the `general` section that provides information about your project.

```yaml
general:
   project_name: my_project
   logs_dir: logs
   saved_models_dir: saved_models
   deterministic_ops: True
```
If you want your experiments to be fully reproducible, you need to activate the `deterministic_ops` attribute and set it to True. Enabling the `deterministic_ops` attribute will restrict TensorFlow to use only deterministic operations on the device, but it may lead to a drop in training performance. It should be noted that not all operations in the used version of TensorFlow can be computed deterministically. If your case involves any such operation, a warning message will be displayed and the attribute will be ignored.

The `logs_dir` attribute is the name of the directory where the MLFlow and TensorBoard files are saved. The `saved_models_dir` attribute is the name of the directory where trained models are saved. These two directories are located under the top-level "hydra" directory (please see [chapter 2.8](#2-8) for Hydra information).

</details>

<details open><summary><a href="#2-3">2.3 Dataset specification</a></summary><a id="2-3"></a>

Information about the dataset you want to use is provided in the `dataset` section of the configuration file, as shown in the YAML code below.

```yaml
dataset:
   dataset_name: DeepSportradar
   training_path: ./datasets/DeepSportradar-ReID/reid_train
   validation_path:
   validation_split: 0.15
   test_query_path: ./datasets/DeepSportradar-ReID/reid_test/query
   test_gallery_path: ./datasets/DeepSportradar-ReID/reid_test/gallery
```
The state machine below describes the rules to follow when handling dataset paths for the training.
<div align="center" style="width:50%; margin: auto;">

![plot](../../common/doc/img/state_machine_training.JPG)
</div>

In this example, no validation set path is provided, so the available data under the *training_path* directory is split in two to create a training set and a validation set. By default, 80% of the data is used for the training set and the remaining 20% is used for the validation set. If you want to use a different split ratio, you need to specify in `validation_split` the ratio to be used for the validation set (value between 0 and 1).

Once the image classification model is trained, it is evaluated on the test set. In re-identification use cases, the test set is made of two subsets: the query set and the gallery set. The query set contains images whose identities are to be predicted. The gallery set contains images with known identities that are used as references for the prediction. Therefore, both the `test_query_path` and `test_gallery_path` attributes must be provided to specify the paths to the query and gallery image folders, respectively.

</details>

<details open><summary><a href="#2-4">2.4 Dataset preprocessing</a></summary><a id="2-4"></a>

The images from the dataset need to be preprocessed before they are presented to the network. This includes rescaling and resizing, as illustrated in the YAML code below.

```yaml
preprocessing:
   rescaling: {scale: 1/127.5, offset: -1}
   resizing: {interpolation: nearest, aspect_ratio: "fit"}
   color_mode: rgb
```

The pixels of the input images are in the interval [0, 255], that is UINT8. If you set `scale` to 1./255 and `offset` to 0, they will be rescaled to the interval [0.0, 1.0]. If you set `scale` to 1/127.5 and `offset` to -1, they will be rescaled to the interval [-1.0, 1.0].

The `resizing` attribute specifies the image resizing methods you want to use:
- The value of `interpolation` must be one of *{"bilinear", "nearest", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"}*.
- The value of `aspect_ratio` must be either *"fit"* or *"crop"*. If you set it to *"fit"*, the resized images will be distorted if their original aspect ratio is not the same as the resizing size. If you set it to *"crop"*, images will be cropped as necessary to preserve the aspect ratio.

The `color_mode` attribute must be one of "*grayscale*", "*rgb*" or "*rgba*".

</details>

<details open><summary><a href="#2-5">2.5 Data augmentation</a></summary><a id="2-5"></a>

Data augmentation is an effective technique to reduce the overfitting of a model when the dataset is too small or the classification problem to solve is too easy for the model.

The data augmentation functions to apply to the input images are specified in the `data_augmentation` section of the configuration file, as illustrated in the YAML code below.

```yaml
data_augmentation:
  random_contrast:
    factor: 0.4
  random_brightness:
    factor: 0.2
  random_flip:
    mode: horizontal_and_vertical
  random_translation:
    width_factor: 0.2
    height_factor: 0.2
  random_rotation:
    factor: 0.15
  random_zoom:
    width_factor: 0.25
    height_factor: 0.25
```

The data augmentation functions with their parameter settings are applied to the input images in their order of appearance in the configuration file. Refer to the data augmentation documentation **[README.md](../../common/data_augmentation/README.md)** for more information about the available functions and their arguments.

A script called *test_data_augment.py* is available in the **data_augmentation** directory. This script reads your configuration file, picks some images from the dataset, applies the data augmentation functions you specified to the images, and displays before/after images side by side. We strongly encourage you to run this script to develop your data augmentation and make sure that it is neither too aggressive nor too weak.

</details>

<details open><summary><a href="#2-6">2.6 Loading a model</a></summary><a id="2-6"></a>

Information about the model you want to train is provided in the `model` section of the configuration file.

The YAML code below shows how you can use a MobileNet V2 model from the Model Zoo.

```yaml
model:
   model_name: mobilenetv2_a035
   pretrained: True
   input_shape: (256, 128, 3)
```

The `pretrained` attribute is set to "True", which indicates that you want to load the weights pretrained on the imagenet dataset and do a *transfer learning* type of training.

If `pretrained` was set to "False", no pretrained weights would be loaded in the model and the training would start *from scratch*, i.e., from randomly initialized weights.

</details>

<details open><summary><a href="#2-7">2.7 Training setup</a></summary><a id="2-7"></a>

The training setup is described in the `training` section of the configuration file, as illustrated in the example below.

```yaml
training:
   batch_size: 128
   epochs: 400
   dropout: 0.3
   optimizer: 
      Adam: {learning_rate: 0.001}
   callbacks:
      ReduceLROnPlateau:
         monitor: val_accuracy
         factor: 0.5
         patience: 10
      EarlyStopping:
         monitor: val_accuracy
         patience: 60
    triplet_loss:
        margin: 0.3  # Optional between [0, 1], default is 0.3
        strategy: semi_hard   #choices=['hard', 'semi_hard', 'simple'], default is 'hard'
        distance_metric: cosine  # Optional, choices=['euclidean', 'cosine'], default is 'cosine'
```

The `batch_size`, `epochs`, and `optimizer` attributes are mandatory. All the others are optional.

The `dropout` attribute only makes sense if your model includes a dropout layer.

All the TensorFlow optimizers can be used in the `optimizer` subsection. All the TensorFlow callbacks can be used in the `callbacks` subsection, except the ModelCheckpoint and TensorBoard callbacks that are built-in and can't be redefined.

A number of learning rate schedulers are provided with the Model Zoo as custom callbacks. The YAML code below shows how to use the LRCosineDecay scheduler that implements a cosine decay function.

```yaml
training:
   batch_size: 128
   epochs: 400
   optimizer: Adam
   callbacks:
      LRCosineDecay:
         initial_learning_rate: 0.01
         decay_steps: 170
         alpha: 0.001
```

A variety of learning rate schedulers are provided with the Model Zoo. If you want to use one of them, just include it in the `callbacks` subsection. Refer to [the learning rate schedulers README](../../common/training/lr_schedulers_README.md) for a description of the available callbacks and learning rate plotting utility.

The `triplet_loss` section controls how the triplet loss function is applied during training of the re-identification model. This loss function is used to make features from the same identity closer together and features from different identities farther apart. It exposes three main parameters:

- `margin`:  
  Sets the margin value used in the triplet loss formulation, which enforces that the distance between an anchor and a negative sample is at least `margin` greater than the distance between the anchor and a positive sample.  
  - Type: floating-point  
  - Optional: yes  
  - Default: `0.3` if not specified.

- `strategy`:  
  Defines the triplet mining strategy, i.e., how anchor–positive–negative triplets are selected from a batch during training. This heavily influences training stability and convergence:  
  - `"hard"`: selects the hardest positive (farthest) and hardest negative (closest) samples for each anchor, generally giving strong gradients but potentially more training instability.  
  - `"semi_hard"`: selects negatives that are harder than the positive but still farther than the positive by less than the margin, often a good trade‑off between convergence speed and stability.  
  - `"simple"`: uses an easier or more random sampling strategy, typically more stable but sometimes slower to converge.  
  - Optional: yes  
  - Default: `"hard"`.

- `distance_metric`:  
  Specifies the distance function used to compute the similarity between feature embeddings within the triplet loss:  
  - `"euclidean"`: uses the Euclidean distance, sensitive to vector magnitude.  
  - `"cosine"`: uses cosine distance between vectors, focusing on angular similarity and often more robust to feature scaling.  
  - Optional: yes  
  - Default: `"cosine"`.

</details>

<details open><summary><a href="#2-8">2.8 Hydra and MLflow settings</a></summary><a id="2-8"></a>

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

</details>
</ul>
</details>

<details open>
<summary><a href="#3"><b>3. Train your model</b></a></summary><a id="3"></a>

To launch your model training using a real dataset, run the following command from the UC folder:

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config.yaml
```
The trained model can be found in the corresponding **experiments_outputs/** folder.

</details>

<details open>
<summary><a href="#4"><b>4. Visualize training results</b></a></summary><a id="4"></a>
<ul>
<details open><summary><a href="#4-1">4.1 Saved results</a></summary><a id="4-1"></a>

All training and evaluation artifacts are saved under the current output simulation directory **"outputs/{run_time}"**.

For example, you can retrieve the plots of the accuracy/loss curves as below:

![plot](./img/Training_curves.png)

</details>

<details open><summary><a href="#4-2">4.2 Run TensorBoard</a></summary><a id="4-2"></a>

To visualize the training curves logged by TensorBoard, go to **"outputs/{run_time}"** and run the following command:

```bash
tensorboard --logdir logs
```

And open the URL `http://localhost:6006` in your browser.

</details>

<details open><summary><a href="#4-3">4.3 Run MLFlow</a></summary><a id="4-3"></a>

MLFlow is an API for logging parameters, code versions, metrics, and artifacts while running machine learning code and for visualizing results. To view and examine the results of multiple trainings, you can simply access the MLFlow Webapp by running the following command:

```bash
mlflow ui
```

And open the given IP address in your browser.

</details>
</ul>
</details>

<details open>
<summary><a href="#5"><b>5. Advanced settings</b></a></summary><a id="5"></a>
<ul>
<details open><summary><a href="#5-1">5.1 Training your own model</a></summary><a id="5-1"></a>

You may want to train your own model rather than a model from the Model Zoo.

This can be done using the `model_path` attribute of the `general:` section to provide the path to the model file to use, as illustrated in the example below.

```yaml
general:

model:
   model_path: <path-to-a-Keras-model-file>    # Path to the model file to use for training

operation_mode: training

dataset:
   training_path: <training-set-root-directory>    # Path to the root directory of the training set.
   validation_split: 0.2                           # Use 20% of the training set to create the validation set.
   test_query_path: <test-query-set-root-directory>        # Path to the root directory of the test set query subset.
   test_gallery_path: <test-gallery-set-root-directory>            # Path to the root directory of the test set gallery subset.

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

The model file must be a Keras model file with a '.keras' filename extension.

The `model:` subsection of the `training:` section is not present as we are not training a model from the Model Zoo. An error will be thrown if it is present when `model_path` is set.

About the model loaded from the file:
- If some layers are frozen in the model, they will be reset to trainable before training. You can use the `frozen_layers` attribute if you want to freeze these layers (or different ones).
- If you set the `dropout` attribute but the model does not include a dropout layer, an error will be thrown. Reciprocally, an error will also occur if the model includes a dropout layer but the `dropout` attribute is not set.
- If the model was trained before, the state of the optimizer won't be preserved as the model is compiled before training.

</details>

<details open><summary><a href="#5-2">5.2 Resuming a training</a></summary><a id="5-2"></a>

You may want to resume a training that you interrupted or that crashed.

When running a training, the model is saved at the end of each epoch in the **'saved_models'** directory that is under the experiment directory (see section "2.2 Output directories and files"). The model file is named 'last_augmented_model.keras'.

To resume a training, you first need to choose the experiment you want to restart from. Then, set the `model_path` attribute of the 'model' section has to be set to True and the model_path attribute of the model section must contain the path to the 'last_augmented_model.keras' file of the experiment. An example is shown below.

```yaml
operation_mode: training

model:
   model_path: <path to the 'last_augmented_model.keras' file of the interrupted/crashed training>

dataset:
   dataset_name: <dataset-name-specification>
   training_path: <training-set-root-directory>
   validation_split: 0.2
   test_query_path: <test-query-set-root-directory>
   test_gallery_path: <test-gallery-set-root-directory>

training:
   batch_size: 64
   epochs: 150      # The number of epochs can be changed for resuming.
   dropout: 0.3 
   frozen_layers: (0:1)
   optimizer:
      Adam:
         learning_rate: 0.001
   callbacks:         
      ReduceLROnPlateau:
         monitor: val_accuracy
         factor: 0.1
         patience: 10
   resume_training: True
```

The configuration file of the training you are resuming should be reused as is, the only exception being the number of epochs. If you make changes to the dropout rate, the frozen layers or the optimizer, they will be ignored and the original settings will be kept. Changes made to the batch size or the callback section will be taken into account. However, they may lead to unexpected results.

The state of the optimizer is saved in the **last_augmented_model.keras** file, so you will restart from where you left off. The model is called 'augmented' because it includes the rescaling and data augmentation preprocessing layers.

There are two other model files in the **saved_models** directory. The one that is called **best_augmented_model.keras** is the best augmented model that was obtained since the beginning of the training. The other one that is called **best_model.keras**, but it does not include the preprocessing layers and cannot be used to resume a training. An error will be thrown if you attempt to do so.

</details>

<details open><summary><a href="#5-3">5.3 Transfer learning</a></summary><a id="5-3"></a>

Transfer learning is a popular training methodology that is used to take advantage of models trained on large datasets, such as imagenet. The Model Zoo features that are available to implement transfer training are presented in the next sections.

<ul>
<details open><summary><a href="#5-3-1">5.3.1 Using imagenet pretrained weights</a></summary><a id="5-3-1"></a>

Weights pretrained on the imagenet dataset are available for the MobileNetv1, MobileNetv2 models, Resnet50v2.

If you want to use imagenet pretrained weights, you need to add the `pretrained` attribute to the `model:` section of the configuration file and set it to 'True', as shown in the YAML code below.

```yaml
model:
   model_name: mobilenetv2_a035
   input_shape: (256, 128, 3)
   pretrained: True
```

By default, no pretrained weights are loaded. If you want to make it explicit that you are not using the imagenet weights, you may add the `pretrained` attribute and set it to `False`.

</details>


<details open><summary><a href="#5-3-2">5.3.2 Freezing layers</a></summary><a id="5-3-2"></a>

Once the pretrained weights have been loaded in the model to train, some layers are often frozen, that is made non-trainable, before training the model. A commonly used approach is to freeze all the layers but the last one, which is the classifier.

By default, all the layers are trainable. If you want to freeze some layers, then you need to add the optional `frozen_layers` attribute to the `training:` section of your configuration file. The indices of the layers to freeze are specified using the Python syntax for indexing into lists and arrays. Below are some examples.

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

Note that if you want to make it explicit that all the layers are trainable, you may add the `frozen_layers` attribute and leave it unset or set to *None*.

</details>

<details open><summary><a href="#5-3-3">5.3.3 Multi-step training</a></summary><a id="5-3-3"></a>

In some cases, better results may be obtained using multiple training steps.

The first training step is generally done with only a few trainable layers, typically the classifier only. Then, more and more layers are made trainable in the subsequent training steps. Some other parameters may also be adjusted from one step to another, in particular the learning rate. Therefore, a different configuration file is needed at each step.

The `model_path` attribute of the `model` section is available to implement such a multi-step training. At a given step, `model_path` is used to load the model that was trained at the previous step.

Assume for example that you are doing a 3 steps training. Then, your 3 configurations would look as shown below.

**Training step #1 configuration file (initial training):**

```yaml

model:
   model_name: mobilenetv2_a035
   input_shape: (256, 128, 3)
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

</details>
</ul>
</details>

<details open><summary><a href="#5-4">5.4 Creating your own custom model</a></summary><a id="5-4"></a>

You can create your own custom model and get it handled as any built-in Model Zoo model. If you want to do that, you need to modify a number of Python source code files that are all located under the */<MODEL-ZOO-ROOT>/re_identification/tf/src* directory root.

An example of a custom model is given in the **models/custom_model.py** located in the */<MODEL-ZOO-ROOT>/re_identification/tf/src/models/*. The model is constructed in the body of the *get_custom_model()* function that returns the model. Modify this function to implement your own model.

In the provided example, the *get_custom_model()* function takes in arguments:
- `model_name`, the name of the model.
- `input_shape`, the input shape of the model.
- `dropout`, the dropout rate if a dropout layer must be included in the model.

As you modify the *get_custom_model()* function, you can add your own arguments. Assume for example that you want to have an argument `alpha` that is a float. Then, just add it to the interface of the function.

Then, your custom model can be used as any other Model Zoo model using the configuration file as shown in the YAML code below:

```yaml
model:
   model_name: custom_model
   input_shape: (256, 128, 3)

training:
   dropout: 0.2
```
</details>

<details open><summary><a href="#5-5">5.5 Train, quantize, benchmark, and evaluate your model</a></summary><a id="5-5"></a>

In case you want to train and quantize a model, you can either launch the training operation mode followed by the quantization operation on the trained model (please refer to the quantization **[README.md](./README_QUANTIZATION.md)** that describes in detail the quantization part) or you can use chained services like launching [chain_tqe](../config_file_examples/chain_tqe_config.yaml) example with the command below:

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_tqe_config.yaml
```

This specific example trains a OSNet model, and quantizes it to 8-bits using quantization_split (10% in this example) of the train dataset for calibration before evaluating the quantized model.

In case you also want to execute a benchmark on top of training and quantizing services, it is recommended to launch the chain service called [chain_tqeb](../config_file_examples/chain_tqeb_config.yaml) that stands for train, quantize, evaluate, benchmark like the example with the command below:

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_tqeb_config.yaml
```

This specific example uses the mobilenet v1 with alpha 0.25, quantizes it to 8-bits for calibration before evaluating the quantized model and benchmarking it.

</details>
</ul>
</details>
