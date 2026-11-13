# Image Classification STM32 Model Zoo

Minimalistic YAML files are available [here](../config_file_examples_pt/) to experiment with specific services. All pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) come with their configuration `.yaml` files used to generate them. These are excellent starting points for your projects!

# Table of Contents

1. [Image Classification Model Zoo Introduction](#1)
   - [1.1 ImageNet Dataset](#1-1)
   - [1.2 Flowers102 Dataset](#1-2)
   - [1.3 Food101 Dataset](#1-3)
   - [1.4 Custom Dataset (ImageNet-like)](#1-4)
2. [Image Classification Tutorial](#2)
   - [2.1 Choose the operation mode](#2-1)
   - [2.2 Global settings](#2-2)
   - [2.3 Model specifications](#2-3)
   - [2.4 Dataset specifications](#2-4)
   - [2.5 Apply image preprocessing](#2-5)
   - [2.6 Use data augmentation](#2-6)
   - [2.7 Set the training parameters](#2-7)
   - [2.8 Model quantization](#2-8)
   - [2.9 Benchmark the model](#2-9)
   - [2.10 Deploy the model](#2-10)
   - [2.11 Hydra and MLflow settings](#2-11)
3. [Run the image classification chained service](#3)
4. [Visualize the chained services results](#4)
   - [4.1 Saved results](#4-1)
   - [4.2 Run TensorBoard](#4-2)
5. [Appendix A: YAML syntax](#A)

<details open><summary><a href="#1"><b>1. Image Classification Model Zoo Introduction</b></a></summary><a id="1"></a>

The image classification model zoo provides a collection of independent services and pre-built chained services for various machine learning functions related to image classification. Individual services include tasks such as training or quantizing the model, while chained services combine multiple services to perform more complex functions, such as training, quantizing, and evaluating the quantized model successively.

To use the services in the image classification model zoo, utilize the model zoo [stm32ai_main.py](../stm32ai_main.py) along with the [user_config_pt.yaml](../user_config_pt.yaml) file as input. The YAML file specifies the service or chained services and a set of configuration parameters such as the model (either from the model zoo or your custom model), the dataset, the number of epochs, and the preprocessing parameters, among others.

More information about the different services and their configuration options can be found in the <a href="#2">next section</a>.

The classification datasets should be structured just like standard datasets. We support ImageNet, Flowers102, Food101, and custom (ImageNet-like).

<ul><details open><summary><a href="#1-1">1.1 ImageNet Dataset</a></summary><a id="1-1"></a>

ImageNet data should be structured as follows:

```yaml
data_dir/
└── imagenet/
    ├── train/
    │   ├── n01440764/
    │   │   ├── n01440764_10026.JPEG
    │   │   ├── n01440764_10027.JPEG
    │   │   └── ...
    │   ├── n01443537/
    │   │   ├── n01443537_10007.JPEG
    │   │   └── ...
    │   └── ...
    └── val/
        ├── n01440764/
        │   ├── ILSVRC2012_val_00000293.JPEG
        │   └── ...
        ├── n01443537/
        │   └── ...
        └── ...
```

### Notes

- ImageNet (ILSVRC2012) contains **1,000 object classes**, each identified by a **WordNet synset ID** (e.g., `n01440764`).
- `train/` is organized into **one subdirectory per class**, each containing training images.
- `val/` images are often **reorganized into class folders** for convenience; originally they are provided as a flat directory with a labels file.
- Class labels are inferred from **directory names** when using loaders like PyTorch `ImageFolder`.


</details></ul>
<ul><details open><summary><a href="#1-2">1.2 Flowers102 Dataset</a></summary><a id="1-2"></a>

Flowers data should be structured as follows (standard format).

```yaml
data_dir/
└── flowers-102/
    ├── jpg/
    │   ├── image_00001.jpg
    │   ├── image_00002.jpg
    │   ├── image_00003.jpg
    │   └── ...
    ├── imagelabels.mat
    └── setid.mat

```

### Notes

- This dataset can be automatically downloaded if you set `data_download`: `True` in the dataset section.
- `jpg/` contains **all images** in a single directory (no class subfolders).
- `imagelabels.mat` maps each image index to a **class label (1–102)**.
- `setid.mat` defines the **train / validation / test splits**.
- This is the **official Oxford Flowers 102 dataset structure** as distributed.
- If you have reorganized this into imagenet format then use [Custom](#1-4) section.


</details></ul>
<ul><details open><summary><a href="#1-3">1.3 Food101 Dataset</a></summary><a id="1-3"></a>

Food data should be structured as follows (standard format).

```yaml
data_dir/
└── food-101/
    ├── images/
    │   ├── apple_pie/
    │   │   ├── 1005649.jpg
    │   │   ├── 1009449.jpg
    │   │   └── ...
    │   ├── baby_back_ribs/
    │   │   ├── 1001022.jpg
    │   │   └── ...
    │   └── ...
    └── meta/
        ├── classes.txt
        ├── train.txt
        └── test.txt
```

### Notes

- This dataset can be automatically downloaded if you set `data_download`: `True` in the dataset section.
- `images/` contains **one folder per class** (101 food categories).
- Each class folder contains **all images** for that category.
- `meta/classes.txt` lists all **class names**.
- `meta/train.txt` and `meta/test.txt` define **train/test splits** using image paths.
- Food-101 does **not provide a validation split** by default (commonly split from train).
- This is the **official Food-101 dataset structure** as distributed.
- If you have reorganized this into imagenet format then use "imagenet" or "custom" as dataset name.


</details></ul>
<ul><details open><summary><a href="#1-4">1.4 Custom Dataset (ImageNet-like)</a></summary><a id="1-4"></a>

If you have your own dataset, you can organize it just like ImageNet as shown below:

```yaml
<training-dataset-root-directory>
   class_a:
      a_image_1.jpg
      a_image_2.jpg
   class_b:
      b_image_1.jpg
      b_image_2.jpg

<validation-dataset-root-directory>
   class_a:
      a_image_1.jpg
      a_image_2.jpg
   class_b:
      b_image_1.jpg
      b_image_2.jpg
```

</details></ul>
</details>
<details open><summary><a href="#2"><b>2. Image Classification Tutorial</b></a></summary><a id="2"></a>

This tutorial demonstrates how to use the `chain_tqeb` services to train, quantize, evaluate, and benchmark the model. Among the various available models in the model zoo, we chose to use the `imagenet` classification dataset and apply transfer learning on the MobileNet image classification model as an example to demonstrate the workflow.

To get started, update the [user_config_pt.yaml](../user_config_pt.yaml) file, which specifies the parameters and configuration options for the services you want to use. Each section of the [user_config_pt.yaml](../user_config_pt.yaml) file is explained in detail in the following sections.

<ul><details open><summary><a href="#2-1">2.1 Choose the operation mode</a></summary><a id="2-1"></a>

The `operation_mode` top-level attribute specifies the operations or the service you want to execute. This may be a single operation or a set of chained operations.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmarking, and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations                                                                                                               |
|:-------------------------|:-------------------------------------------------------------------------------------------------------------------------|
| `training`               | Train a model from the variety of classification models in the model zoo or your own model                               |
| `evaluation`             | Evaluate the accuracy of a float or quantized model on a test or validation dataset                                      |
| `quantization`           | Quantize a float model                                                                                                   |
| `prediction`             | Predict the classes some images belong to using a float or quantized model                                               |
| `benchmarking`           | Benchmark a float or quantized model on an STM32 board                                                                   |
| `deployment`             | Deploy a model on an STM32 board                                                                                         |
| `chain_tqeb`             | Sequentially: training, quantization of trained model, evaluation of quantized model, benchmarking of quantized model    |
| `chain_tqe`              | Sequentially: training, quantization of trained model, evaluation of quantized model                                     |
| `chain_eqe`              | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model                              |
| `chain_qb`               | Sequentially: quantization of a float model, benchmarking of quantized model                                             |
| `chain_eqeb`             | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model |
| `chain_qd`               | Sequentially: quantization of a float model, deployment of quantized model                                               |

You can refer to the README links below that provide typical examples of operation modes and tutorials on specific services:

   - [training, chain_tqe, chain_tqeb](./README_TRAINING_TORCH.md)
   - [quantization, chain_eqe, chain_qb](./README_QUANTIZATION.md)
   - [evaluation, chain_eqeb](./README_EVALUATION.md)
   - [benchmarking](./README_BENCHMARKING.md)
   - [prediction](./README_PREDICTION.md)
   - deployment, chain_qd ([STM32H7](./README_DEPLOYMENT_STM32H7.md), [STM32N6](./README_DEPLOYMENT_STM32N6.md), [STM32MPU](./README_DEPLOYMENT_MPU.md))

In this tutorial, the `operation_mode` used is the `chain_tqeb` as shown below to train a model, quantize, evaluate it, and later deploy it on the STM32 boards.

```yaml
operation_mode: chain_tqeb
```
</details></ul>
<ul><details open><summary><a href="#2-2">2.2 Global settings</a></summary><a id="2-2"></a>

The `general` section and its attributes are shown below.

```yaml
general:
  project_name: 'pt_imagenet'        # Name of the project (used for logging, checkpoints, runs)
  output: ''                         # Output directory for experiment artifacts (empty = default)
  saved_models_dir: saved_models     # Directory where model checkpoints will be saved
  seed: 42                           # Random seed for reproducibility
  gpu_memory_limit: 3                # GPU memory limit in GB (0 or unset = no limit)
  display_figures: False             # Whether to display plots/figures during training
```
The general section defines global settings that control experiment setup, hardware usage, logging, checkpointing, and runtime behavior during training and evaluation.

By default the checkpoints,onnx model, training summary and tensorboard logs will saved at `pt/src/experiments_outputs/<date-and-time>/saved_models`. 

`seed` is Random seed for reproducible training runs. This attribute is optional, with a default value of 42.

`gpu_memory_limit` is currently not in use.

`display_figures` is a Boolean flag that, when true, shows interactive figures you must interact with to continue the process.


</details></ul>
<ul><details open><summary><a href="#2-3">2.3 Model specifications</a></summary><a id="2-3"></a>

```yaml
model:
  framework: 'torch'                      # Deep learning framework used for the model
  model_name: 'mobilenetv4small_pt'       # Model architecture name / identifier
  pretrained: True                        # Whether to load pretrained weights
  pretrained_dataset: "imagenet"          # Dataset used to pretrain the model
  input_shape: [3, 224, 224]              # Model input shape: [channels, height, width]
  #model_path: path to .pt / .onnx file.  # Optional
```
The `model_path` attribute is utilized to indicate the path to the model file that you wish to use for the selected operation mode. The accepted formats for `model_path` are listed in the table below:

| Operation mode | `model_path`                                     |
|:---------------|:-------------------------------------------------|
| 'evaluation'   | .pt or ONNX (float or QDQ) file |
| 'quantization', 'chain_eqe', 'chain_eqeb', 'chain_qb', 'chain_qd' | .pt or ONNX (float or QDQ) model file          |
| 'prediction'   | .pt or ONNX (float or QDQ) file |
| 'benchmarking' | .pt or ONNX (float or QDQ) file |
| 'deployment'   | .pt or ONNX (QDQ) file |

If you are providing `.pt` file for `model_path` then you also need to provide `model_name` so that model definition can be created before loading weights from `.pt` file.

</details></ul>
<ul><details open><summary><a href="#2-4">2.4 Dataset specifications</a></summary><a id="2-4"></a>


<ul><details open><summary><a href="#2-4-1">2.4.1 ImageNet dataset (same is valid for Flowers102 and Food101 dataset)</a></summary><a id="2-4-1"></a>

Expected top level folder structure for all three datasets.

```yaml
data_dir/
└── imagenet/
or
└── flower-102/
or
└── food-101/
```

The `dataset` section and its attributes are shown in the YAML code below.

```yaml
dataset:
  class_names: ''                              # Optional explicit class names
  classes_file_path: ./datasets/deployment_labels_imagenet.txt  # Path to class label file
  dataset_name: "imagenet"                     # Dataset name: "flowers102", "food101", or "imagenet"
  num_classes: 1000                            # Number of classes (update according to the dataset)
  data_dir: "path to data_dir"                 # Root data directory; must contain the "imagenet" folder, as shown above in the introduction section.
  train_split: "train"                         # (Optional, ImageNet only) Training subfolder name
  val_split: "val"                             # (Optional, ImageNet only) Validation subfolder name
  quantization_split: 0.01                     # Fraction of data used for quantization calibration
  test_path: ""                                # Optional test dataset path for evaluation (defaults to validation data)
  quantization_path: ""                        # Optional quantization dataset path (defaults to training data)
  workers: 4                                   # Number of data loader worker processes
  worker_seeding: all
  pin_mem: false                               # Enable pinned memory for DataLoader (faster host→GPU transfer)
  no_prefetcher: true                          # Disable data prefetcher (use standard DataLoader)
  use_multi_epochs_loader: false               # Use multi-epoch DataLoader to reduce startup overhead
  #data_download: True                         # Valid for food101 and flowers102 datasets.
```


The `dataset_name` attribute has four options: `imagenet`, `food101`, `flowers102`, and `custom`. Custom will be explained in the next section. Custom is almost the same as ImageNet data.

The `class_names` attribute specifies the classes in the dataset. This information must be provided in the YAML file. If the `class_names` attribute is absent, the `classes_file_path` argument can be used as an alternative, pointing to a text file containing the class names. Alternatively, the class names can be deduced from the folder names of each class.

`train_split` and `val_split` are optional variables just for the `imagenet` dataset. Useful in scenarios when your training and validation folders are renamed to something other than "train" and "val". The standard ImageNet dataset has "train" and "val" folders inside the "imagenet" directory.

The `quantization_path` attribute is used to specify a dataset for the quantization process. If this attribute is not provided and a training set is available, the training set is used for the quantization. However, training sets can be quite large, and the quantization process can take a long time to run. To avoid this issue, you can set the `quantization_split` attribute to use only a portion of the dataset for quantization.

`workers` is number of worker processes used by the data loader.

`pin_mem` is Enables pinned memory for faster CPU → GPU data transfer.

`no_prefetcher` disables the data prefetcher and uses the standard PyTorch DataLoader.

`use_multi_epochs_loader` uses a multi-epoch DataLoader to reduce worker startup overhead.

`data_download` if set to True it will download the dataset. Only valid for `flowers102` and `food101` dataset.

</details></ul>

<ul><details open><summary><a href="#2-4-2">2.4.2 Custom Dataset (Imagenet-like)</a></summary><a id="2-4-2"></a>

You should rearrange your custom dataset into ImageNet format.

```yaml
<training-dataset-root-directory>
   class_a:
      a_image_1.jpg
      a_image_2.jpg
   class_b:
      b_image_1.jpg
      b_image_2.jpg

<validation-dataset-root-directory>
   class_a:
      a_image_1.jpg
      a_image_2.jpg
   class_b:
      b_image_1.jpg
      b_image_2.jpg
```

```yaml
dataset:
  class_names: ''                              # Optional explicit class names
  classes_file_path: ./datasets/deployment_labels_imagenet.txt  # Path to class label file
  dataset_name: "custom"                       # Dataset name: "flowers102", "food101", or "imagenet"
  num_classes: 1000                            # Number of classes (update according to the dataset)
  training_path: <training-dataset-root-directory>
  validation_path: <validation-dataset-root-directory>
  quantization_split: 0.01                     # Fraction of data used for quantization calibration
  test_path: ""                                # Optional test dataset path for evaluation (defaults to validation data)
  quantization_path: ""                        # Optional quantization dataset path (defaults to training data)
  workers: 4                                   # Number of data loader worker processes
  worker_seeding: all
  pin_mem: false                               # Enable pinned memory for DataLoader (faster host→GPU transfer)
  no_prefetcher: true                          # Disable data prefetcher (use standard DataLoader)
  use_multi_epochs_loader: false               # Use multi-epoch DataLoader to reduce startup overhead
```

Three variables that are important for custom dataset: `dataset_name`, `training_path`, and `validation_path`. </details></ul>

</details></ul>
<ul><details open><summary><a href="#2-5">2.5 Apply image preprocessing</a></summary><a id="2-5"></a>

Images need to be rescaled and resized before they can be used. This is specified in the 'preprocessing' section that is required in all the operation modes.

The 'preprocessing' section for this tutorial is shown below.
**Note:** Only `mean` and `std` are being used in torch image classification. All other values like `scale`, `offset`, `aspect_ratio`, and `color_mode` will not have any impact on the training.

```yaml
preprocessing: 
  rescaling: # Not used for Torch Image Classification
    scale: None # Not used for Torch Image Classification
    offset: None # Not used for Torch Image Classification
  resizing:
    interpolation: random # 'random' means random.choice(["bilinear", "bicubic"]). This will be used for training data and bilinear is fixed for test data 
    aspect_ratio: fit # Not used for Torch Image Classification
  color_mode: rgb # Not used for Torch Image Classification
  normalization:
    mean: [0.485, 0.456, 0.406] # Subtracted from each channel: (x - mean) / std
    std: [0.229, 0.224, 0.225] # Divides each channel after mean subtraction
```

The `preprocessing` section defines how images are prepared before being given to the model. It controls resizing behavior and pixel value normalization so that inputs match what the model expects.

The `rescaling` block is present but not used for Torch Image Classification. Both `scale` and `offset` are set to `None`, so no linear rescaling is applied to pixel values.

The `resizing` block specifies how images are resized. The `interpolation` method is set to `random`, which means that for training data the interpolation is randomly chosen between `bilinear` and `bicubic`, while for test data `bilinear` interpolation is always used. The `aspect_ratio` option is set to `fit`, but it is not used for Torch Image Classification.

The `color_mode` is set to `rgb`, indicating three-channel RGB images. This option is also not used for Torch Image Classification.

The `normalization` block defines per-channel normalization using fixed `mean` and standard deviation values. Each pixel is normalized using the formula (x - `mean`) / `std`. The provided mean and standard deviation values correspond to ImageNet statistics and are applied in RGB channel order.


</details></ul>
<ul><details open><summary><a href="#2-6">2.6 Use data augmentation</a></summary><a id="2-6"></a>

The data augmentation functions to apply to the input images during training are specified in the optional `data_augmentation` section of the configuration file. They are only applied to the images during training.

For this tutorial, the data augmentation section is shown below.

```yaml
data_augmentation:
  no_aug: False  
  crop_range: [0.08, 1.0]
  ratio: [0.75, 1.33]
  horizontal_flip: 0.5
  vertical_flip: 0.0
  hflip: 0.5
  vflip: 0.0
  color_jitter: 0.4
  aa: null 
  aug_repeats: 0
  aug_splits: 0
  jsd_loss: False
  bce_loss: False
  bce_target_thresh: null
  reprob: 0 
  remode: 'pixel' 
  recount: 1 
  resplit: False  
  mixup: 0.0
  cutmix: 0.0
  cutmix_minmax: null  # Example: [0.3, 0.8]
  mixup_prob: 1.0
  mixup_switch_prob: 0.5
  mixup_mode: "batch"
  smoothing: 0.1
  train_interpolation: "random"
  drop: 0.0
  drop_connect: null
  drop_path: null
  drop_block: null
```

The data augmentation functions with their parameter settings are applied to the input images.

`no_aug` – Disables all data augmentation when set to `True`.

`scale` – Range of random resize scale applied during image cropping.

`ratio` – Aspect ratio range used for random resized cropping.

`horizontal_flip` – Probability of applying random horizontal flip.

`vertical_flip` – Probability of applying random vertical flip.

`hflip` – Alias for horizontal flip probability.

`vflip` – Alias for vertical flip probability.

`color_jitter` – Strength of random color jitter (brightness, contrast, saturation).

`aa` – AutoAugment policy to apply (e.g., RandAugment or AutoAugment); null disables it.

`aug_repeats` – Number of times each sample is augmented per epoch.

`aug_splits` – Number of augmented splits per batch (used for JSD training).

`jsd_loss` – Enables Jensen–Shannon Divergence loss for consistency across augmentations.

`bce_loss` – Uses binary cross-entropy loss instead of categorical cross-entropy.

`bce_target_thresh` – Threshold for converting targets to binary labels when using BCE loss.

`reprob` – Probability of applying Random Erasing augmentation.

`remode` – Random Erasing mode (e.g., pixel, const, or rand).

`recount` – Number of Random Erasing operations applied per image.

`resplit` – Applies Random Erasing separately per augmented split.

`mixup` – MixUp alpha value controlling the strength of sample mixing.

`cutmix` – CutMix alpha value controlling region-based sample mixing.

`cutmix_minmax` – Min/max ratio range for CutMix region size.

`mixup_prob` – Probability of applying MixUp or CutMix to a batch.

`mixup_switch_prob` – Probability of switching between MixUp and CutMix when both are enabled.

`mixup_mode` – Scope of MixUp application (batch, pair, or element).

`smoothing` – Label smoothing factor applied to training targets.

`drop` – Standard dropout rate applied during training.

`drop_connect` – DropConnect rate for randomly dropping network connections.

`drop_path` – Stochastic depth rate for dropping entire network paths.

`drop_block` – DropBlock regularization rate for structured feature map dropout.


</details></ul>
<ul><details open><summary><a href="#2-7">2.7 Set the training parameters</a></summary><a id="2-7"></a>

A 'training' section is required in all the operation modes that include training, namely 'training', 'chain_tqeb', and 'chain_tqe'.

In this tutorial, we picked a MobileNet model for transfer learning. The model weights are pre-trained on the imagenet dataset, a large dataset consisting of 1.4M images and 1000 classes. As an example, we will use a MobileNet with alpha = 0.25. To do so, we will need to configure the train section in [user_config_pt.yaml](../user_config_pt.yaml) as follows:

```yaml
training:
  epochs: 2
  batch_size: 128
  validation_batch_size: null
  resume_training_from: null              # checkpoint path to resume training
  optimizer:
    opt: 'sgd' 
    opt-eps: null 
    opt-betas: null 
    momentum: 0.9
    weight_decay: 0.00002
    clip_grad: null
    clip_mode: 'norm'
    layer_decay: null
  lr_scheduler:
    sched: 'cosine'
    sched_on_updates: False
    lr: null
    lr_base: 0.1
    lr_base_size: 256
    lr_base_scale: ''
    lr_noise: null
    lr_noise_pct: 0.67
    lr_noise_std: 1.0
    lr_cycle_mul: 1.0
    lr_cycle_decay: 0.5
    lr_cycle_limit: 1
    lr_k_decay: 1.0
    warmup_lr: 0.00001
    min_lr: 0
    epoch_repeats: 0
    start_epoch: 0
    decay_milestones: [90, 180, 270]
    decay_epochs: 90 
    warmup_epochs: 5
    warmup_prefix: False
    cooldown_epochs: 0
    patience_epochs: 10
    decay_rate: 0.1
  bn_momentum: null
  bn_eps: null
  sync_bn: false
  dist_bn: "reduce"
  split_bn: false
  model_ema: false
  model_ema_force_cpu: false
  model_ema_decay: 0.9998
  log_wandb: false                   # Enable logging to Weights & Biases
  log_tb: false                      # Enable logging to TensorBoard
  log_interval: 50                   # Logging frequency (in training iterations)
  recovery_interval: 0               # Interval for recovery checkpoints (0 = disabled)
  checkpoint_hist: 10                # Number of past checkpoints to keep
  save_images: False                 # Whether to save sample images during training/evaluation
  amp: false                         # Enable Automatic Mixed Precision (AMP)
  amp_dtype: "float16"               # AMP data type (e.g., float16, bfloat16)
  amp_impl: "native"                 # AMP implementation ("native" = PyTorch autocast)
  tta: 0                             # Test-time augmentation factor (0 = disabled)
  eval_metric: "top1"                # Primary evaluation metric (e.g., top1, top5)
  no_ddp_bb: false                   # Disable DDP backbone wrapping (for distributed training)
  synchronize_step: false            # Synchronize GPU operations every step (debug/perf impact)
```

This section controls training duration, optimization strategy, learning rate scheduling, normalization behavior, and stability mechanisms such as EMA and gradient clipping.
The `batch_size` and `epochs` attributes are mandatory. All the others are optional.

`epochs` – Total number of training epochs.

`batch_size` – Number of samples processed per training iteration.

`validation_batch_size` – Batch size used during validation (defaults to training batch size if null).

`resume_training_from` - is set to specify a checkpoint path for resuming training. This can be used if your training ends abruptly and you want to continue training from a specific checkpoints.  

`optimizer.opt` – Optimizer type used for training (e.g., sgd, adam, adamw).

`optimizer.momentum` – Momentum factor for momentum-based optimizers like SGD.

`optimizer.weight_decay` – L2 weight decay coefficient for regularization.

`optimizer.clip_grad` – Maximum gradient value or norm for gradient clipping.

`optimizer.clip_mode` – Gradient clipping mode (norm or value).

`optimizer.layer_decay` – Layer-wise learning rate decay factor (used in transformer-style models).

`lr_scheduler.sched` – Learning rate scheduling strategy (e.g., cosine, step, plateau).

`lr_scheduler.sched_on_updates` – Updates the learning rate per optimizer step instead of per epoch.

`lr_scheduler.lr `– Explicit learning rate override (if set).

`lr_scheduler.lr_base` – Base learning rate before scaling.

`lr_scheduler.lr_base_size` – Reference batch size used for learning rate scaling.

`lr_scheduler.lr_base_scale` – Scaling strategy applied to the base learning rate.

`lr_scheduler.lr_noise` – Range of epochs where learning rate noise is applied.

`lr_scheduler.lr_noise_pct` – Percentage magnitude of learning rate noise.

`lr_scheduler.lr_noise_std` – Standard deviation of learning rate noise.

`lr_scheduler.lr_cycle_mul` – Multiplicative factor for increasing cycle length in cyclic schedulers.

`lr_scheduler.lr_cycle_decay` – Decay factor applied to learning rate after each cycle.

`lr_scheduler.lr_cycle_limit` – Maximum number of learning rate cycles.

`lr_scheduler.lr_k_decay` – Decay factor controlling cosine or polynomial schedule curvature.

`lr_scheduler.warmup_lr` – Initial learning rate used during warmup.

`lr_scheduler.min_lr` – Minimum learning rate after decay.

`lr_scheduler.epoch_repeats` – Number of times each epoch is repeated before advancing.

`lr_scheduler.start_epoch` – Epoch number to resume training from.

`lr_scheduler.decay_milestones` – Epochs at which the learning rate is reduced for step-based schedules.

`lr_scheduler.decay_epochs` – Number of epochs between learning rate decay steps.

`lr_scheduler.warmup_epochs` – Number of warmup epochs before normal scheduling begins.

`lr_scheduler.warmup_prefix` – Applies warmup before the main scheduler starts.

`lr_scheduler.cooldown_epochs` – Number of epochs to keep the learning rate at its minimum.

`lr_scheduler.patience_epochs` – Number of epochs to wait before reducing LR on plateau.

`lr_scheduler.decay_rate` – Factor by which the learning rate is reduced during decay.

`bn_momentum` – Momentum value for BatchNorm running statistics.

`bn_eps` – Numerical stability epsilon for BatchNorm layers.

`sync_bn` – Enables synchronized BatchNorm across GPUs.

`dist_bn` – Distributed BatchNorm reduction method (e.g., reduce or broadcast).

`split_bn` – Uses separate BatchNorm statistics for different data splits.

`model_ema` – Enables exponential moving average tracking of model weights.

`model_ema_force_cpu` – Forces EMA weights to be stored on the CPU.

`model_ema_decay` – Decay rate controlling how fast EMA weights update.

`synchronize_step` forces GPU synchronization at every step (useful for debugging; may impact performance).

`log_interval` Logging frequency measured in training iterations.

`log_wandb` Enables logging to Weights & Biases.

`log_tb` Enables logging to TensorBoard.

`eval_metric` Primary evaluation metric (e.g., top1, top5, loss).

`recovery_interval` Interval for saving recovery checkpoints. Set to 0 to disable.

`checkpoint_hist` Number of historical checkpoints to retain.

`save_images` Whether to save sample images during training or evaluation.

`amp` Enables Automatic Mixed Precision (AMP) for faster training and lower memory usage.

`amp_dtype` Data type used for AMP (e.g., float16, bfloat16).

`amp_impl` AMP backend implementation (native uses PyTorch autocast).

`no_ddp_bb` Disables Distributed Data Parallel (DDP) wrapping for the model backbone.

`tta` Test-time augmentation factor. Set to 0 to disable TTA.

The best model obtained at the end of the training is saved in the `experiments_outputs/<date-and-time>/saved_models` directory.

</details></ul>
<ul><details open><summary><a href="#2-8">2.8 Model quantization</a></summary><a id="2-8"></a>

Configure the quantization section in [user_config.yaml](../user_config.yaml) as follows:

```yaml

quantization:
   quantizer: onnx_quantizer         # mandatory
   quantization_type: PTQ             
   quantization_input_type: float      # float for onnx_quantizer
   quantization_output_type: float     # float for onnx_quantizer
   granularity: per_channel            # Optional, defaults to "per_channel".
   optimize: False                      # Optional, defaults to False.
   target_opset: 17                    # Optional, defaults to 17.
   export_dir: quantized_models        # Optional, defaults to "quantized_models".
   onnx_extra_options: 
      CalibMovingAverage: True         # Optional, default to False
```

This section is used to configure the quantization process, which optimizes the model for efficient deployment on embedded devices by reducing its memory usage (Flash/RAM) and accelerating its inference time, with minimal degradation in model accuracy. The `quantizer` attribute expects the value `onnx_quantizer`, which is used to convert the trained model weights from float to integer values and transfer the model to a .onnx QDQ format. 

The `quantization_type` attribute only allows the value "PTQ," which stands for Post Training Quantization. To specify the quantization type for the model input and output, use the `quantization_input_type` and `quantization_output_type` attributes, respectively. The `quantization_input_type` attribute is a string that can be set to "int8", "uint8," or "float" to represent the quantization type for the model input. Similarly, the `quantization_output_type` attribute is a string that can be set to "int8", "uint8," or "float" to represent the quantization type for the model output. These values are not accounted for when using `onnx_quantizer` as both model input and output types are float and only the weights and activations are quantized.

The `granularity` is either "per_channel" or "per_tensor". If the parameter is not set, it will default to "per_channel". 'per channel' means all weights contributing to a given layer output channel are quantized with one unique (scale, offset) couple.
The alternative is 'per tensor' quantization which means that the full weight tensor of a given layer is quantized with one unique (scale, offset) couple. 
It is obviously more challenging to preserve original float model accuracy using 'per tensor' quantization. But this method is particularly well suited to fully exploit STM32MP2 platforms HW design.

The `target_opset` is an integer parameter. Before doing the ONNX quantization, the ONNX opset of the model is updated to the target_opset. If no value is provided, a default value of 17 is used. The `onnx_extra_options` sub-section only applies to ONNX quantization. This option allows for example min and max smoothing during activation calibration ('CalibMovingAverage': True). We observed that it can sometimes help filter some activations outliers which translates into a lower quantization noise on small amplitude values which in the end can have a positive effect on network accuracy or precision. Other options are possible: please refer to [README_QUANTIZATION.md](./README_QUANTIZATION.md).

By default, the quantized model is saved in the 'quantized_models' directory under the 'experiments_outputs' directory. You may use the optional `export_dir` attribute to change the name of this directory.


</details></ul>
<ul><details open><summary><a href="#2-9">2.9 Benchmark the model</a></summary><a id="2-9"></a>

The [STEdgeAI Developer Cloud](https://stedgeai-dc.st.com/home) allows you to benchmark your model and estimate its footprints and inference time for different STM32 target devices. To use this feature, set the `on_cloud` attribute to True. Alternatively, you can use [STEdgeAI Core](https://www.st.com/en/development-tools/stedgeai-core.html) to benchmark your model and estimate its footprints for STM32 target devices locally. To do this, make sure to add the path to the `stedgeai` executable under the `path_to_stedgeai` attribute and set the `on_cloud` attribute to False.

The `optimization` defines the optimization used to generate the C model, options: `balanced`, `time`, `ram`.

The `board` attribute is used to provide the name of the STM32 board to benchmark the model on. The available boards are 'STM32N6570-DK', 'STM32H747I-DISCO', 'STM32H7B3I-DK', 'STM32F469I-DISCO', 'B-U585I-IOT02A', 'STM32L4R9I-DISCO', 'NUCLEO-H743ZI2', 'STM32H735G-DK', 'STM32F769I-DISCO', 'NUCLEO-G474RE', 'NUCLEO-F401RE', and 'STM32F746G-DISCO'.

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
The `path_to_cubeIDE` attribute is for the deployment service which is not part of the chain `chain_tqeb` used in this tutorial.

</details></ul>
<ul><details open><summary><a href="#2-10">2.10 Deploy the model</a></summary><a id="2-10"></a>

In this tutorial, we are using the `chain_tqeb` toolchain, which does not include the deployment service. However, if you want to deploy the model after running the chain, you can do so by referring to the deployment README and modifying the `deployment_config.yaml` file or by setting the `operation_mode` to `deploy` and modifying the `user_config_pt.yaml` file as described below:

```yaml
model:
   model_path: <path-to-an-ONNX-QDQ-model-file>     # Path to the model file to deploy

dataset:
   class_names: [daisy, dandelion, roses, sunflowers, tulips] 

tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/ST/STEdgeAI/<x.y>/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_<*.*.*>/STM32CubeIDE/stm32cubeide.exe

deployment:
  c_project_path: ../application_code/image_classification/STM32H7/
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32H7
    board: STM32H747I-DISCO
```

In the `model` section, users must provide the path to the ONNX model file that they want to deploy using the `model_path` attribute.

The `dataset` section requires users to provide the names of the classes using the `class_names` or `classes_file_path` attribute. If you use the `classes_file_path`, ensure the classes are listed in alphabetical order or according to the dataset's convention.

This maintains clarity and conciseness while explaining the different ways to specify class names.
The `tools` section includes information about the stedgeai toolchain, such as the version, optimization level, and path to the `stedgeai.exe` file.

Finally, in the `deployment` section, users must provide information about the hardware setup, such as the series and board of the STM32 device, as well as the input and output interfaces. Once all of these sections have been filled in, users can run the deployment service to deploy their model to the STM32 device.

Please refer to the readme below for a complete deployment tutorial:
- on H7-MCU: [README_STM32H7.md](./README_DEPLOYMENT_STM32H7.md)
- on N6-NPU: [README_STM32N6.md](./README_DEPLOYMENT_STM32N6.md)
- on MPU: [README_STM32MPU.md](./README_DEPLOYMENT_MPU.md)

</details></ul>
<ul><details open><summary><a href="#2-11">2.11 Hydra and MLflow settings</a></summary><a id="2-11"></a>
 
The `mlflow` and `hydra` sections must always be present in the YAML configuration file. The `hydra` section can be used to specify the name of the directory where experiment directories are saved and/or the pattern used to name experiment directories. In the YAML code below, it is set to save the outputs as explained in the section <a href="#4">visualize the chained services results</a>:

```yaml
hydra:
   run:
      dir: ./pt/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

The `mlflow` section is used to specify the location and name of the directory where MLflow files are saved, as shown below:

**Note:** For PyTorch, we only output logs for TensorBoard. MLflow is not used.

```yaml
mlflow:
   uri: ./pt/src/experiments_outputs/mlruns
```
</details></ul>
</details>
<details open><summary><a href="#3"><b>3. Run the image classification chained service</b></a></summary><a id="3"></a>

After updating the [user_config_pt.yaml](../user_config_pt.yaml) file, please run the following command:

```bash
python stm32ai_main.py --config-path ./ --config-name user_config_pt.yaml
```

</details>
<details open><summary><a href="#4"><b>4. Visualize the chained services results</b></a></summary><a id="4"></a>

Every time you run the Model Zoo, an experiment directory is created that contains all the directories and files created during the run. The names of experiment directories are all unique as they are based on the date and time of the run.

Experiment directories are managed using the Hydra Python package. Refer to [Hydra Home](https://hydra.cc/) for more information about this package.

By default, all the experiment directories are under the <MODEL-ZOO-ROOT>/image_classification/pt/src/experiments_outputs directory and their names follow the "%Y_%m_%d_%H_%M_%S" pattern.

This is illustrated in the figure below.

```
                                  experiments_outputs
                                          |
                                          |
      +-----------------------------------+
      |                                   |                  
      |                                   |                   
    mlruns                         <date-and-time> 
      |                                   |              
    MLflow                                +--- stm32ai_main.log
    files                                 |
                                          |
      +-----------------------------------+-----------------------------------------------+
      |                                   |                                               |
      |                                   |                                               |
 saved_models                      quantized_models                                    .hydra
      |                                   |                                               |
      +--- last.pth.tar                   +--- *.onnx QDQ model                         Hydra
      +--- model_best.pth.tar                                                           files
      +--- model_name/model_name.onnx
      +--- summary.csv
      +--- tensorboard log file                  
```

The file named 'stm32ai_main.log' under each experiment directory is the log file saved during the execution of the 'stm32ai_main.py' script. The contents of the other files saved under an experiment directory are described in the table below.

| File                                          | Directory        | Contents                                                                                           |
|:----------------------------------------------|:-----------------|:---------------------------------------------------------------------------------------------------|
| model_best.pth.tar                            | saved_models     | Best model saved during training, rescaling and data augmentation layers included                  |
| last.pth.tar                                  | saved_models     | Last model saved at the end of a training, rescaling and data augmentation layers included         |
| model_name.onnx                               | saved_models     | Onnx model obtained at the end of a training                                                       |
| quantized_models/*.onnx                       | quantized_models | Quantized model ONNX QDQ                                                                           |
| summary.csv                                   | metrics          | Training metrics CSV including epochs, losses, accuracies and learning rate                        |

                                                               

All the directory names, including the naming pattern of experiment directories, can be changed using the configuration file. The names of the files cannot be changed.

<ul><details open><summary><a href="#4-1">4.1 Saved results</a></summary><a id="4-1"></a>

All of the training and evaluation artifacts are saved in the current output simulation directory, which is located at **experiments_outputs/<date-and-time>**.

</details></ul>
<ul><details open><summary><a href="#4-2">4.2 Run tensorboard</a></summary><a id="4-2"></a>
 
To visualize the training curves that were logged by TensorBoard, navigate to the **experiments_outputs/<date-and-time>** directory and run the following command:

```bash
tensorboard --logdir logs
```
This will start a server and its address will be displayed. Use this address in a web browser to connect to the server. Then, using the web browser, you will be able to explore the learning curves and other training metrics.

</details></ul>

</details>

<details open><summary><a href="#A"><b>Appendix A: YAML Syntax</b></a></summary><a id="A"></a>

**Example and terminology:**

An example of YAML code is shown below.

```yaml
preprocessing:
   rescaling:
      scale : 1/127.5
      offset : -1
   resizing:
      aspect_ratio: fit
      interpolation: nearest
```

The code consists of a number of nested "key-value" pairs. The colon character is used as a separator between the key and the value.

Indentation is how YAML denotes nesting. The specification forbids tabs because tools treat them differently. A common practice is to use 2 or 3 spaces but you can use any number of them. 

We use "attribute-value" instead of "key-value" as in the YAML terminology, the term "attribute" being more relevant to our application. We may use the term "attribute" or "section" for nested attribute-value pairs constructs. In the example above, we may indifferently refer to "preprocessing" as an attribute (whose value is a list of nested constructs) or as a section.

**Comments:**

Comments begin with a pound sign. They can appear after an attribute value or take up an entire line.

```yaml
preprocessing:
   rescaling:
      scale : 1/127.5   # This is a comment.
      offset : -1
   resizing:
      # This is a comment.
      aspect_ratio: fit
      interpolation: nearest
   color_mode: rgb
```

**Attributes with no value:**

The YAML language supports attributes with no value. The code below shows the alternative syntaxes you can use for such attributes.

```yaml
attribute_1:
attribute_2: ~
attribute_3: null
attribute_4: None     # Model Zoo extension
```
The value *None* is a Model Zoo extension that was made because it is intuitive to Python users.

Attributes with no value can be useful to list in the configuration file all the attributes that are available in a given section and explicitly show which ones were not used.

**Strings:**

You can enclose strings in single or double quotes. However, unless the string contains special YAML characters, you don't need to use quotes.

This syntax:

```yaml
resizing:
   aspect_ratio: fit
   interpolation: nearest
```

is equivalent to this one:

```yaml
resizing:
   aspect_ratio: "fit"
   interpolation: "nearest"
```

**Strings with special characters:**

If a string value includes YAML special characters, you need to enclose it in single or double quotes. In the example below, the string includes the ',' character, so quotes are required.

```yaml
name: "Pepper,_bell___Bacterial_spot"
```

**Strings spanning multiple lines:**

You can write long strings on multiple lines for better readability. This can be done using the '|' (pipe) continuation character as shown in the example below.

This syntax:

```yaml
LearningRateScheduler:
   schedule: |
      lambda epoch, lr:
          (0.0005*epoch + 0.00001) if epoch < 20 else
          (0.01 if epoch < 50 else
          (lr / (1 + 0.0005 * epoch)))
```

is equivalent to this one:

```yaml
LearningRateScheduler:
   schedule: "lambda epoch, lr: (0.0005*epoch + 0.00001) if epoch < 20 else (0.01 if epoch < 50 else (lr / (1 + 0.0005 * epoch)))"
```

Note that when using the first syntax, strings that contain YAML special characters don't need to be enclosed in quotes. In the example above, the string includes the ',' character.

**Booleans:**

The syntaxes you can use for boolean values are shown below. Supported values have been extended to *True* and *False* in the Model Zoo as they are intuitive to Python users.

```yaml
# YAML native syntax
attribute_1: true
attribute_2: false

# Model Zoo extensions
attribute_3: True
attribute_4: False
```

**Numbers and numerical expressions:**

Attribute values can be integer numbers, floating-point numbers or numerical expressions as shown in the YAML code below.

```yaml
ReduceLROnPlateau:
   patience: 10    # Integer value
   factor: 0.1     # Floating-point value
   min_lr: 1e-6    # Floating-point value, exponential notation

rescaling:
   scale: 1/127.5  # Numerical expression, evaluated to 0.00784314
   offset: -1
```

**Lists:**

You can specify lists on a single line or on multiple lines as shown below.

This syntax:

```yaml
class_names: [daisy, dandelion, roses, sunflowers, tulips]
```
is equivalent to this one:

```yaml
class_names:
- daisy
- dandelion
- roses
- sunflowers
- tulips
```

**Multiple attribute-value pairs on one line:**

Multiple attribute-value pairs can be specified on one line as shown below.

This syntax:

```yaml
rescaling: { scale: 1/127.5, offset: -1 }
```
is equivalent to this one:

```yaml
rescaling:
   scale: 1/127.5
   offset: -1
```
</details>
