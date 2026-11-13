# Image Classification STM32 Model Training

This README shows how to train from scratch or apply transfer learning on an image classification model using a custom dataset. As an example, we will demonstrate the workflow on the `imagenet` classification dataset.


<details open><summary><a href="#1"><b>1. Prepare the dataset</b></a></summary><a id="1"></a>

After downloading and extracting the `imagenet` dataset, the dataset directory tree should look as below:

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

If you have your own dataset, you can organize it just like ImageNet as shown below and training on you custom data:

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

</details>

<details open>
<summary><a href="#2"><b>2. Create your training configuration file</b></a></summary><a id="2"></a>
<ul>
<details open><summary><a href="#2-1">2.1 Overview</a></summary><a id="2-1"></a>

All the proposed services like the training of the model are driven by a configuration file written in the YAML language.

For training, the configuration file should include at least the following sections:

- `general`, describes your project, including project name, directory where to save models, etc.
- `operation_mode`, describes the service or chained services to be used. Should be set to `training` for this tutorial.
- `dataset`, describes the dataset you are using, including directory paths, class names, etc.
- `preprocessing`, specifies the methods you want to use for rescaling and resizing the images.
- `model`, describes the inputs reuiqred to define/load a model from the zoo.
- `training`, specifies your training setup, including batch size, number of epochs, optimizer, callbacks, etc.
- `hydra`, specifies the folder to save Hydra logs.

This tutorial only describes the settings needed to train a model. In the first part, we describe basic settings. At the end of this README, you can also find more advanced settings.

</details>

<details open><summary><a href="#2-2">2.2 General settings</a></summary><a id="2-2"></a>

The first section of the configuration file is the `general` section that provides information about your project.

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

</details>

<details open><summary><a href="#2-3">2.3 Dataset specification</a></summary><a id="2-3"></a>

<ul><details open><summary><a href="#2-3-1">2.3.1 Imagenet Dataset </a></summary><a id="2-3-1"></a>

Expected folder structure at top level.

```yaml
data_dir/
└── imagenet/
    ├── train/
    └── val/
```

Information about the dataset you want to use is provided in the `dataset` section of the configuration file, as shown in the YAML code below.

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
  test_path: ""                              # Optional test dataset path for evaluation (defaults to validation data)
  quantization_path: ""                      # Optional quantization dataset path (defaults to training data)
  workers: 4                                   # Number of data loader worker processes
  worker_seeding: all
  pin_mem: false                               # Enable pinned memory for DataLoader (faster host→GPU transfer)
  no_prefetcher: true                          # Disable data prefetcher (use standard DataLoader)
  use_multi_epochs_loader: false               # Use multi-epoch DataLoader to reduce startup overhead
  #data_download: True                         # Valid for food101 and flowers102 datasets.
```

If test set path is not provided to evaluate the model accuracy after training and quantization, then the validation set is used as the test set. If `quantization_path` path is not provided then a `0.01` (i.e. `quantization_split`) portion of training data will be used for quantization. 

The `class_names` attribute specifies the classes in the dataset. This information must be provided in the YAML file. If the `class_names` attribute is absent, the `classes_file_path` argument can be used as an alternative, pointing to a text file containing the class names. Alternatively, the class names can be deduced from the folder names of each class.

`train_split` and `val_split` are optional variables just for the `imagenet` dataset. Useful in scenarios when your training and validation folders are renamed to something other than "train" and "val". The standard ImageNet dataset has "train" and "val" folders inside the "imagenet" directory.

`workers` is number of worker processes used by the data loader.

`pin_mem` is Enables pinned memory for faster CPU → GPU data transfer.

`no_prefetcher` disables the data prefetcher and uses the standard PyTorch DataLoader.

`use_multi_epochs_loader` uses a multi-epoch DataLoader to reduce worker startup overhead.
</ul></details>

<ul><details open><summary><a href="#2-3-2">2.3.2 Custom Dataset (Imagenet-like)</a></summary><a id="2-3-2"></a>

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


</ul></details>

</details>

<details open><summary><a href="#2-4">2.4 Dataset preprocessing</a></summary><a id="2-4"></a>

Images need to be rescaled and resized before they can be used. This is specified in the 'preprocessing' section that is required in all the operation modes.

The 'preprocessing' section for this tutorial is shown below.
**Note:** Only `interpolation`, `mean` and `std` are being used in torch image classification. All other values like `scale`, `offset`, `aspect_ratio`, and `color_mode` will not have any impact on the training.

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

</details>

<details open><summary><a href="#2-5">2.5 Data augmentation</a></summary><a id="2-5"></a>

Data augmentation is an effective technique to reduce the overfitting of a model when the dataset is too small or the classification problem to solve is too easy for the model.

The data augmentation functions to apply to the input images are specified in the `data_augmentation` section of the configuration file, as illustrated in the YAML code below.

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

</details>

<details open><summary><a href="#2-6">2.6 Loading a model</a></summary><a id="2-6"></a>

Information about the model you want to train is provided in the `model` section of the configuration file.

The YAML code below shows how you can use a MobileNet V4 model from the Model Zoo.

```yaml
model:
  framework: 'torch'                      # Deep learning framework used for the model
  model_name: 'mobilenetv4small_pt'       # Model architecture name / identifier
  pretrained: True                        # Whether to load pretrained weights
  pretrained_dataset: "imagenet"          # Dataset used to pretrain the model
  input_shape: [3, 224, 224]              # Model input shape: [channels, height, width]
  #model_path: path to .pt file.          # Optional weights, zoo has imagenet pretrained weights.
```

The `pretrained` attribute is set to "True", which indicates that you want to load the weights pretrained on the imagenet dataset and do a *transfer learning* type of training.

If `pretrained` was set to "False", no pretrained weights would be loaded in the model and the training would start *from scratch*, i.e., from randomly initialized weights.

If you are providing `.pt` file for `model_path` then you also need to provide `model_name` so that model definition can be created before loading weights from your `.pt` file.

</details>

<details open><summary><a href="#2-7">2.7 Training setup</a></summary><a id="2-7"></a>

The training setup is described in the `training` section of the configuration file, as illustrated in the example below.

```yaml
training:
  epochs: 2
  batch_size: 128
  validation_batch_size: null
  resume_training_from: null       # checkpoint path to resume training
  optimizer:
    opt: 'sgd' 
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
The `batch_size`, `epochs`, and `optimizer` attributes are mandatory. All the others are optional.

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
</details>

<details open><summary><a href="#2-8">2.8 Hydra and MLflow settings</a></summary><a id="2-8"></a>

The `mlflow` and `hydra` sections must always be present in the YAML configuration file. The `hydra` section can be used to specify the name of the directory where experiment directories are saved and/or the pattern used to name experiment directories. With the YAML code below, every time you run the Model Zoo, an experiment directory is created that contains all the directories and files created during the run. The names of experiment directories are all unique as they are based on the date and time of the run.

```yaml
hydra:
   run:
      dir: ./pt/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

The `mlflow` section is used to specify the location and name of the directory where MLflow files are saved, as shown below:

```yaml
mlflow:
   uri: ./pt/src/experiments_outputs/mlruns
```

Note: For torch we only output logs for tensorboard. So MLflow is not valid. 

</details>
</ul>
</details>

<details open>
<summary><a href="#3"><b>3. Train your model</b></a></summary><a id="3"></a>

To launch your model training using a real dataset, run the following command from the **stm32ai-modelzoo-services/image_classification/ folder:

**CPU Mode** 
To train the model on CPU (not recommended as the model training is very intensive and needs a lot of computational resources and needs GPUs for realistic training)

```bash
CUDA_VISIBLE_DEVICES=-1 python stm32ai_main.py --config-path ./config_file_examples_pt/ --config-name training_config.yaml
```

```CUDA_VISIBLE_DEVICES=-1``` is used when training the model on CPU where GPU is also present and CUDA is installed. 


**GPU Mode** 
To train the model on GPU, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python stm32ai_main.py --config-path ./config_file_examples_pt/ --config-name training_config.yaml
```
```CUDA_VISIBLE_DEVICES=0``` is used when training the model on GPU:0. This is optional if you have only one GPU in your machine and if not provided, training uses GPU:0 by default. 

**Multi Mode**
Multi GPU model is supported for all models using following command.  

```bash
torchrun --nproc_per_node=4 stm32ai_main.py --config-path ./config_file_examples_pt/ --config-name training_config.ya
```

Above command runs the model on multiple GPUs. Please make sure to use multi-GPU mode in case more than one GPU is available to speedup the training process. Another important point to be considered is that the `batch_size` provided in config file gets multiplied by number of GPUs used for training which means a default `batch_size` of 128 provided in cfg file effectively becomes 128*4 batch size while training on 4 GPUs using multi GPU mode.

The resulting trained `.pth.tar` and `.onnx` model can be found in the corresponding **experiments_outputs/<YYYY-DD-MM-HH-MM>** folder where YYYY-DD-MM-HH-MM is current date and time. 

</details>

<details open>
<summary><a href="#4"><b>4. Visualize training results</b></a></summary><a id="4"></a>
<ul>
<details open><summary><a href="#4-1">4.1 Saved results</a></summary><a id="4-1"></a>

Training summary is provided in the **pt/src/experiments_outputs/<data-time>/saved_models/** folder.

</details>

<details open><summary><a href="#4-2">4.2 Run TensorBoard</a></summary><a id="4-2"></a>

To visualize the training curves logged by TensorBoard, go to **pt/src/experiments_outputs/<data-time>/saved_models/** and run the following command:

```bash
tensorboard --logdir logs
```

This will start a server and its address will be displayed. Use this address in a web browser to connect to the server.
Then, using the web browser, you will able to explore the learning curves and other training metrics.
</details>


</ul>
</details>

<details open>
<summary><a href="#5"><b>5. Advanced settings</b></a></summary><a id="5"></a>




<details open><summary><a href="#5-1">5.1 Transfer learning</a></summary><a id="5-1"></a>

Transfer learning is a popular training methodology that is used to take advantage of models trained on large datasets, such as imagenet. The Model Zoo features that are available to implement transfer training are presented in the next sections.

<ul>
<details open><summary><a href="#5-1-1">5.1.1 Using imagenet pretrained weights</a></summary><a id="5-1-1"></a>

Weights pretrained on the imagenet dataset are available for the MobileNetv1, MobileNetv2, MobileNetv4 etc models.

If you want to use imagenet pretrained weights, you need to add the `pretrained` attribute to the `model:` section of the configuration file and set it to 'True', as shown in the YAML code below.

```yaml
model:
  framework: 'torch'                      # Deep learning framework used for the model
  model_name: 'mobilenetv4small_pt'       # Model architecture name / identifier
  pretrained: True                        # Whether to load pretrained weights
  pretrained_dataset: "imagenet"          # Dataset used to pretrain the model
  input_shape: [3, 224, 224]              # Model input shape: [channels, height, width]
  model_path:                             # Optional: path to your own weights file
```

By default, no pretrained weights are loaded. If you want to make it explicit that you are not using the imagenet weights, you may add the `pretrained` attribute and set it to `False`.

You can provide your own weights file in `model_path`.

</ul>
</details>
</details>
<details open><summary><a href="#5-2">5.2 Train, quantize, benchmark, and evaluate your model</a></summary><a id="5-2"></a>

In case you want to train and quantize a model, you can either launch the training operation mode followed by the quantization operation on the trained model (please refer to the quantization **[README.md](./README_QUANTIZATION.md)** that describes in detail the quantization part) or you can use chained services like launching [chain_tqe](../config_file_examples_pt/chain_tqe_config.yaml) example with the command below:

```bash
python stm32ai_main.py --config-path ./config_file_examples_pt/ --config-name chain_tqe_config.yaml
```

This specific example trains a MobileNet V2 model with imagenet pre-trained weights, fine-tunes it by retraining the latest seven layers but the fifth one (this only as an example), and quantizes it to 8-bits using quantization_split (30% in this example) of the train dataset for calibration before evaluating the quantized model.

In case you also want to execute a benchmark on top of training and quantizing services, it is recommended to launch the chain service called [chain_tqeb](../config_file_examples_pt/chain_tqeb_config.yaml) that stands for train, quantize, evaluate, benchmark like the example with the command below:

```bash
python stm32ai_main.py --config-path ./config_file_examples_pt/ --config-name chain_tqeb_config.yaml
```

</details>
</ul>
</details>

<!-- <details open><summary><a href="#5-3">5.3 Resuming a training</a></summary><a id="5-3"></a>

To be continued. . . 

</details> -->
