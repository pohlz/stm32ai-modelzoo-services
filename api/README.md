## 🧠 `get_model` API

The `get_model` function allows you to dynamically retrieve a model from the internal model registry based on key parameters like architecture, dataset, task type, and framework.

---

### Function Signature

```python
get_model(cfg)
```


### Variables expected from CFG

| Name             | Type      | Required | Description                                                                 |
|------------------|-----------|----------|-----------------------------------------------------------------------------|
| `model_name`     | `str`     | ✅        | Name of the model architecture (e.g., `'resnet18'`, `'mobilenet_v2'`)      |
| `use_case`      | `str`     | ✅        | Task type (e.g., `"image_classification"`, `'segmentation'`)                |
| `framework`      | `str`     | ✅        | Framework to use: `'torch'` or `'tensorflow'`                               |

Plus many more according to model you are loading. 

### Returns

- A model object initialized with the specified parameters.
- If `pretrained=True`, the model includes pretrained weights (if available).

---

### Notes

- If the combination of parameters doesn't match a known model, the function will raise an error.
- You can pass additional model-specific keyword arguments such as `dropout_rate=0.3` or `input_shape=(224, 224, 3)` depending on the framework.


### Example Usage

```python
from api import get_model

# load hydra config with required variables, It must have:
cfg.use_case = "image_classification"
cfg.model.framework = "torch"
cfg.model.model_name = "xcit_tiny_12_p8_224_timm"
model = get_model(cfg)
```

## `list_models` API

A helper function to list all existing models based on text filters.  
You can provide a string or a list of strings to filter by model name, dataset name, task type, framework, or combined keys like `"model_name_dataset_name"`.

---

### Function Signature

```python
list_models(
    filter_string='',
    match_all=True,
    print_table=True,
    with_checkpoint=False,
)
```

### Parameters

| Name              | Type           | Required | Description                                                                                               |
|-------------------|----------------|----------|-----------------------------------------------------------------------------------------------------------|
| `filter_string`   | `str` or `list`| ❌        | String or list of strings containing model name, dataset name, task type, framework, or combined filters. |
| `match_all`       | `bool`         | ❌        | If `True`, only models matching *all* keywords (intersection) are returned. If `False`, models matching *any* keyword are returned. (default: `True`) |
| `print_table`     | `bool`         | ❌        | Whether to print a table of matched models. Returns a list. (default: `True`)           |
| `with_checkpoint` | `bool`         | ❌        | Whether to include only models that have available checkpoints. (default: `False`)                         |

### Returns

- If `print_table=True`, prints a formatted table of matched models.
- If `print_table=False`, returns a list of matched model identifiers.

---

### Notes

- The `filter_string` can be a single string or a list of strings to filter the models.
- Setting `match_all=True` returns models matching *all* provided keywords (logical AND).
- Setting `match_all=False` returns models matching *any* of the keywords (logical OR).
- This function is useful to quickly explore what models are available in the registry based on flexible filters.

### Example Usage

```python

from api import list_models

# List and print models matching both 'resnet' and 'imagenet'
list_models(filter_string=['resnet', 'imagenet'], match_all=True)

# Get list of models matching either 'mobilenet' or 'cifar10', without printing
models = list_models(filter_string=['mobilenet', 'cifar10'], match_all=False, print_table=False)
print(models)
```

### Keywords for model names

```
    'airnext', 'aim', 'alexnet', 'bagnet', 'beit', 'botnet', 'botnext', 'byobnet', 'cait', 'caformer',
    'channelnet', 'coat', 'convformer', 'convnext', 'darknet', 'darts', 'deit', 'densnet', 'dicenet', 'diracnet',
    'dla', 'dpn', 'drnc', 'drnd', 'edgenet', 'edgenext', 'efficientformer', 'efficientnet', 'espnet', 'eva',
    'fasternet', 'fbnet', 'fishnet', 'focalnet', 'gmlp', 'gernet', 'ghostnet', 'googlenet', 'halonet', 'halonext',
    'hardcorenas', 'hardnet', 'hgera', 'hgnet', 'hrnet', 'igc', 'inception', 'irevnet', 'lcnet', 'mambaout',
    'mixer', 'mixnet', 'mobilenet', 'msdnet', 'nasnet', 'nest', 'nfnet', 'pvt', 'peleenet', 'pit', 'proxylessnas',
    'pyramidnet', 'rdnet', 'regnet', 'res2net', 'resattnet', 'resmlp', 'resnet', 'resnest', 'resnext', 'rexnet',
    'scnet', 'selecsls', 'senet', 'sequencer', 'shufflenet', 'sknet', 'sparsenet', 'sqnxt', 'squeezenet',
    'starnet', 'swiftformer', 'swin', 'tinynet', 'tnt', 'tresnet', 'twins', 'vit', 'vitamin', 'vgg', 'volo',
    'vovnet', 'xcit', 'xception', 'zfnet'
```

## `get_dataloaders` API

Loads and returns training and testing dataloaders for a specified dataset, task type, and framework.

### 🔧 Function Signature

```python
get_dataloaders(cfg)
```

### Variables expected from CFG
| Name           | Type   | Required | Description                                                                                                                                                                                                                                                                                                                                                                    |
| -------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data_root`    | `str`  | ✅     | Path to the root directory containing the dataset folder.                                                                                                                                                                                                                                                                                                                             |
| `dataset_name` | `str`  | ✅     | Name of the dataset (e.g., `'imagenet'`, `'flowers102'`, etc.).                                                                                                                                                                                                                                                                                                                       |
| `use_case`    | `str`  | ✅     | Task type (e.g., `"image_classification"`, `'detection'`, `'segmentation'`).                                                                                                                                                                                                                                                                                                         |
| `framework`    | `str`  | ✅     | Framework or model name used to select the appropriate dataset wrapper.     
| `prediction_path`    | `str`  | ❌     | Path to folder containing images or folders of images.                                                                                                                                              
| `qunatization_path`    | `str`  |  ❌    | Path to folder containing images or folders of images.                                                                                                                                                              |
| `**kwargs`     | `dict` | ❌      | Additional keyword arguments passed to the dataset loader. Common options include:<ul><li>`download` (`bool`) – Whether to download the dataset (if supported).</li><li>`batch_size` (`int`) – Batch size for the dataloaders.</li><li>`img_size` (`int`)</li><li>`num_workers` (`int`) – Number of subprocesses for data loading.</li></ul> |


## Returns
Returns a dictionary containing PyTorch dataloaders for training and testing splits.
```python
Dict
{
    'train': torch.utils.data.DataLoader,
    'test' : torch.utils.data.DataLoader,
    'valid': torch.utils.data.DataLoader,
    'pred' : torch.utils.data.DataLoader,
    'quant': torch.utils.data.DataLoader,
}
```

## Example Usage

```python
from api import get_dataloaders

cfg.dataset.data_dir= "/neutrino/datasets/"
cfg.dataset.dataset_name = "imagenet"
cfg.use_case = "image_classification"
cfg.model.framework = "torch"
# plus other variables like batch_size, input_size, aumentations, num_classes etc etc
dataloaders = get_dataloaders(cfg)
train_loader = dataloaders['train']
test_loader = dataloaders['test']
```

## Expected folder structure in standard datasets

- FLOWERS102 shud have 'jpg' folder, 'setid.mat' and 'imagelabels.mat' inside.
- VWW should have 'all' , 'annotations/instances_train.json' and 'annotations/instances_train.json' inside.
- FOOD101 should have 'images' and 'meta' folder inside
- IMAGENET should have 'train' and 'val' folder inside it with sub folders of classes.

## Project Structure Overview
```
project/
├── apis/
│   ├── get_model.py
│   ├── get_dataset.py
│   └── get_trainers.py
│
├── common/
│   ├── blocks/                     # Shared building blocks
│   ├── registry/                  
│   ├── model_registry.py           # Model registry system
│   ├── dataset_registry.py         # Dataset registry system
│   └── trainer_registry.py         # Trainer registry system
│
├── image_classification/
│   ├── config.py                   # Config file (e.g. for args, yaml loading, etc.)
│   ├── main.py                     # Entry point: loads config, gets model/dataset/trainer, trains
│   └── pt/
│       ├── src/
│       │   ├── models/             # Model definitions
│       │   ├── dataset/            # Dataset definitions
│       │   └── trainers/           # Training logic
│       │
│       └── wrapper/
│           └── models/             # Wrapper to unify model interfaces and register with registry
│
└── README.md
```

## System Flow
```
main.py (reads config)
    ↓
apis/ (get_model, get_dataset, get_trainers)
    ↓
common/registry (e.g., model_registry.get)
    ↓
image_classification/pt/wrapper/models/  (model registration)
    ↓
image_classification/pt/src/models/      (actual model implementation)
```


## Important to Avoid Circular Imports

Make sure:

common/ does not import anything from image_classification/

src/trainers/ does not call get_model() from apis/ — it should just accept already-prepared objects

wrapper/models/ only handles registration logic (and doesn't depend on training or dataset logic)

