# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import pickle
import numpy as np
import tensorflow as tf
import onnxruntime
from pathlib import Path
from typing import Tuple, List
from image_classification.tf.src.preprocessing import preprocessing
from omegaconf import OmegaConf, DictConfig
from torchvision.datasets.utils import download_and_extract_archive
import shutil


def _get_path_dataset(path: str,
                     class_names: list[str],
                     seed: int,
                     shuffle: bool = True) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from a dataset root directory path.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            class_a:
                a_image_1.jpg
                a_image_2.jpg
            class_b:
                b_image_1.jpg
                b_image_2.jpg

    Args:
        path (str): Path of the dataset folder.
        class_names (list(str)): List of the classes names.
        seed (int): seed when performing shuffle.
        shuffle (bool): Initial shuffling (or not) of input files names.

    Returns:
        dataset(tf.data.Dataset) -> dataset with a tuple (path, label) of each sample.
    """

    data_list = []
    labels = []

    for d in os.scandir(path):
        if d.is_dir():
            labels.append(d.name)
        elif Path(d.name).suffix.lower() in [".jpg",".jpeg",".png"]:
            data_list.append((os.path.join(path,d.name),0))

    if labels == []:
        imgs = os.listdir(path)
        data_list = sorted(data_list)

    else :
        data_list = []
        for label in class_names:
            assert label in labels, f"[ERROR] label {label} not found in {path}"
        for idx, label in enumerate(sorted(class_names)):
            imgs = os.listdir(os.path.join(path, label))
            data_list.extend(sorted([(os.path.join(path,label,img), idx) for img in imgs]))

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(data_list)

    imgs, labels = zip(*data_list)
    dataset = tf.data.Dataset.from_tensor_slices((list(imgs), list(labels)))

    return dataset


def _preprocess_function(data_x : tf.Tensor,
                         data_y : tf.Tensor,
                         image_size: tuple[int],
                         interpolation: str,
                         aspect_ratio: str,
                         color_mode: str,
                         label_mode: str,
                         num_classes: int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Load images from path and apply necessary transformations.

    Args:
        data_x (tf.Tensor): input image
        data_y (tf.Tensor): input label
        image_size (tuple[int]): Size of the input images to resize them to.
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (str): Cropping method specifying whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        label_mode (str): Mode for generating the labels for the images.
        num_classes (int): number of classes in the considered use-case.

    Returns:
        Pre-processed image and label, with special treatement for imagenet

    """
    # width, height = image_size
    height, width = image_size
    intermediate_resize = 256
    channels = 1 if color_mode == "grayscale" else 3

    image = tf.io.read_file(data_x)
    image = tf.image.decode_image(image, channels=channels, expand_animations=False)
    if aspect_ratio == "fit":
        if height==224 and width==224 and num_classes==1000:
            # imagenet trick...
            image = tf.image.resize(image, [intermediate_resize, intermediate_resize], method=interpolation, preserve_aspect_ratio=False, antialias=True)
            image = tf.image.central_crop(image, central_fraction=height/intermediate_resize)
        else:
            image = tf.image.resize(image, [height, width], method=interpolation, preserve_aspect_ratio=False, antialias=True)
    else:
        image = tf.image.resize_with_crop_or_pad(image, height, width)

    if label_mode == "categorical":
        data_y = tf.keras.utils.to_categorical(data_y, num_classes)

    return image, data_y


def _preprocess_prediction_function(data_x : tf.Tensor,
                                    data_y : tf.Tensor,
                                    image_size: tuple[int],
                                    interpolation: str,
                                    aspect_ratio: str,
                                    color_mode: str,
                                    label_mode: str,
                                    num_classes: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    """
    Load images from path and apply necessary transformations.

    Args:
        data_x (tf.Tensor): input image
        data_y (tf.Tensor): input label
        image_size (tuple[int]): Size of the input images to resize them to.
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (str): Cropping method specifying whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        label_mode (str): Mode for generating the labels for the images.
        num_classes (int): number of classes in the considered use-case.

    Returns:
        Pre-processed image and label, with special treatement for imagenet

    """
#    width, height = image_size
    height, width = image_size
    intermediate_resize = 256
    channels = 1 if color_mode == "grayscale" else 3

    image = tf.io.read_file(data_x)
    image = tf.image.decode_image(image, channels=channels, expand_animations=False)
    if aspect_ratio == "fit":
        if height==224 and width==224 and num_classes==1000:
            # imagenet trick...
            image = tf.image.resize(image, [intermediate_resize, intermediate_resize], method=interpolation, preserve_aspect_ratio=False, antialias=True)
            image = tf.image.central_crop(image, central_fraction=height/intermediate_resize)
        else:
            image = tf.image.resize(image, [height, width], method=interpolation, preserve_aspect_ratio=False, antialias=True)
    else:
        image = tf.image.resize_with_crop_or_pad(image, height, width)

    if label_mode == "categorical":
        data_y = tf.keras.utils.to_categorical(data_y, num_classes)

    return image, data_x


def get_train_val_ds(training_path: str,
                     image_size: tuple[int] = None,
                     label_mode: str = None,
                     class_names: list[str] = None,
                     interpolation: str = None,
                     aspect_ratio: str = None,
                     color_mode: str = None,
                     validation_split: float = None,
                     batch_size: int = None,
                     seed: int = None,
                     shuffle: bool = True,
                     to_cache: bool = False
                     ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the images under a given dataset root directory and returns training
    and validation tf.Data.datasets.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            class_a:
                a_image_1.jpg
                a_image_2.jpg
            class_b:
                b_image_1.jpg
                b_image_2.jpg

    Args:
        training_path (str): Path to the directory containing the training images.
        image_size (tuple[int]): Size of the input images to resize them to.
        label_mode (str): Mode for generating the labels for the images.
        class_names (list[str]): List of class names to use for the images.
        interpolation (float): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        validation_split (float): Fraction of the data to use for validation.
        batch_size (int): Batch size to use for training and validation.
        seed (int): Seed to use for shuffling the data.
        shuffle (bool): Whether or not to reshuffle at each iteration the dataset.
        to_cache (bool): Whether or not to cache the datasets.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    # When calling this function using the config file data, some of the arguments
    # may be used but equal to None (happens when an attribute is missing in the
    # config file or has no value). For this reason, all the arguments in the
    # definition of the function defaults to None and we set default values here
    # in case the function is called in another context with missing arguments.

    label_mode = label_mode if label_mode else "int"
    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    color_mode = color_mode if color_mode else "rgb"
    validation_split = validation_split if validation_split else 0.2
    batch_size = batch_size if batch_size else 32

    preprocess_params = (image_size,
                         interpolation,
                         aspect_ratio,
                         color_mode,
                         label_mode,
                         len(class_names))

    dataset = _get_path_dataset(training_path, class_names, seed=seed)

    train_size = int(len(dataset)*(1-validation_split))
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    if shuffle:
        train_ds = train_ds.shuffle(len(train_ds), reshuffle_each_iteration=True, seed=seed)

    train_ds = train_ds.map(lambda *data : _preprocess_function(*data,*preprocess_params))
    val_ds = val_ds.map(lambda *data : _preprocess_function(*data,*preprocess_params))

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    if to_cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def get_ds(data_path: str = None,
           label_mode: str = None,
           class_names: list[str] = None,
           image_size: tuple[int] = None,
           interpolation: str = None,
           aspect_ratio: str = None,
           color_mode: str = None,
           batch_size: int = None,
           seed: int = None,
           shuffle: bool = True,
           to_cache: bool = False) -> tf.data.Dataset:
    """
    Loads the images from the given dataset root directory and returns a tf.data.Dataset.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            class_a:
                a_image_1.jpg
                a_image_2.jpg
            class_b:
                b_image_1.jpg
                b_image_2.jpg

    Args:
        data_path (str): Path to the directory containing the images.
        label_mode (str): Mode for generating the labels for the images.
        class_names (list[str]): List of class names to use for the images.
        image_size (tuple[int]): Size of the input images to resize them to.
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        batch_size (int): Batch size to use for the dataset.
        seed (int): Seed to use for shuffling the data.
        shuffle (bool): Whether or not to reshuffle the dataset at each iteration.
        to_cache (bool): Whether or not to cache the dataset.

    Returns:
        tf.data.Dataset: Dataset containing the images.
    """
    # When calling this function using the config file data, some of the arguments
    # may be used but equal to None (happens when an attribute is missing in the
    # config file or has no value). For this reason, all the arguments in the
    # definition of the function defaults to None and we set default values here
    # in case the function is called in another context with missing arguments.

    label_mode = label_mode if label_mode else "int"
    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    color_mode = color_mode if color_mode else "rgb"
    batch_size = batch_size if batch_size else 32

    preprocess_params = (image_size,
                         interpolation,
                         aspect_ratio,
                         color_mode,
                         label_mode,
                         len(class_names))

    dataset = _get_path_dataset(data_path, class_names, seed=seed)

    if shuffle:
        dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True, seed=seed)

    dataset = dataset.map(lambda *data: _preprocess_function(*data, *preprocess_params))
    dataset = dataset.batch(batch_size)

    if to_cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_prediction_ds(data_path: str = None,
           label_mode: str = None,
           class_names: list[str] = None,
           image_size: tuple[int] = None,
           interpolation: str = None,
           aspect_ratio: str = None,
           color_mode: str = None,
           batch_size: int = None,
           seed: int = None,
           shuffle: bool = True,
           to_cache: bool = False) -> tf.data.Dataset:
    """
    Loads the images from the given dataset root directory and returns a tf.data.Dataset.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            class_a:
                a_image_1.jpg
                a_image_2.jpg
            class_b:
                b_image_1.jpg
                b_image_2.jpg

    Args:
        data_path (str): Path to the directory containing the images.
        label_mode (str): Mode for generating the labels for the images.
        class_names (list[str]): List of class names to use for the images.
        image_size (tuple[int]): Size of the input images to resize them to.
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        batch_size (int): Batch size to use for the dataset.
        seed (int): Seed to use for shuffling the data.
        shuffle (bool): Whether or not to reshuffle the dataset at each iteration.
        to_cache (bool): Whether or not to cache the dataset.

    Returns:
        tf.data.Dataset: Dataset containing the images and the path.
    """
    # When calling this function using the config file data, some of the arguments
    # may be used but equal to None (happens when an attribute is missing in the
    # config file or has no value). For this reason, all the arguments in the
    # definition of the function defaults to None and we set default values here
    # in case the function is called in another context with missing arguments.

    label_mode = label_mode if label_mode else "int"
    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    color_mode = color_mode if color_mode else "rgb"
    batch_size = batch_size if batch_size else 1

    preprocess_params = (image_size,
                         interpolation,
                         aspect_ratio,
                         color_mode,
                         label_mode,
                         len(class_names))

    dataset = _get_path_dataset(data_path, class_names, seed=seed)

    if shuffle:
        dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True, seed=seed)

    dataset = dataset.map(lambda *data: _preprocess_prediction_function(*data, *preprocess_params))
    dataset = dataset.batch(batch_size)

    if to_cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def preprocess_data(dataloaders: dict = None,
                    scale: float = None,
                    offset: float = None,
                    mean: tuple[float] = None,
                    std: tuple[float] = None
                    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    """
    Load images using dataloaders and apply rescaling/normalization.

    Args:
        dataloaders (dict): dictionnary of tf.data.Dataset
        scale (float): simple scaling of images pixels
        offset (float): simple offsetting of images pixels
        mean (tuple[float]): image normalization, centering channel wise
        std (tuple[float]): image normalization, reduction channel wise

    Returns:
        Dictionnary of rescaled and normalised datasets

    """

    train_ds=dataloaders['train']
    val_ds=dataloaders['valid']
    quantization_ds=dataloaders['quantization']
    test_ds=dataloaders['test']
    predict_ds=dataloaders['predict']

    train_ds = preprocessing(dataset=train_ds,
                             scale=scale,
                             offset=offset,
                             mean=mean,
                             std=std)
    val_ds = preprocessing(dataset=val_ds,
                           scale=scale,
                           offset=offset,
                           mean=mean,
                           std=std)
    quantization_ds = preprocessing(dataset=quantization_ds,
                                    scale=scale,
                                    offset=offset,
                                    mean=mean,
                                    std=std)
    test_ds = preprocessing(dataset=test_ds,
                            scale=scale,
                            offset=offset,
                            mean=mean,
                            std=std)
    predict_ds = preprocessing(dataset=predict_ds,
                               scale=scale,
                               offset=offset,
                               mean=mean,
                               std=std)

    return {'train': train_ds, 'valid': val_ds, 'quantization': quantization_ds, 'test': test_ds, 'predict': predict_ds}


def load_cifar_batch(fpath, label_key="labels") -> Tuple:
    """
    Internal utility for parsing CIFAR data.

    Args:
        fpath (str): File path of the CIFAR data batch.
        label_key (str, optional): Key name for the labels in the CIFAR data. Defaults to "labels".

    Returns:
        data (numpy.ndarray): CIFAR data.
        labels (numpy.ndarray): Labels corresponding to the CIFAR data.
    """
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="bytes")

        # Decode utf8 keys
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode("utf8")] = v
        d = d_decoded

    data = d["data"]
    labels = d[label_key]

    # Reshape the data array
    data = data.reshape(data.shape[0], 3, 32, 32)

    return data, labels



def prepare_kwargs_for_dataloader(cfg: DictConfig):

    """
    Extract image size from the model and prepare dataloader args

    Args:
        cfg (dict): dictionnary of parameters

    Returns:
        Dictionnary of parameters for dataloader

    """
    model_path = cfg.model.model_path
    model_tmp = None
    input_shape = None
    image_size = None

    if model_path is not None:
        file_extension = str(model_path).split('.')[-1]
        input_shape = cfg.model.input_shape
        if file_extension in ['h5', 'keras']:
            model_tmp = tf.keras.models.load_model(model_path, compile=False)
            input_shape = model_tmp.inputs[0].shape[1:]
            image_size = tuple(input_shape)[:-1]
        elif file_extension == 'tflite':
            model_tmp = tf.lite.Interpreter(model_path=model_path)
            model_tmp.allocate_tensors()
            # Get the input details
            input_details = model_tmp.get_input_details()
            input_shape = tuple(input_details[0]['shape'])
            image_size = tuple(input_shape)[-3:-1]
        elif file_extension == 'onnx':
            model_tmp = onnxruntime.InferenceSession(model_path)
            # Get the model input shape
            input_shape = model_tmp.get_inputs()[0].shape
            input_shape = tuple(input_shape)[-3:]
            image_size = tuple(input_shape)[1:]
    else:
        # If no model path is provided, use the input_shape from config
        input_shape = cfg.model.input_shape
        if input_shape is not None:
            image_size = tuple(input_shape)[:-1]

    print("input_shape=", input_shape)
    print("image_size=", image_size)

    # Prepare kwargs
    batch_size = getattr(cfg.training, 'batch_size', 32) if cfg.training else 32
    dataloader_kwargs = {
        'training_path': getattr(cfg.dataset, 'training_path', None),
        'validation_path': getattr(cfg.dataset, 'validation_path', None),
        'quantization_path': getattr(cfg.dataset, 'quantization_path', None),
        'test_path': getattr(cfg.dataset, 'test_path', None),
        'prediction_path': getattr(cfg.dataset, 'prediction_path', None),
        'validation_split': getattr(cfg.dataset, 'validation_split', None),
        'quantization_split': getattr(cfg.dataset, 'quantization_split', None),
        'class_names': getattr(cfg.dataset, 'class_names', None),
        'image_size': image_size,
        'interpolation': getattr(cfg.preprocessing.resizing, 'interpolation', None),
        'aspect_ratio': getattr(cfg.preprocessing.resizing, 'aspect_ratio', None),
        'color_mode': getattr(cfg.preprocessing, 'color_mode', None),
        'batch_size': batch_size,
        'seed': getattr(cfg.dataset, 'seed', 127),
        'rescaling_scale': getattr(cfg.preprocessing.rescaling, 'scale', 1.0/255.0),
        'rescaling_offset': getattr(cfg.preprocessing.rescaling, 'offset', 0),
        'normalization_mean': getattr(cfg.preprocessing.normalization, 'mean', 0.0),
        'normalization_std': getattr(cfg.preprocessing.normalization, 'std', 1.0),
        'data_dir':  getattr(cfg.dataset, 'data_dir', './datasets/'),
        'data_download': getattr(cfg.dataset, 'data_download', True),
    }

    return dataloader_kwargs

def _copy_food_101_images(images_dir:str,
                 file_list_path:str,
                 dest_root:str):
    """
    Copy images listed in the file_list_path from images_dir to dest_root preserving directory structure.

    Args:
        images_dir (str): source images directory.
        file_list_path (str): path to the file containing list of images to copy.
        dest_root (str): destination root directory where images will be copied.

    """
    with open(file_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for relative_path in lines:
        src_path = os.path.join(images_dir, relative_path + '.jpg')
        dest_path = os.path.join(dest_root, relative_path + '.jpg')
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)

def _split_food101(data_root:str='./datasets/'):
    """
    Splits the food-101 dataset into train and test folders based on the meta/train.txt and meta/test.txt files.

    Args:
        data_root (str): root directory where the food-101 dataset is located.

    """
    image_dir = os.path.join(data_root, 'food-101', 'images')
    meta_dir = os.path.join(data_root, 'food-101', 'meta')
    train_paths = os.path.join(meta_dir, 'train.txt')
    test_paths = os.path.join(meta_dir, 'test.txt')
    train_dir = os.path.join(data_root, 'food-101', 'train')
    test_dir = os.path.join(data_root, 'food-101', 'test')
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print('[INFO] : found food-101 train and test folders.')
    else:
        print('[INFO] : Splitting food-101 dataset into train and test folders...')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        print("[INFO] : Copying training images...")
        _copy_food_101_images(images_dir=image_dir,
                     file_list_path=train_paths,
                     dest_root=train_dir)
        print("[INFO] : Copying test images...")
        _copy_food_101_images(images_dir=image_dir,
                     file_list_path=test_paths,
                     dest_root=test_dir)

def _check_dataset_already_exists(data_root:str='./datasets/',
                                  dataset_name:str='',
                                  data_download: bool=True) -> bool:
    """
    Checks if the dataset_dir already exists in the specified root directory.
    Args:
        data_root (str): Directory where to check for the dataset.
        dataset_name (str): name of the dataset to be checked
        data_download (bool): whether to download the dataset if not found
    """
    print(f"[INFO] : Checking if \"{dataset_name}\" dataset is already downloaded in \"{data_root}\"...")
    if dataset_name == 'cifar10':
        data_folder = os.path.join(data_root, 'cifar-10-batches-py')
        files = ["batches.meta", "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "readme.html", "test_batch"]
        _exists =  all(os.path.exists(os.path.join(data_folder, f)) for f in files)

    elif dataset_name == 'cifar100':
        data_folder = os.path.join(data_root, 'cifar-100-python')
        files = ["meta", "test", "train"]
        _exists = all(os.path.exists(os.path.join(data_folder, f)) for f in files)

    elif dataset_name == 'tf_flowers':
        _exists =  os.path.exists(os.path.join(data_root, 'flower_photos'))

    elif dataset_name == 'plant_leaf_diseases':
        _exists = os.path.exists(os.path.join(data_root, 'Plant_leave_diseases_dataset_without_augmentation'))

    elif dataset_name == 'food101':
        images_dir = os.path.join(data_root, 'food-101', 'images')
        meta_dir = os.path.join(data_root, 'food-101', 'meta')
        _exists = all(os.path.exists(folder) and os.path.isdir(folder) for folder in [images_dir, meta_dir])

    elif dataset_name == 'emnist_byclass':
        data_folder = os.path.join(data_root, 'emnist_dataset')
        _exists =  os.path.exists(os.path.join(data_folder, "emnist-byclass.mat"))

    if not _exists and not data_download:
        raise KeyError(f"[ERROR] : The dataset \"{dataset_name}\" was not found in \"{data_root}\" and \"data.data_download\" is set to False. Please set \"data.data_download\" to True to automatically the download the dataset or provide a valid path.")
    if _exists:
        print(f"[INFO] : \"{dataset_name}\" dataset found in \"{data_root}\".")
    return _exists

def download_dataset(data_root:str,
                     dataset_name: str,
                     data_download: bool) -> str:
    """
    This function downloads and extracts the specified dataset in the specified root directory.
    If the dataset is already present, it does not download it again and simply returns the path to the dataset.

    Args:
        data_root (str): Directory where to download the dataset.
        dataset_name (str): name of the dataset to be downloaded. Supported datasets are:
            - cifar10
            - cifar100
            - tf_flowers
            - food101
            - plant_leaf_diseases
            - emnist_byclass
    Returns:
        str: The path to the downloaded dataset based on the dataset structure and requirements.
    """
    if dataset_name == 'cifar10':
        if _check_dataset_already_exists(data_root=data_root,
                                         dataset_name=dataset_name,
                                         data_download=data_download):
            return os.path.join(data_root, 'cifar-10-batches-py')
        print(f'[INFO] : Files not found!\nDownloading {dataset_name} dataset in {data_root}')
        download_and_extract_archive(url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                                     download_root=data_root,
                                     md5='c58f30108f718f92721af3b95e74349a')
        return os.path.join(data_root, 'cifar-10-batches-py')

    elif dataset_name == 'cifar100':
        if _check_dataset_already_exists(data_root=data_root,
                                         dataset_name=dataset_name,
                                         data_download=data_download):
            print('data files found! Using existing files!')
            return os.path.join(data_root, 'cifar-100-python')

        print(f'[INFO] : Files not found!\nDownloading {dataset_name} dataset in {data_root}')
        download_and_extract_archive(url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                                     download_root=data_root,
                                     md5='eb9058c3a382ffc7106e4002c42a8d85')
        return os.path.join(data_root, 'cifar-100-python')

    elif dataset_name == 'tf_flowers':
        if _check_dataset_already_exists(data_root=data_root,
                                         dataset_name=dataset_name,
                                         data_download=data_download):
            return os.path.join(data_root, 'flower_photos')

        print(f'[INFO] : Files not found!\nDownloading {dataset_name} dataset in {data_root}')
        download_and_extract_archive(url='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                 download_root=data_root,
                                 md5='6f87fb78e9cc9ab41eff2015b380011d')
        return os.path.join(data_root, 'flower_photos')

    elif dataset_name == 'food101':
        if _check_dataset_already_exists(data_root=data_root,
                                         dataset_name=dataset_name,
                                         data_download=data_download):
            _split_food101(data_root)
            return os.path.join(data_root, 'food-101/train'), os.path.join(data_root, 'food-101/test')
        else:
            print(f'[INFO] : Files not found!\nDownloading {dataset_name} dataset in {data_root}')
            download_and_extract_archive(url='https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz',
                                         download_root=data_root,
                                         md5='85eeb15f3717b99a5da872d97d918f87')
        _split_food101(data_root)
        return os.path.join(data_root, 'food-101/train'), os.path.join(data_root, 'food-101/test')

    elif dataset_name == 'plant_leaf_diseases':
        if _check_dataset_already_exists(data_root=data_root,
                                         dataset_name=dataset_name,
                                         data_download=data_download):
            return os.path.join(data_root, 'Plant_leave_diseases_dataset_without_augmentation')

        print(f'[INFO] : Files not found!\nDownloading {dataset_name} dataset in {data_root}')
        download_and_extract_archive(url='https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/d29ed9b2-8a5d-4663-8a82-c9174f2c7066',
                                     download_root=data_root,
                                     md5='14ae99240aa7e7ba737bb94bd2bc87e3',
                                     filename='Plant_leave_diseases_dataset_without_augmentation.zip')
        return os.path.join(data_root, 'Plant_leave_diseases_dataset_without_augmentation')

    elif dataset_name == 'emnist_byclass':
        if _check_dataset_already_exists(data_root=data_root,
                                         dataset_name=dataset_name,
                                         data_download=data_download):
            return os.path.join(data_root, 'emnist_dataset')

        print(f'[INFO] : Files not found!\nDownloading {dataset_name} dataset in {data_root}')
        download_and_extract_archive(url='https://biometrics.nist.gov/cs_links/EMNIST/matlab.zip',
                                     download_root=data_root,
                                     md5='1bbb49fdf3462bb70c240eac93fff0e4',
                                     filename='emnist_dataset.zip')
        shutil.move(os.path.join(data_root, 'matlab'), os.path.join(data_root, 'emnist_dataset'))
        return os.path.join(data_root, 'emnist_dataset')
    else:
        raise TypeError('The choosen dataset is not supported! Please choose one of the supported datasets ["cifar10", "cifar100", "tf_flowers", "plant-leaf_diseases", "emnist_byclass", "food101"]')