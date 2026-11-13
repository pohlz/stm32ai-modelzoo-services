# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from image_classification.tf.src.datasets.utils import preprocess_data, download_dataset
from image_classification.tf.src.datasets.emnist_byclass import load_emnist
from image_classification.tf.src.datasets import prepare_kwargs_for_dataloader


__all__ = ['get_emnist']

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='emnist_byclass', use_case="image_classification")
def get_emnist(cfg): 

    """
    Loads the images from the emnist dataset, pre-process them and return training, validation, and test tf.data.Datasets.
    
    Args:
        cfg (dict): configuration parameters
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, 
        quantization, test and prediction datasets once pre-processed.

    """

    # Get dataloader kwargs
    args = prepare_kwargs_for_dataloader(cfg)

    # Add possibility to download the dataset here?
    if args['data_dir'] and args['training_path'] == None and\
        cfg.operation_mode in ['training', 'chain_tqe', 'chain_tqeb']:
        args['training_path'] = download_dataset(data_root=args['data_dir'],
                                                 dataset_name='emnist_byclass',
                                                 data_download=args['data_download'])
    # Creates datasets
    dataloaders = load_emnist(training_path=args['training_path'],
                             quantization_path=args['quantization_path'],
                             test_path=args['test_path'],
                             prediction_path=args['prediction_path'],
                             quantization_split=args['quantization_split'],
                             class_names=args['class_names'],
                             image_size=args['image_size'],
                             interpolation=args['interpolation'],
                             aspect_ratio=args['aspect_ratio'],
                             color_mode=args['color_mode'],
                             batch_size=args['batch_size'],
                             seed=args['seed'])
    
    # Pre-processes loaded data
    dataloaders = preprocess_data(dataloaders=dataloaders,
                                  scale=args['rescaling_scale'],
                                  offset=args['rescaling_offset'],
                                  mean=args['normalization_mean'],
                                  std=args['normalization_std'])
 
    return dataloaders
