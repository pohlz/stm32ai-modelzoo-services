# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from hand_posture.tf.src.preprocessing.data_loader import load_dataset
from hand_posture.tf.src.datasets.utils import prepare_kwargs_for_dataloader

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='ST_handposture_dataset', use_case="hand_posture")
def get_ST_handposture_dataset(cfg):
    # Get dataloader kwargs
    args = prepare_kwargs_for_dataloader(cfg)
    
    # Creates datasets
    train_ds, val_ds,test_ds = load_dataset(dataset_name= args['dataset_name'],
                 training_path=args['training_path'],
                 validation_path=args['validation_path'],
                 test_path=args['test_path'],
                 validation_split=args['validation_split'],
                 class_names=args['class_names'],
                 Max_distance=args['Max_distance'],
                 Min_distance=args['Min_distance'],
                 Background_distance=args['Background_distance'],
                 batch_size=args['batch_size'],
                 seed=args['seed'])
    data_loaders = {'train': train_ds, 'valid': val_ds, 'test': test_ds}

    return data_loaders