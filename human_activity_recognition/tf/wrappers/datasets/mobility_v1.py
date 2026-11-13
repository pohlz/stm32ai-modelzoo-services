# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from human_activity_recognition.tf.src.datasets.mobility_v1 import load_mobility_v1
from human_activity_recognition.tf.src.datasets.utils import prepare_kwargs_for_dataloader

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='mobility_v1', use_case="human_activity_recognition")
def get_mobility_v1(cfg):
    # Get dataloader kwargs
    args = prepare_kwargs_for_dataloader(cfg)
    
    # Creates datasets
    train_ds, val_ds, test_ds = load_mobility_v1(train_path=args['training_path'],
                                                  test_path=args['test_path'],
                                                  validation_split=args['validation_split'],
                                                  class_names=args['class_names'],
                                                  input_shape=args['input_shape'],
                                                  gravity_rot_sup=args['gravity_rot_sup'],
                                                  normalization=args['normalization'],
                                                  batch_size=args['batch_size'],
                                                  seed=args['seed'],
                                                  to_cache=args['to_cache'])
    data_loaders = {'train': train_ds, 'valid': val_ds, 'test': test_ds}

    return data_loaders