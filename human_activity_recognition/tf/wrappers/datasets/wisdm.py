# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from human_activity_recognition.tf.src.datasets.utils import prepare_kwargs_for_dataloader
from human_activity_recognition.tf.src.datasets.wisdm import load_wisdm

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='wisdm', use_case="human_activity_recognition")
def get_wisdm(cfg):

    # Get dataloader kwargs
    args = prepare_kwargs_for_dataloader(cfg)
    # Creates datasets    
    train_ds, val_ds, test_ds = load_wisdm(dataset_path=args['training_path'],
                                            class_names=args['class_names'],
                                            input_shape=args['input_shape'],
                                            gravity_rot_sup=args['gravity_rot_sup'],
                                            normalization=args['normalization'],
                                            val_split=args['validation_split'],
                                            test_split=args['test_split'],
                                            seed=args['seed'],
                                            batch_size=args['batch_size'],
                                            to_cache=args['to_cache'])
    data_loaders = {'train': train_ds, 'valid': val_ds, 'test': test_ds}

    return data_loaders