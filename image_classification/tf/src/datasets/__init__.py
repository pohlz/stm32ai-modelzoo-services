# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from .utils import load_cifar_batch, get_train_val_ds, get_ds, get_prediction_ds, \
                   preprocess_data, prepare_kwargs_for_dataloader
from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .emnist_byclass import load_emnist
from .imagenet import load_imagenet_like
from .custom_dataset import load_custom_dataset
