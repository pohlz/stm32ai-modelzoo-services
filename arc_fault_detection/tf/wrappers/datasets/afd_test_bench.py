# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from arc_fault_detection.tf.src.datasets.afd_test_bench import AFDTestBench
from typing import Dict

__all__ = ['get_afd_test_bench']

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='afd_test_bench', use_case="arc_fault_detection")
def get_afd_test_bench(cfg) -> Dict:
    """Get AFD Test Bench dataset.
    Args:
        cfg (DictConfig): Configuration object. 
    Returns:
        Dict: A dictionary containing the following:
            - train_ds (object): Training dataset.
            - valid_ds (object): Validation dataset.
            - test_ds (object): Test dataset.
            - quantization_ds (object): Quantization dataset.
            - predict_ds (object): Prediction dataset.
    """
    
    dataloaders = AFDTestBench(cfg).load_afd_test_bench()
    return dataloaders
