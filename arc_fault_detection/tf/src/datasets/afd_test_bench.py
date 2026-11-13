# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
from .load_datasets import load_afd_test_bench

class AFDTestBench:
    """Loader for afd test bench dataset."""

    def __init__(self, cfg: DictConfig = None):
        """
        Initialize the loader with  loading configuration.

        Args:
            cfg (DictConfig): data loader configuration.
        """
        self.cfg = cfg

    def load_afd_test_bench(self):
        """Load the AFD test bench dataset."""
        return load_afd_test_bench(cfg=self.cfg)
