
# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import numpy as np
import torch


class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        number_of_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, number_of_holes, length):
        self.num_holes = number_of_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with num_holes of dimension length x length cut out of it.
        """
        height = img.size(1)
        width = img.size(2)

        mask = np.ones((height, width), np.float32)

        for _ in range(self.num_holes):
            y = np.random.randint(height)
            x = np.random.randint(width)

            y1 = np.clip(y - self.length // 2, 0, height)
            y2 = np.clip(y + self.length // 2, 0, height)
            x1 = np.clip(x - self.length // 2, 0, width)
            x2 = np.clip(x + self.length // 2, 0, width)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
