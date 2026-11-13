# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import numpy as np

def downsample_data(data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Downsamples the columns of a 2D array to target_length by selecting evenly spaced columns.

    Parameters:
        data (np.ndarray): Input array of shape (n, m)
        target_length (int): Desired number of columns

    Returns:
        np.ndarray: Downsampled array of shape (n, target_length)
    """
    n, m = data.shape
    if m < target_length:
        raise ValueError(f"Data width {m} is less than or equal to target length {target_length}. Downsampling not possible.")
    # Compute indices of columns to keep
    indices = np.linspace(0, m - 1, target_length, dtype=int)
    return data[:, indices]
