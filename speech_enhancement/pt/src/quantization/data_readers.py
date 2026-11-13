# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from torch.utils.data import default_collate

class DataLoaderDataReader(CalibrationDataReader):
    '''
    DataReader for the caliberation during onnx quantization, built from a pytorch dataloader
    For single input models.
    
    Parameters
    ----------
    quant_dl : Pytorch dataloader containing the quantization samples. Batch size must be 1
    input_name : Name of the ONNX model's input
    change_dl_collate : if True, replace the dataloader's collate function 
        to a collate function that only returns the first item in a pair (which corresponds
        to the noisy sample)

    '''
    def __init__(self,
                 quant_dl,
                 input_name="input",
                 replace_dl_collate=True):

        self.quant_dl = quant_dl  # Dataloader for calibration data
        assert self.quant_dl.batch_size == 1, "Quantization dataloader batch size must be 1"
        self.input_name = input_name
        self.enum_dataloader = None
        if replace_dl_collate: # Use if Dataloader returns a (noisy, clean) tuple instead of just noisy data
            self.quant_dl.collate_fn = self._return_only_noisy_collate
        else:
            self.quant_dl.collate_fn = self._return_numpy_arrays_collate

    def get_next(self):
        if self.enum_dataloader is None:
            self.enum_dataloader = self.quant_dl
            # Create an iterator that generates input dictionaries
            # with input name and corresponding data
            self.enum_data = iter(
                [{self.input_name: data} for data in self.enum_dataloader]
            )
        
        return next(self.enum_data, None)  # Return next item from enumerator

    def rewind(self):
        self.enum_data = None  # Reset the enumeration of calibration data

    @staticmethod
    def _return_only_noisy_collate(batch):
        try:
            return np.expand_dims(batch[0][0].numpy(), axis=0)
        except:
            return np.expand_dims(batch[0][0], axis=0)
        
    @staticmethod
    def _return_numpy_arrays_collate(batch):
        try:
            return default_collate(batch).numpy()
        except:
            return default_collate(batch)
