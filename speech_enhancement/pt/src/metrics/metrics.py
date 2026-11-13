# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

'''Various metrics, both as standalone functions and torch losses.'''
import numpy as np
import torch
import torch.nn as nn

# For some reason this is gone from recent scipy versions
def snr(ref, deg):
    """Signal to noise ratio between clean and degraded or denoised speech. 
       The SNR output is given in decibel scale
       Accepts numpy arrays and torch tensors as input
       Inputs
       ------
       ref : Clean source signal
       deg : Degraded or denoised signal

       Outputs
       -------
       SNR in dB
    """

    try:
        ref = ref.numpy()
        deg = deg.numpy()
    except:
        pass
    ratio = np.sum(ref**2) / (np.sum((deg - ref)**2))
    return 10 * np.log10(ratio)

def si_snr(ref, deg):
    """Scale-invariant signal to noise ratio between clean and degraded or denoised speech. 
       The SNR output is given in decibel scale.
       Accepts numpy arrays and torch tensors as input
       Inputs
       ------
       ref : Clean source signal
       deg : Degraded or denoised signal

       Outputs
       -------
       SI-SNR in dB
    """
    try:
        ref = ref.numpy()
        deg = deg.numpy()
    except:
        pass

    scale_factor = np.dot(ref, deg) / np.sum(ref**2)
    ratio = np.sum((scale_factor * ref)**2) / (np.sum((deg - (scale_factor * ref))**2))

    return 10 * np.log10(ratio)

class SNRLoss(nn.Module):
    '''SNR in torch loss form'''
    def __init__(self, reduction="mean"):
        '''
        Parameters
        ----------
        reduction, str or None : One of "mean", "sum", or None. 
            Reduction mode for this loss. 
            If "mean", returns mean loss over input batch.
            If "sum", returns sum of loss over input batch
            If "None", returns a vector of the losses for each sample.
        '''
        super().__init__()
        self.reduction = reduction
        if self.reduction not in ["mean", "sum", None]:
            raise ValueError(
                f"'reduction' argument should be one of ['mean', 'sum', None], but was {reduction}")
    def forward(self, input, target):
        '''
        Signal to noise ratio between clean and degraded or denoised speech. 
        The SNR output is given in decibel scale.
        Inputs
        ------
        input : Degraded or denoised signal
        target : Clean source signal

        Outputs
        -------
        SNR in dB
        '''
        ratio = torch.sum(target**2, dim=-1) / torch.sum((input - target)**2, dim=-1)
        log_ratio = 10 * torch.log10(ratio)
        if self.reduction == "mean":
            out = -torch.mean(log_ratio)
        elif self.reduction == "sum":
            out = -torch.sum(log_ratio)
        elif self.reduction == None:
            out = -log_ratio
        return out
    
class SISNRLoss(nn.Module):
    '''SISNR in torch loss form'''
    def __init__(self, reduction="mean"):
        '''
        Parameters
        ----------
        reduction, str or None : One of "mean", "sum", or None. 
            Reduction mode for this loss. 
            If "mean", returns mean loss over input batch.
            If "sum", returns sum of loss over input batch
            If "None", returns a vector of the losses for each sample.
        '''
        super().__init__()
        self.reduction = reduction
        if self.reduction not in ["mean", "sum", None]:
            raise ValueError(
                f"'reduction' argument should be one of ['mean', 'sum', None], but was {reduction}")
        
    def forward(self, input, target):
        '''
        Scale-invariants signal to noise ratio between clean and degraded or denoised speech. 
        The SNR output is given in decibel scale.
        Inputs
        ------
        input : Degraded or denoised signal
        target : Clean source signal

        Outputs
        -------
        SI-SNR in dB
        '''
        scale_factor = torch.diagonal(torch.matmul(target, input.transpose(1, 0))) / torch.sum(target**2, axis=-1)
        scale_factor = scale_factor.reshape(scale_factor.shape[0], 1) # For broadcasting
        ratio = torch.sum((scale_factor * target)**2, dim=-1) / (torch.sum((input - (scale_factor * target))**2, dim=-1))
        log_ratio = 10 * torch.log10(ratio)
        if self.reduction == "mean":
            out = -torch.mean(log_ratio)
        elif self.reduction == "sum":
            out = -torch.sum(log_ratio)
        elif self.reduction == None:
            out = -log_ratio
        return out


