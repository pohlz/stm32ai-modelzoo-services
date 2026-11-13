# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf


def _reg_to_heatmaps(tensor: tf.Tensor, mask: tf.Tensor, w: int, h: int):
    '''
    Regressions of (x,y) coordinates to heatmaps

    Args
        tensor (tf.Tensor): shape (batch, N, nb_kpts, 2) FLOAT32 representing the (x,y) coordinates.
        mask   (tf.Tensor): shape (batch, N, nb_kpts) FLOAT32 representing a mask for valid keypoints.
        w            (int): shape (1,) first dimension in heatmap shape
        h            (int): shape (1,) second dimension in heatmap shape

    Returns:
        heatmaps   (tf.Tensor): shape (batch, w, h, nb_kpts) FLOAT32 representing the heatmaps.
    '''

    _,N,nb_kpts,_ = tensor.shape

    x = tf.cast(tensor[:,:,:,0] * tf.cast(h,tf.float32), tf.int32) # shape (batch, N, nb_kpts) INT32
    y = tf.cast(tensor[:,:,:,1] * tf.cast(w,tf.float32), tf.int32) # shape (batch, N, nb_kpts) INT32

    W = tf.one_hot(tf.reshape(y,[-1]),w) # (batch*N*nb_kpts,w)
    H = tf.one_hot(tf.reshape(x,[-1]),h) # (batch*N*nb_kpts,h)

    W = tf.reshape(W,[-1,N,nb_kpts,w]) # (batch,N,nb_kpts,w)
    H = tf.reshape(H,[-1,N,nb_kpts,h]) # (batch,N,nb_kpts,h)

    W *= mask[...,None] # (batch,N,nb_kpts,w)
    H *= mask[...,None] # (batch,N,nb_kpts,h)

    heatmaps = W[:,:,:,:,None] * H[:,:,:,None,:] # create the heatmaps shape (batch,N,nb_kpts,w,h)
    heatmaps = tf.reduce_sum(heatmaps,axis=1) # (batch,nb_kpts,w,h)
    heatmaps = tf.cast(tf.cast(heatmaps,tf.bool),tf.float32) # all floats above 1 are reduced to 1
    heatmaps = tf.transpose(heatmaps,[0,2,3,1]) # (batch,w,h,nb_kpts)

    return heatmaps

def _heatmaps_to_reg(tensor: tf.Tensor, w: int, h: int):
    '''
    Continuous function that maps heatmaps to (x,y) coordinates

    Args
        tensor (tf.Tensor): shape (batch, w, h, nb_kpts) FLOAT32 representing the heatmaps.
        w            (int): shape (1,) first dimension in heatmap shape
        h            (int): shape (1,) second dimension in heatmap shape

    Returns:
        reg    (tf.Tensor): shape (batch, 1, nb_kpts, 2) FLOAT32 representing the x,y coordinates.
    '''

    norm_scale  = tf.reduce_sum(tf.reduce_sum(tensor,1),1)[:,None,None,:] # shape (batch, 1, 1, nb_kpts) FLOAT32
    norm_tensor = tensor/norm_scale # shape (batch, w, h, nb_kpts)

    x_sum = tf.reduce_sum(norm_tensor,1) # shape (batch, h, nb_kpts) FLOAT32
    y_sum = tf.reduce_sum(norm_tensor,2) # shape (batch, w, nb_kpts) FLOAT32

    x_mul = x_sum * (tf.cast(tf.range(h),tf.float32)[None,:,None]+0.5)/tf.cast(h,tf.float32) # shape (batch, h, nb_kpts) FLOAT32
    y_mul = y_sum * (tf.cast(tf.range(w),tf.float32)[None,:,None]+0.5)/tf.cast(w,tf.float32) # shape (batch, w, nb_kpts) FLOAT32

    x = tf.reduce_sum(x_mul,1) # shape (batch, nb_kpts) FLOAT32
    y = tf.reduce_sum(y_mul,1) # shape (batch, nb_kpts) FLOAT32

    reg = tf.stack([x,y],-1)[:,None] # shape (batch,1,nb_kpts,2) FLOAT32

    return reg

def spe_loss(y_true: tf.Tensor, y_pred: tf.Tensor, image_size: tf.Tensor, network_stride: int, output_type: str = 'heatmaps', loss_type: str = 'mse'):
    '''
    Calculate single pose estimation loss

    Args
        y_true (tf.Tensor): shape (batch, N, 5+nb_kpts*3) FLOAT32 representing ground truths.
        y_pred (tf.Tensor): shape (batch, ...) FLOAT32 representing predictions.
        output_type  (str): shape (1,) choices between 'heatmaps' or 'reg'
        loss_type    (str): shape (1,) choices between 'rmse' or 'mse' losses

    Returns:
        loss   (tf.Tensor): shape (1,) FLOAT32 representing the mse.
    '''

    y_true = y_true[:,:,5:] # (batch, N, nb_kpts*3)

    _, N, nb_kpts3 = y_true.shape

    nb_kpts = int(nb_kpts3/3)

    y_true  = tf.reshape(y_true,[-1,N,nb_kpts,3]) # (batch, N, nb_kpts, 3)

    mask    = y_true[..., 2]   # shape (batch,N,nb_kpts) visible keypoints

    wh = tf.cast(image_size / network_stride,tf.int32)

    if output_type == 'heatmaps':
        w, h   = wh[0], wh[1]
        y_true = _reg_to_heatmaps(y_true[:,:,:,:2],mask,w,h) # shape (batch,W,H,nb_kpts)
        mask   = tf.reduce_sum(mask,axis=1) # shape (batch,nb_kpts)
        mask   = tf.cast(tf.cast(mask,tf.bool),tf.float32) # shape (batch,nb_kpts)
        y_pred *= mask[:,None,None,:] # shape (batch,W,H,nb_kpts)
    elif output_type == 'reg':
        # y_pred -> (batch,nb_kpts*3)
        y_pred = tf.reshape(y_pred,[-1,nb_kpts,3])[:,None] # shape (batch,1,nb_kpts,3)
        y_pred = y_pred[...,:2] * mask[...,None] # shape (batch,1,nb_kpts,2)
        y_true = y_true[...,:2] * mask[...,None] # shape (batch,1,nb_kpts,2)
    elif output_type == 'reg_heatmaps':
        w, h   = wh[0], wh[1]
        y_pred = _heatmaps_to_reg(y_pred,w,h)     # shape (batch,1,nb_kpts,2)
        y_pred = y_pred[...,:2] * mask[...,None] # shape (batch,1,nb_kpts,2)
        y_true = y_true[...,:2] * mask[...,None] # shape (batch,1,nb_kpts,2)
    else:
        print('[ERROR] this output type is not supported, currently we only support "heatmaps" or "reg"')

    if loss_type == 'rmse':
        loss = _rmse_loss(y_true,y_pred)
    elif loss_type == 'mse':
        loss = _mse_loss(y_true,y_pred)
    elif loss_type == 'bce':
        loss = _bce_loss(y_true,y_pred)
    elif loss_type == 'cce':
        loss = _cce_loss(y_true,y_pred)
    else:
        print('[ERROR] this loss is not supported, currently we only support "rmse" or "mse"')

    return loss

def _bce_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    '''
    Calculate binary cross entropy loss (mean(y_true*log(y_pred)+(1-y_true)*log(1-y_pred)) if from_logits=True: y_pred=sigmoid(y_pred))

    Args
        y_true   (tf.Tensor): shape (batch, ...) FLOAT32 representing ground truths.
        y_pred   (tf.Tensor): shape (batch, ...) FLOAT32 representing predictions.

    Returns:
        loss    (tf.Tensor): shape (1,) FLOAT32 representing the bce.
    '''

    r_layer = tf.keras.layers.Reshape((-1,y_true.shape[-1]))

    y_true_r = r_layer(y_true) # shape (batch, H*W, kpts)
    y_pred_r = r_layer(y_pred) # shape (batch, H*W, kpts)

    y_true_t = tf.transpose(y_true_r,[0,2,1]) # shape (batch, kpts, H*W)
    y_pred_t = tf.transpose(y_pred_r,[0,2,1]) # shape (batch, kpts, H*W)

    bce_f = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    bce = bce_f(y_true_t,y_pred_t)

    return bce

def _cce_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    '''
    Calculate categorical cross entropy loss (mean(sum_classes(y_true*log(y_pred))), if from_logits=True: y_pred=sotfmax(y_pred))

    Args
        y_true   (tf.Tensor): shape (batch, ...) FLOAT32 representing ground truths.
        y_pred   (tf.Tensor): shape (batch, ...) FLOAT32 representing predictions.

    Returns:
        loss    (tf.Tensor): shape (1,) FLOAT32 representing the cce.
    '''

    r_layer = tf.keras.layers.Reshape((-1,y_true.shape[-1]))

    y_true_r = r_layer(y_true) # shape (batch, H*W, kpts)
    y_pred_r = r_layer(y_pred) # shape (batch, H*W, kpts)

    y_true_t = tf.transpose(y_true_r,[0,2,1]) # shape (batch, kpts, H*W)
    y_pred_t = tf.transpose(y_pred_r,[0,2,1]) # shape (batch, kpts, H*W)

    cce_f = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    eps = 1e-15

    cce = cce_f(y_true_t,(y_pred_t + eps))

    return cce

def _mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    '''
    Calculate mean square error loss

    Args
        y_true   (tf.Tensor): shape (batch, ...) FLOAT32 representing ground truths.
        y_pred   (tf.Tensor): shape (batch, ...) FLOAT32 representing predictions.

    Returns:
        loss    (tf.Tensor): shape (1,) FLOAT32 representing the mse.
    '''

    diff_2 = tf.square(y_true - y_pred)

    loss = tf.math.reduce_mean(diff_2)

    return loss

def _rmse_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    '''
    Calculate root mean square error loss

    Args
        y_true   (tf.Tensor): shape (batch, ...) FLOAT32 representing ground truths.
        y_pred   (tf.Tensor): shape (batch, ...) FLOAT32 representing predictions.

    Returns:
        loss    (tf.Tensor): shape (1,) FLOAT32 representing the rmse.
    '''

    mse = _mse_loss(y_true,y_pred)

    loss = tf.math.sqrt(mse)

    return loss



# if __name__=='__main__':

#     from functools import partial

#     spe = partial(_spe_loss,output_type='reg',loss_type='rmse')

#     true = tf.constant([[0.0,0.0,0.0,0.0,0.0,0.575937557220459, 0.4, 1.0, 0.6234375238418579, 0.2541176378726959, 0.0, 0.559374988079071, 0.30352941155433655, 1.0, 0.676562488079071, 0.3341176509857178, 0.0]],tf.float32)
#                         #[0.0,0.0,0.0,0.0,0.0,0.6, 0.4, 1.0, 0.6234375238418579, 0.2541176378726959, 1.0, 0.559374988079071, 0.7, 1.0, 0.9, 0.3341176509857178, 0.0]],tf.float32)
#     true = true[None] #tf.reshape(true,[1,1,-1])
#     # true = tf.transpose(true,[0,1,3,2])

#     # pred = tf.constant([0.375937557220459, 0.15258824348449706, 0.7, 0.5534375238418579, 0.6541176378726959, 0.34, 0.159374988079071, 0.40352941155433655, 0.01, 0.671562488079071, 0.3541176509857178, 0.345],tf.float32)
#     # pred = tf.reshape(pred,[1,1,3,-1])
#     # pred = tf.transpose(pred,[0,1,3,2])

#     pred = tf.constant([0.575937557220459, 0.35, 1.0, 0.6234375238418579, 0.2541176378726959, 0.0, 0.559374988079071, 0.30352941155433655, 1.0, 0.676562488079071, 0.3341176509857178, 0.0],tf.float32)
#     pred = pred[None,None]

#     # pred = np.zeros((1,10,10,4))

#     # pred[0,5,4,0] = 1.
#     # pred[0,6,2,1] = 1.
#     # pred[0,4,3,2] = 1.
#     # pred[0,6,3,3] = 1. #pred[0,6,3,3] = 1. the loss is still 0 with that setup it is weird !!!!!!!!!!!!!!!!!!!!

#     # pred = tf.constant(pred,tf.float32)

#     #print(pred)

#     print(spe(true, pred))