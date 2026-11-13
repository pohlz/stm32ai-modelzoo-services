# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf
import numpy as np


def heatmaps_spe_postprocess(tensor:tf.Tensor, pred_size:tf.Tensor):
    '''
    Post-process for the single pose estimation heatmaps use-case

    Args
        tensor    (tf.Tensor): shape (batch, res, res, keypoints) FLOAT32 heatmaps outputs of the single pose estimation models
        pred_size (tf.Tensor): shape (2, )                        INT64   size of the prediction heatmaps
    
    Returns:
        detection (tf.Tensor): shape (batch, 1, keypoints*3) FLOAT32 the (x,y,conf) values of all keypoints for the single person
    '''

    sh = [tensor.shape[0],pred_size[0],pred_size[1],tensor.shape[3]] # shape : (4,)
    predictions_flat = tf.reshape(tensor,[-1,sh[1]*sh[2],sh[3]]) # flatten the prediction tensor,         shape : (batch,res*res,keypoints)
    arg_pred = tf.argmax(predictions_flat,1) # find the argmax for each keypoints in this flatten tensor, shape : (batch,keypoints)
    val_pred = tf.reduce_max(predictions_flat,1) # find the max value of this flatten tensor,             shape : (batch,keypoints)

    arg_pred_x = (tf.cast(arg_pred//sh[2],dtype=tf.float32)+0.5) / tf.cast(sh[1],tf.float32) # find the x values of keypoints in [0,1], shape : (batch,keypoints)
    arg_pred_y = (tf.cast(arg_pred%sh[2],dtype=tf.float32)+0.5) / tf.cast(sh[2],tf.float32) # find the y values of keypoints in [0,1],  shape : (batch,keypoints)

    detection = tf.stack([arg_pred_y,arg_pred_x,val_pred],1) # shape : (batch,3,keypoints)
    detection = tf.transpose(detection,[0,2,1])              # shape : (batch,keypoints,3)
    detection = tf.reshape(detection,[-1,sh[3]*3])           # shape : (batch,keypoints*3)
    detection = detection[:,None,:]                          # shape : (batch,1,keypoints*3)

    return detection # shape : (batch,1,keypoints*3)

def spe_postprocess(tensor:tf.Tensor):
    '''
    Post-process for the single pose estimation use-case

    Args
        tensor    (tf.Tensor): shape (batch, 1, keypoints, 3) FLOAT32 the (x,y,conf) values of all keypoints but in a different format
    
    Returns:
        detection (tf.Tensor): shape (batch, 1, keypoints*3) FLOAT32 the (x,y,conf) values of all keypoints for the single person
    '''

    sh = tensor.shape

    x = tensor[:,:,:,1]
    y = tensor[:,:,:,0]
    v = tensor[:,:,:,2]

    detection = tf.stack([x,y,v],-1) # (batch, 1, keypoints, 3)

    detection = tf.reshape(detection,[-1,1,sh[2]*sh[3]]) # (batch, 1, keypoints*3)

    return detection

def _padded_nms(tensor:tf.Tensor,max_output_size:int,iou_threshold:float,score_threshold:float):
    '''
    Function used to apply NMS on each image of the batch independently

    Args
        tensor          (tf.Tensor): shape (5+keypoints*3, num_boxes) FLOAT32 outputs of the YOLO multi pose estimation models
        max_output_size (tf.Tensor): shape (1,) INT32 max number of detections per image
        iou_threshold   (tf.Tensor): shape (1,) FLOAT32 threshold for NMS iou
        score_threshold (tf.Tensor): shape (1,) FLOAT32 threshold to filter detections under a certain score
    
    Returns:
        detection       (tf.Tensor): shape (max_output_size, 5+keypoints*3) FLOAT32 bounding boxes + (x,y,conf) values for all keypoints of all detected persons
    '''

    boxes     = tensor[:4] # shape : (4,num_boxes)
    scores    = tensor[4]  # shape : (num_boxes)
    keypoints = tensor[5:] # shape : (nb_kpts*3,num_boxes)

    # the scores must be of the form [y1, x1, y2, x2] in order for the NMS to work
    # they are initially of the form [x,y,w,h]

    x1 = boxes[0] - boxes[2]/2 # shape : (num_boxes)
    y1 = boxes[1] - boxes[3]/2 # shape : (num_boxes)
    x2 = boxes[0] + boxes[2]/2 # shape : (num_boxes)
    y2 = boxes[1] + boxes[3]/2 # shape : (num_boxes)

    yxboxes = tf.stack([y1, x1, y2, x2])  # shape : (4,num_boxes)
    yxboxes = tf.transpose(yxboxes,[1,0]) # shape : (num_boxes,4)

    selected_indices = tf.image.non_max_suppression(yxboxes,scores,
                                                    max_output_size = max_output_size,
                                                    iou_threshold   = iou_threshold,
                                                    score_threshold = score_threshold)
    indices = selected_indices.shape[0]

    if indices:
        detection = tf.gather(tensor,selected_indices,axis=1) # shape : (5+nb_kpts*3,selected_indices)
        detection = tf.transpose(detection,[1,0])             # shape : (selected_indices,5+nb_kpts*3)
    else:
        detection = tf.zeros((indices,tensor.shape[0]))

    zero_padding = tf.zeros((max_output_size-indices,tensor.shape[0])) # shape : (max_output_size-selected_indices,5+nb_kpts*3)

    detection = tf.concat([detection,zero_padding],0) # shape : (max_output_size,5+nb_kpts*3)

    return detection # shape : (max_output_size, 5+nb_kpts*3)

def yolo_mpe_postprocess(tensor:tf.Tensor,max_output_size:int=20,iou_threshold:float=0.7,score_threshold:float=0.25):
    '''
    Post-process for the multi pose estimation use-case

    Args
        tensor          (tf.Tensor): shape (batch, 5+keypoints*3, num_boxes) FLOAT32 outputs of the YOLO multi pose estimation models
        max_output_size (tf.Tensor): shape (1,) INT32 max number of detections per image
        iou_threshold   (tf.Tensor): shape (1,) FLOAT32 threshold for NMS iou
        score_threshold (tf.Tensor): shape (1,) FLOAT32 threshold to filter detections under a certain score
    
    Returns:
        detections      (tf.Tensor): shape (batch, max_output_size, 5+keypoints*3) FLOAT32 values for all keypoints of all detected persons
    '''


    tensor = tf.constant(tensor,tf.float32)

    args = {'max_output_size':max_output_size,
            'iou_threshold':iou_threshold,
            'score_threshold':score_threshold}

    detections = tf.map_fn(lambda x : _padded_nms(x,**args),tensor)

    return detections # shape : (batch, max_output_size, 5+keypoints*3)


def hand_landmarks_postprocess(tensor:[tf.Tensor]):
    '''
    Post-process for the hand landmarks use-case

    Args
        tensor  list(tf.Tensor): shape [(batch,1),(batch, keypoints*3),(batch,1),(batch, keypoints*3)] FLOAT32 outputs of the hand landmarks
    
    Returns:
        det      (tf.Tensor): shape (batch, keypoints*3) FLOAT32 3D detections of the hand landmarks in pixels
        norm_det (tf.Tensor): shape (batch, keypoints*3) FLOAT32 3D detections of the hand landmarks centered reduced (invariant of hand size and position)
        htype    (tf.Tensor): shape (batch,) FLOAT32 type of hand (right or left) if near 0 -> left, if near 1 -> right
        hprob    (tf.Tensor): shape (batch,) FLOAT32 presence probability of the hand
    '''

    det      = tensor[-1]
    norm_det = tensor[1]

    x_sc = tf.reduce_sum((norm_det[:,0::3]-tf.reduce_mean(norm_det[:,0::3],-1))*(det[:,0::3]-tf.reduce_mean(det[:,0::3],-1)),-1) / tf.reduce_sum((norm_det[:,0::3]-tf.reduce_mean(norm_det[:,0::3],-1))**2,-1)
    y_sc = tf.reduce_sum((norm_det[:,1::3]-tf.reduce_mean(norm_det[:,1::3],-1))*(det[:,1::3]-tf.reduce_mean(det[:,1::3],-1)),-1) / tf.reduce_sum((norm_det[:,1::3]-tf.reduce_mean(norm_det[:,1::3],-1))**2,-1)
    z_sc = tf.reduce_sum((norm_det[:,2::3]-tf.reduce_mean(norm_det[:,2::3],-1))*(det[:,2::3]-tf.reduce_mean(det[:,2::3],-1)),-1) / tf.reduce_sum((norm_det[:,2::3]-tf.reduce_mean(norm_det[:,2::3],-1))**2,-1)

    x_off = tf.reduce_mean(det[:,0::3],-1) - x_sc*tf.reduce_mean(norm_det[:,0::3],-1)
    y_off = tf.reduce_mean(det[:,1::3],-1) - x_sc*tf.reduce_mean(norm_det[:,1::3],-1)
    z_off = tf.reduce_mean(det[:,2::3],-1) - x_sc*tf.reduce_mean(norm_det[:,2::3],-1)

    norm_det[:,0::3] *= x_sc
    norm_det[:,1::3] *= y_sc
    norm_det[:,2::3] *= z_sc

    norm_det[:,0::3] += x_off
    norm_det[:,1::3] += y_off
    norm_det[:,2::3] += z_off

    htype = tensor[0][:,0]
    hprob = tensor[2][:,0]

    return det, norm_det, htype, hprob


def head_landmarks_postprocess(tensor:tf.Tensor):
    '''
    Post-process for the head landmarks use-case

    Args
        tensor   (tf.Tensor): shape (batch,1,1,keypoints*3) FLOAT32 outputs of the head landmarks
    
    Returns:
        det      (tf.Tensor): shape (batch, keypoints*2) FLOAT32 (x,y) detections of the head landmarks in pixels
    '''

    # det   = tensor[:,0,:,0] # shape (batch, 478)
    det   = tensor[:,0,0,:] # shape (batch, 936)

    return det