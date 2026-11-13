# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
    
import tensorflow as tf
from pose_estimation.tf.src.data_augmentation import data_augmentation
from pose_estimation.tf.src.utils import change_model_input_shape

class HMTrainingModel(tf.keras.Model):

    """
    Keras model wrapper for training heatmap-based networks with custom loss,
    metrics and optional data augmentation.

    This class delegates the forward pass to an underlying `model` while
    handling:
    - heatmap loss computation,
    - heatmap-based metrics,
    - optional image/label augmentation,
    - tracking of training and validation metrics,
    - optional dynamic input resolution handling.

    Args:
        model (tf.keras.Model) :
            The underlying Keras model that produces heatmap predictions.
        hm_loss (Callable) :
            Loss function
        hm_metrics (Iterable[Callable]) :
            Metric functions
        data_augmentation_cfg (dict) :
            Configuration object for `data_augmentation`. If it contains
            the key "random_periodic_resizing", the internal model input
            shape will be adapted to support variable resolution.
        pixels_range (tuple) :
            Range of pixel values used by the augmentation function, e.g.
            (-1., 1.) or (0., 1.) for example.
        network_stride (int) :
            Stride of the network used by the loss/metrics to map between
            image space and heatmap space.
    """

    def __init__(self, model: tf.keras.Model, hm_loss: callable, hm_metrics: list[callable], 
                 data_augmentation_cfg: dict = None, pixels_range: tuple = None, network_stride: int = None):
            
        super(HMTrainingModel, self).__init__()

        self.metrics_tracker     = [tf.keras.metrics.Mean(name='loss')] # add the loss to the metrics tracker
        self.val_metrics_tracker = [tf.keras.metrics.Mean(name='val_loss')] # add the loss to the metrics tracker

        for hmm in hm_metrics:
            self.metrics_tracker.append(tf.keras.metrics.Mean(name=hmm.__name__)) # add the other metrics
            self.val_metrics_tracker.append(tf.keras.metrics.Mean(name=hmm.__name__)) # add the other metrics

        self.hm_loss    = hm_loss
        self.hm_metrics = hm_metrics
        self.data_augmentation_cfg = data_augmentation_cfg
        self.pixels_range = pixels_range
        self.network_stride = network_stride
        self.model = model

        if self.data_augmentation_cfg is not None:
            if 'random_periodic_resizing' in self.data_augmentation_cfg:
                self.model, _ = change_model_input_shape(model,(None,None,None,3))

        self.current_res = tf.Variable([0,0], trainable=False, dtype=tf.int64)

    def set_resolution(self, res):
        self.current_res.assign([res[0],res[1]])

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.model.save_weights(
            filepath, overwrite=overwrite) #, save_format=save_format, options=options)

    def load_weights(self, filepath, skip_mismatch=False, by_name=False, options=None):
        return self.model.load_weights(
            filepath, skip_mismatch=skip_mismatch, by_name=by_name, options=options)

    def train_step(self, data):

        images, y_true = data

        current_im_shape = tf.cast(images.shape[1:3], dtype=tf.int64)

        if self.data_augmentation_cfg is not None:
            images, y_true = data_augmentation(images, y_true, self.data_augmentation_cfg, self.pixels_range, self.current_res)
            if 'random_periodic_resizing' in self.data_augmentation_cfg:
                current_im_shape = self.current_res
            
        with tf.GradientTape() as tape:
            y_pred = self.model(images, training=True)
            loss   = self.hm_loss(y_true, y_pred, image_size=current_im_shape, network_stride=self.network_stride)

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        metrcs = []
        for hmm in self.hm_metrics:
            metrcs.append(hmm(y_true, y_pred, image_size=current_im_shape, network_stride=self.network_stride))

        # Update metrics (includes the metric that tracks the loss)
        for i,mt in enumerate(self.metrics_tracker):
            if mt.name == "loss":
                mt.update_state(loss)
            else:
                mt.update_state(metrcs[i-1])

        return {m.name: m.result() for m in self.metrics_tracker}

    def test_step(self, data):
        # The data loader supplies groundtruth boxes in
        images, y_true = data

        current_im_shape = tf.cast(images.shape[1:3], dtype=tf.int64)

        y_pred   = self.model(images, training=False)
        val_loss = self.hm_loss(y_true, y_pred, image_size=current_im_shape, network_stride=self.network_stride)

        val_metrcs = []
        for hmm in self.hm_metrics:
            val_metrcs.append(hmm(y_true, y_pred, image_size=current_im_shape, network_stride=self.network_stride))

        output_dict = {}

        # Update metrics (includes the metric that tracks the val_loss)
        for i,mt in enumerate(self.val_metrics_tracker):
            if mt.name == "val_loss":
                mt.update_state(val_loss)
                output_dict.update({'loss': mt.result()})
            else:
                mt.update_state(val_metrcs[i-1])
                output_dict.update({mt.name: mt.result()})

        return output_dict # {m.name: m.result() for m in self.val_metrics_tracker}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.metrics_tracker + self.val_metrics_tracker