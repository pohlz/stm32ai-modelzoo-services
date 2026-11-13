# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf
from .loss import segmentation_loss
from semantic_segmentation.tf.src.data_augmentation import data_augmentation
from semantic_segmentation.tf.src.utils import change_model_input_shape

            
class SegmentationTrainingModel(tf.keras.Model):

    def __init__(self,
            model,
            num_classes=None,
            image_size=None,
            data_augmentation_cfg=None,
            loss_weights=None,
            pixels_range=None):

        super(SegmentationTrainingModel, self).__init__()

        # self.model = model
        self.model,_ = change_model_input_shape(model,(None,None,None,3))
        self.num_classes = num_classes
        self.data_augmentation_cfg = data_augmentation_cfg
        self.loss_weights = loss_weights
        self.pixels_range = pixels_range

        # Batch info tracker
        self.batch_info = tf.Variable([0, 0, 0], trainable=False, dtype=tf.int64)

        # Training trackers
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="train_acc")

        # Validation trackers
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="val_acc")

    def set_resolution(self, res):
        self.batch_info.assign([0,res[0],res[1]])
      
    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.model.save_weights(
            filepath, overwrite=overwrite) #, save_format=save_format, options=options)

    def load_weights(self, filepath, skip_mismatch=False, by_name=False, options=None):
        return self.model.load_weights(
            filepath, skip_mismatch=skip_mismatch, by_name=by_name, options=options)


    def train_step(self, data):
        images, labels = data

        if self.data_augmentation_cfg is not None:            
            images, labels = data_augmentation(
                        images,
                        labels,
                        config=self.data_augmentation_cfg,
                        pixels_range=self.pixels_range,
                        batch_info=self.batch_info)
       
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = segmentation_loss(labels, predictions,
                                     num_classes=self.num_classes, loss_weights=self.loss_weights)

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update the loss tracker
        self.train_loss_tracker.update_state(loss)
        
        # Update the accuracy tracker
        labels_one_hot = tf.one_hot(labels, depth=self.num_classes)
        labels_one_hot = tf.squeeze(labels_one_hot, axis=-2)
        self.train_acc_tracker.update_state(labels_one_hot, predictions)

        return {
            "loss": self.train_loss_tracker.result(),
            "accuracy": self.train_acc_tracker.result()
        }


    def test_step(self, data):        
        images, labels = data

        predictions = self.model(images, training=False)

        # Update the loss tracker
        loss = segmentation_loss(labels, predictions,
                                 num_classes=self.num_classes, loss_weights=self.loss_weights)
        self.val_loss_tracker.update_state(loss)
        
        # Update the accuracy tracker
        labels_one_hot = tf.one_hot(labels, depth=self.num_classes)
        labels_one_hot = tf.squeeze(labels_one_hot, axis=-2)
        self.val_acc_tracker.update_state(labels_one_hot, predictions)
        
        return {
            "loss": self.val_loss_tracker.result(),
            "accuracy": self.val_acc_tracker.result()
        }

    # List trackers here to avoid the need for calling reset_state()
    @property
    def metrics(self):
        return [
            self.train_loss_tracker,
            self.train_acc_tracker,
            self.val_loss_tracker,
            self.val_acc_tracker
        ]
