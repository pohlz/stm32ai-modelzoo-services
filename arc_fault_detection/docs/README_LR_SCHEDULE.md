# Learning rate schedulers

A number of callbacks are available that implement different learning rate decay functions. The *ReduceLROnPlateau* and *LearningRateScheduler* schedulers are Keras callbacks; the others are provided with the Model Zoo. They all update the learning rate at the beginning of each epoch.
To use one of these learning rate schedulers, simply add it to the list of callbacks in the `training:` section of your configuration file. 

### <summary><a><b>Plotting the learning rate before training</b></a></summary>

A script that plots the learning rate schedule without running training is available. To run it, change the current directory to `arc_fault_detection/tf/src/training/lr_schedule` and execute **plot_lr_schedule.py** as follows:

```bash
python plot_lr_schedule.py --config-path ../../../../ --config-name user_config.yaml --fname plot.png
```
This will plot the learning rate schedule used in `user_config.yaml` and save the curve to `tf/src/training/lr_schedule/plot.png`.

The script reads the `training:` section of your configuration file to get the number of training epochs, and the name and arguments of the learning rate scheduler in the `callbacks:` subsection. 

We encourage you to run this script using the above command. It does not require any extra work, as it only needs your configuration file. It may save you a lot of time to choose a learning rate scheduler and tun its parameters.

For a more detailed description of the available callbacks and the learning rate plotting utility, we refer you to [the learning rate schedulers README](../../common/training/lr_schedulers_README.md).

**_Note that the script cannot be used with the TensorFlow *ReduceLROnPlateau* scheduler, as the learning rate schedule is only available after training._**
