# Using data augmentation


<details open><summary><a href="#1"><b>1. Introduction</b></a></summary><a id="1"></a>

Data augmentation has proven to be an effective technique to reduce overfitting of a network and make it generalize better. It is generally useful when you have a small dataset or a dataset that is too easy for the network to learn.

Also, for the Hand Posture use case, it's interesting to have each posture available for both left-handed and right-handed individuals.

The data augmentation transforms you want to apply to the frames are specified in the YAML configuration file. The transforms are only applied to the frames during training. They are not applied when the model is evaluated.

</details>
<details open><summary><a href="#2"><b>2. Specifying your data augmentation</b></a></summary><a id="2"></a>

The data augmentation transforms to apply to the input images are specified in the configuration file using a `data_augmentation` section, as illustrated in the YAML code below:

```yaml
data_augmentation:
   random_flip:
      mode: horizontal
```

If you don't want to use a specific data augmentation technique, set it to **False**; otherwise, provide an adequate value to be applied:

- `mode` - One of "*horizontal*", "*vertical*", or "*horizontal_and_vertical*". For most postures, only "*horizontal*" is useful.

</details>
