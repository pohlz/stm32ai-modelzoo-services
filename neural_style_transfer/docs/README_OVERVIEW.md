# Neural style transfer STM32 model zoo

Remember that minimalistic yaml files are available [here](../config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!

<details open><summary><a href="#1"><b>1. Neural style transfer Model Zoo introduction</b></a></summary><a id="1"></a>

The Neural style transfer Model Zoo provides a collection of independent services related to machine learning for neural style transfer.

To use the services in the Neural style transfer model zoo, you can utilize the model
zoo [stm32ai_main.py](../stm32ai_main.py) along with [user_config.yaml](../user_config.yaml) file as input. The yaml file specifies the service and a set of configuration parameters such as the model (either from the model zoo or your own custom model), the dataset, and the preprocessing parameters, among others.

More information about the different services and their configuration options can be found in the <a href="#2">next section</a>.

</details>
<details open><summary><a href="#2"><b>2. Neural style transfer services </b></a></summary><a id="2"></a>

To use any service of this use case, you will need to update the [user_config.yaml](../user_config.yaml) file, which specifies the parameters and configuration options for the services that you want to use. Each section of the [user_config.yaml](../user_config.yaml) file is explained in detail in the following sections.

The `operation_mode` is a top-level attribute that specifies the operations or the service you want to execute.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below:

| operation_mode attribute | Operations                                                                                     |
|:-------------------------|:-----------------------------------------------------------------------------------------------|
| `prediction`             | Predict the classes and bounding boxes of some images using a TF-Lite model file.              |
| `benchmarking`           | Benchmark a float or quantized model on an STM32 board.                                        |

You can refer to the README links below that provide typical examples of operation modes and tutorials on specific services:

- [prediction](./README_PREDICTION.md)
- [benchmarking](./README_BENCHMARKING.md)

Deployment on the Neural style transfer is currently not supported.

</details></ul>
</details>
<details open><summary><a href="#A"><b>Appendix A: YAML syntax</b></a></summary><a id="A"></a>

**Example and terminology:**

An example of YAML code is shown below.

```yaml
preprocessing:
  rescaling:
    scale: 1/127.5
    offset: -1
  resizing:
    aspect_ratio: fit
    interpolation: nearest
```

The code consists of a number of nested "key-value" pairs. The colon character is used as a separator between the key and the value.

Indentation is how YAML denotes nesting. The specification forbids tabs because tools treat them differently. A common practice is to use 2 or 3 spaces, but you can use any number of them.

We use "attribute-value" instead of "key-value" as in the YAML terminology, the term "attribute" being more relevant to our application. We may use the term "attribute" or "section" for nested attribute-value pairs constructs. In the example above, we may indifferently refer to "preprocessing" as an attribute (whose value is a list of nested constructs) or as a section.

**Comments:**

Comments begin with a pound sign. They can appear after an attribute value or take up an entire line.

```yaml
preprocessing:
  rescaling:
    scale: 1/127.5   # This is a comment.
    offset: -1
  resizing:
    # This is a comment.
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb
```

**Attributes with no value:**

The YAML language supports attributes with no value. The code below shows the alternative syntaxes you can use for such attributes.

```yaml
attribute_1:
attribute_2: ~
attribute_3: null
attribute_4: None     # Model Zoo extension
```

The value *None* is a Model Zoo extension that was made because it is intuitive to Python users.

Attributes with no value can be useful to list in the configuration file all the attributes that are available in a given section and explicitly show which ones were not used.

**Strings:**

You can enclose strings in single or double quotes. However, unless the string contains special YAML characters, you don't need to use quotes.

This syntax:

```yaml
resizing:
  aspect_ratio: fit
  interpolation: nearest
```

is equivalent to this one:

```yaml
resizing:
  aspect_ratio: "fit"
  interpolation: "nearest"
```

**Strings with special characters:**

If a string value includes YAML special characters, you need to enclose it in single or double quotes. In the example below, the string includes the ',' character, so quotes are required.

```yaml
name: "Pepper,_bell___Bacterial_spot"
```

**Strings Spanning Multiple Lines:**

You can write long strings on multiple lines for better readability. This can be done using the '|' (pipe) continuation character as shown in the example below.

This syntax:

```yaml
LearningRateScheduler:
  schedule: |
    lambda epoch, lr:
        (0.0005*epoch + 0.00001) if epoch < 20 else
        (0.01 if epoch < 50 else
        (lr / (1 + 0.0005 * epoch)))
```

is equivalent to this one:

```yaml
LearningRateScheduler:
  schedule: "lambda epoch, lr: (0.0005*epoch + 0.00001) if epoch < 20 else (0.01 if epoch < 50 else (lr / (1 + 0.0005 * epoch)))"
```

Note that when using the first syntax, strings that contain YAML special characters don't need to be enclosed in quotes. In the example above, the string includes the ',' character.

**Booleans:**

The syntaxes you can use for boolean values are shown below. Supported values have been extended to *True* and *False* in the Model Zoo as they are intuitive to Python users.

```yaml
# YAML native syntax
attribute_1: true
attribute_2: false

# Model Zoo extensions
attribute_3: True
attribute_4: False
```

**Numbers and numerical expressions:**

Attribute values can be integer numbers, floating-point numbers, or numerical expressions as shown in the YAML code below.

```yaml
ReduceLROnPlateau:
  patience: 10    # Integer value
  factor: 0.1     # Floating-point value
  min_lr: 1e-6    # Floating-point value, exponential notation

rescaling:
  scale: 1/127.5  # Numerical expression, evaluated to 0.00784314
  offset: -1
```

**Lists:**

You can specify lists on a single line or on multiple lines as shown below.

This syntax:

```yaml
class_names: [ aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor ]
```

is equivalent to this one:

```yaml
class_names:
  - aeroplane
  - bicycle
  - bird
  - boat
  - bottle
  - bus
  - car
  - cat
  - chair
  - cow
  - diningtable
  - dog
  - horse
  - motorbike
  - person
  - pottedplant
  - sheep
  - sofa
  - train
  - tvmonitor
```

**Multiple attribute-value pairs on one line:**

Multiple attribute-value pairs can be specified on one line as shown below.

This syntax:

```yaml
rescaling: { scale: 1/127.5, offset: -1 }
```

is equivalent to this one:

```yaml
rescaling:
  scale: 1/127.5
  offset: -1
```

</details>
