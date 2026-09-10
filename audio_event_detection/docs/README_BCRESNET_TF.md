# Google kws_streaming BC-ResNet for STM AED

The TensorFlow `custom` hook now builds Google's non-streaming BC-ResNet-1.
Source: [google-research/kws_streaming](https://github.com/google-research/google-research/tree/08a8d6736475776f42ffac23b2c13111a28e5795/kws_streaming),
revision `08a8d6736475776f42ffac23b2c13111a28e5795`.
The pristine `bc_resnet.py`, `sub_spectral_normalization.py`, and Apache-2.0
license are retained under `tf/src/models/kws_streaming/upstream/`.

## Backup

The previous custom model is preserved byte-for-byte as
`tf/src/models/custom_model_before_kws_streaming_20260910.py`
(SHA256 `1467207634fe985478da5f336423eff67f72bd62dcaea11339d8944b32e65673`).
To restore its factory, change the import in `tf/src/models/custom_model.py` to
`from .custom_model_before_kws_streaming_20260910 import get_custom_model`.
Existing YAMLs selecting `model_name: custom` now select the Google model when
building from scratch. Existing saved Keras models retain their stored graphs.
The PyTorch model and its checkpoints are unaffected.

## Train, quantize, evaluate

Use the existing STM TensorFlow environment. From `audio_event_detection`:

```sh
python stm32ai_main.py --config-name user_config_gsc12_gsc_preproc_bcresnet_tf_v03
```

The configuration selects `chain_tqe`: training, evaluation of the best float
model, representative-data TFLite PTQ, and evaluation of the quantized model.
It uses 200 epochs, batch size 100, SGD momentum 0.9, weight decay 0.001,
dropout 0.1, and a five-epoch warmup to LR 0.1 followed by cosine decay.
STM's scheduler updates once per epoch; the PyTorch runner updates per batch.
Keras SGD weight decay is decoupled; PyTorch SGD applies an L2 gradient penalty.
These are comparable configurations, not identical optimization trajectories.

GSC CSV paths, 12 labels, waveform augmentation, held-out `running_tap.wav`
policy, 40 mel bins and 96-frame log-mel patches are retained from the PyTorch
v03 configuration. Additional SpecAugment remains off. There is no pretrained
checkpoint or PyTorch weight conversion: training starts from scratch.

Quantization uses the training-derived `quantization.csv`, per-channel weights,
int8 input and float output, matching the original STM v03 quantization settings.
Float output is intentional; the quantized network's output is dequantized for
the existing evaluator. No STEdgeAI installation is needed for host evaluation.
Hardware benchmarking/deployment is a separate workflow and is not validated here.

Results go under `tf/src/experiments_outputs/bcresnet_tf_v03_<timestamp>/`,
including STM training curves, float/quantized confusion matrices, best `.keras`
model, quantized `.tflite`, logs and MLflow records. `display_figures: false`
prevents opening GUI windows; PNG reports are still saved.

To quantize/evaluate a saved model without retraining:

```sh
python stm32ai_main.py --config-name user_config_gsc12_gsc_preproc_bcresnet_tf_v03 operation_mode=chain_eqe model.model_path=/path/to/best_model.keras
```

Individual services are selected with `operation_mode=training`,
`operation_mode=quantization model.model_path=/path/to/best_model.keras`, or
`operation_mode=evaluation model.model_path=/path/to/model.tflite`.

## Architecture and adaptation

`kws_streaming_model.py` expresses the upstream non-streaming graph with built-in
functional Keras layers, so save/reload needs no custom objects or Lambda layers.
It uses externally computed log-mel features instead of Google's raw-audio
frontend. The input adapter permutes STM `[N, frequency, time, 1]` to Google's
`[N, time, frequency, 1]`; the output is softmax for STM's categorical loss.
Streaming/causal state wrappers and QAT annotations are intentionally outside
this PTQ integration; installing the whole `kws_streaming` package is unnecessary.

The implementation preserves Google's default stage widths `[8,12,16,20]`,
dilations `[(1,1),(2,1),(3,1),(3,1)]`, and frequency strides `[1,2,2,1]`.
Upstream `blocks_n=[2,2,4,4]` counts normal blocks **in addition to** one transition
per stage: 16 blocks total. Its stem has bias and no BN/activation; its final
depthwise classifier uses SAME padding and no intermediate BN/activation.
SubSpectralNorm uses Google's direct frequency/channel reshape, not the previous
Qualcomm-oriented folding. These differences from Qualcomm are preserved.
Google's default dropout 0.5 is overridden to 0.1 by the training YAML.

Frequency means use full-frequency average pooling. Residual addition uses
Keras Add broadcasting, preserving upstream addition order. Expand/squeeze
operations at the boundaries use Permute/Reshape. No arbitrary TensorFlow ops
are applied to symbolic Keras tensors outside layers.

## Checks

```sh
python -m unittest discover -s tests -p test_bcresnet_tf.py -v
```

Tests compare transition/normal block outputs against the pristine Google source
classes with shared weights, verify all 16 blocks, run a training step, and
check `.keras` reload without custom objects. Full 200-epoch accuracy is not
claimed by these checks.
