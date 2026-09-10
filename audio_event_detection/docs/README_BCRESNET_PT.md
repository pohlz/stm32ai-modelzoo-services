# Original BC-ResNet in PyTorch

`pt/src/models/bcresnet.py` and `subspectralnorm.py` come from
[Qualcomm-AI-research/bcresnet](https://github.com/Qualcomm-AI-research/bcresnet),
the implementation accompanying Kim et al., *Broadcasted Residual Learning for
Efficient Keyword Spotting*, Interspeech 2021. The only source change is the
relative import of SubSpectralNorm. Copyright notices and the upstream
BSD-3-Clause-Clear license are retained in `pt/src/models/LICENSE.bcresnet`.
Source revision: `656f947a2a757ab5f36f649ed303ea0191cf553e` (verified against
the pinned raw sources, including the license).

The custom factory returns the original `BCResNets(int(tau * 8), num_classes)`.
It preserves upstream state-dict keys, stages `[2, 2, 4, 4]`, frequency strides,
five-band SubSpectralNorm, temporal dilation, SiLU, broadcast residuals,
Dropout2d(0.1), and the classifier. Inputs are float NCHW `[N, 1, 40, 96]`;
outputs are `[N, 12]` logits for cross-entropy, with class order from the YAML.
Apply `softmax(dim=1)` only if probabilities are needed for inference.

## Run

Use a PyTorch environment with `pip install -r pt/requirements.txt`.
From `stm32ai-modelzoo-services/audio_event_detection`:

```sh
python stm32ai_main_pt.py --config-name user_config_gsc12_gsc_preproc_bcresnet_py_v03
```

This is a separate native PyTorch entry point. The existing `stm32ai_main.py`,
TensorFlow model registry, Keras custom model and TFLite chain remain available
for the original configurations. The PyTorch runner supports `training` and
`evaluation`; quantization, deployment, prediction services and exact optimizer/RNG
resume are not implemented. Unsupported operation modes fail explicitly.

Paths in the YAML resolve from the launch directory. Hydra keeps that directory
and saves each run under `pt/src/experiments_outputs/`. Outputs include
`config.yaml`, `history.json`, `metrics.json`, `saved_models/best_model.pth`
(lowest validation loss), and `saved_models/last_model.pth`.

Evaluate a checkpoint, using the same tau, input features and class order:

```sh
python stm32ai_main_pt.py operation_mode=evaluation model.model_path=/path/to/best_model.pth
```

To fine-tune, provide `model.model_path`, set `training.fine_tune=true`, and
override the learning rate/epochs/warmup as appropriate. This loads weights
and starts a new optimizer and schedule. Raw upstream state dictionaries are
also accepted with strict shape/key checking, but their class order and feature
recipe must be supplied correctly by the caller. Keras weights are incompatible.
There are no bundled pretrained weights.

## Proposed changes from the TF v03 configuration

The proposal is implemented in the new `user_config_gsc12_gsc_preproc_bcresnet_py_v03.yaml`;
the original `user_config_gsc12_gsc_preproc_bcresnet_ft_v03.yaml` is preserved.

| Setting | TF v03 | PyTorch v03 proposal |
| --- | --- | --- |
| Entry point | `stm32ai_main.py` | `stm32ai_main_pt.py` |
| Operation | `chain_tqe` | `training` |
| Model | Keras custom BC-ResNet | Original Qualcomm PyTorch custom BC-ResNet-1 |
| Input shape | `[40, 96, 1]` | `[1, 40, 96]` |
| Initialization | fine_tune=true without weights | Train from scratch; explicit `.pth` path for fine-tuning |
| Optimizer | Adam, 0.001 | SGD, 0.1, momentum 0.9, weight decay 0.001 |
| Schedule | Plateau reduction, early stopping | 5 epochs warmup, per-step cosine decay |
| Training | 4 epochs, batch 32 | 200 epochs, batch 100 |
| Dropout | training.dropout=0.0 | Original built-in Dropout2d(0.1) |
| Quantization/tools | TFLite/STEdgeAI sections | Omitted; no PyTorch quantization chain |

The optimization defaults follow upstream `main.py`. This is **not a claim of
reproducing the paper's accuracy**: the existing v03 Librosa feature settings,
96-frame first patch, CSV balancing and held-out noise policy are retained.
Waveforms reuse the existing TensorFlow-free `GSCWaveformPipeline` module via
direct file loading, avoiding its parent package's TensorFlow imports. Features
use natural log(mel + 1e-6), matching the existing GSC path. Augmentation is
online for training only; validation/test are deterministic. Silence filenames
are checked for reserved-source leakage. Reuse CSVs from `tools/prepare_gsc_csv.py`.
The ordered 12 labels are preserved, not alphabetically sorted.

Additional SpecAugment is disabled, consistent with upstream tau=1. Larger tau
values are supported architecturally, but their upstream SpecAugment recipes
are not implemented here. DataLoader workers remain zero so the waveform RNG
is not copied into multiple workers. MLflow/ClearML and TF-specific callbacks
are not used by this runner.

## Verification

```sh
python -m unittest discover -s tests -p test_bcresnet_pt.py -v
```

Tests cover all upstream width scales, gradients, checkpoint round-trip and
class-order rejection, configuration/schedule checks, deterministic features,
silence leakage rejection, and a tiny synthetic train/save/evaluate cycle.

Verified in the existing WSL environment with PyTorch 2.7.1: all four tests pass.
A separate smoke check loaded the local 36,961 training / 4,437 validation CSV
rows, confirmed exact feature equality against the existing Librosa frequency
pipeline on a real GSC sample, and completed a backward pass without importing
TensorFlow. Full 200-epoch training and paper-level accuracy were not measured.
