# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from audio_event_detection.tf.src.datasets import ESC10TFDataset
from .utils import get_pipelines

__all__ = ['get_esc10']

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='esc10', use_case="audio_event_detection")
def get_esc10(cfg):
    """
    Build TensorFlow datasets for the ESC-10 audio dataset.

    Parameters
    ----------
    cfg : DictConfig
        User configuration

    Returns
    -------
    dict
        Dictionary with keys: `train_ds`, `val_ds`, `quantization_ds`, `test_ds`,
        `val_clip_labels`, `test_clip_labels`, `pred_ds`, `pred_clip_labels`, `pred_filenames`.
    """

    time_pipeline, freq_pipeline = get_pipelines(cfg)

    # Get dataset related args

    training_csv_path = cfg.dataset.get("training_csv_path", None)
    training_audio_path = cfg.dataset.get("training_audio_path", None)
    validation_csv_path = cfg.dataset.get("validation_csv_path", None)
    validation_audio_path = cfg.dataset.get("validation_audio_path", None)
    validation_split = cfg.dataset.get("validation_split", None)
    quantization_csv_path = cfg.dataset.get("quantization_csv_path", None)
    quantization_audio_path = cfg.dataset.get("quantization_audio_path", None)
    test_csv_path = cfg.dataset.get("test_csv_path", None)
    test_audio_path = cfg.dataset.get("test_audio_path", None)
    # Keep compatibility with the old yaml format for prediction files path
    if cfg.prediction is not None:
        pred_audio_path = cfg.prediction.get("test_files_path", None)
    # Overwrite with audio path in dataset section of config file if present
    pred_audio_path = cfg.dataset.get("prediction_audio_path", None)

    class_names = cfg.dataset.get("class_names", None)
    use_garbage_class = cfg.dataset.get("use_garbage_class", False)
    file_extension = cfg.dataset.get("file_extension", ".wav")
    n_samples_per_garbage_class = cfg.dataset.get("n_samples_per_garbage_class", None)
    seed = cfg.dataset.get("seed", 133)
    to_cache = cfg.dataset.get("to_cache", True)
    shuffle = cfg.dataset.get("shuffle", True)

    if cfg.training:
        batch_size = cfg.training.get('batch_size', 16)
    elif cfg.general:
        batch_size = cfg.general.get('batch_size', 16)
    else:
        batch_size = 16

    dataset = ESC10TFDataset(
        time_pipeline=time_pipeline,
        freq_pipeline=freq_pipeline,
        training_csv_path=training_csv_path,
        training_audio_path=training_audio_path,
        validation_csv_path=validation_csv_path,
        validation_audio_path=validation_audio_path,
        validation_split=validation_split,
        quantization_csv_path=quantization_csv_path,
        quantization_audio_path=quantization_audio_path,
        test_csv_path=test_csv_path,
        test_audio_path=test_audio_path,
        pred_audio_path=pred_audio_path,
        class_names=class_names,
        use_garbage_class=use_garbage_class,
        file_extension=file_extension,
        n_samples_per_garbage_class=n_samples_per_garbage_class,
        expand_last_dim=True,
        seed=seed
    )

    tf_datasets = dataset.get_tf_datasets(batch_size=batch_size,
                                          shuffle=shuffle,
                                          to_cache=to_cache)
    
    return tf_datasets