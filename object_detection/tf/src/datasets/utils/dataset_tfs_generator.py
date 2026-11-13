import os
import glob
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from .dataset_analyzers import parse_label_file


def add_tfs_files_to_dataset(dataset_path: str,
                             annotations_path: str,
                             output_path: str,
                             exclude_unlabeled_images: bool = False,
                             padded_labels_size: int = 20) -> None:
    """
    Create .tfs files from images + YOLO txt labels.

    Args:
        dataset_path: directory containing images (.jpg)
        annotations_path: directory containing temporary YOLO .txt labels
        output_path: directory where .tfs files will be written
        exclude_unlabeled_images: skip images with empty labels
        padded_labels_size: maximum number of detections per image

    Returns:
        None
    """
    print("\nAdding .tfs labels files to dataset:")
    print("----------------------------------")
    print("Images root:", dataset_path)
    print("Annotations root:", annotations_path)
    print("Output root:", output_path)
    print("Padded labels size:", padded_labels_size)

    if not os.path.isdir(dataset_path):
        raise ValueError(f"dataset_path '{dataset_path}' does not exist or is not a directory")
    if not os.path.isdir(annotations_path):
        raise ValueError(f"annotations_path '{annotations_path}' does not exist or is not a directory")

    os.makedirs(output_path, exist_ok=True)

    jpg_file_paths = glob.glob(os.path.join(dataset_path, "*.jpg"))
    if len(jpg_file_paths) == 0:
        raise ValueError(f"Could not find any .jpg file in dataset root directory {dataset_path}")

    if padded_labels_size is None:
        padded_labels_size = 20
    if padded_labels_size <= 0:
        raise ValueError("Error getting maximum number of detections")

    # Remove previous .tfs files
    for path in glob.glob(os.path.join(output_path, "*.tfs")):
        os.remove(path)

    discarded = 0
    background = 0
    num_images = len(jpg_file_paths)

    for jpg_path in tqdm(jpg_file_paths):
        stem = Path(jpg_path).stem
        txt_path = os.path.join(annotations_path, stem + ".txt")
        out_path = os.path.join(output_path, stem + ".tfs")

        labels = parse_label_file(txt_path)
        if not labels:
            background += 1
            if exclude_unlabeled_images:
                continue
            labels = [[0., 0., 0., 0., 0.]]

        if len(labels) > padded_labels_size:
            discarded += 1
            continue

        padded_labels = []
        for i in range(padded_labels_size):
            if i < len(labels):
                padded_labels.append(labels[i])
            else:
                padded_labels.append([0., 0., 0., 0., 0.])

        data = tf.convert_to_tensor(padded_labels, dtype=tf.float32)
        data = tf.io.serialize_tensor(data)
        tf.io.write_file(out_path, data)

    print("Number of image files:", num_images)
    remaining = num_images - background if exclude_unlabeled_images else num_images
    print("Discarded examples due to number of labels greater than padding size: {}  ({:.1f}%)".format(
        discarded, 100 * discarded / remaining if remaining else 0.0
    ))
    remaining -= discarded
    print("Remaining examples: {}   ({:.1f}%)".format(
        remaining, 100 * remaining / num_images if num_images else 0.0
    ))
    print("Background images: {}   ({:.1f}%)".format(
        background, 100 * background / num_images if num_images else 0.0
    ))