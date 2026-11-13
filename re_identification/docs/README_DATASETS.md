# Re-Identification STM32 model zoo

A ReID dataset is a flat directory of images (no nested class folders). Files belonging to the same identity must share an exact prefix; the part after the prefix can encode camera or frame information.

## Expected directory layout

```text
dataset_root_dir/
    id1_1.jpg
    id1_2.jpg
    id2_1.jpg
    id2_2.jpg
```

- Identity naming: `X_*` where `X` is the identity label (e.g., `id1`, `id2`, `00034`). Avoid spaces or special characters; zero-pad if you want lexical ordering to match numeric ordering.
- File naming: The suffix after the first underscore can be any unique token (index, camera, timestamp), for example `id12_camA_0045.jpg`.


## Directory components

`datasets/` is a placeholder root for re-identification datasets.