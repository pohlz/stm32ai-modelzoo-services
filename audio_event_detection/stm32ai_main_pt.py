"""Native PyTorch AED entry point (training/evaluation, no TensorFlow imports)."""

import hydra
from hydra.core.hydra_config import HydraConfig

from pt.src.runner import run


@hydra.main(version_base=None, config_path=".", config_name="user_config_gsc12_gsc_preproc_bcresnet_py_v03")
def main(cfg):
    run(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
