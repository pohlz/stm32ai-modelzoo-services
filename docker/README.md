
# STM32AI Model Zoo – Docker Setup & Usage

Docker ensures the reproducibility and portability of the execution environment, regardless of the operating system used (Windows, Linux, etc.). With Docker, all dependencies, tools, and required configurations are encapsulated in a single image, avoiding compatibility issues and manual setup. This greatly simplifies deployment, execution, and sharing of the project, while ensuring that results will be identical on any machine.

This guide explains how to build the STM32AI Model Zoo Docker image and run it on GPU or CPU with launcher.sh, including dataset mounting and typical training workflows.

## Requirements
- Docker 24+
- GPU on Linux: NVIDIA driver + NVIDIA Container Toolkit (enables `docker run --gpus all`)
- GPU on Windows (Docker Desktop, WSL2): enable GPU support in Docker Desktop; ensure NVIDIA driver on Windows (no manual `nvidia-docker` install)
- Internet access for base image and Python dependencies

## Quick Start
```bash
# Use GPU (default)
bash docker/launcher.sh

# Use CPU
bash docker/launcher.sh --cpu
```

## Image Contents
- Base: `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`
- Python 3.12, virtualenv, pip
- Project copied to `/workspace/stm32ai-modelzoo-services`
- Virtualenv at `/workspace/stm32ai-modelzoo-services/.venv` with `requirements.txt` installed
- `PYTHONPATH` includes `/workspace/stm32ai-modelzoo-services`

## Run via launcher.sh (recommended)
`docker/launcher.sh` builds the image and runs an interactive container.

**Key features:**
- GPU toggle: `--gpu` (default) or `--cpu` or env `USE_GPU=false`
- Shared memory tuning: `SHM_SIZE=8g` (default) or `USE_IPC_HOST=true`
- Mounts experiment output folders back to your host
- Mounts datasets you specify in the script’s `DATASETS` map

Examples:
```bash
# Use GPU (default)
bash docker/launcher.sh

# Force CPU
bash docker/launcher.sh --cpu

# Increase shared memory
SHM_SIZE=16g bash docker/launcher.sh

# Use host IPC (Linux only)
USE_IPC_HOST=true bash docker/launcher.sh
```

### Configure dataset mounts
Edit the `DATASETS` map at the top of `docker/launcher.sh`:
```bash
DATASETS["image_classification"]="/absolute/path/to/ic_flower_photos.zip"
```
At launch, each path is mounted under `/workspace/stm32ai-modelzoo-services/<use_case>/datasets/`.

Verify mounts inside the container:
```bash
ls -la /workspace/stm32ai-modelzoo-services/image_classification/datasets
```

### Experiments outputs
Experiment outputs generated inside the container are saved on the host via bind mounts: 
- Tensorflow: `<use_case>/tf/src/experiments_outputs`
- PyTorch (IC/OD): `<use_case>/pt/src/experiments_outputs`

## Run inside the container
The container starts in `/workspace`. The venv is auto-activated via `~/.bashrc` in interactive shells, but you can source it explicitly:
```bash
source /workspace/stm32ai-modelzoo-services/.venv/bin/activate
cd /workspace/stm32ai-modelzoo-services/image_classification
python stm32ai_main.py --config-name user_config_pt.yaml
```

### Optional: Clone stm32ai-modelzoo inside container
Repository: [stm32ai-modelzoo (GitHub)](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/main)
You can ask the image build to clone the external repo into `/workspace/stm32ai-modelzoo`:
> [!NOTE]
To enable on-demand cloning of the `stm32ai-modelzoo` repo into `/workspace/stm32ai-modelzoo`, set `CLONE_STM32AI_MODELZOO=true` (optionally `STM32AI_MODELZOO_BRANCH` and `STM32AI_MODELZOO_DIRNAME`) before build; cloning occurs at image build time and is cached by Docker.
Example:
```bash
export CLONE_STM32AI_MODELZOO=true
export STM32AI_MODELZOO_BRANCH=main
bash docker/launcher.sh
```
<br>

### Proxy support (if needed)
The launcher forwards common proxy args automatically if set in your environment:
- `HTTP_PROXY`, `http_proxy`, `HTTPS_PROXY`, `https_proxy`, `NO_PROXY`, `no_proxy`

Example manual build with proxies:
```bash
docker build \
  --build-arg HTTP_PROXY=$HTTP_PROXY \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  -t modelzoo_docker -f docker/Dockerfile docker/..
```


## Troubleshooting
- No GPU visible: ensure NVIDIA Container Toolkit is installed and use `--gpus all`.
- Datasets not found: verify host paths in `DATASETS` exist; they must be absolute.
- Increase shared memory if DataLoader errors: set `SHM_SIZE=16g` or `USE_IPC_HOST=true`.

- Diagnose shared memory issues:
  - Inside container:
    ```bash
    df -h /dev/shm
    cat /proc/mounts | grep /dev/shm
    ```

> [!NOTE]
With `--ipc=host`, the container uses the host `/dev/shm` and `--shm-size` is ignored.

- GPU compatibility check:
  - The base image uses CUDA 12.6; ensure your host NVIDIA driver supports CUDA 12.x. Validate with `nvidia-smi` on the host and a quick container check:
    ```bash
    docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
    ```
## License
This is licensed under the SLA0044 License. See the [LICENSE](./LICENSE.md) file.
