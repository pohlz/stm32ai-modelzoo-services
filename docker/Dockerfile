# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025-2026 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

#optional: proxy settings
# ARG HTTP_PROXY
# ARG http_proxy
# ARG HTTPS_PROXY
# ARG https_proxy
# ARG NO_PROXY
# ARG no_proxy

# Optional: conditionally clone stm32ai-modelzoo during build
ARG CLONE_STM32AI_MODELZOO=true
ARG STM32AI_MODELZOO_BRANCH=main
ARG STM32AI_MODELZOO_DIRNAME=stm32ai-modelzoo

ENV DEBIAN_FRONTEND=noninteractive

#optional: proxy settings
# ENV HTTP_PROXY=$HTTP_PROXY
# ENV http_proxy=$http_proxy
# ENV HTTPS_PROXY=$HTTPS_PROXY
# ENV https_proxy=$https_proxy
# ENV NO_PROXY=$NO_PROXY
# ENV no_proxy=$no_proxy

RUN apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    libgl1 \
    git \
    git-lfs \
    nano \
    wget \
    unzip \
    curl \
    libglib2.0-0 \
    ca-certificates && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN mkdir -p /opt/datasets

WORKDIR /workspace

ENV PYTHONPATH=/workspace/stm32ai-modelzoo-services:/workspace
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN mkdir -p /workspace/stm32ai-modelzoo-services
COPY . /workspace/stm32ai-modelzoo-services

## On-demand clone of stm32ai-modelzoo into /workspace with branch fallback
RUN if [ "$CLONE_STM32AI_MODELZOO" = "true" ]; then \
            set -e; \
            echo "[Dockerfile] Cloning stm32ai-modelzoo (requested branch: $STM32AI_MODELZOO_BRANCH)"; \
            DEST="/workspace/$STM32AI_MODELZOO_DIRNAME"; \
            if [ -d "$DEST/.git" ] || [ -d "$DEST" ]; then \
                echo "[Dockerfile] Target exists: $DEST — skipping clone."; \
            else \
                if git ls-remote --exit-code --heads https://github.com/STMicroelectronics/stm32ai-modelzoo.git "$STM32AI_MODELZOO_BRANCH" >/dev/null 2>&1; then \
                    git clone --depth 1 -b "$STM32AI_MODELZOO_BRANCH" https://github.com/STMicroelectronics/stm32ai-modelzoo.git "$DEST"; \
                else \
                    echo "[Dockerfile] Branch '$STM32AI_MODELZOO_BRANCH' not found; falling back to 'main'"; \
                    git clone --depth 1 -b main https://github.com/STMicroelectronics/stm32ai-modelzoo.git "$DEST"; \
                fi; \
            fi; \
        else \
            echo "[Dockerfile] Skipping stm32ai-modelzoo clone"; \
        fi

RUN python -m venv /workspace/stm32ai-modelzoo-services/.venv && \
    /bin/bash -c "source /workspace/stm32ai-modelzoo-services/.venv/bin/activate && pip install --upgrade pip && pip install --no-cache-dir -r /workspace/stm32ai-modelzoo-services/requirements.txt"

RUN echo "source /workspace/stm32ai-modelzoo-services/.venv/bin/activate" >> ~/.bashrc

RUN git lfs install

CMD ["/bin/bash"]