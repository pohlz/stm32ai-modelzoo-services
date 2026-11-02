# Run inside your existing Docker terminal

Requires Python 3 and working `nvidia-smi` inside the container. Both GPUs must already be exposed to the container. No extra Python packages are needed.

Wait for either GPU to be idle, then run training:

```bash
python3 wait_gpu.py -- python3 train.py --epochs 100
```

Wait for at least 12 GiB free on either GPU, even if it is busy:

```bash
python3 wait_gpu.py --memory 12G -- python3 train.py --epochs 100
```

Memory accepts MiB (`8192` or `8192MiB`) or GiB (`8G` or `8GiB`). The eligible GPU with the most free memory wins. Without `--memory`, idle means no compute processes and utilization at most 1%.

The command inherits your terminal, working directory, and environment, with `CUDA_VISIBLE_DEVICES` replaced by the selected GPU UUID. Your training should use **cuda:0** (or its default GPU), even when physical GPU 1 was selected. The UUID selection is supported by [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

Options go before `--`:

```bash
python3 wait_gpu.py --memory 8G --interval 5 --timeout 3600 -- python3 train.py
```

Default polling is every 10 seconds with no timeout. Timeout exits with code 124; once started, your command receives signals and determines the exit code. For shell expressions, pass `bash -c 'your command'` explicitly. The script targets full NVIDIA GPUs, not MIG instances.

Free memory is a snapshot, not a reservation or enforced memory limit. Concurrent launchers or other users can take the memory before your training allocates it. Leave headroom for peak memory use. This launcher does not coordinate a queue between multiple submissions.
