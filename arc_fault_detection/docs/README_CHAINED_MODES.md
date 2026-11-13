# Chained Modes in Arc Fault Detection (AFD) Model Zoo

This guide explains how to use the AFD pipeline's chained modes. Chained modes allow you to automate multiple steps—such as training, quantization, evaluation, and benchmarking—using a single command. This streamlines your workflow and helps you focus on results.

## Table of Contents

<details open><summary><a href="#1"><b>1. Chained Modes Overview</b></a></summary></details>
<details open><summary><a href="#2"><b>2. Configuration Guidelines</b></a></summary></details>
<details open><summary><a href="#3"><b>3. Chained Mode Descriptions</b></a></summary></details>
<details open><summary><a href="#4"><b>4. Additional Notes</b></a></summary></details>
<details open><summary><a href="#5"><b>5. Running Chained Modes</b></a></summary></details>

---

<details open><summary><b><a href="#1">1. Chained Modes Overview</a></b></summary><a id="1"></a>

Chained modes specify the sequence of operations performed in the AFD pipeline. The mode is selected using the `operation_mode` field in your YAML configuration.

| Mode         | Steps Included                                               |
|--------------|-------------------------------------------------------------|
| `chain_tb`   | Training → Benchmarking                                     |
| `chain_tbqeb`| Training → Benchmarking → Quantization → Evaluation → Benchmarking |
| `chain_tqe`  | Training → Quantization → Evaluation                        |
| `chain_eqe`  | Evaluation → Quantization → Evaluation                      |
| `chain_qb`   | Quantization → Benchmarking                                 |
| `chain_eqeb` | Evaluation → Quantization → Evaluation → Benchmarking       |


 **Note:** Not all modes require training. Some start from a pretrained model and only perform quantization, evaluation, or benchmarking.

</details>

---

<details open><summary><b><a href="#2">2. Configuration Guidelines</a></b></summary><a id="2"></a>

1. **Select your `operation_mode`**: Choose the mode that fits your workflow.
2. **Prepare your YAML configuration**:
   - Always include: `general`, `model`, `mlflow`, and `hydra`.
   - Add: `dataset`, `preprocessing`, `training`, `quantization`, `evaluation`, and `benchmarking` if required by your chosen mode.
3. **Customize each section** for your project needs.

For complete configuration examples, refer to the YAML files  [here](../config_file_examples).

</details>

---

<details open><summary><b><a href="#3">3. Chained Mode Descriptions</a></b></summary><a id="3"></a>

Below is a summary of each chained mode, its workflow, and the required configuration sections.

<ul><details open><summary><a href="#3-1">chain_tb: Training and Benchmarking</a></summary><a id="3-1"></a>

Trains a model and then benchmarks it on your target board.
- **Required config sections:** `training`, `benchmarking`

</details></ul>
<ul><details open><summary><a href="#3-2">chain_tbqeb: Training, Benchmarking, Quantization, Evaluation, Benchmarking</a></summary><a id="3-2"></a>

Trains a model, benchmarks the float model, quantizes it, evaluates the quantized model, and benchmarks the quantized model.
- **Required config sections:** `training`, `quantization`, `evaluation`, `benchmarking`

</details></ul>
<ul><details open><summary><a href="#3-3">chain_tqe: Training, Quantization, Evaluation</a></summary><a id="3-3"></a>

Trains a model, quantizes it, and evaluates the quantized version.
- **Required config sections:** `training`, `quantization`, `evaluation`

</details></ul>
<ul><details open><summary><a href="#3-4">chain_eqe: Evaluation, Quantization, Evaluation</a></summary><a id="3-4"></a>

Starts from a pretrained model, evaluates it, quantizes it, and evaluates the quantized version.
- **Required config sections:** `evaluation`, `quantization`

</details></ul>
<ul><details open><summary><a href="#3-5">chain_qb: Quantization and Benchmarking</a></summary><a id="3-5"></a>

Quantizes a pretrained model and benchmarks the quantized version.
- **Required config sections:** `quantization`, `benchmarking`

</details></ul>
<ul><details open><summary><a href="#3-6">chain_eqeb: Evaluation, Quantization, Evaluation, Benchmarking</a></summary><a id="3-6"></a>

Starts from a pretrained model, evaluates it, quantizes it, evaluates the quantized version, and benchmarks the quantized model.
- **Required config sections:** `evaluation`, `quantization`, `benchmarking`

</details></ul>

</details>

---

<details open><summary><b><a href="#4">4. Additional Notes</a></b></summary><a id="4"></a>

- Every configuration must include `general`, `model`, `hydra`, and `mlflow`.
- The `operation_mode` field controls which steps are run.
- Only add `dataset`,  `preprocessing`, `training`, `quantization`, `evaluation`, or `benchmarking` if your chosen mode requires them.

</details>

---

<details open><summary><b><a href="#5">5. Running Chained Modes</a></b></summary><a id="5"></a>

After preparing your YAML configuration, you can launch any chained mode from your project folder with:

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name <config_name>.yaml
```

For example, to run the `chain_eqeb` mode:

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_eqeb_config.yaml
```

**Tips:**
- Ensure your config file matches the mode you want to run (see the descriptions above).
- You can create and customize your own config files by copying and editing the examples in [`arc_fault_detection/config_file_examples/`](../config_file_examples).

**Viewing Results:**
- All outputs—logs, trained or quantized models, evaluation metrics, and benchmarking results—are saved in the `experiments_outputs/<date-and-time>` directory (or as specified in your config).
- Detailed logs are available in `stm32ai_main.log` within the same directory.
- For experiment tracking and visualizations, launch MLflow with:
  ```bash
  cd tf/src/experiments_outputs
  mlflow ui
  ```
  Then open the provided link in your browser to explore your experiment history interactively.

</details>