#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <training-output-folder-name>" >&2
  echo "Example: $0 bcresnet_tf_v04_2026_09_11_05_08_38" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

run_name=$1
if [[ -z "$run_name" || "$run_name" == "." || "$run_name" == ".." || "$run_name" == */* || "$run_name" == *\\* ]]; then
  echo "Error: pass only the output folder name, not a path: $run_name" >&2
  usage
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
experiments_dir="$script_dir/tf/src/experiments_outputs"
run_dir="$experiments_dir/$run_name"
model_linux="$run_dir/quantized_models/quantized_model.tflite"
analysis_dir="$run_dir/NeuralART_analysis"
stedgeai="/mnt/c/ST/STEdgeAI/4.0/Utilities/windows/stedgeai.exe"

if [[ ! -d "$run_dir" ]]; then
  echo "Error: training output folder does not exist: $run_dir" >&2
  exit 1
fi
if [[ ! -f "$model_linux" ]]; then
  echo "Error: quantized model does not exist: $model_linux" >&2
  exit 1
fi
if [[ ! -x "$stedgeai" ]]; then
  echo "Error: STEdgeAI executable was not found: $stedgeai" >&2
  exit 1
fi

mkdir -p -- "$analysis_dir"
model_windows=$(wslpath -w "$model_linux")
analysis_windows=$(wslpath -w "$analysis_dir")

echo "Analyzing: $model_linux"
echo "Results:   $analysis_dir"

set +e
"$stedgeai" analyze \
  --model "$model_windows" \
  --target stm32n6 \
  --st-neural-art \
  --optimization balanced \
  --output "$analysis_windows" \
  --workspace "$analysis_windows" \
  2>&1 | tee "$analysis_dir/stedgeai_analyze.log"
status=${PIPESTATUS[0]}
set -e

if [[ $status -ne 0 ]]; then
  echo "Neural-ART analysis failed (exit $status). See: $analysis_dir/stedgeai_analyze.log" >&2
  exit "$status"
fi

echo "Neural-ART analysis completed: $analysis_dir"
