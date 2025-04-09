#!/bin/bash

# Usage:
# ./batch_dampen_bvh.sh <output_suffix> <n_jobs>
# Example:
# ./batch_dampen_bvh.sh delayed_bvh 16

# Check args
if [ $# -ne 2 ]; then
  echo "Usage: $0 <output_suffix> <n_jobs>"
  exit 1
fi

OUTPUT_SUFFIX="$1"
N_JOBS="$2"
BASE_FOLDER=~/data/dnd/Session_1
CONFIG_PATH=./configs/joint_params.yaml
LABELS=("a" "b" "c" "j" "l")

for LABEL in "${LABELS[@]}"; do
  INPUT_PATH="${BASE_FOLDER}/${LABEL}/${LABEL}_chunks"
  OUTPUT_PATH="${BASE_FOLDER}/${LABEL}/${OUTPUT_SUFFIX}"

  echo "Processing $LABEL with $N_JOBS jobs..."
  uv run tools/bvh_tools.py dampen \
    --input-path "$INPUT_PATH" \
    --config-path "$CONFIG_PATH" \
    --output-path "$OUTPUT_PATH" \
    --n_jobs "$N_JOBS"
done
