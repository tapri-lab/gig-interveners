#!/bin/bash

# Usage:
# ./batch_extract_bvh.sh <input_folder_suffix> <output_suffix>
# Example:
# ./batch_extract_bvh.sh l_chunks world_pos.zip

# Check that both arguments are provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <input_folder_suffix> <output_suffix>"
  exit 1
fi

INPUT_SUFFIX="$1"
OUTPUT_SUFFIX="$2"
BASE_FOLDER=~/data/dnd/Session_1
LABELS=("a" "b" "c" "j" "l")
JOINT_NAMES=("LeftHand" "LeftArm" "RightHand" "RightArm")

for LABEL in "${LABELS[@]}"; do
  INPUT_FOLDER="${BASE_FOLDER}/${LABEL}/${INPUT_SUFFIX}"
  OUTPUT_PATH="${BASE_FOLDER}/${LABEL}/${LABEL}_${OUTPUT_SUFFIX}"

  echo "Processing $LABEL..."
  uv run tools/bvh_tools.py extract \
    --folder "$INPUT_FOLDER" \
    --joint-names "${JOINT_NAMES[@]}" \
    --output-path "$OUTPUT_PATH"
done
