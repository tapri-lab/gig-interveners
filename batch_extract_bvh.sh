#!/bin/bash

# Usage:
# ./batch_extract_bvh.sh <output_suffix>
# Example:
# ./batch_extract_bvh.sh world_pos.zip

# Check that suffix was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <output_suffix>"
  exit 1
fi

OUTPUT_SUFFIX="$1"
BASE_FOLDER=~/data/dnd/Session_1
LABELS=("a" "b" "c" "j" "l")
JOINT_NAMES=("LeftHand" "LeftArm" "RightHand" "RightArm")

for LABEL in "${LABELS[@]}"; do
  INPUT_FOLDER="${BASE_FOLDER}/${LABEL}/${LABEL}_chunks"
  OUTPUT_PATH="${BASE_FOLDER}/${LABEL}/${LABEL}_${OUTPUT_SUFFIX}"

  echo "Processing $LABEL..."
  uv run tools/bvh_tools.py extract \
    --folder "$INPUT_FOLDER" \
    --joint-names "${JOINT_NAMES[@]}" \
    --output-path "$OUTPUT_PATH"
done
