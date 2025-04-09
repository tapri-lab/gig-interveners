#!/bin/bash

# Usage:
# ./batch_split_bvh.sh [chunk_suffix]
# Example:
# ./batch_split_bvh.sh a_chunks  # optional, defaults to '<label>_chunks'

BASE_FOLDER=~/data/dnd/Session_1
LABELS=("a" "b" "c" "j" "l")

# Optional suffix for output dirs, default is <label>_chunks
CHUNK_SUFFIX=$1

for LABEL in "${LABELS[@]}"; do
  INPUT_BVH="${BASE_FOLDER}/${LABEL}/${LABEL}.bvh"
  OUTPUT_DIR="${BASE_FOLDER}/${LABEL}/${CHUNK_SUFFIX:-${LABEL}_chunks}"

  echo "Splitting $INPUT_BVH into $OUTPUT_DIR..."
  uv run tools/bvh_tools.py split \
    --input-bvh-path "$INPUT_BVH" \
    --output-dir "$OUTPUT_DIR"
done
