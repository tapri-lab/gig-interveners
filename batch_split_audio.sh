#!/bin/bash

# Usage:
# ./batch_split_audio.sh <segment_length> <offset> <output_suffix>
# Example:
# ./batch_split_audio.sh 30 0.5 delay_m0.5

# Check args
if [ $# -ne 3 ]; then
  echo "Usage: $0 <segment_length> <offset> <output_suffix>"
  exit 1
fi

SEGMENT_LENGTH="$1"
OFFSET="$2"
OUTPUT_SUFFIX="$3"

BASE_FOLDER=~/data/dnd/Session_1
LABELS=("a" "b" "c" "j" "l")

for LABEL in "${LABELS[@]}"; do
  INPUT_FILE="${BASE_FOLDER}/${LABEL}/${LABEL}-clean.wav"
  OUTPUT_DIR="${BASE_FOLDER}/${LABEL}/${LABEL}_${OUTPUT_SUFFIX}"

  echo "Splitting $INPUT_FILE into segments of $SEGMENT_LENGTHs with offset $OFFSET..."
  uv run tools/audio_tools.py split_audio \
    --input-file "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --segment-length "$SEGMENT_LENGTH" \
    --offset "$OFFSET"
done

