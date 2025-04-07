#!/bin/bash

# Usage:
# ./batch_limit_f0.sh <max_deviation_hz> <output_suffix>
# Example:
# ./batch_limit_f0.sh 60 shift_60hz

# Check arguments
if [ $# -ne 2 ]; then
  echo "Usage: $0 <max_deviation_hz> <output_suffix>"
  exit 1
fi

MAX_DEVIATION_HZ="$1"
OUTPUT_SUFFIX="$2"
BASE_FOLDER=~/data/dnd/Session_1
LABELS=("a" "b" "c" "j" "l")

for LABEL in "${LABELS[@]}"; do
  AUDIO_PATH="${BASE_FOLDER}/${LABEL}/audio_segments"
  OUTPUT_PATH="${BASE_FOLDER}/${LABEL}/${OUTPUT_SUFFIX}"

  echo "Processing $LABEL with max deviation ${MAX_DEVIATION_HZ}Hz..."
  uv run tools/audio_tools.py limit_f0 \
    --audio-path "$AUDIO_PATH" \
    --output-path "$OUTPUT_PATH" \
    --max-deviation-hz "$MAX_DEVIATION_HZ"
done
