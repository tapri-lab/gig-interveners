# Interveners

This repository contains tools for creating intervened sequences from motion capture (BVH) and audio data. It's designed for research on multimodal synchronization in group interactions, allowing you to apply controlled interventions (dampening gestures, limiting vocal pitch) and measure their effects using various synchronization metrics.

## Setup Instructions

**You will need to install `uv` first, if you do not already have it**
Instructions to install `uv` can be found here: [uv installation instructions.](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

If you have `brew` installed on macOS, you can do: `brew install uv`

`uv` will automatically install dependencies when running commands for the below tools.

## Tools

To see all options available for a given script/command use the `-h` help flag.

### Vocal Pitch Limiter

Limit the fundamental frequency (F0) variability in speech to stay within a maximum deviation from the mean F0.

```sh
uv run python vocal_features/pitch_shifter.py --audio-path PATH --output-path PATH [--max-deviation-hz FLOAT]
```

**Example:**
```sh
uv run python vocal_features/pitch_shifter.py --audio-path input.wav --output-path output.wav --max-deviation-hz 50.0
```

### Motion Dampener (BVH)

Dampen the motion of specific joints in BVH files using Gaussian smoothing on rotation data. Requires a YAML configuration file specifying which joints to dampen and their parameters.

```sh
uv run python tools/bvh_tools.py dampen --input-path PATH --config-path PATH --output-path PATH [--n-jobs N]
```

**Example:**
```sh
uv run python tools/bvh_tools.py dampen --input-path motion.bvh --config-path configs/joint_params.yaml --output-path output_dir/
```

See `configs/joint_params.yaml` for configuration examples.

### SMPLX Parameter Dampener

Dampen SMPLX pose parameters for specified joints using Gaussian smoothing.

```sh
uv run python tools/bvh_tools.py dampen_smpl --input-path PATH --config-path PATH --output-path PATH [--n-jobs N]
```

**Example:**
```sh
uv run python tools/bvh_tools.py dampen_smpl --input-path params.npz --config-path configs/joint_params_smplx.yaml --output-path output_dir/
```

### World Position Extraction

Extract world positions of specified joints from BVH files and store them in compressed Zarr archives (`.zip`).

```sh
uv run python tools/bvh_tools.py extract --folder PATH --joint-names JOINT1 JOINT2 ... [--output-path PATH]
```

**Example:**
```sh
uv run python tools/bvh_tools.py extract --folder bvh_chunks/ --joint-names LeftHand RightHand LeftArm RightArm --output-path world_positions.zip
```

### BVH Splitter

Split a BVH file into multiple chunks based on time duration (default: 30 seconds).

```sh
uv run python tools/bvh_tools.py split --input-bvh-path PATH --output-dir PATH [--chunk-duration-sec FLOAT]
```

**Example:**
```sh
uv run python tools/bvh_tools.py split --input-bvh-path long_motion.bvh --output-dir chunks/ --chunk-duration-sec 30.0
```

## Video Processing

Tools for stitching, cropping, and splitting videos. Supports both direct CLI usage and YAML config-driven workflows.

### Direct CLI Usage

```sh
# Stitch 4 videos into a 2x2 grid, optionally with a side video and audio
uv run python tools/video.py stitch --global-video-path DIR --output-path DIR [--individual-video-path PATH] [--audio-paths PATH ...] [--audio-offset FLOAT]

# Center-crop videos to a specified resolution
uv run python tools/video.py crop --input-path DIR --output-path DIR --crop-width INT --crop-height INT [--preset STR]

# Split videos into fixed-length chunks
uv run python tools/video.py split --input-path DIR --output-path DIR --chunk-length INT
```

### Config-Driven Usage

Run operations from a YAML config file:

```sh
uv run python tools/video.py config --config-path PATH [--operation OPERATION]
```

`--operation` can be `stitch`, `crop`, `split`, or `batch_stitch` (defaults to `stitch`).

#### Stitch Config

Stitches 4 global camera views into a 2x2 grid, optionally with a side video and audio overlay.

```yaml
stitch:
  global_video_path: "path/to/global_videos/"  # directory with exactly 4 .mp4 files
  individual_video_path: "path/to/side_video.mp4"  # optional vertical side video
  audio_paths:
    - "path/to/audio1.wav"
    - "path/to/audio2.wav"
  audio_offset: 7.4  # delay audio by this many seconds
  output_path: "path/to/output/"
```

#### Batch Stitch Config

Runs multiple stitch jobs in sequence. Supports a `{person}` placeholder in paths for automatic per-person discovery from directory structure.

```yaml
batch_stitch:
  - name: "skeleton_grp_no_intervention"
    global_video_path: "renders/global/skeleton/{person}/*/*.mp4"
    individual_video_path: "renders/individual/skeleton/{person}/*.mp4"
    audio_paths:
      - "audio/session_1/*.wav"
    audio_offset: 7.4
    output_path: "output/stitched/"

  - name: "skeleton_grp_dampened"
    global_video_path: "renders/global/dampened/{person}/*/*.mp4"
    audio_paths:
      - "audio/session_1/*.wav"
    audio_offset: 7.4
    output_path: "output/stitched_dampened/"
```

#### Crop Config

```yaml
crop:
  input_path: "path/to/videos/"
  output_path: "path/to/cropped/"
  crop_width: 1920
  crop_height: 1080
  preset: "fast"
```

#### Split Config

```yaml
split:
  input_path: "path/to/videos/"
  output_path: "path/to/chunks/"
  chunk_length: 30  # seconds
```

See `configs/video_skel.yaml` and `configs/video_smplx.yaml` for full working examples.

## Metrics Analysis

Compute synchronization metrics (RQA, SDTW, beat consistency) across participants. Requires a YAML configuration file specifying data paths and which metrics to compute.

```sh
uv run python metrics/analyze.py --cfg-path PATH [--n-jobs N] [--output-dir PATH]
```

**Example:**
```sh
uv run python metrics/analyze.py --cfg-path configs/metrics.yaml --n-jobs -1 --output-dir results/
```

See `configs/metrics.yaml` for configuration examples. Available metrics include:
- Individual and cross-person RQA (recurrence quantification analysis)
- Individual and cross-person SDTW (soft dynamic time warping)
- Beat consistency (audio-motion synchrony)
- Pitch variability analysis

## Batch Processing

The repository includes bash scripts for batch processing multiple files:
- `batch_limit_f0.sh` - Process multiple audio files
- `batch_dampen_bvh.sh` - Dampen multiple BVH files
- `batch_split_bvh.sh` - Split multiple BVH files
- `batch_extract_bvh.sh` - Extract world positions from multiple files
- `batch_split_audio.sh` - Split multiple audio files

## Configuration Files

Configuration files are stored in `configs/`:
- `joint_params.yaml` - Joint dampening parameters for BVH files
- `joint_params_smplx.yaml` - Joint dampening parameters for SMPLX
- `metrics.yaml` - Metrics analysis configuration
- `motion_damped.yaml` - Motion dampening configurations
- `pitch_shifted.yaml` - Pitch shifting configurations
- `video_skel.yaml` - Video stitching for skeleton renders
- `video_skel_damp.yaml` - Video stitching for dampened skeleton renders
- `video_smplx.yaml` - Video stitching for SMPLX renders
- `video_smplx_damp.yaml` - Video stitching for dampened SMPLX renders

## Code Formatting

The project uses Ruff for linting and formatting:

```sh
uv run ruff format .
uv run ruff check .
```