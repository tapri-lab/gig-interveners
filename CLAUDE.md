# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains tools for creating intervened sequences from gesture and audio data, primarily for analyzing multimodal synchronization in group interactions. The project processes BVH motion capture files and audio data to apply controlled interventions (dampening gestures, limiting vocal pitch) and measure their effects using various synchronization metrics.

## Development Setup

The project uses `uv` for dependency management. Install it via: `brew install uv` (macOS) or follow [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

All commands should be run with `uv run python <script>` to ensure dependencies are automatically installed and managed.

## Core Architecture

### Main Entry Points

The repository has three primary CLI tools accessed through `tools/bvh_tools.py`:

1. **Motion Dampener** (`dampen` subcommand): Applies Gaussian smoothing to specific joints in BVH files
2. **SMPLX Dampener** (`dampen_smpl` subcommand): Applies Gaussian smoothing to SMPLX pose parameters
3. **World Position Extractor** (`extract` subcommand): Extracts world positions of joints and stores in `.npz` files
4. **BVH Splitter** (`split` subcommand): Splits BVH files into chunks by duration

Vocal pitch limiting is handled by `vocal_features/pitch_shifter.py`.

### Key Commands

```bash
# Dampen motion in BVH files
uv run python tools/bvh_tools.py dampen --input-path PATH --config-path PATH --output-path PATH [--n-jobs N]

# Dampen SMPLX parameters
uv run python tools/bvh_tools.py dampen_smpl --input-path PATH --config-path PATH --output-path PATH [--n-jobs N]

# Extract world positions
uv run python tools/bvh_tools.py extract --folder PATH --joint-names JOINT1 JOINT2 ... [--output-path PATH]

# Split BVH by duration
uv run python tools/bvh_tools.py split --input-bvh-path PATH --output-dir PATH [--chunk-duration-sec FLOAT]

# Limit vocal pitch range
uv run python vocal_features/pitch_shifter.py --audio-path PATH --output-path PATH [--max-deviation-hz FLOAT]

# Run metrics analysis
uv run python metrics/analyze.py --cfg-path PATH [--n-jobs N] [--output-dir PATH]
```

### Configuration System

The project uses YAML configuration files (stored in `configs/`) to specify intervention parameters:

- **joint_params.yaml**: Defines which joints to dampen and their parameters (sigma, damping_factor, iterations)
- **metrics.yaml**: Configures which metrics to compute, data paths, and analysis settings
- Joint dampening uses Gaussian filtering on rotation data in scaled axis-angle representation

### Directory Structure

- `tools/`: Core utilities for BVH/audio processing and video generation
- `vocal_features/`: Vocal pitch manipulation using Parselmouth/Praat
- `metrics/`: Synchronization analysis (RQA, SDTW, beat consistency)
- `exploration/`: Jupyter notebooks and scripts for data exploration
- `sweeper/`: Parameter sweep utilities for batch processing
- `viz/`: Visualization tools (SMPL viewer, kinematic tree processing)
- `configs/`: YAML configuration files for interventions and metrics

### Data Pipeline

1. **Input**: Raw BVH motion capture files + audio WAV files
2. **Chunking**: BVH/audio split into 30s segments using batch scripts or split commands
3. **Intervention**: Apply dampening (motion) or pitch limiting (audio) with specified parameters
4. **Extraction**: Convert BVH to world positions stored in Zarr archives
5. **Analysis**: Compute synchronization metrics (RQA, SDTW, beat consistency) across participants

### Key Dependencies

- `pymotion`: BVH file I/O and skeletal operations (FK, quaternion math)
- `parselmouth`: Praat-based pitch manipulation
- `tyro`: Type-based CLI parsing
- `polars`: Fast dataframe operations for analysis results
- `zarr`: Compressed storage for extracted world positions
- `joblib`: Parallel processing for batch operations

### Metrics Module

The `metrics/analyze.py` orchestrates multiple synchronization analyses:
- Individual/cross-person RQA (recurrence quantification analysis)
- Individual/cross-person SDTW (soft dynamic time warping)
- Beat consistency (audio-motion synchrony via EMD)
- Pitch variability analysis

Metrics are configured via YAML (see `configs/metrics.yaml` for schema) and run in parallel using joblib.

### Code Formatting

The project uses Ruff for linting/formatting with 120-char line length and space indentation. Run: `uv run ruff format .` or `uv run ruff check .`

## Important Implementation Details

- BVH dampening converts quaternions to scaled axis-angle, applies Gaussian filter, then converts back
- SMPLX dampening operates directly on axis-angle pose parameters (165D for 55 joints)
- Batch scripts (batch_*.sh) demonstrate common processing workflows for multiple files
- The project expects a specific data structure: chunked BVH/audio organized by participant with world position Zarr files
- All file paths should use `Path(...).expanduser()` to handle `~` in paths
