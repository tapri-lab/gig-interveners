# Interveners

This repo holds the code for creating intervened sequences from data.

# Setup Instructions

**You will need to install `uv` first, if you do not already have it**
Instructions to install `uv` can be found here: [uv installation instructions.](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

If you have `brew` installed on macOS, you can do: `brew install uv`

`uv` will automatically install dependencies when running commands for the below tools.

# Tools

There are currently three tools, one for limiting vocal pitch range, another for dampening keypoints in the bvh directly and finally a tool to extract world positions of joints from a bvh and storing it into `.npz` files.

To see all options available for a given script/command use the `-h` help flag.

## Vocal range limiter

```sh
uv run python pitch_shifter.py [-h] --audio-path PATH --output-path PATH [--max-deviation-hz FLOAT]
```

## Motion Dampener

```sh
uv run python dampener.py dampen [-h] --input-path PATH --config-path PATH [--output-path {None}|PATH]
```

## World Position Extraction
```sh
uv run python dampener.py extract [-h] --file-path PATH --joint-names [STR [STR ...]] [--output-path {None}|PATH]
```