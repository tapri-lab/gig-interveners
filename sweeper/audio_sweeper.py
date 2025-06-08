import re
import subprocess
import tempfile
from pathlib import Path
from typing import List

import tyro
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def run_delay_experiments(
    offsets: List[float],
    segment_length: float = 30.0,
    base_config: Path = Path("configs/audio_delay.yaml"),
    n_jobs: int = -1,
    keep_configs: bool = False,
) -> None:
    """Run delay analysis for multiple offsets.

    This helper will:
    1. Generate delayed audio using ``batch_split_audio.sh``.
    2. Create a temporary config with the correct audio paths.
    3. Run ``metrics/analyze.py`` for each offset.

    Args:
        offsets: List of offsets to test (in seconds).
        segment_length: Segment length to pass to ``batch_split_audio.sh``.
        base_config: Template config to modify. Should contain ``delay_m0.5`` as
            the audio folder suffix which will be replaced per offset.
        n_jobs: Number of parallel jobs for ``metrics/analyze.py``.
        keep_configs: Whether to keep the generated config files.
    """

    base_config = Path(base_config)
    config_template = base_config.read_text()
    match = re.search(r"delay_m[0-9.]+", config_template)
    if not match:
        raise ValueError("base config missing delay placeholder like 'delay_m0.5'")
    placeholder = match.group(0)

    for offset in offsets:
        suffix = f"delay_m{offset}"
        subprocess.run(
            ["./batch_split_audio.sh", str(segment_length), str(offset), suffix],
            check=True,
        )

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write(config_template.replace(placeholder, suffix))
            cfg_path = Path(tmp.name)

        output_dir = Path("results") / suffix
        subprocess.run(
            [
                "uv",
                "run",
                "metrics/analyze.py",
                "--cfg-path",
                str(cfg_path),
                "--n-jobs",
                str(n_jobs),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
        )

        if not keep_configs:
            cfg_path.unlink()


if __name__ == "__main__":
    tyro.cli(run_delay_experiments)
