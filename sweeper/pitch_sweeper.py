import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List

import tyro

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def run_pitch_variance_experiments(
    deviations: List[float],
    base_config: Path = Path("configs/pitch_shifted.yaml"),
    n_jobs: int = -1,
    keep_configs: bool = False,
) -> None:
    """Run pitch variance analysis for multiple deviations.

    This helper will:
    1. Generate pitch-limited audio using ``batch_limit_f0.sh``.
    2. Create a temporary config with the correct folder names.
    3. Run ``metrics/analyze.py`` for each deviation.

    Args:
        deviations: Maximum deviation values in Hz.
        base_config: Template config containing a ``shift_30hz`` suffix that will be
            replaced per deviation.
        n_jobs: Number of parallel jobs for ``metrics/analyze.py``.
        keep_configs: Whether to keep the generated config files.
    """

    base_config = Path(base_config)
    config_template = base_config.read_text()
    match = re.search(r"shift_[0-9]+hz", config_template)
    if not match:
        raise ValueError("base config missing pitch placeholder like 'shift_30hz'")
    placeholder = match.group(0)

    for deviation in deviations:
        suffix = f"shift_{deviation}hz"
        subprocess.run(["./batch_limit_f0.sh", str(deviation), suffix], check=True)

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
    tyro.cli(run_pitch_variance_experiments)
