import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List

import tyro
import yaml

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def run_dampening_experiments(
    levels: List[int],
    base_config: Path = Path("configs/motion_damped.yaml"),
    joint_params_path: Path = Path("configs/joint_params.yaml"),
    n_jobs: int = -1,
    keep_configs: bool = False,
) -> None:
    """Run motion dampening analysis for multiple levels.

    This helper will:
    1. Generate dampened BVH files using ``batch_dampen_bvh.sh``.
    2. Extract world positions from those BVH files via ``batch_extract_bvh.sh``.
    3. Create a temporary metrics config with the correct folder names.
    4. Run ``metrics/analyze.py`` for each level.

    Args:
        levels: Level numbers to test, e.g. ``[1, 2]`` for ``d1`` and ``d2`` suffices.
        base_config: Template config containing a ``d1`` suffix that will be
            replaced per level.
        joint_params_path: Base YAML file containing joint dampening parameters
            (``sigma`` values). ``sigma`` is scaled by the level when running
            ``batch_dampen_bvh.sh``.
        n_jobs: Number of parallel jobs for both dampening and analysis.
        keep_configs: Whether to keep the generated config files.
    """

    base_config = Path(base_config)
    joint_params_path = Path(joint_params_path)

    config_template = base_config.read_text()
    match = re.search(r"d[0-9]+", config_template)
    if not match:
        raise ValueError("base config missing dampening placeholder like 'd1'")
    placeholder = match.group(0)

    base_joint_params = yaml.safe_load(joint_params_path.read_text())
    original_joint_params = joint_params_path.read_text()

    for level in levels:
        suffix = f"d{level}"

        # Scale sigma for each joint by the level and temporarily update the
        # joint_params file expected by ``batch_dampen_bvh.sh``.
        level_params = yaml.safe_load(yaml.safe_dump(base_joint_params))
        for params in level_params.get("joint_params", {}).values():
            if "sigma" in params:
                params["sigma"] = params["sigma"] * level

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as j_tmp:
            yaml.safe_dump(level_params, j_tmp)
            tmp_joint_path = Path(j_tmp.name)

        # Replace the expected config file with our temporary version
        joint_params_path.write_text(tmp_joint_path.read_text())

        try:
            subprocess.run(["./batch_dampen_bvh.sh", suffix, str(n_jobs)], check=True)
            subprocess.run(
                ["./batch_extract_bvh.sh", suffix, f"world_pos_{suffix}.zip"],
                check=True,
            )
        finally:
            joint_params_path.write_text(original_joint_params)
            tmp_joint_path.unlink()

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
    tyro.cli(run_dampening_experiments)
