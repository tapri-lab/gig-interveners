from pymotion.io.bvh import BVH
from pathlib import Path
import os
import tyro


def main(raw_bvh: Path, output_dir: Path):
    """
    Process a BVH file and save it to the specified output directory.

    Args:
        raw_bvh (Path): Path to the input BVH file.
        output_dir (Path): Directory where the processed BVH file will be saved.
    """

    bvh = BVH()

    for root_str, _, files in os.walk(raw_bvh.expanduser()):
        root = Path(root_str)
        for file in files:
            if file.endswith(".bvh"):
                raw_bvh = root / file
                bvh.load(raw_bvh)
                bvh.set_scale(0.1)
                # ensure the output directory exists
                (output_dir / root.stem).mkdir(parents=True, exist_ok=True)
                bvh.save(output_dir / root.stem / file)


if __name__ == "__main__":
    tyro.cli(main)
