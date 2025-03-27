from pathlib import Path
from typing import Dict, List, Optional

import bvhio
import glm
import numpy as np
import yaml
import zarr
import zarr.storage
from scipy.signal import savgol_filter
from tqdm.auto import tqdm, trange
from tyro.extras import subcommand_cli_from_dict
from wasabi import msg


def dampen_multiple_joints(file_path: Path, joint_params: Dict, output_path: Optional[Path] = None):
    """
    Dampen the motion of multiple joints in a BVH file using Savitzky-Golay filter.
    Args:
        file_path: Path to the input BVH file.
        joint_params: Dictionary containing joint names as keys and their parameters as values.
            Each value should be a dictionary with the following keys:
            - damping_factor: The factor by which the smoothed position is mixed with the original position.
            - window_size: The length of the filter window (must be an odd integer).
        output_path: Path to save the output BVH file. If not provided, the input file will be overwritten.
    """
    root = bvhio.readAsHierarchy(file_path)
    frame_range = root.getKeyframeRange()[1] + 1
    print(f"Analyzing {frame_range} frames")

    for joint_name, params in joint_params.items():
        target_joint = root.filter(joint_name)[0]
        damping_factor = params.get("damping_factor", 0.5)
        window_size = params.get("window_size", 15)

        print(f"\nProcessing {joint_name}")
        print("Sample of original local positions:")

        positions = []
        original_positions = []
        for frame in range(frame_range):
            target_joint.loadPose(frame)
            pos = target_joint.Position
            positions.append([pos.x, pos.y, pos.z])
            if frame < 5:
                print(f"Frame {frame}: {pos}")
                original_positions.append(pos)

        positions = np.array(positions)

        # Apply Savitzky-Golay filter to local positions
        smoothed_x = savgol_filter(positions[:, 0], window_size, 3)
        smoothed_y = savgol_filter(positions[:, 1], window_size, 3)
        smoothed_z = savgol_filter(positions[:, 2], window_size, 3)

        print("\nSample of smoothed local positions:")
        smoothed_positions = []

        for frame in range(frame_range):
            original = glm.vec3(positions[frame])
            smoothed = glm.vec3(smoothed_x[frame], smoothed_y[frame], smoothed_z[frame])
            final_pos = glm.mix(original, smoothed, damping_factor)

            if frame < 5:
                print(f"Frame {frame}: {final_pos}")
                smoothed_positions.append(final_pos)

            new_transform = bvhio.Transform()
            new_transform.Position = final_pos
            new_transform.Rotation = target_joint.getKeyframe(frame).Rotation
            new_transform.Scale = target_joint.getKeyframe(frame).Scale

            target_joint.setKeyframe(frame, new_transform)

        max_diff = max(glm.length(p1 - p2) for p1, p2 in zip(original_positions, smoothed_positions))
        print(f"\nMaximum position difference in first 5 frames: {max_diff}")

    output_path = output_path or file_path
    bvhio.writeHierarchy(output_path, root, 1 / 30)


def extract_world_positions(folder: Path, joint_names: List[str], output_path: Optional[Path] = None):
    """
    Extract the world positions of multiple joints in a BVH file.
    Args:
        file_path: Path to the input BVH file.
        joint_names: List of joint names to extract world positions.
    Returns:
        Dict: Dictionary containing joint names as keys and their world positions as values.
    """
    files = folder.rglob("*.bvh")
    output_path = Path("world_positions.zip") if output_path is None else output_path
    parent_dir = output_path.parent
    parent_dir.mkdir(exist_ok=True, parents=True)
    store = zarr.storage.ZipStore(output_path, mode="w")
    zarr_root = zarr.create_group(store=store)
    persons = {}
    for file_path in files:
        root = bvhio.readAsHierarchy(str(file_path))
        chunk = file_path.stem[-3:]
        person = zarr_root.create_group(chunk)
        frame_range = root.getKeyframeRange()[1] + 1

        # Dict to store positions for each joint
        joint_positions = {}

        msg.info(f"Extracting world positions for {joint_names} joints")

        for joint_name in (pbar := tqdm(joint_names)):
            pbar.set_description(f"Processing {joint_name}")
            try:
                target_joint = root.filter(joint_name)[0]
            except IndexError as e:
                msg.warn(f"Joint {joint_name} not found in the BVH file")
                raise e
            positions = []

            for frame in (pbar2 := trange(frame_range)):
                root.loadPose(frame, recursive=True)
                pos = target_joint.PositionWorld
                positions.append([pos.x, pos.y, pos.z])

                if frame < 5:
                    print(f"Frame {frame}: {pos}")
            positions = np.array(positions)
            p = person.create_array(name=joint_name, shape=positions.shape, dtype="float32")
            p[:] = positions
    store.close()
    msg.info(f"Saving world positions to {output_path}")

    return persons


def main(input_path: Path, config_path: Path, output_path: Optional[Path] = None):
    """
    Dampen the motion of multiple joints in a BVH file using Savitzky-Golay filter.
    Args:
        input_path: Path to the input BVH file.
        config_path: Path to the YAML configuration file.
        output_path: Path to save the output BVH file. If not provided, the input file will be overwritten.
    """
    cfg = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    msg.info(f"Loaded config from {config_path}")
    msg.divider("Config")
    print(cfg)
    dampen_multiple_joints(input_path, cfg["joint_params"], output_path)


if __name__ == "__main__":
    subcommand_cli_from_dict(
        dict(
            dampen=main,
            extract=extract_world_positions,
        )
    )
