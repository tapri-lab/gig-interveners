from pathlib import Path
from typing import Dict, List, Optional

import bvhio
import glm
import numpy as np
import yaml
import zarr
import zarr.storage
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from tqdm.auto import tqdm, trange
from tyro.extras import subcommand_cli_from_dict
from wasabi import msg
from pymotion.io.bvh import BVH
import pymotion.rotations.quat as quat
from scipy.ndimage import gaussian_filter1d
from pymotion.ops.skeleton import fk


def split_bvh_by_duration(input_bvh_path: Path, output_dir: Path, chunk_duration_sec: float = 30.0):
    """
    Splits a BVH file into multiple chunks based on a time duration (default: 30s).

    Parameters:
    - input_bvh_path (Path): Path to the input BVH file.
    - output_dir (Path): Directory to store the chunked BVH files.
    - chunk_duration_sec (float): Duration (in seconds) of each chunk.
    """
    bvh = BVH()
    bvh.load(input_bvh_path.expanduser())
    fname = input_bvh_path.stem

    # Extract data
    local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()
    frame_time = bvh.data["frame_time"]
    total_frames = local_rotations.shape[0]

    # Frames per chunk
    frames_per_chunk = int(chunk_duration_sec / frame_time)
    num_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk  # ceil division

    # Create output directory if it doesn't exist
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in (pbar := trange(num_chunks)):
        start = i * frames_per_chunk
        end = min((i + 1) * frames_per_chunk, total_frames)

        # Slice data
        chunk_rotations = local_rotations[start:end]
        chunk_positions = local_positions[start:end]

        # Set and save
        bvh.set_data(chunk_rotations, chunk_positions)
        chunk_filename = output_dir / f"{fname}_chunk_{i:03d}.bvh"
        bvh.save(chunk_filename)
        pbar.set_description(f"Saved chunk {i + 1}/{num_chunks}: {chunk_filename}")


def dampen_multiple_joints(file_path: Path, joint_params: Dict, output_dir: Optional[Path] = None):
    """
    Dampen the motion of multiple joints in a BVH file using Gaussian smoothing on rotation data.

    Args:
        file_path: Path to the input BVH file.
        joint_params: Dictionary containing joint names as keys and their parameters as values.
            Each value should be a dictionary with the following keys:
            - sigma: Standard deviation for Gaussian filter (controls smoothing amount)
        output_dir: Path to save the output BVH file.
    """

    if output_dir is None:
        output_dir = file_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{file_path.stem}.bvh"

    bvh = BVH()
    bvh.load(file_path.expanduser())

    # Extract data
    local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()

    msg.info(f"File: {file_path}")
    print(f"Analyzing {local_rotations.shape[0]} frames")

    for joint_name, params in joint_params.items():
        try:
            # Find index of the joint
            joint_index = bvh.data["names"].tolist().index(joint_name)
        except ValueError:
            msg.warn(f"Joint '{joint_name}' not found in BVH file, skipping")
            continue

        sigma = params.get("sigma", 3.0)
        print(f"\nProcessing {joint_name} with sigma={sigma}")

        # Convert quaternion to scaled axis angle for smoothing
        axis_angles = quat.to_scaled_angle_axis(local_rotations[:, joint_index])

        # Show sample of original values
        print("Sample of original axis-angle rotations:")
        for i in range(min(5, axis_angles.shape[0])):
            print(f"Frame {i}: {axis_angles[i]}")

        # Apply Gaussian smoothing to each dimension of the axis-angle representation
        smoothed = np.stack([gaussian_filter1d(axis_angles[:, i], sigma=sigma) for i in range(3)], axis=-1)

        # Show sample of smoothed values
        print("\nSample of smoothed axis-angle rotations:")
        for i in range(min(5, smoothed.shape[0])):
            print(f"Frame {i}: {smoothed[i]}")

        # Convert back to quaternion
        smoothed_quat = quat.from_scaled_angle_axis(smoothed)

        # Replace original joint's rotations
        local_rotations[:, joint_index] = smoothed_quat

    # Set modified data back and save
    bvh.set_data(local_rotations, local_positions)
    bvh.save(output_path)

    msg.good(f"Saved smoothed animation to: {output_path}")
    return output_path


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


def dampener(input_path: Path, config_path: Path, output_path: Path, n_jobs: int = -1):
    """
    Dampen the motion of multiple joints in a BVH file using Savitzky-Golay filter.
    Args:
        input_path: Path to the input BVH file or folder with BVH files.
        config_path: Path to the YAML configuration file.
        output_path: Path to save the output BVH file.
        n_jobs: Number of parallel jobs to run. Default is -1, which uses all available cores.
    """
    cfg = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    msg.info(f"Loaded config from {config_path}")
    msg.divider("Config")
    print(cfg)
    msg.info("Creating output directory")
    output_path.mkdir(parents=True, exist_ok=True)
    if input_path.is_dir():
        files = input_path.rglob("*.bvh")
    else:
        files = [input_path]

    with Parallel(n_jobs=n_jobs) as pll_exec:
        _ = pll_exec(
            delayed(dampen_multiple_joints)(file_path, cfg["joint_params"], output_path) for file_path in files
        )


if __name__ == "__main__":
    subcommand_cli_from_dict(
        dict(
            dampen=dampener,
            extract=extract_world_positions,
            split=split_bvh_by_duration,
        )
    )
