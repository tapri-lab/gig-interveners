from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pymotion.rotations.quat as quat
import yaml
import zarr
import zarr.storage
from joblib import Parallel, delayed
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm, trange
from tyro.extras import subcommand_cli_from_dict
from wasabi import msg


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


def extract_world_pos(folder: Path, joint_names: List[str], output_path: Optional[Path] = None):
    """
    Extract the world positions of multiple joints in a BVH file.
    Args:
        folder: Path to the input BVH file.
        joint_names: List of joint names to extract world positions.
        output_path: Path to save the output BVH file.
    Returns: None
    """
    files = folder.rglob("*.bvh")
    output_path = Path("world_positions.zip") if output_path is None else output_path
    parent_dir = output_path.parent
    parent_dir.mkdir(exist_ok=True, parents=True)
    store = zarr.storage.ZipStore(output_path, mode="w")
    zarr_root = zarr.create_group(store=store)
    msg.info(f"Extracting world positions for {joint_names} joints")

    for file_path in (pbar := tqdm(list(files))):
        pbar.set_description(f"Processing {file_path}")
        bvh = BVH()
        bvh.load(file_path.expanduser())
        fname = file_path.stem
        chunk = fname[-3:]
        zarr_chunk = zarr_root.create_group(chunk)
        local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()
        frame_time = bvh.data["frame_time"]
        frame_range = local_rotations.shape[0]
        world_pos, rotmats = fk(local_rotations, local_positions[:, 0, :], offsets, parents)
        for joint_name in joint_names:
            joint_index = bvh.data["names"].tolist().index(joint_name)
            pos = world_pos[:, joint_index, :]
            tmp = zarr_chunk.create_array(name=joint_name, shape=pos.shape, dtype="float32")
            tmp[:] = pos
    store.close()
    msg.info(f"Saving world positions to {output_path}")


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


def dampen_smpl_parameters(
    input_npz_path: Path,
    joint_params: Dict,
    output_dir: Path,
    smooth_trans: bool = False,
):
    """
    Dampen SMPLX pose parameters for specified joints using Gaussian smoothing, similar to BVH dampening.

    Args:
        input_npz_path: Path to input NPZ file with SMPLX parameters.
        joint_params: Dictionary with joint names (e.g., 'right_wrist') and their params (must include 'sigma').
        output_dir: Path to save the smoothed NPZ file.
        smooth_trans: Whether to smooth translation parameters (default: False).
    """
    # SMPLX joint names list (55 joints, pose is 165D)
    SMPLX_JOINT_NAMES = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "jaw",
        "left_eye",
        "right_eye",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
    ]

    output_npz_path = output_dir / f"{input_npz_path.stem}_smoothed.npz"

    data = np.load(input_npz_path.expanduser())
    poses = data["poses"].copy()  # Shape: [num_frames, 165] (55 joints * 3)
    betas = data["betas"].copy()  # Shape: [10] or [300] for SMPLX expression/shape
    trans = data["trans"].copy()

    msg.info(f"Processing SMPLX file: {input_npz_path}")
    print(f"Analyzing {poses.shape[0]} frames")

    for joint_name, params in joint_params.items():
        try:
            joint_idx = SMPLX_JOINT_NAMES.index(joint_name)
        except ValueError:
            msg.warn(f"Joint '{joint_name}' not found in SMPLX, skipping")
            continue

        sigma = params.get("sigma", 3.0)
        print(f"Smoothing {joint_name} (joint {joint_idx}) with sigma={sigma}")

        # Extract 3D axis-angle for this joint
        start_idx = joint_idx * 3
        axis_angles = poses[:, start_idx : start_idx + 3]

        # Apply Gaussian smoothing to each dimension
        smoothed = np.stack([gaussian_filter1d(axis_angles[:, i], sigma=sigma) for i in range(3)], axis=-1)

        # Replace in poses
        poses[:, start_idx : start_idx + 3] = smoothed

    if smooth_trans:
        print("Smoothing translations (global sigma=3.0)")
        smoothed_trans = np.stack([gaussian_filter1d(trans[:, i], sigma=3.0) for i in range(3)], axis=-1)
        trans = smoothed_trans

    # Save smoothed data (copy all keys from source and update modified ones)
    save_dict = dict(data)  # Copy all keys from the original NPZ
    save_dict["poses"] = poses
    save_dict["betas"] = betas
    save_dict["trans"] = trans
    np.savez(output_npz_path, **save_dict)
    msg.good(f"Saved smoothed SMPLX parameters to: {output_npz_path}")


def dampener_smpl(input_path: Path, config_path: Path, output_path: Path, n_jobs: int = -1):
    """
    Dampen the motion of multiple joints in SMPLX NPZ files using Gaussian smoothing.
    Args:
        input_path: Path to the input NPZ file or folder with NPZ files.
        config_path: Path to the YAML configuration file.
        output_path: Path to save the output NPZ files.
        n_jobs: Number of parallel jobs to run. Default is -1, which uses all available cores.
    """
    cfg = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    msg.info(f"Loaded config from {config_path}")
    msg.divider("Config")
    print(cfg)
    msg.info("Creating output directory")
    output_path.mkdir(parents=True, exist_ok=True)
    if input_path.is_dir():
        files = input_path.rglob("*.npz")
    else:
        files = [input_path]

    with Parallel(n_jobs=n_jobs) as pll_exec:
        _ = pll_exec(
            delayed(dampen_smpl_parameters)(file_path, cfg["joint_params"], output_path) for file_path in files
        )


if __name__ == "__main__":
    subcommand_cli_from_dict(
        dict(
            dampen=dampener,
            dampen_smpl=dampener_smpl,
            extract=extract_world_pos,
            split=split_bvh_by_duration,
        )
    )
