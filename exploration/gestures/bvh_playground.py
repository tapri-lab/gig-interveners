import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.signal import savgol_filter
    import bvhio
    import glm
    from copy import deepcopy
    from typing import Dict, Union, List
    from pathlib import Path
    from pyprojroot import here
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Optional
    from scipy.ndimage import gaussian_filter1d
    from pymotion.io.bvh import BVH
    import pymotion.rotations.quat as quat
    from pymotion.ops.skeleton import fk
    from tqdm.auto import tqdm, trange
    return (
        BVH,
        Dict,
        List,
        Optional,
        Path,
        Union,
        bvhio,
        deepcopy,
        fk,
        gaussian_filter1d,
        glm,
        here,
        mo,
        np,
        plt,
        quat,
        savgol_filter,
        tqdm,
        trange,
    )


@app.cell
def _(Path, here):
    bvh_path = Path("~/projects/data/Session_1/c/c_chunks/c_chunk001.bvh").expanduser()
    out_path = here() / "samples" / "delayed.bvh"
    return bvh_path, out_path


@app.cell
def _(BVH, gaussian_filter1d, np, quat):
    def dampen_joint_motion(input_bvh_path, output_bvh_path, joint_name, sigma=3):
        """
        Load a BVH file, dampen the motion of a specific joint, and save the modified animation.

        Parameters:
        - input_bvh_path (str): Path to the input BVH file.
        - output_bvh_path (str): Path to save the modified BVH file.
        - joint_name (str): Name of the joint to dampen.
        - damping_factor (float): Factor to scale the joint rotation. 1.0 = no change, 0.0 = no motion.
        """
        bvh = BVH()
        bvh.load(input_bvh_path)

        # Extract data
        local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()

        # Find index of the joint
        joint_index = bvh.data["names"].tolist().index(joint_name)

        # Convert quaternion to scaled axis, scale, then back to quaternion
        axis_angles = quat.to_scaled_angle_axis(local_rotations[:, joint_index])
        # scaled_axis *= damping_factor
        # dampened_quats = quat.from_scaled_angle_axis(scaled_axis)
        smoothed = np.stack([gaussian_filter1d(axis_angles[:, i], sigma=sigma) for i in range(3)], axis=-1)

        # Convert back to quaternion
        smoothed_quat = quat.from_scaled_angle_axis(smoothed)

        # Replace original joint's rotations
        local_rotations[:, joint_index] = smoothed_quat

        # Set modified data back and save
        bvh.set_data(local_rotations, local_positions)
        bvh.save(output_bvh_path)

        print(f"Dampened joint '{joint_name}' by factor {sigma}. Saved to: {output_bvh_path}")
    return (dampen_joint_motion,)


@app.cell
def _(bvh_path, dampen_joint_motion, out_path):
    dampen_joint_motion(bvh_path, out_path, "RightArm", 20)
    dampen_joint_motion(out_path, out_path, "RightForeArm", 20)
    return


@app.cell
def _(bvh_path, out_path, plot_joint_positions_comparison):
    plot_joint_positions_comparison(bvh_path, out_path, "RightHand")
    return


@app.cell
def _(bvh_path, out_path, plot_joint_rotation_comparison):
    plot_joint_rotation_comparison(bvh_path, out_path, "LeftHand")
    return


@app.cell(hide_code=True)
def _(BVH, fk, np, plt, quat):
    def plot_joint_positions_comparison(bvh_path_1, bvh_path_2, joint_name):
        bvh1 = BVH()
        bvh1.load(bvh_path_1)

        bvh2 = BVH()
        bvh2.load(bvh_path_2)

        joint_index = bvh1.data["names"].tolist().index(joint_name)

        # FK to get global positions
        lr1, lp1, parents1, offsets1, *_ = bvh1.get_data()
        pos1, _ = fk(lr1, lp1[:, 0, :], offsets1, parents1)

        lr2, lp2, parents2, offsets2, *_ = bvh2.get_data()
        pos2, _ = fk(lr2, lp2[:, 0, :], offsets2, parents2)

        # Extract XYZ over time
        xyz1 = pos1[:, joint_index, :]
        xyz2 = pos2[:, joint_index, :]

        # Plot
        time = np.arange(len(xyz1))
        plt.figure(figsize=(12, 4))
        for i, axis in enumerate("XYZ"):
            plt.subplot(1, 3, i + 1)
            plt.plot(time, xyz1[:, i], label="Original")
            plt.plot(time, xyz2[:, i], label="Dampened")
            plt.title(f"{joint_name} {axis}-Position")
            plt.xlabel("Frame")
            plt.ylabel("Position")
            plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_joint_rotation_comparison(bvh_path_1, bvh_path_2, joint_name):
        bvh1 = BVH()
        bvh1.load(bvh_path_1)

        bvh2 = BVH()
        bvh2.load(bvh_path_2)

        joint_index = bvh1.data["names"].tolist().index(joint_name)

        rot1, *_ = bvh1.get_data()
        rot2, *_ = bvh2.get_data()

        # Convert to scaled angle axis and extract the angle (norm of the axis vector)
        angles1 = np.linalg.norm(quat.to_scaled_angle_axis(rot1[:, joint_index]), axis=1)
        angles2 = np.linalg.norm(quat.to_scaled_angle_axis(rot2[:, joint_index]), axis=1)

        time = np.arange(len(angles1))
        plt.figure(figsize=(6, 4))
        plt.plot(time, np.degrees(angles1), label="Original")
        plt.plot(time, np.degrees(angles2), label="Dampened")
        plt.title(f"{joint_name} Rotation Angle")
        plt.xlabel("Frame")
        plt.ylabel("Degrees")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return plot_joint_positions_comparison, plot_joint_rotation_comparison


@app.cell
def _(BVH, Path, mo):
    def split_bvh_by_duration(input_bvh_path: Path, output_dir: Path, chunk_duration_sec: float = 30.0):
        """
        Splits a BVH file into multiple chunks based on a time duration (default: 30s).

        Parameters:
        - input_bvh_path (Path): Path to the input BVH file.
        - output_dir (Path): Directory to store the chunked BVH files.
        - chunk_duration_sec (float): Duration (in seconds) of each chunk.
        """
        bvh = BVH()
        bvh.load(input_bvh_path)
        fname = input_bvh_path.stem

        # Extract data
        local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()
        frame_time = bvh.data["frame_time"]
        total_frames = local_rotations.shape[0]

        # Frames per chunk
        frames_per_chunk = int(chunk_duration_sec / frame_time)
        num_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk  # ceil division

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in (pbar := mo.status.progress_bar(range(num_chunks), title="Processing Chunks")):
            start = i * frames_per_chunk
            end = min((i + 1) * frames_per_chunk, total_frames)

            # Slice data
            chunk_rotations = local_rotations[start:end]
            chunk_positions = local_positions[start:end]

            # Set and save
            bvh.set_data(chunk_rotations, chunk_positions)
            chunk_filename = output_dir / f"{fname}_chunk_{i:03d}.bvh"
            bvh.save(chunk_filename)
            # pbar.update(title=f"Saved chunk {i + 1}/{num_chunks}: {chunk_filename}")
    return (split_bvh_by_duration,)


@app.cell
def _(Path, here, split_bvh_by_duration):
    split_bvh_by_duration(
        Path("~/projects/data/Session_1/c/c.bvh").expanduser(), here() / "samples" / "chunks", chunk_duration_sec=30.0
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
