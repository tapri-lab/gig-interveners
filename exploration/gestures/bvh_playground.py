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
    return (
        Dict,
        List,
        Optional,
        Path,
        Union,
        bvhio,
        deepcopy,
        glm,
        here,
        mo,
        np,
        plt,
        savgol_filter,
    )


@app.cell
def _(Path, bvhio):
    bvh_path = Path("~/data/dnd/Session_1/c/c_chunks/c_chunk001.bvh").expanduser()
    root = bvhio.readAsHierarchy(bvh_path).loadPose(100)
    return bvh_path, root


@app.cell
def _(root):
    root.printTree()
    return


@app.cell
def _(root):
    joint = root.filter("LeftHand")[0]
    return (joint,)


@app.cell
def _(bvhio, joint):
    bvhio.Euler.fromQuatTo(joint.getKeyframe(100).RotationWorld)
    return


@app.cell
def _(joint):
    frame_range = joint.getKeyframeRange()
    return (frame_range,)


@app.cell
def _(frame_range):
    total_frames = frame_range[1] + 1
    total_frames
    return (total_frames,)


@app.cell
def _(bvhio, glm, np, savgol_filter):
    def dampen_multiple_joints(file_path: str, joint_params: dict, output_path: str = None):
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

        bvhio.writeHierarchy(output_path, root, 1 / 30)
    return (dampen_multiple_joints,)


@app.cell
def _():
    joint_params = {
        "RightHand": {"damping_factor": 0.8, "window_size": 51},
        "RightArm": {"damping_factor": 0.8, "window_size": 51},
        "RightForeArm": {"damping_factor": 0.8, "window_size": 51},
    }
    return (joint_params,)


@app.cell
def _(bvh_path, dampen_multiple_joints, here, joint_params):
    dampen_multiple_joints(
        bvh_path,
        joint_params,
        output_path=here() / "samples" / "delayed.bvh",
    )
    return


@app.cell
def _(Path, bvhio, np, plt):
    def plot_joint_trajectory(bvh_path: Path, joint_name: str, label: str, color: str):
        root = bvhio.readAsHierarchy(bvh_path)
        frame_range = root.getKeyframeRange()[1] + 1
        positions = []

        for frame in range(frame_range):
            root.loadPose(frame, recursive=True)
            joint = root.filter(joint_name)[0]
            pos = joint.PositionWorld
            positions.append([pos.x, pos.y, pos.z])

        positions = np.array(positions)
        plt.plot(positions[:, 0], label=f"{label} - X", linestyle='--', color=color)
        plt.plot(positions[:, 1], label=f"{label} - Y", linestyle='-', color=color)
        plt.plot(positions[:, 2], label=f"{label} - Z", linestyle=':', color=color)
        plt.show()
        return positions
    return (plot_joint_trajectory,)


@app.cell
def _(here, plot_joint_trajectory):
    plot_joint_trajectory(here() / "samples" / "delayed.bvh", "RightHand", "damped", "red")
    return


@app.cell
def _(bvh_path, plot_joint_trajectory):
    plot_joint_trajectory(bvh_path, "RightHand", "original", "blue")
    return


@app.cell
def _(Dict, Optional, Path, bvhio, glm, np, savgol_filter):
    def dampen_joint_recursively(joint, frame_range, joint_params, default_damping=0.5, default_window=15):
        joint_name = joint.Name
        params = joint_params.get(joint_name, {})
        damping_factor = params.get("damping_factor", default_damping)
        window_size = params.get("window_size", default_window)

        positions, rotations = [], []

        for frame in range(frame_range):
            joint.loadPose(frame)
            key = joint.getKeyframe(frame)
            positions.append([key.Position.x, key.Position.y, key.Position.z])
            rotations.append([key.Rotation.x, key.Rotation.y, key.Rotation.z])  # Euler XYZ

        positions = np.array(positions)
        rotations = np.array(rotations)

        # Smooth positions
        smoothed_pos = np.array([
            savgol_filter(positions[:, i], window_size, 3) for i in range(3)
        ]).T

        # Smooth rotations
        smoothed_rot = np.array([
            savgol_filter(rotations[:, i], window_size, 3) for i in range(3)
        ]).T

        for frame in range(frame_range):
            # Blend position
            original_pos = glm.vec3(positions[frame])
            smooth_pos = glm.vec3(smoothed_pos[frame])
            final_pos = glm.mix(original_pos, smooth_pos, damping_factor)

            # Blend rotation (Euler angles in degrees)
            original_rot = glm.vec3(rotations[frame])
            smooth_rot = glm.vec3(smoothed_rot[frame])
            final_rot = glm.mix(original_rot, smooth_rot, damping_factor)

            transform = bvhio.Transform()
            transform.Position = final_pos
            transform.Rotation = final_rot  # Assuming Euler XYZ
            transform.Scale = joint.getKeyframe(frame).Scale

            joint.setKeyframe(frame, transform)

        # Recurse
        for child in joint.Children:
            dampen_joint_recursively(child, frame_range, joint_params)


    def dampen_all_joints_recursively(file_path: Path, joint_params: Dict, output_dir: Optional[Path] = None):
        """
        Dampen the motion of all joints in a BVH file recursively, respecting hierarchy.
        """
        root = bvhio.readAsHierarchy(file_path)
        frame_range = root.getKeyframeRange()[1] + 1
        print(f"Analyzing {frame_range} frames in {file_path.name}")

        # Start recursive smoothing from the root
        dampen_joint_recursively(root, frame_range, joint_params)

        # Write back
        output_file = output_dir / f"{file_path.stem}_recursive.bvh"
        bvhio.writeHierarchy(output_file, root, 1 / 30)
        print(f"Written smoothed file to {output_file}")
    return dampen_all_joints_recursively, dampen_joint_recursively


@app.cell
def _(bvh_path, dampen_all_joints_recursively, here, joint_params):
    dampen_all_joints_recursively(bvh_path, joint_params, output_dir=here() / "samples")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
