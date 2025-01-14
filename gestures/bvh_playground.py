import marimo

__generated_with = "0.10.12"
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
    return Dict, List, Union, bvhio, deepcopy, glm, mo, np, savgol_filter


@app.cell
def _(bvhio):
    root = bvhio.readAsHierarchy("/Users/ojas/projects/data/c-cut.bvh")
    return (root,)


@app.cell
def _(root):
    root.printTree()
    return


@app.cell
def _(root):
    joint = root.filter("LeftHand")[0]
    return (joint,)


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

        output_path = output_path or file_path
        bvhio.writeHierarchy(output_path, root, 1 / 30)
    return (dampen_multiple_joints,)


@app.cell
def _():
    joint_params = {
        "RightHand": {"damping_factor": 0.9, "window_size": 51, "iterations": 8},
        "RightArm": {"damping_factor": 0.9, "window_size": 51, "iterations": 8},
        "RightForeArm": {"damping_factor": 0.9, "window_size": 51, "iterations": 8},
    }
    return (joint_params,)


@app.cell
def _(dampen_multiple_joints, joint_params):
    dampen_multiple_joints(
        "/Users/ojas/projects/data/c-cut.bvh",
        joint_params,
        output_path="/Users/ojas/projects/data/c-cut-dampened.bvh",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
