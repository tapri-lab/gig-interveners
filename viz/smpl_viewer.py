import json
import os
from pathlib import Path

import numpy as np
import tyro
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

from kintree_constants import BODY_HAND_KINTREE
from pyprojroot import here
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils.path import lock_to_node

C.smplx_models = here() / "smplx"
C.window_type = "pyglet"
C.auto_set_floor = False


def load_in_aitviewer(smpl_path: Path, kp_path: Path, frame_limit: int = 1000):
    """
    Load SMPL sequences and keypoint data into AITViewer for visualization.
    :param smpl_path: Path to the directory containing SMPL sequences in .npz format.
    :param kp_path: Path to the directory containing keypoint data in .json format.
    :param frame_limit: Maximum number of frames to load from each sequence. (only for SMPL sequences)
    :return:
    """
    smplx_layer = SMPLLayer(model_type="smplx", gender="neutral", device=C.device)

    # To load a SMPL layer (without X), use this poses_body_end and change the model_type above to "smpl"
    # Remember that SMPL does not have hands, so those need to be removed from the SMPLSequence below
    # poses_body_end = 24 * 3

    poses_body_end = 22 * 3
    poses_left_hand_start = 25 * 3
    poses_left_hand_end = 40 * 3
    poses_right_hand_start = poses_left_hand_end

    smpl_seqs = {}

    for root_str, _, files in list(os.walk(smpl_path)):
        root = Path(root_str)
        for filename in files:
            if filename.endswith(".npz"):
                input_path = root / filename

                data = np.load(input_path)

                smpl_seqs[root.stem] = SMPLSequence(
                        smpl_layer=smplx_layer,
                        poses_body=data["poses"][:frame_limit, 3:poses_body_end],
                        poses_root=data["poses_root"][:frame_limit, :3],
                        betas=data["betas"][:frame_limit],
                        trans=data["trans"][:frame_limit],
                        poses_left_hand=data["poses"][:frame_limit, poses_left_hand_start:poses_left_hand_end],
                        poses_right_hand=data["poses"][:frame_limit, poses_right_hand_start:],
                    )


    point_clouds = []
    points = []

    for root_str, _, files in list(os.walk(kp_path)):
        root = Path(root_str)
        for filename in sorted(files):
            if filename.endswith(".json"):
                input_path = root / filename
                with open(input_path) as f:
                    data = json.load(f)
                    assert len(data) == 1  # Only one person
                    points.append(data[0]["keypoints3d"])

    points = np.array(points)

    point_clouds.append(PointClouds(points=points[:, :, :3]))

    skeleton = add_body25_skeleton(points, icon="body25")

    # Add to scene and render
    v = Viewer()
    camera_rel_pos = {
        "c": np.array([2, 1, 0]),
        "a": np.array([2, 1, 2]),
    }
    for body, smpl_seq in smpl_seqs.items():
        v.scene.add(smpl_seq)
        positions, targets = lock_to_node(smpl_seq, relative_position=camera_rel_pos[body], smooth_sigma=10.0)
        cam = PinholeCamera(
            position=positions,
            target=targets,
            cols=1280,
            rows=720,
            fov=60.0,
        )
        v.scene.add(cam)
        v.set_temp_camera(cam)
        v0 = smpl_seq.joints[0, 15]
        v1 = smpl_seq.joints[0, 0]
        d = v0 - v1
        print(d / np.linalg.norm(d))

    for pc_seq in point_clouds:
        v.scene.add(pc_seq)

    v.scene.add(skeleton)

    v.run()


def add_body25_skeleton(
    points: np.ndarray, icon="skeleton", kintree=BODY_HAND_KINTREE, color=(1.0, 0, 1 / 255, 1.0)
) -> Skeletons:
    skeleton = Skeletons(
        joint_positions=points[:, :, :3],
        joint_connections=kintree,
        icon=icon,
        color=color,
    )

    # Remove the lines that we don't have data for by making them transparent
    line_colors = np.zeros((len(kintree), 4))
    line_colors[:] = color
    for i, connection in enumerate(kintree):
        if points[0, connection[0], 3] == 0:
            line_colors[i] = [1, 0, 0, 0]

        if points[0, connection[1], 3] == 0:
            line_colors[i] = [1, 0, 0, 0]

    skeleton.lines.line_colors = line_colors
    skeleton.spheres.color = (50 / 255, 50 / 255, 1 / 255, 1.0)
    return skeleton


def main():
    tyro.cli(load_in_aitviewer)


if __name__ == "__main__":
    main()
