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

from kintree_constants import BODY_HAND_KINTREE, DND_BODY_HAND_KINTREE

C.smplx_models = "/home/ojas/projects/generative-interactions/data/body_models/smplx_models/"
C.window_type = "pyglet"
C.auto_set_floor = False


def load_in_aitviewer(
    smpl_folder: Path, kp_folder: Path, body_model: str = "smplx", gender="neutral", frame_limit=1000
):  # pc_folder: Path, smpl_folder: Path):
    smplx_layer = SMPLLayer(model_type=body_model, gender=gender, device=C.device)

    # This is the number of joints for SMPL
    # poses_body_end = 24 * 3

    poses_body_end = 22 * 3
    poses_left_hand_start = 25 * 3
    poses_left_hand_end = 40 * 3
    poses_right_hand_start = poses_left_hand_end

    smpl_seqs = []

    for root_str, _, files in list(os.walk(smpl_folder)):
        root = Path(root_str)
        for filename in files:
            if filename.endswith(".npz"):
                input_path = root / filename

                data = np.load(input_path)

                smpl_seqs.append(
                    SMPLSequence(
                        smpl_layer=smplx_layer,
                        poses_body=data["poses"][:frame_limit, 3:poses_body_end],
                        poses_root=data["poses_root"][:frame_limit, :3],
                        betas=data["betas"][:frame_limit],
                        trans=data["trans"][:frame_limit],
                        poses_left_hand=data["poses"][:frame_limit, poses_left_hand_start:poses_left_hand_end]
                        if body_model == "smplx"
                        else None,
                        poses_right_hand=data["poses"][:frame_limit, poses_right_hand_start:]
                        if body_model == "smplx"
                        else None,
                    )
                )

    point_clouds = []
    points = []

    for root_str, _, files in list(os.walk(kp_folder)):
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

    data_dnd_raw = np.genfromtxt(
        kp_folder / "a.csv",
        delimiter=",",
        max_rows=2,
    )
    data_dnd_raw = data_dnd_raw[:, 1:].reshape(-1, 67, 3)
    data_dnd_raw = np.array(data_dnd_raw) / 1000  # mm to m
    data_dnd = np.ones(data_dnd_raw.shape[0:1] + (63, 4))
    data_dnd[:, 0, :3] = data_dnd_raw[:, 3]
    data_dnd[:, 1:4, :3] = data_dnd_raw[:, :3]
    data_dnd[:, 4:23, :3] = data_dnd_raw[:, 4:23]
    data_dnd[:, 23:43, :3] = data_dnd_raw[:, 24:44, :3]
    data_dnd[:, 43:63, :3] = data_dnd_raw[:, 46:66, :3]
    skeleton_dnd = add_body25_skeleton(
        data_dnd, icon="DnD", kintree=DND_BODY_HAND_KINTREE, color=(10 / 255, 1, 10 / 255, 1)
    )

    # Add to scene and render
    v = Viewer()

    for smpl_seq in smpl_seqs:
        v.scene.add(smpl_seq)

    for pc_seq in point_clouds:
        v.scene.add(pc_seq)

    v.scene.add(skeleton)
    v.scene.add(skeleton_dnd)

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


if __name__ == "__main__":
    tyro.cli(load_in_aitviewer)
