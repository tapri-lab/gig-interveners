import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import tyro
from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils import path
from aitviewer.utils.so3 import aa2rot_torch
from aitviewer.viewer import Viewer
from kintree_constants import BODY_HAND_KINTREE
from pyprojroot import here
from scipy.ndimage import gaussian_filter1d
import itertools

C.smplx_models = here() / "smplx"
C.window_type = "pyglet"
C.auto_set_floor = False


def collect_bvh_seqs(bvh_path: Path) -> Dict[str, SMPLSequence]:
    bvh_seqs = {}
    for root_str, _, files in list(os.walk(bvh_path.expanduser())):
        root = Path(root_str)
        for filename in files:
            if filename.endswith(".bvh"):
                input_path = root / filename
                bvh_seqs[root.stem] = Skeletons.from_bvh(input_path)
    return bvh_seqs


def collect_smpl_sequences(
    smpl_path: Path, frame_limit: int = 1000
) -> Tuple[Dict[str, SMPLSequence], List[np.ndarray]]:
    """
    Collect SMPL sequences from the specified path.
    :param smpl_path: Path to the directory containing SMPL sequences in .npz format.
    :param frame_limit: Maximum number of frames to load from each sequence.
    :return: Dictionary of SMPLSequence objects keyed by their directory names.
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
    root_trans = []
    # collect all SMPL sequences
    for root_str, _, files in list(os.walk(smpl_path.expanduser())):
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
                    color=(22 / 255, 125 / 255, 127 / 255, 1.0),
                )
                root_trans.append(smpl_seqs[root.stem].trans[500].cpu().numpy())
    return smpl_seqs, root_trans


def collect_kp_seqs(kp_path: Path) -> Dict[str, np.ndarray]:
    points = {}
    for root_str, _, files in list(os.walk(kp_path.expanduser())):
        root = Path(root_str)
        print(root)
        points[root.stem] = []
        for filename in sorted(files):
            if filename.endswith(".json"):
                input_path = root / filename
                with open(input_path) as f:
                    data = json.load(f)
                    assert len(data) == 1  # Only one person
                    points[root.stem].append(data[0]["keypoints3d"])
        points[root.stem] = np.array(points[root.stem])
    return points


def camera_positions_from_smpl(smpl_seq: SMPLSequence, sigma: float = 10.0) -> (np.ndarray, np.ndarray):
    """
    Calculate camera positions and targets based on SMPL sequence root positions and orientations.
    :param smpl_seq: SMPLSequence object containing the SMPL data.
    :param sigma: Smoothing factor for camera positions and targets.
    :return: Tuple of camera positions and targets as numpy arrays.
    """
    root_positions = smpl_seq.trans
    root_orientations_aa = smpl_seq.poses_root
    root_orientations_rot = aa2rot_torch(root_orientations_aa.float())

    # Define the standard forward vector (-Z axis).
    forward_vec = torch.tensor([0.0, 0.0, -1.0]).float().to(C.device)

    # Rotate the forward vector by the root orientation for each frame.
    forward_directions = torch.einsum("fab,b->fa", root_orientations_rot, forward_vec)

    # Define the camera's distance and height relative to the model.
    camera_distance = 3.0  # meters
    camera_height = 0.5  # meters

    # Calculate the camera position for each frame.
    # We move the camera "behind" the model along the forward vector.
    cam_positions = root_positions.cpu() - camera_distance * forward_directions.cpu().numpy()
    cam_positions[:, 1] += camera_height  # Adjust camera height
    cam_positions = cam_positions.cpu().numpy()

    # The camera should always look at the model's root.
    cam_targets = root_positions
    cam_targets = cam_targets.cpu().numpy()

    if sigma > 0:
        cam_positions = gaussian_filter1d(cam_positions, sigma=sigma, axis=0)
        cam_targets = gaussian_filter1d(cam_targets, sigma=sigma, axis=0)
    return cam_positions, cam_targets


def render_smpl_sequences(
    smpl_path: Path,
    bvh_path: Path,
    frame_limit: int = 1000,
    sigma: float = 10.0,
    global_scene: bool = False,
    skeleton: bool = False,
):
    """
    Render SMPL sequences in headless mode using AITViewer.
    :param skeleton: If True, renders skeletons from BVH files alongside SMPL sequences.
    :param bvh_path: Path to the directory containing BVH files for skeletons.
    :param smpl_path: Path to the directory containing SMPL sequences in .npz format.
    :param frame_limit: Maximum number of frames to load from each sequence.
    :param sigma: Smoothing factor for camera positions and targets. If > 0, applies Gaussian smoothing.
    :param global_scene: If True, renders the full scene with all SMPL sequences in one video.
    :return:
    """
    smpl_seqs, root_trans = collect_smpl_sequences(smpl_path, frame_limit=frame_limit)
    v = HeadlessRenderer()
    v.scene.origin.enabled = False
    bvh_seqs = collect_bvh_seqs(bvh_path=bvh_path)

    if not global_scene:
        for body, smpl_seq in smpl_seqs.items():
            v.scene.add(smpl_seq)
            v.scene.fps = 30
            v.playback_fps = 30
            cam_positions, cam_targets = camera_positions_from_smpl(smpl_seq, sigma=sigma)
            cam = PinholeCamera(
                position=cam_positions,
                target=cam_targets,
                cols=1280,
                rows=720,
                fov=60.0,
            )
            v.scene.add(cam)
            v.set_temp_camera(cam)
            if skeleton:
                v.scene.add(bvh_seqs[body])
                v.scene.get_node_by_name(smpl_seq.name).enabled = False
            v.save_video(
                video_dir=os.path.join(
                    here(), "export", f"headless/individual/{'skeleton' if skeleton else 'smplx'}/{body}.mp4"
                ),
                output_fps=30,
            )
            v.reset()
    elif global_scene:
        for body, smpl_seq in smpl_seqs.items():
            v.scene.add(smpl_seq)
            smpl_seq.color = (22 / 255, 125 / 255, 127 / 255, 1.0)
            v.scene.fps = 30
            v.playback_fps = 30
            if skeleton:
                bvh_seqs[body].color = (22 / 255, 125 / 255, 127 / 255, 1.0)
                v.scene.add(bvh_seqs[body])
                v.scene.get_node_by_name(smpl_seq.name).enabled = False
                smpl_seqs[body] = bvh_seqs[body]
        center = np.mean(root_trans, axis=0)
        center[1] += 0.5  # Raise the camera a bit
        r = 5
        d = 8
        gcam_pos = [
            path.circle(center=center, radius=r, num=int(314 * 2 * r / d), start_angle=360, end_angle=i * 90)[-1]
            for i in range(1, 5)
        ]
        global_cams = [
            PinholeCamera(
                pos,
                center,
                v.window_size[0],
                v.window_size[1],
                viewer=v,
                fov=60.0,
            )
            for pos in gcam_pos
        ]

        for (idx, cam), (body, smpl_seq) in itertools.product(enumerate(global_cams), smpl_seqs.items()):
            smpl_seq.color = (136 / 255, 123 / 255, 176 / 255, 1.0)
            v.scene.add(cam)
            v.set_temp_camera(cam)
            v.save_video(
                video_dir=os.path.join(
                    here(),
                    "export",
                    "headless",
                    "global",
                    "smplx" if not skeleton else "skeleton",
                    f"cam_{idx}_{body}.mp4",
                ),
                output_fps=30,
                animation_range=[0, 5000],
            )
            smpl_seq.color = (22 / 255, 125 / 255, 127 / 255, 1.0)


def view_in_aitviewer(
    smpl_path: Path, kp_path: Path, frame_limit: int = 1000, sigma: float = 10.0, skeleton: bool = False
):
    """
    Load SMPL sequences and keypoint data into AITViewer for visualization.
    :param smpl_path: Path to the directory containing SMPL sequences in .npz format.
    :param kp_path: Path to the directory containing keypoint data in .json format.
    :param frame_limit: Maximum number of frames to load from each sequence. (only for SMPL sequences)
    :param sigma: Smoothing factor for cameras. If > 0, applies Gaussian smoothing to camera positions and targets.
    :param headless: Run in headless mode (no GUI) - only for rendering.
    :return:
    """

    # smpl_seqs, _ = collect_smpl_sequences(smpl_path, frame_limit=frame_limit)
    v = Viewer()
    #
    # points = collect_kp_seqs(kp_path)
    # root_trans = []
    #
    # for body, kp_seq in points.items():
    #     v.scene.add(PointClouds(points=kp_seq[:, :, :3]))
    #     skeleton = add_body25_skeleton(kp_seq, icon=f"body25_{body}")
    #     v.scene.add(skeleton)
    #
    # for body, smpl_seq in smpl_seqs.items():
    #     v.scene.add(smpl_seq)
    #     # smpl_seq.mesh_seq.enabled = False
    #     root_trans.append(smpl_seq.trans[500].cpu().numpy())
    #     v.playback_fps = 30
    #     v.scene.fps = 30
    #     cam_positions, cam_targets = camera_positions_from_smpl(smpl_seq, sigma=sigma)
    #
    #     if sigma > 0:
    #         cam_positions = gaussian_filter1d(cam_positions, sigma=sigma, axis=0)
    #         cam_targets = gaussian_filter1d(cam_targets, sigma=sigma, axis=0)
    #
    #     cam = PinholeCamera(
    #         position=cam_positions,
    #         target=cam_targets,
    #         cols=1280,
    #         rows=720,
    #         fov=60.0,
    #     )
    #     v.scene.add(cam)
    #     # v.set_temp_camera(cam)
    # center = np.mean(root_trans, axis=0)
    # center[0] += 0  # Move the camera back a bit
    # center[1] += 0.5  # Raise the camera a bit
    # center[2] += 0
    # r = 5
    # d = 10
    #
    # gcam_pos = [
    #     path.circle(center=center, radius=r, num=int(314 * 2 * r / d), start_angle=360, end_angle=i * 90)[-1]
    #     for i in range(1, 5)
    # ]
    # global_cams = [
    #     PinholeCamera(
    #         pos,
    #         center,
    #         v.window_size[0],
    #         v.window_size[1],
    #         viewer=v,
    #         fov=60.0,
    #     )
    #     for pos in gcam_pos
    # ]

    # v.scene.add(*global_cams)
    # v.set_temp_camera(global_cams[3])
    s = Skeletons.from_bvh("/home/ojas/data/dnd/Session_1/scaled_bvh/b/b.bvh")
    s.color = (22 / 255, 125 / 255, 127 / 255, 1.0)
    v.scene.add(s)
    s.color = (136 / 255, 123 / 255, 176 / 255, 1.0)
    v.run()


def add_body25_skeleton(
    points: np.ndarray,
    icon="skeleton",
    kintree=BODY_HAND_KINTREE,
    color=(1.0, 0, 1 / 255, 1.0),
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
    tyro.extras.subcommand_cli_from_dict(
        {"view": view_in_aitviewer, "render": render_smpl_sequences},
    )


if __name__ == "__main__":
    main()
