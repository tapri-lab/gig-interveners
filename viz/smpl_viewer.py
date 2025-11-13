import itertools
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
    smpl_path: Path, frame_limit: Tuple[int, int] = (0, 1000)
) -> Tuple[Dict[str, SMPLSequence], List[np.ndarray]]:
    """
    Collect SMPL sequences from the specified path.
    :param smpl_path: Path to the directory containing SMPL sequences in .npz format.
    :param frame_limit: Maximum number of frames to load from each sequence.
    :return: Dictionary of SMPLSequence objects keyed by their directory names.
    """
    smplx_layer = SMPLLayer(model_type="smplx", gender="neutral", device=C.device)
    frame_start, frame_end = frame_limit

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
                    poses_body=data["poses"][frame_start:frame_end, 3:poses_body_end],
                    poses_root=data["poses_root"][frame_start:frame_end, :3],
                    betas=data["betas"],
                    trans=data["trans"][frame_start:frame_end],
                    poses_left_hand=data["poses"][frame_start:frame_end, poses_left_hand_start:poses_left_hand_end],
                    poses_right_hand=data["poses"][frame_start:frame_end, poses_right_hand_start:],
                    color=(22 / 255, 125 / 255, 127 / 255, 1.0),
                )
                # Use middle frame or frame 300 if available, otherwise use last frame
                num_frames = smpl_seqs[root.stem].trans.shape[0]
                frame_idx = min(300, num_frames - 1) if num_frames > 0 else 0
                root_trans.append(smpl_seqs[root.stem].trans[frame_idx].cpu().numpy())
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
    camera_distance = 2.8  # meters
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
    bvh_path: Path | None = None,
    sigma: float = 10.0,
    global_scene: bool = False,
    global_unified: bool = False,
    skeleton: bool = False,
    frame_range: List[int] = [0, 5000],
):
    """
    Render SMPL sequences in headless mode using AITViewer.
    :param skeleton: If True, renders skeletons from BVH files alongside SMPL sequences.
    :param bvh_path: Path to the directory containing BVH files for skeletons.
    :param smpl_path: Path to the directory containing SMPL sequences in .npz format.
    :param sigma: Smoothing factor for camera positions and targets. If > 0, applies Gaussian smoothing.
    :param global_scene: If True, renders the full scene with all SMPL sequences in one video per camera-person pair.
    :param global_unified: If True, renders the full scene with all SMPL sequences in one video per camera (all persons together).
    :param frame_range: Range of frames to render in the video.
    :return:
    """
    if not skeleton:
        smpl_seqs, root_trans = collect_smpl_sequences(smpl_path, frame_limit=(frame_range[0], frame_range[1]))
    else:
        smpl_seqs, root_trans = collect_smpl_sequences(smpl_path, frame_limit=(0, 600))
    v = HeadlessRenderer()
    v.scene.origin.enabled = False

    # Only load BVH sequences if skeleton rendering is enabled
    if skeleton:
        if bvh_path is None:
            raise ValueError("bvh_path must be provided when skeleton=True")
        bvh_seqs = collect_bvh_seqs(bvh_path=bvh_path)
    else:
        bvh_seqs = {}

    if not global_scene and not global_unified:
        for body, smpl_seq in smpl_seqs.items():
            smpl_seq.color = (229 / 255, 91 / 255, 19 / 255, 1.0)
            v.scene.add(smpl_seq)
            v.scene.fps = 25
            v.playback_fps = 25
            cam_positions, cam_targets = camera_positions_from_smpl(smpl_seq, sigma=sigma)
            cam = PinholeCamera(
                position=cam_positions[500],
                target=cam_targets[500],
                cols=1280,
                rows=720,
                fov=60.0,
            )
            v.scene.add(cam)
            v.set_temp_camera(cam)
            if skeleton:
                bvh_seqs[body].color = (229 / 255, 91 / 255, 19 / 255, 1.0)
                v.scene.add(bvh_seqs[body])
                v.scene.get_node_by_name(smpl_seq.name).enabled = False
            v.save_video(
                video_dir=os.path.join(
                    here(),
                    "export",
                    "headless",
                    "individual",
                    f"{'skeleton' if skeleton else 'smplx'}",
                    body,
                    f"{body}.mp4",
                ),
                output_fps=25,
                animation_range=frame_range,
            )
            v.reset()
    elif global_unified:
        # Add all sequences to the scene with consistent colors
        for body, smpl_seq in smpl_seqs.items():
            v.scene.add(smpl_seq)
            smpl_seq.color = (22 / 255, 125 / 255, 127 / 255, 1.0)
            v.scene.fps = 25
            v.playback_fps = 25
            if skeleton:
                bvh_seqs[body].color = (22 / 255, 125 / 255, 127 / 255, 1.0)
                v.scene.add(bvh_seqs[body])
                v.scene.get_node_by_name(smpl_seq.name).enabled = False

        # Set up cameras
        center = np.mean(root_trans, axis=0)
        center[1] += 0.1  # Raise the camera a bit
        r = 3.5
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

        # Render once per camera (not per camera-person pair)
        for idx, cam in enumerate(global_cams):
            v.scene.add(cam)
            v.set_temp_camera(cam)
            v.save_video(
                video_dir=os.path.join(
                    here(),
                    "export",
                    "headless",
                    "global_unified",
                    "smplx" if not skeleton else "skeleton",
                    f"cam_{idx}",
                    f"cam_{idx}_all.mp4",
                ),
                output_fps=25,
                animation_range=frame_range if skeleton else None,
            )
    elif global_scene:
        for body, smpl_seq in smpl_seqs.items():
            v.scene.add(smpl_seq)
            smpl_seq.color = (22 / 255, 125 / 255, 127 / 255, 1.0)
            v.scene.fps = 25
            v.playback_fps = 25
            if skeleton:
                bvh_seqs[body].color = (22 / 255, 125 / 255, 127 / 255, 1.0)
                v.scene.add(bvh_seqs[body])
                v.scene.get_node_by_name(smpl_seq.name).enabled = False
                smpl_seqs[body] = bvh_seqs[body]
        center = np.mean(root_trans, axis=0)
        center[1] += 0.1  # Raise the camera a bit
        r = 3.5
        d = 8
        gcam_pos = [
            path.circle(center=center, radius=r, num=int(314 * 2 * r / d), start_angle=360, end_angle=i * 90)[-1]
            for i in range(1, 5)
        ]
        global_cams = [
            PinholeCamera(
                pos,  # Raise the camera a bit,
                center,
                v.window_size[0],
                v.window_size[1],
                viewer=v,
                fov=60.0,
            )
            for pos in gcam_pos
        ]

        for (idx, cam), (body, smpl_seq) in itertools.product(enumerate(global_cams), smpl_seqs.items()):
            smpl_seq.color = (229 / 255, 91 / 255, 19 / 255, 1.0)
            v.scene.add(cam)
            v.set_temp_camera(cam)
            v.save_video(
                video_dir=os.path.join(
                    here(),
                    "export",
                    "headless",
                    "global",
                    "smplx" if not skeleton else "skeleton",
                    body,
                    f"cam_{idx}",
                    f"cam_{idx}_{body}.mp4",
                ),
                output_fps=25,
                animation_range=frame_range if skeleton else None,
            )
            smpl_seq.color = (22 / 255, 125 / 255, 127 / 255, 1.0)


def view_in_aitviewer(
    smpl_path: Path,
    bvh_path: Path | None = None,
    frame_limit: int = 1000,
    sigma: float = 10.0,
    skeleton: bool = False,
):
    """
    Load SMPL sequences and keypoint data into AITViewer for visualization.
    :param smpl_path: Path to the directory containing SMPL sequences in .npz format.
    :param bvh_path: Path to the directory containing keypoint data in .json format.
    :param frame_limit: Maximum number of frames to load from each sequence. (only for SMPL sequences)
    :param sigma: Smoothing factor for cameras. If > 0, applies Gaussian smoothing to camera positions and targets.
    :param skeleton: Don't show smpl bodies.
    :return:
    """

    smpl_seqs, root_trans = collect_smpl_sequences(smpl_path, frame_limit=frame_limit)
    v = Viewer()

    # Only load BVH sequences if skeleton rendering is enabled
    if skeleton:
        if bvh_path is None:
            raise ValueError("bvh_path must be provided when skeleton=True")
        bvh_seqs = collect_bvh_seqs(bvh_path=bvh_path)
    else:
        bvh_seqs = {}

    for body, smpl_seq in smpl_seqs.items():
        if not skeleton:
            v.scene.add(smpl_seq)
        else:
            v.scene.add(bvh_seqs[body])
            # smpl_seq.mesh_seq.enabled = False
            bvh_seqs[body].color = (229 / 255, 91 / 255, 19 / 255, 1.0)
        v.playback_fps = 25
        # v.scene.fps = 30
        cam_positions, cam_targets = camera_positions_from_smpl(smpl_seq, sigma=sigma)

        cam = PinholeCamera(
            position=cam_positions[400],
            target=cam_targets[400],
            cols=1280,
            rows=720,
            fov=60.0,
        )
        v.scene.add(cam)
        v.set_temp_camera(cam)
    center = np.mean(root_trans, axis=0)
    center[1] += 0.1  # Raise the camera a bit
    r = 3.5
    d = 10

    gcam_pos = [
        path.circle(center=center, radius=r, num=int(314 * 2 * r / d), start_angle=360, end_angle=i * 90)[-1]
        for i in range(1, 5)
    ]
    global_cams = [
        PinholeCamera(
            pos + np.array([0, 0.5, 0]),  # Raise the camera a bit
            center,
            v.window_size[0],
            v.window_size[1],
            viewer=v,
            fov=65.0,
        )
        for pos in gcam_pos
    ]

    v.scene.add(*global_cams)
    v.set_temp_camera(global_cams[3])

    v.run()


def _add_sequences_to_scene(
    viewer: HeadlessRenderer,
    smpl_seqs: Dict[str, SMPLSequence],
    bvh_seqs: Dict[str, Skeletons],
    skeleton_mode: bool,
    highlight_body: str | None = None,
) -> None:
    """
    Add SMPL sequences and skeletons to the scene.

    :param viewer: The HeadlessRenderer instance
    :param smpl_seqs: Dictionary of SMPL sequences to add
    :param bvh_seqs: Dictionary of skeleton sequences to add if skeleton_mode=True
    :param skeleton_mode: If True, add skeletons and disable SMPL meshes
    :param highlight_body: Optional body name to highlight in orange, others in teal
    """
    for body, smpl_seq in smpl_seqs.items():
        # Set color based on whether this body is highlighted
        if highlight_body is None or highlight_body == body:
            color = (229 / 255, 91 / 255, 19 / 255, 1.0)  # Orange for highlighted
        else:
            color = (22 / 255, 125 / 255, 127 / 255, 1.0)  # Teal for background

        smpl_seq.color = color
        viewer.scene.add(smpl_seq)

        # In skeleton mode, add skeleton and disable the SMPL mesh
        if skeleton_mode and body in bvh_seqs:
            bvh_seqs[body].color = color
            viewer.scene.add(bvh_seqs[body])
            viewer.scene.get_node_by_name(smpl_seq.name).enabled = False


def _cleanup_chunk_scene(
    viewer: HeadlessRenderer,
    smpl_seqs: Dict[str, SMPLSequence],
    bvh_seqs: Dict[str, Skeletons],
    cam: PinholeCamera,
    skeleton_mode: bool,
) -> None:
    """
    Clean up SMPL sequences, skeletons, and camera from the scene after rendering a chunk.

    :param viewer: The HeadlessRenderer instance
    :param smpl_seqs: Dictionary of SMPL sequences to remove
    :param bvh_seqs: Dictionary of skeleton sequences to remove
    :param cam: Camera to remove
    :param skeleton_mode: If True, also remove skeletons from scene
    """
    import gc

    for smpl_seq in smpl_seqs.values():
        viewer.scene.remove(smpl_seq)

    # Remove skeletons to prevent memory leak
    # if skeleton_mode:
    #     for body in smpl_seqs.keys():
    #         if body in bvh_seqs:
    #             viewer.scene.remove(bvh_seqs[body])

    viewer.scene.remove(cam)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def merge_video_chunks(chunk_paths: List[Path], output_path: Path) -> None:
    """
    Merge multiple video chunks into a single video using ffmpeg.
    :param chunk_paths: List of paths to video chunks in order.
    :param output_path: Path to the output merged video.
    """
    import subprocess
    import tempfile

    # Create a temporary file listing all chunks
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for chunk_path in chunk_paths:
            f.write(f"file '{chunk_path.absolute()}'\n")
        concat_file = Path(f.name)

    try:
        # Use ffmpeg to concatenate videos
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output_path),
                "-y",  # Overwrite output file if it exists
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Merged {len(chunk_paths)} chunks into {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        raise
    finally:
        # Clean up temporary concat file
        # concat_file.unlink()
        pass


def render_smpl_sequences_chunked(
    smpl_path: Path,
    bvh_path: Path | None = None,
    sigma: float = 10.0,
    global_scene: bool = False,
    global_unified: bool = False,
    skeleton: bool = False,
    frame_range: List[int] = [0, 5000],
    chunk_size: int = 1000,
    keep_chunks: bool = False,
):
    """
    Render SMPL sequences in chunks to manage CUDA memory, then merge the results.
    This function loads SMPL sequences in chunks to avoid CUDA OOM errors when creating SMPLSequence objects.

    :param smpl_path: Path to the directory containing SMPL sequences in .npz format.
    :param bvh_path: Path to the directory containing BVH files for skeletons (required if skeleton=True).
    :param sigma: Smoothing factor for camera positions and targets.
    :param global_scene: If True, renders the full scene with all SMPL sequences in one video per camera-person pair.
    :param global_unified: If True, renders the full scene with all SMPL sequences in one video per camera (all persons together).
    :param skeleton: If True, renders skeletons from BVH files alongside SMPL sequences.
    :param frame_range: Range of frames to render [start, end].
    :param chunk_size: Number of frames per chunk (default 1000).
    :param keep_chunks: If True, keeps individual chunk files after merging.
    """
    import gc

    v = HeadlessRenderer()
    start_frame, end_frame = frame_range
    total_frames = end_frame - start_frame

    # Calculate chunks
    chunks = []
    for chunk_start in range(start_frame, end_frame, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_frame)
        chunks.append([chunk_start, chunk_end])

    print(f"Rendering {total_frames} frames in {len(chunks)} chunks of size {chunk_size}")

    # Validate skeleton mode
    if skeleton and bvh_path is None:
        raise ValueError("bvh_path must be provided when skeleton=True")

    # Determine output paths based on rendering mode
    if not global_scene and not global_unified:
        # Get list of bodies directly from filesystem (no CUDA memory usage)
        body_names = []
        for root_str, _, files in os.walk(smpl_path.expanduser()):
            root = Path(root_str)
            for filename in files:
                if filename.endswith(".npz"):
                    body_names.append(root.stem)
                    break  # Only need one file per body
        body_names = list(set(body_names))  # Remove duplicates
        print(f"Found {len(body_names)} bodies: {body_names}")

        # Load BVH sequences once if in skeleton mode (they don't use CUDA memory)
        bvh_seqs = {}
        if skeleton:
            assert bvh_path is not None  # Already validated above
            bvh_seqs = collect_bvh_seqs(bvh_path=bvh_path)

        # Render each body separately
        for body in body_names:
            chunk_paths = []

            for chunk_idx, chunk_range in enumerate(chunks):
                print(f"Rendering {body} chunk {chunk_idx + 1}/{len(chunks)}: frames {chunk_range[0]}-{chunk_range[1]}")

                # Load SMPL sequences for this chunk only (this is where CUDA memory is allocated)
                smpl_seqs, _ = collect_smpl_sequences(smpl_path, frame_limit=(chunk_range[0], chunk_range[1]))

                # Check if this body exists in this chunk
                if body not in smpl_seqs:
                    print(f"Warning: {body} not found in chunk {chunk_idx}, skipping")
                    continue

                smpl_seq = smpl_seqs[body]

                # Set up renderer
                v.scene.origin.enabled = False
                v.scene.fps = 25
                v.playback_fps = 25

                # Add SMPL sequences to scene (with skeleton overlay if needed)
                _add_sequences_to_scene(v, {body: smpl_seq}, bvh_seqs, skeleton_mode=skeleton, highlight_body=body)

                # Calculate camera from SMPL sequence
                cam_positions, cam_targets = camera_positions_from_smpl(smpl_seq, sigma=sigma)

                # Use middle frame for camera position
                cam_frame_idx = len(cam_positions) // 2
                cam = PinholeCamera(
                    position=cam_positions[cam_frame_idx],
                    target=cam_targets[cam_frame_idx],
                    cols=1280,
                    rows=720,
                    fov=60.0,
                )
                v.scene.add(cam)
                v.set_temp_camera(cam)

                # Set up chunk output path
                chunk_output = Path(
                    here(),
                    "export",
                    "headless",
                    "individual",
                    f"{'skeleton' if skeleton else 'smplx'}",
                    body,
                    f"{body}_chunk_{chunk_idx:04d}.mp4",
                )
                chunk_output.parent.mkdir(parents=True, exist_ok=True)

                # Render and save
                v.save_video(
                    video_dir=str(chunk_output),
                    output_fps=25,
                    animation_range=[chunk_range[0], chunk_range[1]],  # Absolute frame range for skeleton
                    ensure_no_overwrite=False,
                )

                chunk_paths.append(chunk_output)

                # Clean up CUDA memory after each chunk
                _cleanup_chunk_scene(v, smpl_seqs, bvh_seqs, cam, skeleton_mode=skeleton)
                del smpl_seqs
                del smpl_seq
                del cam

                print(f"  Completed chunk {chunk_idx + 1}/{len(chunks)}, freed CUDA memory")
            # Merge chunks for this body
            if chunk_paths:
                final_output = Path(
                    here(),
                    "export",
                    "headless",
                    "individual",
                    f"{'skeleton' if skeleton else 'smplx'}",
                    body,
                    f"{body}.mp4",
                )
                merge_video_chunks(chunk_paths, final_output)

                # Clean up chunks if requested
                if not keep_chunks:
                    for chunk_path in chunk_paths:
                        chunk_path.unlink()

    elif global_unified:
        # Global unified: all bodies in one video per camera
        # Get body names from filesystem
        body_names = []
        for root_str, _, files in os.walk(smpl_path.expanduser()):
            root = Path(root_str)
            for filename in files:
                if filename.endswith(".npz"):
                    body_names.append(root.stem)
                    break
        body_names = list(set(body_names))
        print(f"Found {len(body_names)} bodies: {body_names}")

        # Load BVH sequences once if in skeleton mode (they don't use CUDA memory)
        bvh_seqs = {}
        if skeleton:
            assert bvh_path is not None  # Already validated above
            bvh_seqs = collect_bvh_seqs(bvh_path=bvh_path)

        # We need root_trans for camera setup - load first chunk to get it
        temp_seqs, temp_root_trans = collect_smpl_sequences(smpl_path, frame_limit=(start_frame, start_frame + 1))
        center = np.mean(temp_root_trans, axis=0)
        center[1] += 0.1  # Raise the camera a bit
        del temp_seqs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Set up cameras (4 cameras around the scene)
        r = 3.5
        d = 8
        gcam_pos = [
            path.circle(center=center, radius=r, num=int(314 * 2 * r / d), start_angle=360, end_angle=i * 90)[-1]
            for i in range(1, 5)
        ]

        # Render each camera separately
        for cam_idx in range(len(gcam_pos)):
            chunk_paths = []

            for chunk_idx, chunk_range in enumerate(chunks):
                print(
                    f"Rendering camera {cam_idx} chunk {chunk_idx + 1}/{len(chunks)}: frames {chunk_range[0]}-{chunk_range[1]}"
                )

                # Load SMPL sequences for this chunk
                smpl_seqs, _ = collect_smpl_sequences(smpl_path, frame_limit=(chunk_range[0], chunk_range[1]))

                v.scene.origin.enabled = False
                v.scene.fps = 25
                v.playback_fps = 25

                # Add all SMPL sequences to scene (no highlighting in unified mode)
                _add_sequences_to_scene(v, smpl_seqs, bvh_seqs, skeleton_mode=skeleton, highlight_body=None)

                # Create camera for this angle
                pos = gcam_pos[cam_idx]
                cam = PinholeCamera(
                    pos + np.array([0, 0.5, 0]),
                    center,
                    v.window_size[0],
                    v.window_size[1],
                    viewer=v,
                    fov=60,
                )
                v.scene.add(cam)
                v.set_temp_camera(cam)

                # Set up chunk output path
                chunk_output = Path(
                    here(),
                    "export",
                    "headless",
                    "global_unified",
                    "smplx" if not skeleton else "skeleton",
                    f"cam_{cam_idx}",
                    f"cam_{cam_idx}_all_chunk_{chunk_idx:04d}.mp4",
                )
                chunk_output.parent.mkdir(parents=True, exist_ok=True)

                # Render and save
                v.save_video(
                    video_dir=str(chunk_output),
                    output_fps=25,
                    animation_range=[chunk_range[0], chunk_range[1]],  # Absolute frame range for skeletons
                    ensure_no_overwrite=False,
                )

                chunk_paths.append(chunk_output)

                # Clean up CUDA memory after each chunk
                _cleanup_chunk_scene(v, smpl_seqs, bvh_seqs, cam, skeleton_mode=skeleton)
                del smpl_seqs
                del cam

                print(f"  Completed chunk {chunk_idx + 1}/{len(chunks)}, freed CUDA memory")

            # Merge chunks for this camera
            if chunk_paths:
                final_output = Path(
                    here(),
                    "export",
                    "headless",
                    "global_unified",
                    "smplx" if not skeleton else "skeleton",
                    f"cam_{cam_idx}",
                    f"cam_{cam_idx}_all.mp4",
                )
                merge_video_chunks(chunk_paths, final_output)

                # Clean up chunks if requested
                if not keep_chunks:
                    for chunk_path in chunk_paths:
                        chunk_path.unlink()

    elif global_scene:
        # Global scene: all bodies visible, but each body highlighted in separate video per camera
        # Get body names from filesystem
        body_names = []
        for root_str, _, files in os.walk(smpl_path.expanduser()):
            root = Path(root_str)
            for filename in files:
                if filename.endswith(".npz"):
                    body_names.append(root.stem)
                    break
        body_names = list(set(body_names))
        print(f"Found {len(body_names)} bodies: {body_names}")

        # Load BVH sequences once if in skeleton mode (they don't use CUDA memory)
        bvh_seqs = {}
        if skeleton:
            assert bvh_path is not None  # Already validated above
            bvh_seqs = collect_bvh_seqs(bvh_path=bvh_path)

        # We need root_trans for camera setup - load first chunk to get it
        temp_seqs, temp_root_trans = collect_smpl_sequences(smpl_path, frame_limit=(start_frame, start_frame + 1))
        center = np.mean(temp_root_trans, axis=0)
        center[1] += 0.1  # Raise the camera a bit
        del temp_seqs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Set up cameras (4 cameras around the scene)
        r = 3.5
        d = 8
        gcam_pos = [
            path.circle(center=center, radius=r, num=int(314 * 2 * r / d), start_angle=360, end_angle=i * 90)[-1]
            for i in range(1, 5)
        ]

        # Render each camera-body combination
        for cam_idx in range(len(gcam_pos)):
            for body in body_names:
                chunk_paths = []

                for chunk_idx, chunk_range in enumerate(chunks):
                    print(
                        f"Rendering camera {cam_idx} body {body} chunk {chunk_idx + 1}/{len(chunks)}: frames {chunk_range[0]}-{chunk_range[1]}"
                    )

                    # Load SMPL sequences for this chunk
                    smpl_seqs, _ = collect_smpl_sequences(smpl_path, frame_limit=(chunk_range[0], chunk_range[1]))

                    # Check if this body exists in this chunk
                    if body not in smpl_seqs:
                        print(f"Warning: {body} not found in chunk {chunk_idx}, skipping")
                        continue

                    v.scene.origin.enabled = False
                    v.scene.fps = 25
                    v.playback_fps = 25

                    # Add all SMPL sequences with the target body highlighted
                    _add_sequences_to_scene(v, smpl_seqs, bvh_seqs, skeleton_mode=skeleton, highlight_body=body)

                    # Create camera for this angle
                    pos = gcam_pos[cam_idx]
                    cam = PinholeCamera(
                        pos,
                        center,
                        v.window_size[0],
                        v.window_size[1],
                        viewer=v,
                        fov=60,
                    )
                    v.scene.add(cam)
                    v.set_temp_camera(cam)

                    # Set up chunk output path
                    chunk_output = Path(
                        here(),
                        "export",
                        "headless",
                        "global",
                        "smplx" if not skeleton else "skeleton",
                        body,
                        f"cam_{cam_idx}",
                        f"cam_{cam_idx}_{body}_chunk_{chunk_idx:04d}.mp4",
                    )
                    chunk_output.parent.mkdir(parents=True, exist_ok=True)

                    # Render and save
                    v.save_video(
                        video_dir=str(chunk_output),
                        output_fps=25,
                        animation_range=[chunk_range[0], chunk_range[1]],  # Absolute frame range for skeletons
                        ensure_no_overwrite=False,
                    )

                    chunk_paths.append(chunk_output)

                    # Clean up CUDA memory after each chunk
                    _cleanup_chunk_scene(v, smpl_seqs, bvh_seqs, cam, skeleton_mode=skeleton)
                    del smpl_seqs
                    del cam

                    print(f"  Completed chunk {chunk_idx + 1}/{len(chunks)}, freed CUDA memory")

                # Merge chunks for this camera-body combination
                if chunk_paths:
                    final_output = Path(
                        here(),
                        "export",
                        "headless",
                        "global",
                        "smplx" if not skeleton else "skeleton",
                        body,
                        f"cam_{cam_idx}",
                        f"cam_{cam_idx}_{body}.mp4",
                    )
                    merge_video_chunks(chunk_paths, final_output)

                    # Clean up chunks if requested
                    if not keep_chunks:
                        for chunk_path in chunk_paths:
                            chunk_path.unlink()

    else:
        raise ValueError("At least one of global_scene, global_unified, or individual mode must be selected")


def main():
    tyro.extras.subcommand_cli_from_dict(
        {
            "view": view_in_aitviewer,
            "render": render_smpl_sequences,
            "render_chunked": render_smpl_sequences_chunked,
        },
    )


if __name__ == "__main__":
    main()
