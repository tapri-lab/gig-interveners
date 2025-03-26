#!/usr/bin/env python3
import math
import os
import sys
from pathlib import Path

import bpy


# Function to split animation into chunks of specified duration
def split_bvh_animation(bvh_file_path, chunk_duration=30):
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(bvh_file_path))[0]
    output_dir = Path(os.path.dirname(bvh_file_path))
    output_dir = output_dir / f"{base_filename}_chunks"
    output_dir.mkdir(exist_ok=True)
    output_dir = str(output_dir)

    # Clear existing objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import the BVH file
    bpy.ops.import_anim.bvh(
        filepath=bvh_file_path,
        filter_glob="*.bvh",
        target="ARMATURE",
        global_scale=1.0,
        frame_start=1,
        use_fps_scale=True,
        use_cyclic=False,
        rotate_mode="NATIVE",
        axis_forward="-Z",
        axis_up="Y",
    )

    # In Blender 4.2, we need to ensure the armature is selected
    bpy.ops.object.select_all(action="DESELECT")

    # Get the armature (imported BVH)
    # In Blender 4.2, we need to find the armature differently
    armature = None
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            armature = obj
            break

    if not armature:
        print("Error: Could not find imported armature.")
        sys.exit(1)

    # Select the armature
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Get scene info
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    frame_count = int(armature.animation_data.action.frame_range[1])

    # Calculate frames per chunk
    frames_per_chunk = int(chunk_duration * fps)
    num_chunks = math.ceil(frame_count / frames_per_chunk)

    print(f"FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Frames per chunk: {frames_per_chunk}")
    print(f"Number of chunks: {num_chunks}")

    # Process each chunk
    for chunk_idx in range(num_chunks):
        # Calculate frame range for this chunk
        start_frame = chunk_idx * frames_per_chunk + 1
        end_frame = min((chunk_idx + 1) * frames_per_chunk, frame_count)

        # Set the frame range for the scene
        scene.frame_start = start_frame
        scene.frame_end = end_frame

        # Create output filename
        output_filename = f"{base_filename}_chunk{chunk_idx + 1:03d}.bvh"
        output_path = os.path.join(output_dir, output_filename)

        # Export the chunk as BVH
        # For Blender 4.2, ensure proper selection before export
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature

        # Export with Blender 4.2 compatible parameters
        bpy.ops.export_anim.bvh(
            filepath=output_path,
            check_existing=True,
            filter_glob="*.bvh",
            global_scale=1.0,
            frame_start=start_frame,
            frame_end=end_frame,
            rotate_mode="NATIVE",
            root_transform_only=False,
        )

        print(f"Exported chunk {chunk_idx + 1}/{num_chunks}: {output_filename}")

    return num_chunks


# Main execution
if __name__ == "__main__":
    # Get command line arguments passed to the script
    # First arg is the script name, second should be the BVH file
    argv = sys.argv

    argv = argv[argv.index("--") + 1 :]  # get all args after "--"

    # Check if BVH file was provided
    if len(argv) < 2:
        print("Usage: blender --background --python bvh_splitter.py -- <path_to_bvh_file> [chunk_duration]")
        sys.exit(1)

    bvh_file_path = argv[0]

    # Check if optional chunk duration was provided
    chunk_duration = 30  # Default: 30 seconds
    if len(argv) > 2:
        try:
            chunk_duration = float(argv[1])
        except ValueError:
            print(f"Invalid chunk duration: {argv[1]}. Using default of 30 seconds.")

    # Verify the file exists
    if not os.path.isfile(bvh_file_path):
        print(f"Error: File '{bvh_file_path}' not found.")
        sys.exit(1)

    # Verify it's a BVH file
    if not bvh_file_path.lower().endswith(".bvh"):
        print(f"Error: File '{bvh_file_path}' is not a BVH file.")
        sys.exit(1)

    print(f"Processing BVH file: {bvh_file_path}")
    print(f"Chunk duration: {chunk_duration} seconds")

    # Split the animation
    num_chunks = split_bvh_animation(bvh_file_path, chunk_duration)

    print(f"Successfully split into {num_chunks} chunks.")
    sys.exit(0)
