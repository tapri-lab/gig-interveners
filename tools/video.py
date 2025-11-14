import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import ffmpeg
from omegaconf import OmegaConf
from tyro.extras import subcommand_cli_from_dict


def crop_center(input_path: Path, output_path: Path, crop_width: int, crop_height: int, preset: str = "fast"):
    """
    Crop the center of a video or audio file to a specified width and height.
    :param input_path: Path to the input video file(s).
    :param output_path: Path to save the cropped video file(s).
    :param crop_width: Width to crop the video to.
    :param crop_height: Height to crop the video to.
    :param preset: FFmpeg preset for encoding speed (e.g., "fast", "medium", "slow").
    :return:
    """
    crop_filter = f"crop={crop_width}:{crop_height}:(in_w-{crop_width})/2:(in_h-{crop_height})/2"

    for root_str, _, files in list(os.walk(input_path.expanduser())):
        root = Path(root_str)
        output_path_real = output_path / root.stem
        if not output_path_real.exists():
            output_path_real.mkdir(parents=True, exist_ok=True)
        for filename in files:
            if filename.endswith(".wav") or filename.endswith(".mp4"):
                (
                    ffmpeg.input(str(root / filename))
                    .output(
                        str(output_path_real / filename),
                        vf=crop_filter,
                        vcodec="h264_nvenc",
                        preset=preset,
                        acodec="copy",
                    )
                    .run(overwrite_output=True)
                )


def build_graph(in_files: List, gap_px=10, pad_color="black"):
    """
    Build the filter graph:
        • Pads each input: +gap all around (here: 10 px)
        • hstack top row, hstack bottom row, then vstack
        • Returns the final video stream and (optionally) audio stream #0
    """
    padded_streams = []
    pad_w_expr = f"iw+{gap_px}"
    pad_h_expr = f"ih+{gap_px}"
    # left = gap_px/2 so total gutter = gap_px
    pad_offset = gap_px // 2

    for idx, inp in enumerate(in_files):
        tag = f"p{idx}"
        stream = (
            ffmpeg.input(inp)
            .video.filter("pad", width=pad_w_expr, height=pad_h_expr, x=pad_offset, y=pad_offset, color=pad_color)
            .setpts("PTS-STARTPTS")  # keep timestamps aligned
            .filter("setsar", "1")  # square pixels
        )
        padded_streams.append(stream.set_name(tag))

    # Stack: (a b) on top, (c d) bottom, then stack the two rows
    top = ffmpeg.filter([padded_streams[0], padded_streams[1]], "hstack", inputs=2)
    bot = ffmpeg.filter([padded_streams[2], padded_streams[3]], "hstack", inputs=2)
    outv = ffmpeg.filter([top, bot], "vstack", inputs=2)

    # Keep audio from first file if present
    ina = ffmpeg.input(in_files[0]).audio
    return outv, ina


def has_audio_stream(filepath):
    """Returns True if file has an audio stream"""
    probe = ffmpeg.probe(filepath)
    return any(stream["codec_type"] == "audio" for stream in probe["streams"])


def stitch_videos(
    global_video_path: Path,
    output_path: Path,
    individual_video_path: Optional[Path] = None,
    audio_paths: List[Path] = [],
    audio_offset: float = 0.0,
):
    """
    Stitch multiple video files into a 2x2 grid, optionally with a vertical side video,
    and optionally merge audio from separate files with an offset.

    :param global_video_path: Directory containing 4 global .mp4 files
    :param output_path: Output directory for final video
    :param individual_video_path: Optional vertical side .mp4 video file
    :param audio_paths: List of .wav files to merge (optional)
    :param audio_offset: Delay applied to audio in seconds (optional)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    input_files = sorted(list(global_video_path.rglob("*.mp4")))
    if len(input_files) != 4:
        raise ValueError("Exactly 4 global video files are required.")

    streams = [ffmpeg.input(str(file)) for file in input_files]

    # Pad and stack 2x2 grid
    padded_streams = [s.video.filter("pad", "iw+10", "ih+10", 5, 5, color="black") for s in streams]
    top = ffmpeg.filter([padded_streams[0], padded_streams[1]], "hstack")
    bot = ffmpeg.filter([padded_streams[2], padded_streams[3]], "hstack")
    grid = ffmpeg.filter([top, bot], "vstack")

    if individual_video_path:
        side_stream = ffmpeg.input(str(individual_video_path))
        # Get video height from one stream to match vertical video height
        probe = ffmpeg.probe(str(input_files[0]))
        ih = next(s for s in probe["streams"] if s["codec_type"] == "video")["height"]
        grid_height = ih * 2 + 20

        # Pad vertical video to match grid height
        side_scaled = side_stream.video.filter("scale", -1, grid_height).filter(
            "pad", "iw", grid_height, 0, "(oh-ih)/2", color="black"
        )

        # Stack vertically padded side with grid
        final_video = ffmpeg.filter([grid, side_scaled], "hstack")
    else:
        final_video = grid

    # === Audio handling ===
    output_file = str(output_path / "grid_pad_audio.mp4")

    if audio_paths:
        audio_inputs = [ffmpeg.input(str(path)) for path in audio_paths]
        audio_streams = [a.audio for a in audio_inputs]

        # Mix multiple audio inputs
        if len(audio_streams) == 1:
            mixed_audio = audio_streams[0]
        else:
            mixed_audio = ffmpeg.filter(audio_streams, "amix", inputs=len(audio_streams), duration="longest")

        # Optional delay
        if audio_offset > 0.0:
            delay_ms = int(audio_offset * 1000)
            mixed_audio = mixed_audio.filter("adelay", f"{delay_ms}|{delay_ms}")

        # Output with audio (shortest=None ensures video duration determines output length)
        # Use CQ mode with high quality VBR and bitrate cap
        out = ffmpeg.output(
            final_video,
            mixed_audio,
            output_file,
            vcodec="hevc_nvenc",
            acodec="aac",
            audio_bitrate="192k",
            shortest=None,
            **{"rc:v": "vbr_hq", "cq:v": "19", "b:v": "8M", "maxrate:v": "10M", "bufsize:v": "16M"},
        ).overwrite_output()
    else:
        # Output without audio
        # Use CQ mode with high quality VBR and bitrate cap
        out = ffmpeg.output(
            final_video,
            output_file,
            vcodec="hevc_nvenc",
            **{"rc:v": "vbr_hq", "cq:v": "19", "b:v": "8M", "maxrate:v": "10M", "bufsize:v": "16M"},
        ).overwrite_output()

    out.run()


def split_videos(input_path: Path, output_path: Path, chunk_length: int):
    """
    Splits all videos in a directory recursively into chunks of a specified length.

    :param input_path: Path to the input directory of video files.
    :param output_path: Path to save the chunked video files.
    :param chunk_length: Length of each chunk in seconds.
    """
    for root_str, _, files in os.walk(input_path.expanduser()):
        root = Path(root_str)
        relative_path = root.relative_to(input_path)
        output_dir = output_path / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            if filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
                input_file = root / filename
                output_pattern = output_dir / f"{input_file.stem}_%03d{input_file.suffix}"

                try:
                    (
                        ffmpeg.input(str(input_file))
                        .output(
                            str(output_pattern),
                            c="copy",
                            map=0,
                            f="segment",
                            segment_time=chunk_length,
                            reset_timestamps=1,
                        )
                        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                    )
                    print(f"Successfully split {input_file} into {chunk_length}s chunks.")
                except ffmpeg.Error as e:
                    print(f"Error splitting {input_file}:")
                    print(e.stderr.decode())


def _resolve_paths(path_spec: str | Path, pattern: str = "*.mp4") -> List[Path]:
    """
    Resolve a path specification to a list of file paths.
    Supports:
    - Direct file path
    - Directory path (finds all files matching pattern)
    - Glob pattern (e.g., "/path/*/*.mp4")
    """
    path = Path(path_spec).expanduser()

    # If it contains glob characters, use glob
    if "*" in str(path) or "?" in str(path):
        from pathlib import Path as P

        # Get the base path without glob parts
        parts = path.parts
        base_parts = []
        glob_pattern_parts = []
        found_glob = False

        for part in parts:
            if "*" in part or "?" in part:
                found_glob = True
            if found_glob:
                glob_pattern_parts.append(part)
            else:
                base_parts.append(part)

        base = P(*base_parts) if base_parts else P(".")
        glob_pattern = str(P(*glob_pattern_parts)) if glob_pattern_parts else pattern

        return sorted(base.glob(glob_pattern))

    # If it's a file, return it directly
    if path.is_file():
        return [path]

    # If it's a directory, find all matching files
    if path.is_dir():
        return sorted(path.glob(pattern))

    # Path doesn't exist
    return []


def process_from_config(config_path: Path, operation: Optional[str] = None):
    """
    Process videos based on a YAML configuration file with OmegaConf variable interpolation.

    :param config_path: Path to the YAML configuration file
    :param operation: Specific operation to run (stitch, crop, split, or batch_stitch). If None, runs 'stitch'.
    """
    config_path = Path(config_path).expanduser()

    # Load config with OmegaConf for automatic variable interpolation
    config = OmegaConf.load(config_path)

    # Default to 'stitch' if no operation specified
    if operation is None:
        operation = "stitch"

    if operation == "stitch":
        stitch_cfg = config.get("stitch", {})
        _process_stitch_config(OmegaConf.to_container(stitch_cfg, resolve=True))
    elif operation == "crop":
        crop_cfg = config.get("crop", {})
        _process_crop_config(OmegaConf.to_container(crop_cfg, resolve=True))
    elif operation == "split":
        split_cfg = config.get("split", {})
        _process_split_config(OmegaConf.to_container(split_cfg, resolve=True))
    elif operation == "batch_stitch":
        batch_cfg_list = config.get("batch_stitch", [])
        batch_configs = OmegaConf.to_container(batch_cfg_list, resolve=True)
        if not isinstance(batch_configs, list):
            raise ValueError("batch_stitch must be a list in the config")
        for idx, batch_cfg in enumerate(batch_configs):
            if not isinstance(batch_cfg, dict):
                raise ValueError(f"batch_stitch item {idx} must be a dict")
            name = batch_cfg.get("name", f"batch_{idx}")
            print(f"\n=== Processing batch job: {name} ===")
            _process_stitch_config(batch_cfg)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def _process_stitch_config(config: Dict[str, Any]):
    """Process a stitch operation from config, supporting multiple people."""
    # Check if we're processing multiple people
    people = config.get("people", [])
    people_dir = config.get("people_dir")

    if people:
        # Process each person separately
        for person_cfg in people:
            _process_single_stitch(config, person_cfg)
    elif people_dir:
        # Auto-discover people from directory structure
        people_dir_path = Path(people_dir).expanduser()
        individual_pattern = config.get("individual_video_pattern", "*/*.mp4")
        individual_videos = _resolve_paths(people_dir_path, individual_pattern)

        for video_path in individual_videos:
            # Extract person name from directory structure
            # Handle nested paths like individual/smplx/crop/person_a/video.mp4
            person_name = video_path.parent.name
            person_cfg = {"name": person_name, "individual_video_path": str(video_path)}
            _process_single_stitch(config, person_cfg)
    elif "{person}" in config.get("global_video_path", "") or "{person}" in config.get("individual_video_path", ""):
        # Auto-discover people from template paths
        people_list = _discover_people_from_templates(config)
        if not people_list:
            raise ValueError(
                "Template path contains {person} placeholder but no people were discovered. "
                "Use 'people_dir' or 'people' list to specify people."
            )
        for person_cfg in people_list:
            _process_single_stitch(config, person_cfg)
    else:
        # Single stitch operation (no people list)
        _process_single_stitch(config, None)


def _discover_people_from_templates(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Auto-discover people from template paths containing {person} placeholder.
    E.g., global/smplx/{person}/*/video.mp4 -> finds all person directories
    """
    people = []
    discovered_names = set()

    # Try to discover from global_video_path
    global_template = config.get("global_video_path", "")
    if "{person}" in global_template:
        # Extract the parent path before {person}
        template_path = Path(global_template)
        parts = template_path.parts
        person_idx = next((i for i, part in enumerate(parts) if "{person}" in part), None)

        if person_idx is not None:
            base_path = Path(*parts[:person_idx]) if person_idx > 0 else Path(".")
            base_path = base_path.expanduser()

            if base_path.exists() and base_path.is_dir():
                # List subdirectories at the person level
                for person_dir in base_path.iterdir():
                    if person_dir.is_dir():
                        person_name = person_dir.name
                        discovered_names.add(person_name)

    # Try to discover from individual_video_path
    individual_template = config.get("individual_video_path", "")
    if "{person}" in individual_template:
        template_path = Path(individual_template)
        parts = template_path.parts
        person_idx = next((i for i, part in enumerate(parts) if "{person}" in part), None)

        if person_idx is not None:
            base_path = Path(*parts[:person_idx]) if person_idx > 0 else Path(".")
            base_path = base_path.expanduser()

            if base_path.exists() and base_path.is_dir():
                for person_dir in base_path.iterdir():
                    if person_dir.is_dir():
                        person_name = person_dir.name
                        discovered_names.add(person_name)

    # Create person configs
    for person_name in sorted(discovered_names):
        people.append({"name": person_name})

    return people


def _process_single_stitch(config: Dict[str, Any], person_cfg: Optional[Dict[str, Any]]):
    """Process a single stitch operation for one person or no person."""
    # Resolve global video paths - support per-person paths
    person_name = person_cfg.get("name") if person_cfg else None

    # Check for person-specific global video path first
    if person_cfg and "global_video_path" in person_cfg:
        global_path = person_cfg["global_video_path"]
        global_pattern = person_cfg.get("global_video_pattern", "*.mp4")
    else:
        global_path = config.get("global_video_path")
        global_pattern = config.get("global_video_pattern", "*.mp4")

        # Replace {person} placeholder if present
        if person_name and global_path:
            global_path = global_path.replace("{person}", person_name)

    if not global_path:
        raise ValueError("global_video_path is required for stitch operation")

    global_videos = _resolve_paths(global_path, global_pattern)
    if len(global_videos) != 4:
        raise ValueError(
            f"Expected 4 global videos for {person_name or 'base'}, found {len(global_videos)}: {global_videos}"
        )

    # Resolve individual video (optional)
    individual_video = None
    # Try person-specific path first, then fall back to main config
    individual_path = None
    individual_pattern = "*.mp4"

    if person_cfg and "individual_video_path" in person_cfg:
        # Get individual video from person config
        individual_path = person_cfg.get("individual_video_path")
        individual_pattern = person_cfg.get("individual_video_pattern", "*.mp4")
    elif config.get("individual_video_path"):
        # Get individual video from main config
        individual_path = config.get("individual_video_path")
        individual_pattern = config.get("individual_video_pattern", "*.mp4")

    if individual_path:
        # Replace {person} placeholder if present
        if person_name and "{person}" in individual_path:
            individual_path = individual_path.replace("{person}", person_name)

        individual_videos = _resolve_paths(individual_path, individual_pattern)
        if len(individual_videos) == 1:
            individual_video = individual_videos[0]
        elif len(individual_videos) == 0:
            print(f"Warning: No individual video found matching pattern: {individual_path}")
        elif len(individual_videos) > 1:
            raise ValueError(f"Expected 1 individual video, found {len(individual_videos)}: {individual_videos}")

    # Resolve audio paths (optional)
    audio_files = []
    # Check person-specific audio first
    if person_cfg and "audio_paths" in person_cfg:
        audio_paths_cfg = person_cfg["audio_paths"]
    else:
        audio_paths_cfg = config.get("audio_paths", [])

    if isinstance(audio_paths_cfg, list):
        for audio_spec in audio_paths_cfg:
            resolved = _resolve_paths(audio_spec, "*.wav")
            audio_files.extend(resolved)
    elif audio_paths_cfg:
        # Single path/pattern
        audio_files = _resolve_paths(audio_paths_cfg, "*.wav")

    # Alternative: audio_dir + audio_pattern
    if not audio_files and "audio_dir" in config:
        audio_dir = config["audio_dir"]
        audio_pattern = config.get("audio_pattern", "*.wav")
        audio_files = _resolve_paths(audio_dir, audio_pattern)

    audio_offset = config.get("audio_offset", 0.0)

    # Determine output path
    output_path_str = config["output_path"]
    if person_cfg:
        person_name = person_cfg["name"]
        # Replace {person} placeholder in output path
        output_path_str = output_path_str.replace("{person}", person_name)
        # If no placeholder, append person name as subdirectory
        if "{person}" not in config["output_path"]:
            output_path = Path(output_path_str).expanduser() / person_name
        else:
            output_path = Path(output_path_str).expanduser()
        print(f"\n=== Processing person: {person_name} ===")
    else:
        output_path = Path(output_path_str).expanduser()

    print(f"Global videos ({len(global_videos)}): {[str(p.name) for p in global_videos]}")
    if individual_video:
        print(f"Individual video: {individual_video}")
    else:
        print("Individual video: None (will create 2x2 grid only)")
    if audio_files:
        print(f"Audio files ({len(audio_files)}): {[str(p.name) for p in audio_files]}")
    else:
        print("Audio files: None (video will have no audio)")
    print(f"Output: {output_path}")

    # Create a temp directory with the 4 global videos for stitch_videos
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for video in global_videos:
            (tmp_path / video.name).symlink_to(video.absolute())

        stitch_videos(
            global_video_path=tmp_path,
            output_path=output_path,
            individual_video_path=individual_video,
            audio_paths=audio_files,
            audio_offset=audio_offset,
        )


def _process_crop_config(config: Dict[str, Any]):
    """Process a crop operation from config."""
    input_path = Path(config["input_path"]).expanduser()
    output_path = Path(config["output_path"]).expanduser()
    crop_width = config.get("crop_width", 1920)
    crop_height = config.get("crop_height", 1080)
    preset = config.get("preset", "fast")

    print(f"Cropping videos from {input_path} to {output_path}")
    print(f"Crop size: {crop_width}x{crop_height}, preset: {preset}")

    crop_center(input_path, output_path, crop_width, crop_height, preset)


def _process_split_config(config: Dict[str, Any]):
    """Process a split operation from config."""
    input_path = Path(config["input_path"]).expanduser()
    output_path = Path(config["output_path"]).expanduser()
    chunk_length = config.get("chunk_length", 30)

    print(f"Splitting videos from {input_path} to {output_path}")
    print(f"Chunk length: {chunk_length} seconds")

    split_videos(input_path, output_path, chunk_length)


if __name__ == "__main__":
    subcommand_cli_from_dict(
        {
            "crop": crop_center,
            "stitch": stitch_videos,
            "split": split_videos,
            "config": process_from_config,
        },
        description="Video processing tools",
    )
