import os
from pathlib import Path
from typing import List, Optional

import ffmpeg
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

        # Output with audio
        out = ffmpeg.output(
            final_video, mixed_audio, output_file, vcodec="hevc_nvenc", acodec="aac", audio_bitrate="192k"
        ).overwrite_output()
    else:
        # Output without audio
        out = ffmpeg.output(final_video, output_file, vcodec="hevc_nvenc").overwrite_output()

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


if __name__ == "__main__":
    subcommand_cli_from_dict(
        {
            "crop": crop_center,
            "stitch": stitch_videos,
            "split": split_videos,
        },
        description="Video processing tools",
    )
