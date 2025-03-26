import math
from pathlib import Path
from typing import Optional

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from tqdm.auto import trange
from tyro.extras import subcommand_cli_from_dict


def split_wav_file(
    input_file: Path,
    output_dir: Optional[Path] = None,
    segment_length: float = 30.0,
    offset: float = 0.0,
) -> list[Path]:
    """
    Splits a .wav file into smaller segments using `soundfile`.

    Args:
        input_file: Path to input .wav file
        output_dir: Directory to save output segments
        segment_length: Length of each segment in seconds
        offset: Time offset in seconds. Positive values trim from beginning,
               negative values add silence at the beginning.
    Returns:
        List of paths to the output segments
    """

    output_dir = output_dir or Path(input_file.parent, "audio_segments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # base file name without extension
    base_name = input_file.stem

    data, sr = sf.read(str(input_file.expanduser()))

    # Handle the offset
    offset_samples = int(offset * sr)
    if offset_samples > 0:
        # Positive offset: trim from beginning
        if offset_samples >= len(data):
            raise ValueError(f"Offset ({offset}s) exceeds audio length ({len(data) / sr:.2f}s)")
        data = data[offset_samples:]
    elif offset_samples < 0:
        # Negative offset: add silence at the beginning
        silence = np.zeros(
            (-offset_samples, data.shape[1]) if len(data.shape) > 1 else -offset_samples, dtype=data.dtype
        )
        data = np.concatenate([silence, data])

    # Calculate samples per segment
    samples_per_segment = int(segment_length * sr)

    # Calculate number of segments
    total_segments = math.ceil(len(data) / samples_per_segment)

    output_files = []

    for seg_idx in range(total_segments):
        start = seg_idx * samples_per_segment
        end = min(start + samples_per_segment, len(data))

        segment_data = data[start:end]

        output_file = output_dir / f"{base_name}_{seg_idx + 1:03d}.wav"
        sf.write(str(output_file), segment_data, sr)
        output_files.append(output_file)

    return output_files


def limit_f0_variability(
    audio_path: Path,
    output_path: Path,
    max_deviation_hz: float = 50.0,
) -> parselmouth.Sound:
    """
    Limit the f0 variability in speech to stay within a max deviation from the mean f0.

    Args:
        audio_path: Path to audio file
        output_path: Path to save the processed audio
        max_deviation_hz: Maximum allowed deviation from mean f0 in Hz
    Returns:
        parselmouth.Sound: Processed sound object
    """
    # Load sound
    audio_path = Path(audio_path).expanduser()
    output_path = str(Path(output_path).expanduser())
    sound = parselmouth.Sound(str(audio_path))

    # Create manipulation object
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)

    # Extract pitch tier
    pitch_tier = call(manipulation, "Extract pitch tier")
    pitch = sound.to_pitch()

    # Get mean f0 of voiced regions
    pitch_values = pitch.selected_array["frequency"]
    voiced_f0 = pitch_values[pitch_values > 0]
    mean_f0 = np.mean(voiced_f0)

    # Get points and limit their deviation
    num_points = call(pitch_tier, "Get number of points")
    for i in trange(1, num_points + 1):
        time = call(pitch_tier, "Get time from index", i)
        f0 = call(pitch_tier, "Get value at index", i)

        # Limit deviation
        limited_f0 = np.clip(f0, mean_f0 - max_deviation_hz, mean_f0 + max_deviation_hz)

        # Replace point
        call(pitch_tier, "Remove point", i)
        call(pitch_tier, "Add point", time, limited_f0)

    # Replace pitch tier and resynthesize
    call([pitch_tier, manipulation], "Replace pitch tier")
    modified_sound = call(manipulation, "Get resynthesis (overlap-add)")
    modified_sound.save(output_path, "WAV")
    return modified_sound


if __name__ == "__main__":
    subcommand_cli_from_dict({"limit_f0": limit_f0_variability, "split_audio": split_wav_file})
