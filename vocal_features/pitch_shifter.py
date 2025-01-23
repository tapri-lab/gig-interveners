from pathlib import Path
from typing import Tuple

import numpy as np
import parselmouth
import tyro
from parselmouth.praat import call
from tqdm.auto import trange


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
    tyro.cli(limit_f0_variability)
