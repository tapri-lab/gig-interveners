from pathlib import Path

import librosa
import numpy as np
import tyro
from scipy.signal import savgol_filter


def f0_limited(audio_path, fmin_hz=65, fmax_hz=400, smoothing_window=51, polyorder=3):
    y, sr = librosa.load(audio_path)

    # extract f0
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))

    # clip f0 values to the desired range
    f0_clipped = np.clip(f0, fmin_hz, fmax_hz)

    # smooth the f0 contour
    f0_smoothed = savgol_filter(f0_clipped, smoothing_window, polyorder)

    return y, sr, f0_smoothed, voiced_flag


def limit_f0_variability(
    audio_path: Path, max_deviation_hz: float = 50.0, smoothing_window: int = 51, polyorder: int = 3
):
    """
    Limit the f0 variability in speech to stay within a max deviation from the mean f0.

    Args:
        audio_path: Path to audio file
        max_deviation_hz: Maximum allowed deviation from mean f0 in Hz
        smoothing_window: Window size for smoothing (must be odd)
        polyorder: Order of polynomial for smoothing for the Savitzky-Golay filter
    """
    # Get f0 contour using existing function
    y, sr, f0 = f0_limited(audio_path, smoothing_window=smoothing_window, polyorder=polyorder)

    # Find voiced frames
    voiced_mask = f0 > 0

    # Calculate mean f0 of voiced segments
    mean_f0 = np.mean(f0[voiced_mask])

    # Define allowed range
    min_f0 = mean_f0 - max_deviation_hz
    max_f0 = mean_f0 + max_deviation_hz

    # Limit f0 range
    f0_clipped = np.clip(f0, min_f0, max_f0)

    # Calculate pitch shift ratio at each point
    shift_ratio = np.ones_like(f0)
    shift_ratio[voiced_mask] = f0_clipped[voiced_mask] / f0[voiced_mask]

    # Apply time-varying pitch shift
    frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
    shifted_frames = np.zeros_like(frames)

    for i in range(frames.shape[1]):
        frame = frames[:, i]
        ratio = shift_ratio[i] if i < len(shift_ratio) else 1.0
        if not np.isnan(ratio):
            n_steps = 12 * np.log2(ratio)
            shifted_frames[:, i] = librosa.effects.pitch_shift(frame, sr=sr, n_steps=float(n_steps))

    # Reconstruct audio
    shifted_audio = librosa.util.fix_length(librosa.overlap_add(shifted_frames, 512), size=len(y))

    return shifted_audio, sr


if __name__ == "__main__":
    tyro.cli(limit_f0_variability)
