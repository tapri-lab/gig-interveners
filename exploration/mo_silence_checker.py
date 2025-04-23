

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import librosa
    import numpy as np
    from pathlib import Path
    from scipy.io import wavfile
    from joblib import Parallel, delayed
    import polars as pl
    import os


@app.function
def is_mostly_silent(audio_path, silence_threshold_db=-40.0):
    y, sr = librosa.load(audio_path, sr=None)
    rms_db = librosa.amplitude_to_db([np.sqrt(np.mean(y**2))])[0]
    # print(f"RMS dB: {rms_db}")
    return rms_db < silence_threshold_db


@app.function
def is_mostly_silent2(filepath, silence_thresh=0.01):
    sr, data = wavfile.read(filepath)
    if data.ndim > 1:  # stereo to mono
        data = data.mean(axis=1)
    data = data / np.max(np.abs(data))  # normalize
    rms = np.sqrt(np.mean(data**2))
    print(f"RMS: {rms}")
    return rms < silence_thresh


@app.cell(hide_code=True)
def _():
    base_data_path = mo.ui.file_browser(
        initial_path="../data/", label="Select DND Session Folder", selection_mode="directory", multiple=False
    )
    base_data_path
    return (base_data_path,)


@app.cell
def _(base_data_path):
    audio_paths = [i / "audio_chunks/" for i in base_data_path.path().glob("*") if not i.stem.startswith(".")]
    return (audio_paths,)


@app.cell
def _(audio_paths):
    res = []
    for audio_path in mo.status.progress_bar(audio_paths, completion_title="Processed all audio files for silence."):
        person = audio_path.parent.stem
        audio_files = list(audio_path.rglob("*.wav"))
        for file in mo.status.progress_bar(audio_files, remove_on_exit=True):
            chunk = file.stem[-3:]
            is_silent = is_mostly_silent(file)
            res.append((person, chunk, is_silent))
    return (res,)


@app.cell
def _(res):
    silence = pl.DataFrame(res, schema=["person", "chunk", "is_silent"], orient="row")
    silence = silence.with_columns(pl.col("is_silent").cast(pl.Boolean), pl.col("chunk").cast(pl.Int32))
    silence
    return (silence,)


@app.cell
def _(base_data_path, silence):
    silence.write_parquet(f"{base_data_path.path().stem}_silence.parquet")
    return


if __name__ == "__main__":
    app.run()
