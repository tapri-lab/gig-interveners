

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import librosa
    import numpy as np
    from ott.geometry.costs import SoftDTW
    from pathlib import Path
    import parselmouth
    from tslearn.metrics import soft_dtw, dtw, soft_dtw_alignment
    import matplotlib.pyplot as plt
    import seaborn as sns
    import polars as pl
    return Path, SoftDTW, librosa, mo, np, sns, soft_dtw_alignment


@app.cell
def _(Path):
    normal_audio_path = Path("../data/Session_1/c/audio_chunks/").resolve()
    intervened_audio_path = Path("../data/Session_1/c/shift_30hz/").resolve()
    return intervened_audio_path, normal_audio_path


@app.cell
def _(intervened_audio_path, normal_audio_path):
    paired_audio = list(zip(sorted(normal_audio_path.glob("*.wav")), sorted(intervened_audio_path.glob("*.wav"))))
    return (paired_audio,)


@app.cell
def _(librosa, mo, np, soft_dtw_alignment):
    def process_audios(paired_audio, person: str, gamma: float = 1.0):
        res = {
            "normal_path": [],
            "normal_sim": [],
            "intervened_path": [],
            "intervened_sim": [],
            "chunk": [],
            "person": [],
            "gamma": [],
        }
        for normal_audio, int_audio in mo.status.progress_bar(paired_audio):
            chunk = normal_audio.stem.split("_")[-1]
            y1, sr1 = librosa.load(normal_audio, sr=None)
            y2, sr2 = librosa.load(int_audio, sr=None)

            f0_1 = librosa.yin(y1, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
            f0_2 = librosa.yin(y2, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))

            # Filter out unvoiced frames (NaNs)
            f0_1 = f0_1[~np.isnan(f0_1)]
            f0_2 = f0_2[~np.isnan(f0_2)]

            # Compute the soft-DTW distance
            path1, sim1 = soft_dtw_alignment(f0_1, f0_1, gamma=gamma)
            path2, sim2 = soft_dtw_alignment(f0_1, f0_2, gamma=gamma)

            res["normal_path"].append(path1)
            res["normal_sim"].append(sim1)
            res["intervened_path"].append(path2)
            res["intervened_sim"].append(sim2)
            res["chunk"].append(chunk)
            res["person"].append(person)
            res["gamma"].append(gamma)
        return res
    return (process_audios,)


@app.cell
def _(paired_audio, process_audios):
    res = process_audios(paired_audio, "c")
    return


@app.cell
def _(Path, librosa):
    y1, sr1 = librosa.load(Path("../data/Session_1/c/audio_chunks/c-clean_001.wav"), sr=None)
    y2, sr2 = librosa.load(Path("../data/Session_1/c/shift_30hz/c-clean_001_shifted.wav"), sr=None)
    return y1, y2


@app.cell
def _(librosa, np, y1, y2):
    f0_1 = librosa.yin(y1, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0_2 = librosa.yin(y2, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    # Filter out unvoiced frames (NaNs)
    f0_1 = f0_1[~np.isnan(f0_1)]
    f0_2 = f0_2[~np.isnan(f0_2)]
    return f0_1, f0_2


@app.cell
def _(SoftDTW):
    s = SoftDTW(gamma=0.5)
    return


@app.cell
def _(f0_1, f0_2, soft_dtw_alignment):
    path, sim = soft_dtw_alignment(f0_1, f0_2, gamma=1)
    return (path,)


@app.cell
def _(path, sns):
    sns.heatmap(path, cmap="viridis", xticklabels=False, yticklabels=False)
    return


if __name__ == "__main__":
    app.run()
