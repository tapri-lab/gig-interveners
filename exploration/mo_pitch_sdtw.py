

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium", app_title="Pitch Var Results")


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
    from pyprojroot import here
    import altair as alt
    from scipy import stats
    import pingouin as pg
    return (
        Path,
        alt,
        here,
        librosa,
        mo,
        np,
        pg,
        pl,
        plt,
        sns,
        soft_dtw_alignment,
    )


@app.cell
def _(Path):
    normal_audio_path = Path("../data/Session_1/c/audio_chunks/").resolve()
    intervened_audio_path = Path("../data/Session_1/c/shift_30hz/").resolve()
    return intervened_audio_path, normal_audio_path


@app.cell
def _(intervened_audio_path, normal_audio_path):
    paired_audio = list(zip(sorted(normal_audio_path.glob("*.wav")), sorted(intervened_audio_path.glob("*.wav"))))
    return (paired_audio,)


@app.cell(hide_code=True)
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


@app.cell(disabled=True)
def _(paired_audio, process_audios):
    res = process_audios(paired_audio, "c")
    return (res,)


@app.cell
def _(res):
    res["intervened_sim"][0].item()
    return


@app.cell
def _(Path, librosa):
    y1, sr1 = librosa.load(Path("../data/Session_1/c/shift_30hz/c-clean_001_shifted.wav"), sr=None)
    # y2, sr2 = librosa.load(Path("../data/Session_1/b/shift_30hz/b-clean_001_shifted.wav"), sr=None)

    y2, sr2 = librosa.load(Path("../data/Session_1/c/audio_chunks/c-clean_001.wav"), sr=None)
    # y2, sr2 = librosa.load(Path("../data/Session_1/b/audio_chunks/b-clean_001.wav"), sr=None)
    return sr1, sr2, y1, y2


@app.cell
def _(librosa, np, y1, y2):
    f0_1 = librosa.yin(y1, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0_2 = librosa.yin(y2, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    # Filter out unvoiced frames (NaNs)
    f0_1 = f0_1[~np.isnan(f0_1)]
    f0_2 = f0_2[~np.isnan(f0_2)]
    return f0_1, f0_2


@app.cell
def _(librosa, plt, sr1, sr2, y1, y2):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.waveshow(y1, sr=sr1, ax=ax[0])
    librosa.display.waveshow(y2, sr=sr2, ax=ax[1])
    plt.show()
    return


@app.cell
def _(f0_1, f0_2, soft_dtw_alignment):
    path, sim = soft_dtw_alignment(f0_1, f0_2, gamma=1)
    # path, sim = soft_dtw_alignment(y1, y2, gamma=1)
    sim
    return (path,)


@app.cell
def _(here, path, sns):
    sns.set_theme()
    axh = sns.heatmap(path[256:, 256:], cmap="viridis", xticklabels=False, yticklabels=False)
    axh.set_title("Soft-DTW Alignment")
    figh = axh.get_figure()
    figh.savefig(here() / "results" / "plots" / f"soft_dtw_alignment_path.pdf")
    axh
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis""")
    return


@app.cell
def _():
    baseline_color = "#74BDCB"
    intervened_color = "#FFA384"
    return baseline_color, intervened_color


@app.cell(hide_code=True)
def _(here, mo):
    silence_file_path = mo.ui.file_browser(
        initial_path=here(), filetypes=[".parquet"], selection_mode="file", label="Select Silence File", multiple=False
    )

    pitch_var_res_path = mo.ui.file_browser(
        initial_path=here() / "results",
        filetypes=[".parquet"],
        selection_mode="file",
        label="Select Pitch Variance File",
        multiple=False,
    )

    mo.hstack([silence_file_path, pitch_var_res_path])
    return pitch_var_res_path, silence_file_path


@app.cell(hide_code=True)
def _(pl, silence_file_path):
    silence_df = pl.read_parquet(silence_file_path.path())
    person_mapping = {
        "person1": "a",
        "person2": "c",
        "person3": "b",
        "person4": "j",
        "person5": "l",
    }
    return person_mapping, silence_df


@app.cell(hide_code=True)
def _(person_mapping, pitch_var_res_path, pl, silence_df):
    sdtw_pitch_var_res = pl.read_parquet(pitch_var_res_path.path())
    sdtw_pitch_var_res = sdtw_pitch_var_res.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32), pl.col("person").replace(person_mapping)
    )
    sdtw_pitch_var_res = sdtw_pitch_var_res.join(silence_df, on=["chunk", "person"], how="inner")
    sdtw_pitch_var_res
    return (sdtw_pitch_var_res,)


@app.cell(hide_code=True)
def _(pl, sdtw_pitch_var_res):
    dist_int = pl.col("Distance_Intervened")
    dist_non_int = pl.col("Distance_Non_Intervened")
    val_col = pl.col("Value")

    pitch_var_agg = (
        sdtw_pitch_var_res.with_columns(
            pl.col("Metric").replace({"Distance_Intervened": "Intervened", "Distance_Non_Intervened": "Base"})
        )
        .filter(pl.col("is_silent") == False)
        .rename({"Metric": "condition"})
        .with_columns(
            pl.col("person").str.to_uppercase(),
            ((val_col - val_col.min()) / (val_col.max() - val_col.min())).alias("Normalized_Value"),
        )
        .group_by(["person", "condition"])
        .agg(
            pl.col("Normalized_Value").mean().alias("mean"),
            pl.col("Normalized_Value").std().alias("std"),
        )
    )
    return dist_int, dist_non_int, pitch_var_agg


@app.cell
def _(alt, baseline_color, here, intervened_color, mo, pitch_var_agg):
    alt.theme.enable("ggplot2")
    base = alt.Chart(pitch_var_agg).encode(
        x=alt.X("person:N", title="Persons"),
        y=alt.Y(
            f"mean:Q",
            title=f"Normalized - SoftDTW Distance",
        ),
        color=alt.Color("condition:N", title="Condition").scale(range=[baseline_color, intervened_color]),
        shape=alt.Shape("condition:N", title="Condition", legend=None),
        strokeDash=alt.StrokeDash("condition:N", title="Condition", legend=None),
    )

    points = base.mark_point(filled=True, size=60)
    lines = base.mark_line(point=False)
    # For error bars, use separate chart but keep consistent encoding
    error_bars = (
        alt.Chart(pitch_var_agg)
        .mark_errorbar(clip=True, ticks=True, size=25, thickness=3)
        .encode(
            x="person:N",
            y=alt.Y(f"mean:Q", title="").scale(zero=False),
            yError=alt.YError(f"std:Q"),
            color=alt.Color("condition:N", title="Condition"),
        )
    )

    chart = (
        alt.layer(lines, points, error_bars)
        .resolve_scale(y="shared")
        .properties(width=500, height=400, title=f"No Intervention vs Pitch Variance Reduction").configure_title(fontSize=18)
    )
    chart.save(here() / "results" / "plots" / f"pitch_var_error_bar_plot.pdf")
    mo.ui.altair_chart(chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Wilcoxon test""")
    return


@app.cell
def _(dist_int, dist_non_int, pl, sdtw_pitch_var_res):
    wilcx = (
        sdtw_pitch_var_res.pivot(index=["chunk", "person", "normal_audio", "altered_audio", "is_silent"], on=["Metric"])
        .filter(pl.col("is_silent") == False)
        .with_columns(
            ((dist_int - dist_int.min()) / (dist_int.max() - dist_int.min())).alias("Int_Normalized_Value"),
            ((dist_non_int - dist_non_int.min()) / (dist_non_int.max() - dist_non_int.min())).alias("Normalized_Value"),
        )
        .with_columns(
            (pl.col("Int_Normalized_Value") - pl.col("Normalized_Value")).alias("deltas"),
        )
    )
    wilcx
    return (wilcx,)


@app.cell(hide_code=True)
def _(pg, wilcx):
    pg.wilcoxon(wilcx.select("Normalized_Value"), wilcx.select("Int_Normalized_Value"))
    return


if __name__ == "__main__":
    app.run()
