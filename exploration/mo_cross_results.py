

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    import altair as alt
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pyprojroot import here
    import numpy as np


@app.cell
def _():
    results_base = here() / "results_base"
    return (results_base,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""# CRQA - Joints""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Base Results""")
    return


@app.cell
def _():
    crdf_base = pl.read_parquet(here() / "results_base/cross_joint_recurrence.parquet")
    crdf_base = crdf_base.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
    crdf_base
    return (crdf_base,)


@app.cell
def _(crdf_base):
    r_radius_base = crdf_base.filter(pl.col("Metric").eq("Recurrence Radius"))
    r_radius_base = r_radius_base.with_columns(
        (pl.col("Value") - pl.col("Value").min()) / (pl.col("Value").max() - pl.col("Value").min())
    )
    r_radius_base = (
        r_radius_base.filter(pl.col("Metric").eq("Recurrence Radius"))
        .group_by(["person1", "person2"])
        .agg(pl.col("Value").mean())
        .unique(["person1", "person2"])
    )
    r_radius_base
    return (r_radius_base,)


@app.function
def df_to_pairwise_mat(df: pl.DataFrame, n: int) -> np.ndarray:
    pairwise_mat = np.zeros((n, n))
    for row in df.iter_rows(named=True):
        p1 = row["person1"][-1]
        p2 = row["person2"][-1]
        p1, p2 = list(map(int, [p1, p2]))
        p1 -= 1
        p2 -= 1
        val = row["Value"]
        pairwise_mat[p1, p2] = val
    return pairwise_mat


@app.cell
def _(r_radius_base):
    pairwise_crqa = df_to_pairwise_mat(r_radius_base, 5)
    return (pairwise_crqa,)


@app.cell
def _(pairwise_crqa):
    sns.set_theme("paper")
    # sns.set_style("whitegrid")
    plt.style.use("ggplot")
    heat_plot = sns.heatmap(
        pairwise_crqa,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Scaled Recurrence Radius"},
        mask=np.triu(pairwise_crqa),
    )
    fig = heat_plot.get_figure()
    fig.savefig(here() / "results_base/crqa_pairwise_radius_base.png", dpi=300, bbox_inches="tight")
    fig
    return


@app.cell
def _(r_radius_base):
    alt.theme.enable("ggplot2")
    alt.Chart(r_radius_base.unique(["person1", "person2"]).with_columns((pl.col("Value")))).mark_rect().encode(
        x=alt.X("person1:N").title("Person 1"),
        y=alt.Y("person2:N").title("Person 2"),
        color=alt.Color(
            "Value:Q", scale=alt.Scale(domain=[0, 1], scheme="viridis", reverse=True), title="Scaled Recurrence Radius"
        ),
    ).properties(width=400, height=400).configure_view(strokeWidth=0).configure_axis(
        labelFontSize=12, titleFontSize=12
    ).configure_legend(labelFontSize=12, titleFontSize=12)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Cross BC""")
    return


@app.cell
def _():
    cross_bc_base = pl.read_parquet(here() / "results_base/cross_beat_consistency.parquet")
    cross_bc_base = cross_bc_base.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
    cross_bc_base
    return (cross_bc_base,)


@app.cell
def _(cross_bc_base):
    cross_bc_base.filter(pl.col("Metric").is_in(["raw_vs_raw", "raw_vs_imf1", "raw_vs_imf2"])).group_by(
        ["person1", "person2"]
    ).agg(pl.col("Value").mean()).unique(["person1", "person2"])
    return


@app.cell
def _(cross_bc_base):
    pairwise_bc = df_to_pairwise_mat(
        cross_bc_base.filter(pl.col("Metric").is_in(["raw_vs_raw", "raw_vs_imf1", "raw_vs_imf2"]))
        .group_by(["person1", "person2"])
        .agg(pl.col("Value").mean())
        .unique(["person1", "person2"]),
        5,
    )
    return (pairwise_bc,)


@app.cell
def _(pairwise_bc):
    bc_heatmap_base = sns.heatmap(
        pairwise_bc,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Beat Consistency"},
        mask=np.triu(pairwise_bc),
    )
    fig2 = bc_heatmap_base.get_figure()
    fig2.savefig(here() / "results_base/bc_pairwise_base.png", dpi=300, bbox_inches="tight")
    fig2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Cross SDTW""")
    return


@app.function
def cross_sdtw_plotter(raw_df: pl.DataFrame, n: int):
    raw_df = raw_df.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
    df = raw_df.with_columns(
        ((pl.col("Value") - pl.col("Value").min()) / (pl.col("Value").max() - pl.col("Value").min()))
    )
    df = df.group_by(["person1", "person2"]).agg(pl.col("Value").mean()).unique(["person1", "person2"])
    sdtw_mat = df_to_pairwise_mat(df, 5)
    fig = sns.heatmap(
        sdtw_mat,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "SDTW Normalised"},
        mask=np.triu(sdtw_mat),
    )
    return fig


@app.cell
def _(results_base):
    base_sdtw = pl.read_parquet(results_base / "sdtw_results.parquet")
    base_sdtw = base_sdtw.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
    base_sdtw.head()
    return (base_sdtw,)


@app.cell
def _(base_sdtw):
    bz = base_sdtw.with_columns(
        ((pl.col("Value") - pl.col("Value").min()) / (pl.col("Value").max() - pl.col("Value").min()))
    )
    bz.head()
    return


@app.cell
def _():
    df = pl.read_parquet(here() / "results_pitch30hz/sdtw_results.parquet").with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)
    )
    df = df.with_columns(
        ((pl.col("Value") - pl.col("Value").min()) / (pl.col("Value").max() - pl.col("Value").min()))
    )
    df = df.group_by(["person1", "person2"]).agg(pl.col("Value").mean()).unique(["person1", "person2"])
    df.head()
    return


@app.cell
def _():
    cross_sdtw_plotter(pl.read_parquet(here() / "results_pitch30hz/sdtw_results.parquet"), 5)
    return


if __name__ == "__main__":
    app.run()
