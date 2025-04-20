

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pyprojroot import here
    import altair as alt
    import pandas as pd
    from typing import List
    import numpy as np
    return List, alt, here, mo, pl


@app.cell
def _(alt):
    alt.theme.enable("default")
    return


@app.cell
def _(here):
    results_base = here() / "results_base"
    motion_damped = here() / "results_damp1"
    pitch_shifted = here() / "results_pitch30hz"
    delayed = here() / "results_delaym0.5"
    return motion_damped, pitch_shifted, results_base


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Individual - Joint Level Recurrence Analysis""")
    return


@app.cell
def _(motion_damped, pl, process_ijr, results_base):
    ijr_base = pl.read_parquet(results_base / "indiv_joint_recurrence.parquet")
    ijr_int = pl.read_parquet(motion_damped / "indiv_joint_recurrence.parquet")

    ijr_base = process_ijr(ijr_base, "Recurrence Radius", ["LeftHand"])
    ijr_int = process_ijr(ijr_int, "Recurrence Radius", ["LeftHand"])
    return ijr_base, ijr_int


@app.cell
def _(ijr_base):
    ijr_base.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plotting Recurrence Radius Box-Plots""")
    return


@app.cell
def _(alt, ijr_base, ijr_int, mo):
    chart_base = (
        alt.Chart(ijr_base, width=400)
        .mark_boxplot()
        .encode(x="person", y="Value", color="person:N")
        .properties(title="Recurrence Radius - No Intervention")
    )
    chart_int = (
        alt.Chart(ijr_int, width=400)
        .mark_boxplot()
        .encode(x="person:N", y="Value:Q", color="person:N")
        .properties(title="Recurrence Radius - Dampened")
    )
    chart_base = mo.ui.altair_chart(chart_base)
    chart_int = mo.ui.altair_chart(chart_int)
    return chart_base, chart_int


@app.cell
def _(chart_base, chart_int, mo):
    mo.vstack([chart_base, chart_int], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Self Beat Consistency""")
    return


@app.cell
def _(pitch_shifted, pl, process_bc, results_base):
    bdf = pl.read_parquet(results_base / "beat_consistency.parquet")
    bdf_int = pl.read_parquet(pitch_shifted / "beat_consistency.parquet")
    bdf, bdf_int = list(map(lambda x: x.filter(pl.col("Metric").ne("fail")), [bdf, bdf_int]))
    bfiltered = process_bc(bdf, ["raw_vs_imf2", "raw_vs_raw", "raw_vs_imf1"])
    bfil_int = process_bc(bdf_int, ["raw_vs_imf2", "raw_vs_raw", "raw_vs_imf1"])
    return bfil_int, bfiltered


@app.cell
def _(bfiltered, pl):
    bfiltered.group_by("person").agg(pl.col("Value").mean())
    return


@app.cell
def _(alt, bfil_int, bfiltered, mo, pl):
    chart_bc_base = (
        alt.Chart(bfiltered.group_by("person").agg(pl.col("Value").mean()), width=500)
        .mark_circle(size=60)
        .encode(x="person:N", y="Value:Q", color=alt.Color("person:N", legend=None))
        .properties(title="Beat Consistency - No Intervention")
    )

    chart_bc_int = (
        alt.Chart(bfil_int.group_by("person").agg(pl.col("Value").mean()), width=500)
        .mark_circle(size=60)
        .encode(x="person:N", y="Value:Q", color=alt.Color("person:N", legend=None))
        .properties(title="Beat Consistency - Pitch Shifted")
    )
    chart_bc_base = mo.ui.altair_chart(chart_bc_base)
    chart_bc_int = mo.ui.altair_chart(chart_bc_int)
    return chart_bc_base, chart_bc_int


@app.cell
def _(alt, bfil_int, pl):
    alt.Chart(bfil_int.filter(pl.col("Metric").eq("raw_vs_raw")), width=500).mark_boxplot().encode(
        x="person:N", y="Value:Q", color=alt.Color("person:N", legend=None)
    ).properties(title="Beat Consistency - Pitch Variation Intervention")
    return


@app.cell
def _(alt, bfil_int, pl):
    alt.Chart(bfil_int.filter(pl.col("Metric").eq("raw_vs_imf1")), width=500).mark_boxplot().encode(
        x="person:N", y="Value:Q", color=alt.Color("person:N", legend=None)
    ).properties(title="Beat Consistency - Pitch Variation Intervention")
    return


@app.cell
def _(alt, bfil_int, pl):
    alt.Chart(bfil_int.filter(pl.col("Metric").eq("raw_vs_imf2")), width=500).mark_boxplot().encode(
        x="person:N", y="Value:Q", color=alt.Color("person:N", legend=None)
    ).properties(title="Beat Consistency - Pitch Variation Intervention")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Avg Plots""")
    return


@app.cell(hide_code=True)
def _(chart_bc_base, chart_bc_int, mo):
    mo.hstack([chart_bc_base, chart_bc_int], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Violin Plots""")
    return


@app.cell
def _(bfil_int, bfiltered, mo, violin_plotter):
    pitch_base_vio = mo.ui.altair_chart(violin_plotter(bfiltered, "Value", ["person"], "Pitch Base"))
    pitch_int_vio = mo.ui.altair_chart(violin_plotter(bfil_int, "Value", ["person"], "Pitch Shifted"))
    return pitch_base_vio, pitch_int_vio


@app.cell
def _(mo, pitch_base_vio, pitch_int_vio):
    mo.vstack([pitch_base_vio, pitch_int_vio])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Helper Functions""")
    return


@app.cell(hide_code=True)
def _(List, alt, pl):
    def violin_plotter(df: pl.DataFrame, value: str, groupby: List[str], title: str, extent: List[float] = [-1, 2]):
        chart = (
            alt.Chart(df, width=150)
            .transform_density(
                value,
                as_=[value, "density"],
                groupby=groupby,
                extent=extent,
            )
            .mark_area(orient="horizontal")
            .encode(
                alt.X("density:Q")
                .stack("center")
                .impute(None)
                .title(None)
                .axis(labels=False, values=[0], grid=False, ticks=True),
                alt.Y(f"{value}:Q"),
                alt.Color(f"{groupby[0]}:N"),
                alt.Column(f"{groupby[0]}:N")
                .spacing(0)
                .header(titleOrient="bottom", labelOrient="bottom", labelPadding=0, title=f"{groupby[0].capitalize()}"),
            )
            .properties(title=title)
            .configure_view(stroke=None)
        )

        return chart
    return (violin_plotter,)


@app.cell(hide_code=True)
def _(List, pl):
    def process_bc(bdf: pl.DataFrame, metrics: List[str]):
        bdf = bdf.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
        bdf = bdf.filter(pl.col("Metric").is_in(metrics))
        return bdf
    return (process_bc,)


@app.cell(hide_code=True)
def _(List, pl):
    def process_ijr(df: pl.DataFrame, metric: str, joint: List[str]):
        df = df.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
        df = df.filter(pl.col("Metric") == metric).filter(pl.col("joint").is_in(joint))
        return df
    return (process_ijr,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
