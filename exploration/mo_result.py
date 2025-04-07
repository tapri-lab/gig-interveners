import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


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
    return List, alt, here, mo, pd, pl, plt, sns


@app.cell
def _(alt):
    alt.theme.enable("default")
    return


@app.cell
def _(here):
    results_base = here() / "results_base"
    motion_damped = here() / "results_dampened"
    pitch_shifted = here() / "results_pitch60hz"
    return motion_damped, pitch_shifted, results_base


@app.cell
def _(List, pl):
    def process_ijr(df: pl.DataFrame, metric: str, joint: List[str]):
        df = df.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
        df = df.filter(pl.col("Metric") == metric).filter(pl.col("joint").is_in(joint))
        return df
    return (process_ijr,)


@app.cell
def _(motion_damped, pl, process_ijr, results_base):
    ijr_base = pl.read_parquet(results_base / "indiv_joint_recurrence.parquet")
    ijr_int = pl.read_parquet(motion_damped / "indiv_joint_recurrence.parquet")

    ijr_base = process_ijr(ijr_base, "Recurrence Rate", ["LeftHand"])
    ijr_int = process_ijr(ijr_int, "Recurrence Rate", ["LeftHand"])
    return ijr_base, ijr_int


@app.cell
def _(alt, ijr_base, ijr_int, mo):
    chart_base = (
        alt.Chart(ijr_base)
        .mark_boxplot()
        .encode(x="person", y="Value", color="person:N")
        .properties(title="Recurrence Rate - No Intervention")
    )
    chart_int = (
        alt.Chart(ijr_int)
        .mark_boxplot()
        .encode(x="person:N", y="Value:Q", color="person:N")
        .properties(title="Recurrence Rate - Dampened")
    )
    chart_base = mo.ui.altair_chart(chart_base)
    chart_int = mo.ui.altair_chart(chart_int)
    return chart_base, chart_int


@app.cell
def _(chart_base, chart_int, mo):
    mo.vstack([chart_base, chart_int])
    return


@app.cell
def _(pitch_shifted, pl, results_base):
    def process_bc(bdf: pl.DataFrame, metric: str):
        bdf = bdf.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
        bdf = bdf.filter(pl.col("Metric") == metric)
        return bdf


    bdf = pl.read_parquet(results_base / "beat_consistency.parquet")
    bdf_int = pl.read_parquet(pitch_shifted / "beat_consistency.parquet")
    bfiltered = process_bc(bdf, "imf1_vs_imf2")
    bfil_int = process_bc(bdf_int, "imf1_vs_imf2")
    return bdf, bdf_int, bfil_int, bfiltered, process_bc


@app.cell
def _(bdf_int):
    bdf_int.head()
    return


@app.cell
def _(alt, bfil_int, bfiltered, mo):
    chart_bc_base = (
        alt.Chart(bfiltered)
        .mark_boxplot()
        .encode(x="person:N", y="Value:Q", color="person:N")
        .properties(title="Beat Consistency - No Intervention")
    )

    chart_bc_int = (
        alt.Chart(bfil_int)
        .mark_boxplot()
        .encode(x="person:N", y="Value:Q", color="person:N")
        .properties(title="Beat Consistency - Pitch Shifted")
    )
    chart_bc_base = mo.ui.altair_chart(chart_bc_base)
    chart_bc_int = mo.ui.altair_chart(chart_bc_int)
    return chart_bc_base, chart_bc_int


@app.cell
def _(chart_bc_base, chart_bc_int, mo):
    mo.vstack([chart_bc_base, chart_bc_int])
    return


@app.cell
def _(List, alt, pl):
    def violin_plotter(df: pl.DataFrame, value: str, groupby: List[str], title: str):
        chart = (
            alt.Chart(df, width=150)
            .transform_density(
                value,
                as_=[value, "density"],
                groupby=groupby,
                extent=[-1, 2],
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


@app.cell
def _(bfil_int, bfiltered, mo, violin_plotter):
    pitch_base_vio = mo.ui.altair_chart(violin_plotter(bfiltered, "Value", ["person"], "Pitch Base"))
    pitch_int_vio = mo.ui.altair_chart(violin_plotter(bfil_int, "Value", ["person"], "Pitch Shifted"))
    return pitch_base_vio, pitch_int_vio


@app.cell
def _(mo, pitch_base_vio, pitch_int_vio):
    mo.vstack([pitch_base_vio, pitch_int_vio])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
