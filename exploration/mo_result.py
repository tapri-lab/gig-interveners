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
    return alt, here, mo, pd, pl, plt, sns


@app.cell
def _(alt):
    alt.theme.enable('ggplot2')
    return


@app.cell
def _(here):
    results_base = here() / "results_base"
    motion_damped = here() / "results_dampened"
    pitch_shifted = here() / "results_pitch_shift"
    return motion_damped, pitch_shifted, results_base


@app.cell
def _(pl, results_base):
    ijr = pl.read_parquet(results_base / "indiv_joint_recurrence.parquet")
    ijr = ijr.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)
    )
    ijr
    return (ijr,)


@app.cell
def _(ijr, mo):
    filtered = mo.sql(
        f"""
        SELECT * FROM ijr where joint == 'LeftHand' and Metric == 'Recurrence Rate'
        """
    )
    return (filtered,)


@app.cell
def _(alt, filtered, mo):
    chart = (
        alt.Chart(filtered)
        .mark_boxplot()
        .encode(
            x="person",
            y="Value",
            color="person:N"
        )
        .properties(
            title="Recurrence Rate"
        )
    )
    chart = mo.ui.altair_chart(chart)
    return (chart,)


@app.cell
def _(pl, results_base):
    bdf = pl.read_parquet(results_base / "beat_consistency.parquet")
    bdf = bdf.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)
    )
    bdf
    return (bdf,)


@app.cell
def _(bdf, mo):
    bfiltered = mo.sql(
        f"""
        select * from bdf where metric == 'imf1_vs_imf2'
        """
    )
    return (bfiltered,)


@app.cell
def _(alt, bfiltered, mo):
    chart2 = (
        alt.Chart(bfiltered)
        .mark_boxplot()
        .encode(
            x="person:N",
            y="Value:Q",
            color="person:N"
        ).properties(
            title="Beat Consistency"
        )
    )
    chart2 = mo.ui.altair_chart(chart2)
    return (chart2,)


@app.cell
def _(chart, chart2, mo):
    mo.vstack([chart, chart2])
    return


@app.cell
def _(here, pl):
    bcdf = pl.read_parquet(here() / "results_base/cross_beat_consistency.parquet")
    bcdf = bcdf.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)
    )
    bcdf
    return (bcdf,)


@app.cell
def _(bcdf, mo):
    bcfiltered = mo.sql(
        f"""
        select * from bcdf where metric == 'raw_vs_imf1' and person1 == 'person1'
        """
    )
    return (bcfiltered,)


@app.cell
def _(alt, bcfiltered, mo):
    chart3 = (
        alt.Chart(bcfiltered)
        .mark_line()
        .encode(
            x="chunk",
            y="Value",
        )
    )
    chart3 = mo.ui.altair_chart(chart3)
    return (chart3,)


@app.cell
def _(chart3):
    chart3
    return


if __name__ == "__main__":
    app.run()
