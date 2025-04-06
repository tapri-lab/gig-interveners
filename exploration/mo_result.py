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
def _(motion_damped, pl, results_base):
    ijr_base = pl.read_parquet(results_base / "indiv_joint_recurrence.parquet")
    ijr_int = pl.read_parquet(motion_damped / "indiv_joint_recurrence.parquet")
    out = map(lambda x: x.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)
    ), [ijr_base, ijr_int])
    ijr_base, ijr_int = list(out)
    return ijr_base, ijr_int, out


@app.cell
def _(ijr_base, mo):
    filtered = mo.sql(
        f"""
        SELECT * FROM ijr_base where joint == 'LeftHand' and Metric == 'Recurrence Rate'
        """
    )
    return (filtered,)


@app.cell
def _(ijr_int, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM ijr_int where joint == 'LeftHand' and Metric == 'Recurrence Rate'
        """
    )
    return


@app.cell
def _(alt, chart, filtered, mo):
    chart_base = (
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
    return chart, chart_base


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
def _():
    return


if __name__ == "__main__":
    app.run()
