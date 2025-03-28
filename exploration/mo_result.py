import marimo

__generated_with = "0.11.30"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pyprojroot import here
    import altair as alt
    return alt, here, mo, pl, plt, sns


@app.cell
def _(here, pl):
    df = pl.read_parquet(here() / "results/indiv_joint_recurrence.parquet")
    return (df,)


@app.cell
def _(df, pl):
    xs = df.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)
    )
    xs
    return (xs,)


@app.cell
def _(mo, xs):
    filtered = mo.sql(
        f"""
        SELECT * FROM xs where person == 'person1' and joint == 'LeftHand' and Metric == 'Recurrence Rate'
        """
    )
    return (filtered,)


@app.cell
def _(alt, filtered, mo):
    chart = (
        alt.Chart(filtered)
        .mark_point()
        .encode(
            x="chunk",
            y="Value",
        )
    )
    chart = mo.ui.altair_chart(chart)
    return (chart,)


@app.cell
def _(chart, mo):
    mo.vstack([chart, chart.value.head()])
    return


@app.cell
def _(here, pl):
    bdf = pl.read_parquet(here() / "results/beat_consistency.parquet")
    bdf = bdf.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)
    )
    bdf
    return (bdf,)


@app.cell
def _(bdf, mo):
    bfiltered = mo.sql(
        f"""
        select * from bdf where metric == 'raw_vs_imf1' and person == 'person1'
        """
    )
    return (bfiltered,)


@app.cell
def _(alt, bfiltered, mo):
    chart2 = (
        alt.Chart(bfiltered)
        .mark_bar()
        .encode(
            x="chunk",
            y="Value",
        )
    )
    chart2 = mo.ui.altair_chart(chart2)
    return (chart2,)


@app.cell
def _(chart2, mo):
    mo.vstack([chart2, chart2.value.head()])
    return


@app.cell
def _(here, pl):
    bcdf = pl.read_parquet(here() / "results/cross_beat_consistency.parquet")
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
