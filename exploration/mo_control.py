

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import altair as alt
    import seaborn as sns
    import polars as pl
    from pyprojroot import here
    return alt, here, mo, pl


@app.cell
def _(alt):
    baseline_color = "#74BDCB"
    stick_color = "#FFA384"
    alt.theme.enable("ggplot2")
    return baseline_color, stick_color


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# ASAQ""")
    return


@app.cell(hide_code=True)
def _(here, pl):
    asaq = pl.read_csv(here() / "res_control/ASAQ_baseline_groupNotIntervened_longFormat.csv")
    asaq
    return (asaq,)


@app.cell
def _(asaq, mo):
    _df = mo.sql(
        f"""
        select distinct "condition" from asaq
        """
    )
    return


@app.cell(hide_code=True)
def _(asaq, pl):
    asaq_base = asaq.filter((pl.col("ordered_rating") != "NA") & (pl.col("condition") == "baseline"))
    asaq_base
    return (asaq_base,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ASAQ Base Video Histogram""")
    return


@app.cell
def _(alt, asaq_base, baseline_color):
    alt.theme.enable("ggplot2")
    asaq_base_chart = (
        alt.Chart(asaq_base, width=400)
        .mark_bar(color=baseline_color)
        .encode(
            x=alt.X("ordered_rating:O", title="Rating"),
            y=alt.Y("count():Q", title="Count"),
        )
        .properties(title="ASAQ Baseline on Videos")
    )
    return (asaq_base_chart,)


@app.cell
def _(asaq, pl):
    asaq_gni = asaq.filter((pl.col("ordered_rating") != "NA") & (pl.col("condition") == "gni"))
    asaq_gni
    return (asaq_gni,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ASAQ GNI Histogram""")
    return


@app.cell
def _(alt, asaq_gni, stick_color):
    asaq_gni_chart = (
        alt.Chart(asaq_gni, width=400)
        .mark_bar(color=stick_color)
        .encode(
            x=alt.X("ordered_rating:O", title="Rating"),
            y=alt.Y("count():Q", title="Count"),
        )
        .properties(title="ASAQ Non-Intervened")
    )
    return (asaq_gni_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Saving and Display""")
    return


@app.cell
def _(asaq_base_chart, asaq_gni_chart, here):
    asaq_base_chart.save(here() / "res_control/asaq_base_chart.pdf")
    asaq_gni_chart.save(here() / "res_control/asaq_gni_chart.pdf")
    return


@app.cell
def _(asaq_base_chart, asaq_gni_chart, mo):
    mo.hstack([asaq_base_chart, asaq_gni_chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# PCQ""")
    return


@app.cell
def _(here, pl):
    pcq = pl.read_csv(here() / "res_control/PCQ_baseline_groupNotIntervened_longFormat.csv")
    pcq
    return (pcq,)


@app.cell
def _(pcq, pl):
    pcq_base = pcq.filter((pl.col("ordered_rating").is_not_null()) & (pl.col("condition").eq("baseline")))
    # pcq_base = pcq.filter(pl.col("ordered_rating").is_not_null())
    pcq_base
    return (pcq_base,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## PCQ Base Video Histogram""")
    return


@app.cell
def _(alt, baseline_color, pcq_base):
    alt.theme.enable("ggplot2")
    pcq_base_chart = (
        alt.Chart(pcq_base, width=400)
        .mark_bar(color=baseline_color)
        .encode(
            x=alt.X("ordered_rating:O", title="Rating"),
            y=alt.Y("count():Q", title="Count"),
            # color=alt.Color("condition:N", title="Condition"),
        )
        .properties(title="PCQ Baseline on Videos")
    )
    return (pcq_base_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## PCQ GNI Histogram""")
    return


@app.cell
def _(alt, pcq, pl, stick_color):
    pcq_gni = pcq.filter((pl.col("ordered_rating").is_not_null()) & (pl.col("condition").eq("gni")))
    pcq_gni_chart = (
        alt.Chart(pcq_gni, width=400)
        .mark_bar(color=stick_color)
        .encode(
            x=alt.X("ordered_rating:O", title="Rating"),
            y=alt.Y("count():Q", title="Count"),
        )
        .properties(title="PCQ Non-Intervened")
    )
    return (pcq_gni_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Saving and Display""")
    return


@app.cell
def _(here, pcq_base_chart, pcq_gni_chart):
    pcq_base_chart.save(here() / "res_control/pcq_base_chart.pdf")
    pcq_gni_chart.save(here() / "res_control/pcq_gni_chart.pdf")
    return


@app.cell
def _(mo, pcq_base_chart, pcq_gni_chart):
    mo.hstack([pcq_base_chart, pcq_gni_chart])
    return


if __name__ == "__main__":
    app.run()
