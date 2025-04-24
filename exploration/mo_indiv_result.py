

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium", app_title="Individual Results")

with app.setup:
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pyprojroot import here
    import altair as alt
    import pandas as pd
    from typing import List
    import numpy as np
    import statsmodels.formula.api as smf
    from scipy import stats
    import pingouin as pg


@app.cell
def _():
    baseline_color = "#74BDCB"
    intervened_color = "#FFA384"
    return baseline_color, intervened_color


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Silence file loading""")
    return


@app.cell(hide_code=True)
def _():
    silence_file_path = mo.ui.file_browser(
        initial_path=here(), filetypes=[".parquet"], selection_mode="file", label="Select Silence File", multiple=False
    )
    silence_file_path
    return (silence_file_path,)


@app.cell(hide_code=True)
def _(silence_file_path):
    silence_df = pl.read_parquet(silence_file_path.path(index=0))
    person_mapping = {
        "person1": "a",
        "person2": "c",
        "person3": "b",
        "person4": "j",
        "person5": "l",
    }
    return person_mapping, silence_df


@app.cell
def _(silence_df):
    alt.Chart(silence_df.with_columns(pl.col("is_silent").not_())).mark_bar().encode(
        x=alt.X("person:N"), y=alt.Y("sum(is_silent):Q")
    ).properties(title="Silence Count")
    return


@app.cell(hide_code=True)
def _(silence_df):
    alt.theme.enable("default")
    silence_base = alt.Chart(
        silence_df.with_columns(
            pl.col("is_silent").not_(),
            pl.col("person").str.to_uppercase(),
        )
    ).encode(
        color=alt.Color("person:N", title="Person"),
        theta=alt.Theta("sum(is_silent):Q").stack(True),
    )
    silence_pie = silence_base.mark_arc(outerRadius=120)
    silence_text = silence_base.mark_text(radius=140, size=20).encode(text="person:N")
    pie = (silence_pie + silence_text).properties(title="Relative Amount of Time Spoken")
    pie.save(here() / "results" / "plots" / "silence_pie_chart.pdf")
    pie
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Individual - Joint Level Recurrence Analysis""")
    return


@app.cell(hide_code=True)
def _():
    base_ijr_path = mo.ui.file_browser(
        initial_path=here() / "results/" / "results_base",
        filetypes=[".parquet"],
        selection_mode="file",
        label="Select RQA Base File",
    )
    intervened_ijr_path = mo.ui.file_browser(
        initial_path=here() / "results/", filetypes=[".parquet"], selection_mode="file", label="Select RQA Intervened File"
    )
    mo.hstack([base_ijr_path, intervened_ijr_path], align="center")
    return base_ijr_path, intervened_ijr_path


@app.cell(hide_code=True)
def _(base_ijr_path, intervened_ijr_path):
    ijr_base = pl.read_parquet(base_ijr_path.path(index=0))
    ijr_int = pl.read_parquet(intervened_ijr_path.path(index=0))

    ijr_base = process_ijr(ijr_base, "Recurrence Radius", ["LeftHand"])
    ijr_int = process_ijr(ijr_int, "Recurrence Radius", ["LeftHand"])
    return ijr_base, ijr_int


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Error Bar Plots""")
    return


@app.cell(hide_code=True)
def _(ijr_base, ijr_int, person_mapping, rqa_metric_choice):
    ijr_joined = ijr_base.join(ijr_int, how="inner", on=["chunk", "person", "joint", "Metric"], suffix="_intervened")
    ijr_joined_filtered = (
        ijr_joined.filter(pl.col("Metric").eq(rqa_metric_choice.value))
        .unpivot(
            index=["Metric", "person", "chunk", "joint"],
            on=["Value", "Value_intervened"],
            variable_name="condition",
            value_name=rqa_metric_choice.value,
        )
        .with_columns(
            pl.col("condition").replace({"Value": "Base", "Value_intervened": "Intervened"}),
            pl.col("person").replace(person_mapping).str.to_uppercase(),
        )
        .filter(pl.col(f"{rqa_metric_choice.value}") > 0)
    )
    ijr_joined_filtered
    return ijr_joined, ijr_joined_filtered


@app.cell(hide_code=True)
def _(ijr_base):
    rqa_metric_choice = mo.ui.dropdown.from_series(
        ijr_base["Metric"],
        label="Select RQA Metric",
    )
    rqa_metric_choice
    return (rqa_metric_choice,)


@app.cell(hide_code=True)
def _(
    baseline_color,
    ijr_joined_filtered,
    intervened_color,
    rqa_metric_choice,
):
    alt.theme.enable("ggplot2")
    ijr_agg = ijr_joined_filtered.group_by(["person", "condition"]).agg(
        pl.col(f"{rqa_metric_choice.value}").mean().alias("mean"),
        pl.col(f"{rqa_metric_choice.value}").std().alias("std"),
    )
    base_ijr = alt.Chart(ijr_agg).encode(
        x=alt.X("person:N", title="Persons"),
        y=alt.Y(
            f"mean:Q",
            title=f"{rqa_metric_choice.value.replace('_', ' ')}",
        ),
        color=alt.Color("condition:N", title="Condition").scale(range=[baseline_color, intervened_color]),
        shape=alt.Shape("condition:N", title="Condition", legend=None),
        strokeDash=alt.StrokeDash("condition:N", title="Condition", legend=None),
    )

    points_ijr = base_ijr.mark_point(filled=True, size=60)
    lines_ijr = base_ijr.mark_line(point=False)
    # For error bars, use separate chart but keep consistent encoding
    error_bars_ijr = (
        alt.Chart(ijr_agg)
        .mark_errorbar(clip=True, ticks=True, size=25, thickness=3)
        .encode(
            x="person:N",
            y=alt.Y(f"mean:Q", title="").scale(zero=False),
            yError=alt.YError(f"std:Q"),
            color=alt.Color("condition:N", title="Condition"),
        )
    )
    chart_ijr = (
        alt.layer(lines_ijr, points_ijr, error_bars_ijr)
        .resolve_scale(y="shared")
        .properties(
            width=500, height=400, title=f"{rqa_metric_choice.value.replace('_', ' ')} - No Intervention vs Intervention"
        )
    )

    chart_ijr.save(here() / "results" / "plots" / f"{rqa_metric_choice.value.replace('_', '')}_indiv_error_bar_plot.pdf")

    mo.ui.altair_chart(chart_ijr)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Box-Plots""")
    return


@app.cell(hide_code=True)
def _(ijr_base, ijr_int, rqa_metric_choice):
    alt.theme.enable("default")
    chart_base = (
        alt.Chart(ijr_base.filter(pl.col("Metric").eq(rqa_metric_choice.value)).filter(pl.col("Value") > 0), width=400)
        .mark_boxplot()
        .encode(x="person", y="Value", color="person:N")
        .properties(title=f"{rqa_metric_choice.value.replace('_', ' ')} - No Intervention")
    )
    chart_int = (
        alt.Chart(ijr_int.filter(pl.col("Metric").eq(rqa_metric_choice.value)).filter(pl.col("Value") > 0), width=400)
        .mark_boxplot()
        .encode(x="person:N", y="Value:Q", color="person:N")
        .properties(title=f"{rqa_metric_choice.value.replace('_', ' ')} - Intervened")
    )
    chart_base = mo.ui.altair_chart(chart_base)
    chart_int = mo.ui.altair_chart(chart_int)
    return chart_base, chart_int


@app.cell(hide_code=True)
def _(chart_base, chart_int):
    mo.vstack([chart_base, chart_int], align="center")
    return


@app.cell(hide_code=True)
def _(baseline_color, intervened_color, rqa_metric_choice):
    def error_bar_plotter(df: pl.DataFrame, metric: str, domain: List[float] = [0.7, 1]):
        alt.theme.enable("ggplot2")
        df = df.with_columns(pl.col("person").str.to_titlecase())
        base = alt.Chart(df).encode(
            x=alt.X("person:N", title="Persons"),
            y=alt.Y(
                f"mean({metric}):Q",
                title=f"Average {metric.replace('_', ' ')}",
            ),
            color=alt.Color("condition:N", title="Condition").scale(range=[baseline_color, intervened_color]),
            # shape=alt.Shape("condition:N", title="Condition"),
            strokeDash=alt.StrokeDash("condition:N", title="Condition", legend=None),
        )

        points = base.mark_point(filled=True, size=50)
        lines = base.mark_line(point=False)
        err_df = df.group_by(["person", "condition"]).agg(
            pl.col(f"{rqa_metric_choice.value}").mean().alias("mean"),
            pl.col(f"{rqa_metric_choice.value}").std().alias("std"),
        )
        # For error bars, use separate chart but keep consistent encoding
        error_bars = (
            alt.Chart(err_df)
            .mark_errorbar(clip=True, ticks=True, size=10, thickness=3)
            .encode(
                x="person:N",
                y=alt.Y(f"mean:Q", title="").scale(zero=False),
                yError=alt.YError(f"std:Q"),
                color=alt.Color("condition:N", title="Condition"),
            )
        )

        chart = alt.layer(lines, points, error_bars).resolve_scale(y="shared").properties(width=500, height=400)
        return chart
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## LMEM""")
    return


@app.cell(hide_code=True)
def _(rqa_metric_choice):
    rqa_metric_choice
    return


@app.cell
def _(ijr_joined_filtered, rqa_metric_choice):
    tmp = ijr_joined_filtered.filter(pl.col("joint").is_in(["LeftHand", "RightHand", "LeftArm"])).to_pandas()
    rqa_lmem_model = smf.mixedlm(
        f"{rqa_metric_choice.value} ~  condition * joint",
        tmp,
        groups=tmp["chunk"],
        re_formula="~1",
        # vc_formula={"pair": "0 + C(pair)"},
    )
    return (rqa_lmem_model,)


@app.cell
def _(rqa_lmem_model):
    rqa_lmem_res = rqa_lmem_model.fit(reml=False)
    return (rqa_lmem_res,)


@app.cell
def _(rqa_lmem_res, rqa_metric_choice):
    with open(here() / "results" / "latex" / f"{rqa_metric_choice.value}_lmem.tex", "w") as f:
        f.write(rqa_lmem_res.summary().as_latex())
    mo.md(rqa_lmem_res.summary().as_html())
    return


@app.cell
def _(ijr_joined_filtered):
    ijr_joined_filtered
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Wilcoxon""")
    return


@app.cell
def _(ijr_joined, rqa_metric_choice):
    wilcx_ijr = ijr_joined.filter(pl.col("Metric") == rqa_metric_choice.value).with_columns(
        (pl.col("Value_intervened") - pl.col("Value")).alias("deltas"),
    )
    return (wilcx_ijr,)


@app.cell(hide_code=True)
def _(wilcx_ijr):
    pg.wilcoxon(wilcx_ijr.select("deltas").to_numpy())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Self Beat Consistency""")
    return


@app.cell(hide_code=True)
def _():
    base_bc_path = mo.ui.file_browser(
        initial_path=here() / "results/" / "results_base",
        filetypes=[".parquet"],
        selection_mode="file",
        label="Select BC Base File",
        multiple=False,
    )
    intervened_bc_path = mo.ui.file_browser(
        initial_path=here() / "results/",
        filetypes=[".parquet"],
        selection_mode="file",
        label="Select BC Intervened File",
        multiple=False,
    )
    mo.hstack([base_bc_path, intervened_bc_path], align="center")
    return base_bc_path, intervened_bc_path


@app.cell(hide_code=True)
def _(base_bc_path, intervened_bc_path):
    bdf = pl.read_parquet(base_bc_path.path(index=0))
    bdf_int = pl.read_parquet(intervened_bc_path.path(index=0))
    bdf, bdf_int = list(map(lambda x: x.filter(pl.col("Metric").ne("fail")), [bdf, bdf_int]))
    bfil_base = process_bc(bdf, ["raw_vs_imf2", "raw_vs_raw", "raw_vs_imf1"])
    bfil_int = process_bc(bdf_int, ["raw_vs_imf2", "raw_vs_raw", "raw_vs_imf1"])
    return bfil_base, bfil_int


@app.cell(hide_code=True)
def _(bfil_base, bfil_int, person_mapping, silence_df):
    bfil_joined = bfil_base.join(bfil_int, how="inner", on=["chunk", "person", "Metric"], suffix="_intervened")
    bfil_joined = bfil_joined.unpivot(
        index=["Metric", "person", "chunk"],
        on=["Value", "Value_intervened"],
        variable_name="condition",
        value_name="Value",
    ).with_columns(
        pl.col("condition").replace({"Value": "Base", "Value_intervened": "Intervened"}),
    )
    # bfil_joined = bfil_joined.filter(pl.col("Value") > 0)
    bfil_joined = (
        bfil_joined.with_columns(pl.col("person").replace(person_mapping))
        .join(silence_df, how="left", on=["chunk", "person"])
        .with_columns(pl.col("is_silent").fill_null(False))
    )
    bfil_joined
    return (bfil_joined,)


@app.cell
def _(bfil_joined):
    bfil_joined.group_by(["condition"]).agg(
        pl.col("Value").median(),
    )
    return


@app.cell(hide_code=True)
def _(bfil_joined):
    bfil_agg = (
        bfil_joined.filter(pl.col("is_silent") == False)
        .group_by(["person", "condition"])
        .agg(
            pl.col("Value").mean().alias("mean"),
            pl.col("Value").std().alias("std"),
        )
    )
    bfil_agg
    return (bfil_agg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Error Bar Plots""")
    return


@app.cell(hide_code=True)
def _(baseline_color, bfil_agg, intervened_bc_path, intervened_color):
    alt.theme.enable("ggplot2")
    df = bfil_agg.with_columns(pl.col("person").str.to_titlecase())
    base = alt.Chart(df).encode(
        x=alt.X("person:N", title="Persons"),
        y=alt.Y(
            f"mean:Q",
            title=f"Beat Consistency",
        ),
        color=alt.Color("condition:N", title="Condition").scale(range=[baseline_color, intervened_color]),
        shape=alt.Shape("condition:N", title="Condition", legend=None),
        strokeDash=alt.StrokeDash("condition:N", title="Condition", legend=None),
    )

    points = base.mark_point(filled=True, size=60)
    lines = base.mark_line(point=False)
    # For error bars, use separate chart but keep consistent encoding
    error_bars = (
        alt.Chart(df)
        .mark_errorbar(clip=True, ticks=True, size=25, thickness=3)
        .encode(
            x="person:N",
            y=alt.Y(f"mean:Q", title="").scale(zero=False),
            yError=alt.YError(f"std:Q"),
            color=alt.Color("condition:N", title="Condition"),
        )
    )
    vs_title = "Audio Delay" if "delay" in intervened_bc_path.path(index=0).parent.name else "Visual Delay"
    chart_bc = (
        alt.layer(lines, points, error_bars)
        .resolve_scale(y="shared")
        .properties(width=500, height=400, title=f"Beat Consistency - Individual (Self)").configure_title(fontSize=18)
    )
    chart_bc.save(here() / "results" / "plots" / f"bc_{vs_title.replace(' ', '_')}_error_bar_plot.pdf")
    mo.ui.altair_chart(chart_bc)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## T-Test and Wilcoxon""")
    return


@app.cell
def _(bfil_joined):
    pivot_bfil_joined = (
        bfil_joined.pivot(
            index=["person", "chunk", "Metric", "is_silent"],
            on=["condition"],
        )
        .with_columns(
            (pl.col("Intervened") - pl.col("Base")).alias("deltas"),
        )
    )
    return (pivot_bfil_joined,)


@app.cell
def _(pivot_bfil_joined):
    pg.qqplot(x=pivot_bfil_joined["deltas"])
    return


@app.cell
def _(pivot_bfil_joined):
    pg.ttest(
        pivot_bfil_joined["Base"].to_numpy(),
        pivot_bfil_joined["Intervened"].to_numpy(),
        paired=True,
    )
    return


@app.cell
def _(pivot_bfil_joined):
    pg.wilcoxon(pivot_bfil_joined["Base"], pivot_bfil_joined["Intervened"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Helper Functions""")
    return


@app.function(hide_code=True)
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


@app.function(hide_code=True)
def process_bc(bdf: pl.DataFrame, metrics: List[str]):
    bdf = bdf.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
    bdf = bdf.filter(pl.col("Metric").is_in(metrics))
    return bdf


@app.function(hide_code=True)
def process_ijr(df: pl.DataFrame, metric: str, joint: List[str]):
    df = df.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
    df = df.filter(pl.col("Value").is_not_nan())
    df = df.with_columns(pl.col("Metric").str.replace(" ", "_"))
    return df


if __name__ == "__main__":
    app.run()
