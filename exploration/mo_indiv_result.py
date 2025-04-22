

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")

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

    alt.data_transformers.enable("vegafusion")


@app.cell
def _():
    baseline_color = "#74BDCB"
    intervened_color = "#FFA384"
    return baseline_color, intervened_color


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


@app.cell
def _(base_ijr_path, intervened_ijr_path):
    ijr_base = pl.read_parquet(base_ijr_path.path(index=0))
    ijr_int = pl.read_parquet(intervened_ijr_path.path(index=0))

    ijr_base = process_ijr(ijr_base, "Recurrence Radius", ["LeftHand"])
    ijr_int = process_ijr(ijr_int, "Recurrence Radius", ["LeftHand"])
    return ijr_base, ijr_int


@app.cell(hide_code=True)
def _(ijr_base):
    rqa_metric_choice = mo.ui.dropdown.from_series(
        ijr_base["Metric"],
        label="Select RQA Metric",
    )
    rqa_metric_choice
    return (rqa_metric_choice,)


@app.cell(hide_code=True)
def _(ijr_base, ijr_int, rqa_metric_choice):
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
        )
        .filter(pl.col(f"{rqa_metric_choice.value}") > 0)
    )
    ijr_joined_filtered
    return (ijr_joined_filtered,)


@app.cell
def _(ijr_joined_filtered, rqa_metric_choice):
    ijr_joined_filtered.group_by(["person", "condition"]).agg(
        pl.col(f"{rqa_metric_choice.value}").mean().alias("mean"),
        pl.col(f"{rqa_metric_choice.value}").std().alias("std"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Plotting Recurrence Radius Box-Plots""")
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

        points = base.mark_point(filled=True)
        lines = base.mark_line(point=False)
        err_df = df.group_by(["person", "condition"]).agg(
            pl.col(f"{rqa_metric_choice.value}").mean().alias("mean"),
            pl.col(f"{rqa_metric_choice.value}").std().alias("std"),
        )
        # For error bars, use separate chart but keep consistent encoding
        error_bars = (
            alt.Chart(err_df)
            .mark_errorbar(clip=True, ticks=True, size=10, thickness=2)
            .encode(
                x="person:N",
                y=alt.Y(f"mean:Q", title="").scale(zero=False),
                yError=alt.YError(f"std:Q"),
                color=alt.Color("condition:N", title="Condition"),
            )
        )

        chart = alt.layer(lines, points, error_bars).resolve_scale(y="shared").properties(width=500, height=400)
        return chart
    return (error_bar_plotter,)


@app.cell(hide_code=True)
def _(rqa_metric_choice):
    rqa_metric_choice
    return


@app.cell(hide_code=True)
def _(error_bar_plotter, ijr_joined_filtered, rqa_metric_choice):
    mo.ui.altair_chart(
        error_bar_plotter(ijr_joined_filtered, rqa_metric_choice.value, domain=[0, 15]).properties(
            title=f"{rqa_metric_choice.value.replace('_', ' ')} - No Intervention vs Intervention"
        )
    )
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
    rqa_lmem_model = smf.mixedlm(
        f"{rqa_metric_choice.value} ~ condition * joint",
        ijr_joined_filtered.to_pandas(),
        groups=ijr_joined_filtered.to_pandas()["chunk"],
        re_formula="~1",
        # vc_formula={"pair": "0 + C(pair)"},
    )
    return (rqa_lmem_model,)


@app.cell
def _(rqa_lmem_model):
    rqa_lmem_res = rqa_lmem_model.fit(reml=False)
    return (rqa_lmem_res,)


@app.cell
def _(rqa_lmem_res):
    mo.md(rqa_lmem_res.summary().as_html())
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
    )
    intervened_bc_path = mo.ui.file_browser(
        initial_path=here() / "results/", filetypes=[".parquet"], selection_mode="file", label="Select BC Intervened File"
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
def _(bfil_base, bfil_int):
    bfil_joined = bfil_base.join(bfil_int, how="inner", on=["chunk", "person", "Metric"], suffix="_intervened")
    bfil_joined = bfil_joined.unpivot(
        index=["Metric", "person", "chunk"],
        on=["Value", "Value_intervened"],
        variable_name="condition",
        value_name="Value",
    ).with_columns(
        pl.col("condition").replace({"Value": "Base", "Value_intervened": "Intervened"}),
    )
    bfil_joined
    return (bfil_joined,)


@app.cell(hide_code=True)
def _(bfil_joined):
    bfil_agg = bfil_joined.group_by(["person", "condition"]).agg(
        pl.col("Value").mean().alias("mean"),
        pl.col("Value").std().alias("std"),
    )
    bfil_agg.head()
    return (bfil_agg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Error Bar Plots""")
    return


@app.cell(hide_code=True)
def _(baseline_color, bfil_agg, intervened_color):
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

    points = base.mark_point(filled=True, size=50)
    lines = base.mark_line(point=False)
    # For error bars, use separate chart but keep consistent encoding
    error_bars = (
        alt.Chart(df)
        .mark_errorbar(clip=True, ticks=True, size=10, thickness=2)
        .encode(
            x="person:N",
            y=alt.Y(f"mean:Q", title="").scale(zero=False),
            yError=alt.YError(f"std:Q"),
            color=alt.Color("condition:N", title="Condition"),
        )
    )

    chart_bc = (
        alt.layer(lines, points, error_bars)
        .resolve_scale(y="shared")
        .properties(width=500, height=400, title="Beat Consistency - No Intervention vs Dampened Movement")
    )
    chart_bc
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## T-Test and Wilcoxon""")
    return


@app.cell
def _(bfil_joined):
    pivot_bfil = bfil_joined.pivot(
        index=["person", "chunk", "Metric"],
        on=["condition"],
    ).with_columns(
        (pl.col("Intervened") - pl.col("Base")).alias("deltas"),
    )
    pivot_bfil
    return (pivot_bfil,)


@app.cell
def _(bfil_joined):
    sns.set_theme()
    stats.probplot(
        bfil_joined.pivot(
            index=["person", "chunk", "Metric"],
            on=["condition"],
        )
        .with_columns(
            (pl.col("Intervened") - pl.col("Base")).alias("deltas"),
        )
        .select(pl.col("deltas"))
        .to_numpy()
        .squeeze(),
        dist="norm",
        plot=plt,
    )
    plt.show()
    return


@app.cell
def _(pivot_bfil):
    alt.theme.enable("default")
    b = (
        alt.Chart(pivot_bfil)
        .transform_quantile("deltas", step=0.01, as_=["p", "v"])
        .transform_calculate(uniform="quantileUniform(datum.p)", normal="quantileNormal(datum.p)")
        .mark_point()
        .encode(alt.Y("v:Q"))
    )

    mo.ui.altair_chart(b.encode(x="uniform:Q") | b.encode(x="normal:Q"))
    return


@app.cell
def _(pivot_bfil):
    rt = stats.ttest_rel(
        pivot_bfil.select("Base").to_numpy(),
        pivot_bfil.select("Intervened").to_numpy(),
    )
    mo.md(rf"""
    | Type  | Value  |
    |---|---|
    | T-statistic  | {rt.statistic.item()}  |
    | p-value  |  {rt.pvalue.item()} |
    """)
    return


@app.cell
def _(pivot_bfil):
    rw = stats.wilcoxon(pivot_bfil.select("deltas").to_numpy())
    mo.md(rf"""
    | Type  | Value  |
    |---|---|
    | Wilcoxon-statistic  | {rw.statistic.item()}  |
    | p-value  |  {rw.pvalue.item()} |
    """)
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
