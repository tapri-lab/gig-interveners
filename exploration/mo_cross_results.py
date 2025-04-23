

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
    from scipy.stats import ttest_rel, wilcoxon, shapiro
    import scipy.stats as stats
    import statsmodels.formula.api as smf

    alt.data_transformers.enable("vegafusion")


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Silence File Loading""")
    return


@app.cell(hide_code=True)
def _():
    silence_file_path = mo.ui.file_browser(
        initial_path=here(), filetypes=[".parquet"], selection_mode="file", label="Select Silence File", multiple=False
    )
    silence_file_path
    return (silence_file_path,)


@app.cell
def _(silence_file_path):
    person_mapping = {
        "person1": "a",
        "person2": "c",
        "person3": "b",
        "person4": "j",
        "person5": "l",
    }
    session1_silence = pl.read_parquet(silence_file_path.path(index=0))
    return person_mapping, session1_silence


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Colors""")
    return


@app.cell
def _():
    baseline_color = "#74BDCB"
    intervened_color = "#FFA384"
    return baseline_color, intervened_color


@app.cell(hide_code=True)
def _():
    mo.md(r"""# CRQA - Joints""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Helper Functions""")
    return


@app.function(hide_code=True)
def crqa_plotter(df: pl.DataFrame, n: int, metric: str):
    df = df.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
    df = df.filter(pl.col("Value").is_not_nan())
    df = df.filter(pl.col("Metric").eq(metric))
    # df = df.with_columns((pl.col("Value") - pl.col("Value").min()) / (pl.col("Value").max() - pl.col("Value").min()))
    df = (
        df.filter(pl.col("Metric").eq(metric))
        .group_by(["person1", "person2"])
        .agg(pl.col("Value").mean())
        .unique(["person1", "person2"])
    )
    pairwise_crqa = df_to_pairwise_mat(df, n)
    axlabels = [f"Person {i + 1}" for i in range(n)]
    heat_plot = sns.heatmap(
        pairwise_crqa,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": f"{metric}"},
        mask=np.triu(pairwise_crqa),
        xticklabels=axlabels,
        yticklabels=axlabels,
    )
    return heat_plot


@app.function(hide_code=True)
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Base Results""")
    return


@app.cell(hide_code=True)
def _():
    base_crqa_path = mo.ui.file_browser(
        initial_path=here() / "results/results_base/",
        selection_mode="file",
        label="Select the base CRQA file",
        filetypes=[".parquet"],
    )
    intervened_crqa_path = mo.ui.file_browser(
        initial_path=here() / "results/",
        selection_mode="file",
        label="Select the Intervened CRQA file",
        filetypes=[".parquet"],
    )
    mo.hstack([base_crqa_path, intervened_crqa_path])
    return base_crqa_path, intervened_crqa_path


@app.cell
def _(base_crqa_path):
    crdf_base = pl.read_parquet(base_crqa_path.path(index=0))
    crdf_base = crdf_base.with_columns(
        pl.col("Value").cast(pl.Float32),
        pl.col("chunk").cast(pl.Int32),
        pl.col("Metric").str.replace(" ", "_"),
    ).filter(
        pl.col("Value").is_not_nan(),
    )
    crdf_base
    return (crdf_base,)


@app.cell(hide_code=True)
def _(crdf_base):
    crqa_metric_choice = mo.ui.dropdown(options=crdf_base["Metric"].unique().to_list(), label="Select Metric")
    crqa_metric_choice
    return (crqa_metric_choice,)


@app.cell
def _(crdf_base, crqa_metric_choice):
    crdf_base.filter(pl.col("Value").is_not_nan()).filter(pl.col("Metric").eq(crqa_metric_choice.value))
    return


@app.cell
def _(crdf_base, crqa_metric_choice):
    sns.set_theme("paper")
    # sns.set_style("whitegrid")
    plt.style.use("ggplot")
    heat_plot = crqa_plotter(crdf_base, 5, crqa_metric_choice.value)
    fig = heat_plot.get_figure()
    fig.savefig(here() / "results/results_base/crqa_pairwise_radius_base.png", dpi=300, bbox_inches="tight")
    fig
    return


@app.cell
def _(crqa_metric_choice, intervened_crqa_path):
    intervened_crqa = pl.read_parquet(intervened_crqa_path.path(index=0))
    intervened_crqa = intervened_crqa.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32), pl.col("Metric").str.replace(" ", "_")
    ).filter(pl.col("Value").is_not_nan())
    crqa_plotter(intervened_crqa, 5, crqa_metric_choice.value)
    return (intervened_crqa,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Normality Tests""")
    return


@app.cell
def _(crdf_base, crqa_metric_choice, intervened_crqa):
    joined = (
        crdf_base.join(intervened_crqa, how="inner", on=["person1", "person2", "chunk", "joint", "Metric"])
        .filter(pl.col("Metric").eq(crqa_metric_choice.value))
        .with_columns((pl.col("Value_right") - pl.col("Value")).alias("deltas"))
    )
    joined = joined.with_columns(
        [
            pl.min_horizontal(["person1", "person2"]).alias("person1"),
            pl.max_horizontal(["person1", "person2"]).alias("person2"),
        ]
    ).unique(subset=["Metric", "person1", "person2", "joint", "chunk"], keep="first")
    joined = joined.with_columns(
        pl.concat_str([pl.col("person1"), pl.col("person2")], separator="_").alias("pair"),
        pl.concat_str([pl.col("person1"), pl.col("person2"), pl.col("chunk")], separator="_").alias("pair_chunk"),
    ).unique(subset=["pair_chunk", "Metric", "joint"], keep="first")
    joined = (
        joined.with_columns(
            pl.concat_list([pl.col("person1"), pl.col("person2")]).alias("person_list"),
        )
        .explode("person_list")
        .rename({"person_list": "person"})
    )
    joined
    return (joined,)


@app.cell(hide_code=True)
def _(crqa_metric_choice):
    crqa_metric_choice
    return


@app.cell(hide_code=True)
def _(joined):
    alt.Chart(joined, width=500).mark_bar().encode(
        x=alt.X("deltas", bin=True),
        y="count()",
    )
    return


@app.cell
def _(joined):
    stats.probplot(
        joined.select(pl.col("deltas")).to_numpy().squeeze(),
        dist="norm",
        plot=plt,
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Error Plots""")
    return


@app.cell(hide_code=True)
def _(crqa_metric_choice):
    crqa_metric_choice
    return


@app.cell(hide_code=True)
def _(crqa_metric_choice, joined, person_mapping):
    joined_unpivot = joined.unpivot(
        index=["Metric", "person", "pair", "chunk", "pair_chunk", "joint"],
        on=["Value", "Value_right"],
        variable_name="condition",
        value_name=crqa_metric_choice.value,
    ).with_columns(
        pl.col("condition").replace({"Value": "Base", "Value_right": "Intervened"}),
    )

    crdf_agg = joined_unpivot.group_by(["person", "condition"]).agg(
        pl.col(crqa_metric_choice.value).mean().alias("mean"),
        pl.col(crqa_metric_choice.value).std().alias("std"),
    )
    crdf_agg = crdf_agg.with_columns(pl.col("person").replace(person_mapping).str.to_titlecase())
    crdf_agg
    return crdf_agg, joined_unpivot


@app.cell(hide_code=True)
def _(baseline_color, crdf_agg, crqa_metric_choice, intervened_color):
    alt.theme.enable("ggplot2")
    base = alt.Chart(crdf_agg).encode(
        x=alt.X("person:N", title="Persons"),
        y=alt.Y(
            f"mean:Q",
            title=f"{crqa_metric_choice.value.replace('_', ' ').title()}",
        ),
        color=alt.Color("condition:N", title="Condition").scale(range=[baseline_color, intervened_color]),
        shape=alt.Shape("condition:N", title="Condition", legend=None),
        strokeDash=alt.StrokeDash("condition:N", title="Condition", legend=None),
    )

    points = base.mark_point(filled=True, size=50)
    lines = base.mark_line(point=False)
    # For error bars, use separate chart but keep consistent encoding
    error_bars = (
        alt.Chart(crdf_agg)
        .mark_errorbar(clip=True, ticks=True, size=10, thickness=2)
        .encode(
            x="person:N",
            y=alt.Y(f"mean:Q", title="").scale(zero=False),
            yError=alt.YError(f"std:Q"),
            color=alt.Color("condition:N", title="Condition"),
        )
    )

    chart_crqa = (
        alt.layer(lines, points, error_bars)
        .resolve_scale(y="shared")
        .properties(width=500, height=400, title=f"CRQA - No Intervention vs Dampened")
    )

    mo.ui.altair_chart(chart_crqa)
    return


@app.cell
def _(crqa_metric_choice, joined_unpivot):
    sns.violinplot(
        data=joined_unpivot.filter(pl.col(crqa_metric_choice.value) > 0),
        x="person",
        y=crqa_metric_choice.value,
        hue="condition",
        palette="Pastel1",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## LMEM""")
    return


@app.cell(hide_code=True)
def _(crqa_metric_choice):
    crqa_metric_choice
    return


@app.cell(hide_code=True)
def _(crqa_metric_choice, joined):
    # crqa_lmem_df = joined.with_columns(
    #     pl.concat_str([pl.col("person1"), pl.col("person2")], separator="_").alias("pair"),
    #     pl.concat_str([pl.col("person1"), pl.col("person2"), pl.col("chunk")], separator="_").alias("pair_chunk"),
    # ).unique(subset=["pair_chunk", "Metric", "joint"], keep="first")
    # crqa_lmem_df.melt(
    #     id_vars=["Metric", "person1", "person2", "pair", "chunk", "joint"],
    #     value_vars=["Value", "Value_right"],
    #     variable_name="condition",
    #     value_name=crqa_metric_choice.value,
    # )
    crqa_lmem_df = joined.unpivot(
        index=["Metric", "person1", "person2", "pair", "chunk", "pair_chunk", "joint"],
        on=["Value", "Value_right"],
        variable_name="condition",
        value_name=crqa_metric_choice.value,
    ).with_columns(
        pl.col("condition").replace({"Value": "raw", "Value_right": "damped"}).cast(pl.Categorical),
        pl.col("joint").cast(pl.Categorical),
        pl.col("pair").cast(pl.Categorical),
        pl.col("pair_chunk").cast(pl.Categorical),
        pl.col("Metric").cast(pl.Categorical),
    )
    crqa_lmem_df
    return (crqa_lmem_df,)


@app.cell
def _(crqa_lmem_df, crqa_metric_choice):
    model = smf.mixedlm(
        f"{crqa_metric_choice.value} ~ condition * joint",
        crqa_lmem_df.to_pandas(),
        groups=crqa_lmem_df.to_pandas()["pair_chunk"],
        re_formula="~1",
        # vc_formula={"pair": "0 + C(pair)"},
    )
    return (model,)


@app.cell
def _(model):
    result = model.fit(reml=False)
    return (result,)


@app.cell
def _(result):
    mo.md(result.summary().as_html())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## T-Test""")
    return


@app.cell
def _(crqa_metric_choice, joined):
    res_crqa = ttest_rel(
        joined.filter(pl.col("Metric").eq(crqa_metric_choice.value)).select(pl.col("Value")).to_numpy(),
        joined.filter(pl.col("Metric").eq(crqa_metric_choice.value)).select(pl.col("Value_right")).to_numpy(),
    )
    mo.md(f"**T-Test p-value:** {res_crqa.pvalue.item()}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Wilcoxon""")
    return


@app.cell
def _(crqa_metric_choice, joined):
    resw = wilcoxon(
        joined.filter(pl.col("Metric").eq(crqa_metric_choice.value)).select(pl.col("deltas")).to_numpy(),
    )
    mo.md(f"**Wilcoxon p-value:** {resw.pvalue.item()}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Cross BC""")
    return


@app.cell(hide_code=True)
def _():
    cross_bc_base_path = mo.ui.file_browser(
        initial_path=here() / "results/results_base/",
        selection_mode="file",
        label="Select the base Cross BC file",
        filetypes=[".parquet"],
        multiple=False,
    )

    cross_bc_int_path = mo.ui.file_browser(
        initial_path=here() / "results/",
        selection_mode="file",
        label="Select the base Intervened Cross BC file",
        filetypes=[".parquet"],
        multiple=False,
    )
    mo.hstack([cross_bc_base_path, cross_bc_int_path])
    return cross_bc_base_path, cross_bc_int_path


@app.cell(hide_code=True)
def _(cross_bc_base_path, cross_bc_int_path):
    cross_bc_base = pl.read_parquet(cross_bc_base_path.path(index=0))
    cross_bc_base = cross_bc_base.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)).filter(
        pl.col("Value").is_not_nan()
    )
    cross_bc_int = pl.read_parquet(cross_bc_int_path.path(index=0))
    cross_bc_int = cross_bc_int.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)).filter(
        pl.col("Value").is_not_nan()
    )
    return cross_bc_base, cross_bc_int


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Joined and Exploded Table""")
    return


@app.cell(hide_code=True)
def _(cross_bc_base, cross_bc_int):
    # Joining the results of the two dataframes for comparison
    bc_joined = (
        cross_bc_base.join(cross_bc_int, how="inner", on=["person1", "person2", "chunk", "Metric"])
        .filter(pl.col("Metric").ne("fail"))
        .with_columns(
            # doing this to ensure only 1 pair remains, although this may not be strictly symmetric
            pl.min_horizontal(["person1", "person2"]).alias("person1"),
            pl.max_horizontal(["person1", "person2"]).alias("person2"),
        )
        .unique(subset=["Metric", "person1", "person2", "chunk"], keep="first")
        .unpivot(
            index=["Metric", "person1", "person2", "chunk"],
            on=["Value", "Value_right"],
            variable_name="condition",
            value_name="Value",
        )
        .with_columns(pl.col("condition").replace({"Value": "Base", "Value_right": "Intervened"}).cast(pl.Categorical))
        .with_columns(
            # adding a new column to get the unique pairs out
            pl.concat_str([pl.col("person1"), pl.col("person2")], separator="_").alias("pair"),
            pl.concat_str([pl.col("person1"), pl.col("person2"), pl.col("chunk")], separator="_").alias("pair_chunk"),
        )
        .unique(subset=["pair_chunk", "Metric", "condition"], keep="first")
        .with_columns(
            # adding a new column to get the unique rows associated to each person, allowing for summary stats calculation
            pl.concat_list([pl.col("person1"), pl.col("person2")]).alias("person_list"),
        )
        .explode("person_list")
        .rename({"person_list": "person"})
    )
    bc_joined
    return (bc_joined,)


@app.cell
def _(bc_joined, person_mapping, session1_silence):
    bc_silence = (
        bc_joined.with_columns(pl.col("person").replace(person_mapping))
        .join(
            session1_silence.with_columns(pl.col("chunk") - 1),
            how="left",
            left_on=["person", "chunk"],
            right_on=["person", "chunk"],
        )
        .with_columns(pl.col("is_silent").fill_null(False))
    )
    bc_silence
    return (bc_silence,)


@app.cell
def _(bc_silence):
    bc_agg = (
        bc_silence.filter(pl.col("is_silent").eq(True))
        .group_by(["person", "condition"])
        .agg(
            pl.col("Value").mean().alias("mean"),
            pl.col("Value").std().alias("std"),
        )
        .with_columns(pl.col("condition").cast(pl.String), pl.col("person").str.to_uppercase())
    )
    bc_agg
    return (bc_agg,)


@app.cell
def _(baseline_color, bc_agg, intervened_color):
    alt.theme.enable("ggplot2")
    bc_base = alt.Chart(bc_agg).encode(
        x=alt.X("person:N", title="Persons"),
        y=alt.Y(
            f"mean:Q",
            title=f"Beat Consistency",
        ),
        color=alt.Color("condition:N", title="Condition").scale(range=[baseline_color, intervened_color]),
        shape=alt.Shape("condition:N", title="Condition", legend=None),
        strokeDash=alt.StrokeDash("condition:N", title="Condition", legend=None),
    )

    bc_points = bc_base.mark_point(filled=True, size=50)
    bc_lines = bc_base.mark_line(point=False)
    # For error bars, use separate chart but keep consistent encoding
    bc_error_bars = (
        alt.Chart(bc_agg)
        .mark_errorbar(clip=True, ticks=True, size=10, thickness=2)
        .encode(
            x="person:N",
            y=alt.Y(f"mean:Q", title="").scale(zero=False),
            yError=alt.YError(f"std:Q"),
            color=alt.Color("condition:N", title="Condition"),
        )
    )

    chart_bc = (
        alt.layer(bc_lines, bc_points, bc_error_bars)
        .resolve_scale(y="shared")
        .properties(width=500, height=400, title=f"Beat Consistency")
    )

    mo.ui.altair_chart(chart_bc)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Cross SDTW""")
    return


@app.function(hide_code=True)
def cross_sdtw_plotter(raw_df: pl.DataFrame, n: int):
    raw_df = raw_df.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32))
    df = raw_df.with_columns(
        ((pl.col("Value") - pl.col("Value").min()) / (pl.col("Value").max() - pl.col("Value").min()))
    )
    df = df.filter(pl.col("Value").is_not_nan())
    axlabels = [f"Person {i + 1}" for i in range(n)]
    df = df.group_by(["person1", "person2"]).agg(pl.col("Value").mean()).unique(["person1", "person2"])
    sdtw_mat = df_to_pairwise_mat(df, 5)
    fig = sns.heatmap(
        sdtw_mat,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "SDTW Normalised"},
        mask=np.triu(sdtw_mat),
        xticklabels=axlabels,
        yticklabels=axlabels,
    )
    return fig


@app.cell(hide_code=True)
def _():
    base_sdtw_path = mo.ui.file_browser(
        initial_path=here() / "results/results_base",
        selection_mode="file",
        label="Select the base SDTW file",
        filetypes=[".parquet"],
        multiple=False,
    )
    intervened_sdtw_path = mo.ui.file_browser(
        initial_path=here() / "results",
        selection_mode="file",
        label="Select the base SDTW file",
        filetypes=[".parquet"],
        multiple=False,
    )
    mo.hstack([base_sdtw_path, intervened_sdtw_path])
    return base_sdtw_path, intervened_sdtw_path


@app.cell
def _(base_sdtw_path):
    sdtw_base = pl.read_parquet(base_sdtw_path.path(index=0))
    sdtw_base = sdtw_base.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32), pl.col("Metric").str.replace(" ", "_")
    ).filter(pl.col("Value").is_not_nan())
    return (sdtw_base,)


@app.cell
def _(intervened_sdtw_path):
    sdtw_intervened = pl.read_parquet(intervened_sdtw_path.path(index=0))
    sdtw_intervened = sdtw_intervened.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32), pl.col("Metric").str.replace(" ", "_")
    ).filter(pl.col("Value").is_not_nan())
    return (sdtw_intervened,)


@app.cell
def _(person_mapping, sdtw_base, sdtw_intervened, session1_silence):
    joined_sdtw = sdtw_base.join(sdtw_intervened, how="inner", on=["person1", "person2", "chunk", "joint", "Metric"])
    joined_sdtw = joined_sdtw.with_columns(
        pl.min_horizontal(["person1", "person2"]).alias("person1"),
        pl.max_horizontal(["person1", "person2"]).alias("person2"),
    ).unique(subset=["Metric", "person1", "person2", "joint", "chunk"], keep="first")
    joined_sdtw = (
        joined_sdtw.unpivot(
            index=["Metric", "person1", "person2", "joint", "chunk"],
            on=["Value", "Value_right"],
            variable_name="condition",
            value_name="Value",
        )
        .with_columns(
            pl.col("condition").replace(
                {"Value": "Base", "Value_right": "Intervened"}
            ),  # adding a new column to get the unique pairs out
            pl.concat_str([pl.col("person1"), pl.col("person2")], separator="_").alias("pair"),
            pl.concat_str([pl.col("person1"), pl.col("person2"), pl.col("chunk")], separator="_").alias("pair_chunk"),
        )
        .unique(subset=["pair_chunk", "Metric", "joint", "condition"], keep="first")
        .with_columns(
            # adding a new column to get the unique rows associated to each person, allowing for summary stats calculation
            pl.concat_list([pl.col("person1"), pl.col("person2")]).alias("person_list"),
        )
        .explode("person_list")
        .rename({"person_list": "person"})
        .with_columns(pl.col("person").replace(person_mapping))
        .join(
            session1_silence,
            how="left",
            on=["person", "chunk"],
        )
        .with_columns(pl.col("is_silent").fill_null(True))
    )
    joined_sdtw = joined_sdtw.with_columns(((pl.col("Value") - pl.col("Value").mean()) / pl.col("Value").std()))
    joined_sdtw
    return (joined_sdtw,)


@app.cell
def _(joined_sdtw):
    joined_sdtw_agg = joined_sdtw.group_by(["person", "condition"]).agg(
        pl.col("Value").mean().alias("mean"), pl.col("Value").std().alias("std")
    ).with_columns(pl.col("person").str.to_uppercase())
    joined_sdtw_agg
    return (joined_sdtw_agg,)


@app.cell
def _(baseline_color, intervened_color, joined_sdtw_agg):
    alt.theme.enable("ggplot2")
    sdtw_base_chart = alt.Chart(joined_sdtw_agg).encode(
        x=alt.X("person:N", title="Persons"),
        y=alt.Y(
            f"mean:Q",
            title=f"SoftDTW",
        ),
        color=alt.Color("condition:N", title="Condition").scale(range=[baseline_color, intervened_color]),
        shape=alt.Shape("condition:N", title="Condition", legend=None),
        strokeDash=alt.StrokeDash("condition:N", title="Condition", legend=None),
    )

    sdtw_points = sdtw_base_chart.mark_point(filled=True, size=50)
    sdtw_lines = sdtw_base_chart.mark_line(point=False)
    # For error bars, use separate chart but keep consistent encoding
    sdtw_error_bars = (
        alt.Chart(joined_sdtw_agg)
        .mark_errorbar(clip=True, ticks=True, size=10, thickness=2)
        .encode(
            x="person:N",
            y=alt.Y(f"mean:Q", title="").scale(zero=False),
            yError=alt.YError(f"std:Q"),
            color=alt.Color("condition:N", title="Condition"),
        )
    )

    chart_sdtw = (
        alt.layer(sdtw_lines, sdtw_points, sdtw_error_bars)
        .resolve_scale(y="shared")
        .properties(width=500, height=400, title=f"SoftDTW")
    )

    mo.ui.altair_chart(chart_sdtw)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## LMEM""")
    return


@app.cell
def _(joined_sdtw):
    sdtw_lmem_df = joined_sdtw.rename({"Value": "Distance"})
    sdtw_lmem_df
    return (sdtw_lmem_df,)


@app.cell
def _(sdtw_lmem_df):
    sdtw_lmem_model = smf.mixedlm(
        f"Distance ~ condition * joint",
        sdtw_lmem_df.to_pandas(),
        groups=sdtw_lmem_df.to_pandas()["pair_chunk"],
        re_formula="~1",
        # vc_formula={"pair": "0 + C(pair)"},
    )
    return (sdtw_lmem_model,)


@app.cell
def _(sdtw_lmem_model):
    lmem_sdtw_res = sdtw_lmem_model.fit(reml=False)
    return (lmem_sdtw_res,)


@app.cell(hide_code=True)
def _(lmem_sdtw_res):
    mo.md(lmem_sdtw_res.summary().as_html())
    return


if __name__ == "__main__":
    app.run()
