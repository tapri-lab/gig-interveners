

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


@app.cell
def _():
    results_base = here() / "results_base"
    return


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
    mo.md(r"""## LMEM""")
    return


@app.cell(hide_code=True)
def _(crqa_metric_choice):
    crqa_metric_choice
    return


@app.cell(hide_code=True)
def _(crqa_metric_choice, joined):
    crqa_lmem_df = joined.with_columns(
        pl.concat_str([pl.col("person1"), pl.col("person2")], separator="_").alias("pair"),
        pl.concat_str([pl.col("person1"), pl.col("person2"), pl.col("chunk")], separator="_").alias("pair_chunk"),
    ).unique(subset=["pair_chunk", "Metric", "joint"], keep="first")
    # crqa_lmem_df.melt(
    #     id_vars=["Metric", "person1", "person2", "pair", "chunk", "joint"],
    #     value_vars=["Value", "Value_right"],
    #     variable_name="condition",
    #     value_name=crqa_metric_choice.value,
    # )
    crqa_lmem_df = crqa_lmem_df.unpivot(
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
        initial_path=here() / "results/",
        selection_mode="file",
        label="Select the base Cross BC file",
        filetypes=[".parquet"],
    )
    cross_bc_base_path
    return (cross_bc_base_path,)


@app.cell
def _(cross_bc_base_path):
    cross_bc_base = pl.read_parquet(cross_bc_base_path.path(index=0))
    cross_bc_base = cross_bc_base.with_columns(pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32)).filter(
        pl.col("Value").is_not_nan()
    )
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
    fig2.savefig(here() / "results" / "results_base/bc_pairwise_base.png", dpi=300, bbox_inches="tight")
    fig2
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
        initial_path=here() / "results/", selection_mode="file", label="Select the base SDTW file", filetypes=[".parquet"]
    )
    base_sdtw_path
    return (base_sdtw_path,)


@app.cell
def _(base_sdtw_path):
    sdtw_base = pl.read_parquet(base_sdtw_path.path(index=0))
    sdtw_base = sdtw_base.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32), pl.col("Metric").str.replace(" ", "_")
    ).filter(pl.col("Value").is_not_nan())
    cross_sdtw_plotter(sdtw_base, 5)
    return (sdtw_base,)


@app.cell
def _():
    intervened_sdtw_path = mo.ui.file_browser(
        initial_path=here() / "results", selection_mode="file", label="Select the base SDTW file", filetypes=[".parquet"]
    )
    intervened_sdtw_path
    return (intervened_sdtw_path,)


@app.cell
def _(intervened_sdtw_path):
    sdtw_intervened = pl.read_parquet(intervened_sdtw_path.path(index=0))
    sdtw_intervened = sdtw_intervened.with_columns(
        pl.col("Value").cast(pl.Float32), pl.col("chunk").cast(pl.Int32), pl.col("Metric").str.replace(" ", "_")
    ).filter(pl.col("Value").is_not_nan())
    cross_sdtw_plotter(sdtw_intervened, 5)
    return (sdtw_intervened,)


@app.cell
def _(sdtw_base, sdtw_intervened):
    joined_sdtw = sdtw_base.join(sdtw_intervened, how="inner", on=["person1", "person2", "chunk", "joint", "Metric"])
    joined_sdtw = joined_sdtw.with_columns(
        pl.min_horizontal(["person1", "person2"]).alias("person1"),
        pl.max_horizontal(["person1", "person2"]).alias("person2"),
    ).unique(subset=["Metric", "person1", "person2", "joint", "chunk"], keep="first")
    return (joined_sdtw,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## LMEM""")
    return


@app.cell
def _(joined_sdtw):
    sdtw_lmem_df = joined_sdtw.with_columns(
        pl.concat_str([pl.col("person1"), pl.col("person2")], separator="_").alias("pair"),
        pl.concat_str([pl.col("person1"), pl.col("person2"), pl.col("chunk")], separator="_").alias("pair_chunk"),
    ).unique(subset=["pair_chunk", "Metric", "joint"], keep="first")

    sdtw_lmem_df = sdtw_lmem_df.unpivot(
        index=["Metric", "person1", "person2", "pair", "chunk", "pair_chunk", "joint"],
        on=["Value", "Value_right"],
        variable_name="condition",
        value_name="Distance",
    ).with_columns(
        pl.col("condition").replace({"Value": "raw", "Value_right": "damped"}).cast(pl.Categorical),
        pl.col("joint").cast(pl.Categorical),
        pl.col("pair").cast(pl.Categorical),
        pl.col("pair_chunk").cast(pl.Categorical),
        pl.col("Metric").cast(pl.Categorical),
    )
    # Z-score standardization
    dist_col = pl.col("Distance")
    sdtw_lmem_df = sdtw_lmem_df.with_columns((dist_col - dist_col.mean()) / dist_col.std())
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
