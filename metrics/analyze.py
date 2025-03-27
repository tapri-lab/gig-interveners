import itertools as it
from pathlib import Path

import polars as pl
import tyro
from analysis_utils import (
    beat_consistency,
    cross_person_joint_level_recurrence,
    indiv_joint_level_recurrence,
    merge_results,
)
from cmd_utils import Config, load_file_paths, read_zarr_into_dict
from joblib.parallel import Parallel, delayed
from omegaconf import OmegaConf
from pyprojroot import here
from wasabi import msg


def load_config(cfg_path: Path):
    """Load and merge configuration"""
    schema = OmegaConf.structured(Config)
    config = OmegaConf.load(cfg_path)
    return OmegaConf.merge(schema, config)


def prepare_analysis_inputs(df):
    """Prepare inputs for analysis"""
    persons = df["person"].unique().to_list()
    all_chunks = df["chunk_name"].unique().to_list()
    all_pairs = list(it.product(persons, persons))
    all_pairs = [pair for pair in all_pairs if pair[0] != pair[1]]
    return persons, all_chunks, all_pairs


def run_indiv_joint_analysis(pll_exec, zarr_paths, person_joint_pairs, all_chunks, rqa_settings) -> pl.DataFrame:
    """Run individual joint level recurrence analysis"""
    msg.divider("Self Joint Level Recurrence Analysis")
    indiv_rqa_per_joint = (
        delayed(indiv_joint_level_recurrence)(
            read_zarr_into_dict(zarr_paths[person], chunk, joint),
            person,
            joint,
            chunk,
            rqa_settings.threshold,
            rqa_settings.recurrence_rate,
        )
        for person, joint in person_joint_pairs
        for chunk in all_chunks
    )
    indiv_out = pll_exec(indiv_rqa_per_joint)
    indiv_out = merge_results(indiv_out)
    print(indiv_out.head())
    return indiv_out


def run_cross_person_analysis(pll_exec, zarr_paths, all_pairs, all_chunks, rqa_settings) -> pl.DataFrame:
    """Run joint level cross recurrence analysis"""
    msg.divider("Joint Level Cross Recurrence Analysis")
    cross_rqa = (
        delayed(cross_person_joint_level_recurrence)(
            read_zarr_into_dict(zarr_paths, chunk),
            person1,
            person2,
            chunk,
            rqa_settings.threshold,
            rqa_settings.recurrence_rate,
        )
        for person1, person2 in all_pairs
        for chunk in all_chunks
    )
    cross_out = pll_exec(cross_rqa)
    cross_out = [x for xs in cross_out for x in xs]
    cross_out = merge_results(cross_out)
    print(cross_out.head())
    return cross_out


def run_beat_consistency_analysis(pll_exec, df) -> pl.DataFrame:
    """Run beat consistency analysis for individuals"""
    msg.divider("Self Beat Consistency Scores")
    bec_tables = (
        delayed(beat_consistency)(
            row["bvh"],
            row["audio"],
            row["person"],
            row["chunk_name"],
            plot=False,
        )
        for row in df.iter_rows(named=True)
    )
    bec_tables = pll_exec(bec_tables)
    bec_tables = merge_results(bec_tables)
    print(bec_tables.head())
    return bec_tables


def run_cross_beat_consistency_analysis(pll_exec, df) -> pl.DataFrame:
    """Run beat consistency analysis across persons"""
    msg.divider("Beat Consistency Scores Cross Person")
    bec_subset = (
        df.select(["person", "bvh", "audio", "chunk_name"])
        .join(df.select(["person", "bvh", "audio", "chunk_name"]), how="cross")
        .filter(pl.col("person") != pl.col("person_right"))
        .select(["person", "person_right", "bvh", "audio_right", "chunk_name"])
        .unique(subset=["person", "person_right", "chunk_name"])
    )
    bec_tables_cross = (
        delayed(beat_consistency)(
            row["bvh"],
            row["audio_right"],
            row["person"] + "_" + row["person_right"],
            row["chunk_name"],
            plot=False,
        )
        for row in bec_subset.iter_rows(named=True)
    )
    bec_tables_cross = pll_exec(bec_tables_cross)
    bec_tables_cross = merge_results(bec_tables_cross)
    print(bec_tables_cross.head())
    return bec_tables_cross


def main(cfg_path: Path, n_jobs: int = -1, output_dir: Path = Path(here() / "results")) -> int:
    """
    Analyze the synchronization metrics.
    Args:
        cfg_path: Path to the configuration file.
        n_jobs: Number of parallel jobs to run. Default is -1, which uses all available cores.
    """
    # Load configuration
    config = load_config(cfg_path)

    # Load data
    df, zarr_paths = load_file_paths(config.bvh_audio_folder_paths)

    # Prepare analysis inputs
    persons, all_chunks, all_pairs = prepare_analysis_inputs(df)

    # Setup RQA parameters
    rqa_settings = config.rqa_settings
    rqa_joints = rqa_settings.indiv_joints
    person_joint_pairs = list(it.product(persons, rqa_joints))
    output_dir.mkdir(exist_ok=True, parents=True)
    msg.info(f"Output directory: {output_dir}")

    # Run analyses in parallel
    with Parallel(n_jobs=n_jobs) as pll_exec:
        # Individual joint analysis
        indiv_out = run_indiv_joint_analysis(pll_exec, zarr_paths, person_joint_pairs, all_chunks, rqa_settings)
        indiv_out.write_parquet(output_dir / "indiv_joint_recurrence.parquet")

        # Cross-person analysis
        cross_out = run_cross_person_analysis(pll_exec, zarr_paths, all_pairs, all_chunks, rqa_settings)
        cross_out.write_parquet(output_dir / "cross_joint_recurrence.parquet")

        # Beat consistency analysis
        bec_tables = run_beat_consistency_analysis(pll_exec, df)
        bec_tables.write_parquet(output_dir / "beat_consistency.parquet")

        # Cross beat consistency analysis
        bec_tables_cross = run_cross_beat_consistency_analysis(pll_exec, df)
        bec_tables_cross.write_parquet(output_dir / "cross_beat_consistency.parquet")


if __name__ == "__main__":
    tyro.cli(main)
