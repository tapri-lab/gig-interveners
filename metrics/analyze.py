import itertools as it
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
import tyro
from analysis_utils import (
    beat_consistency,
    cross_person_joint_level_recurrence,
    indiv_joint_level_recurrence,
    merge_results,
    run_cross_person_sdtw,
    run_pitch_var_sdtw,
    run_indiv_person_sdtw,
)
from cmd_utils import Config, RQASettings, SDTWSettings, load_file_paths, read_zarr_into_dict
from joblib.parallel import Parallel, delayed
from omegaconf import OmegaConf
from pyprojroot import here
from wasabi import msg


def load_config(cfg_path: Path) -> Config:
    """Load and merge configuration"""
    schema = OmegaConf.structured(Config)
    config = OmegaConf.load(cfg_path)
    return OmegaConf.merge(schema, config)


def prepare_analysis_inputs(df: pl.DataFrame) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Prepare inputs for analysis"""
    persons = df["person"].unique().to_list()
    all_chunks = df["chunk_name"].unique().to_list()
    all_pairs = list(it.product(persons, persons))
    all_pairs = [pair for pair in all_pairs if pair[0] != pair[1]]
    return persons, all_chunks, all_pairs


def run_indiv_joint_analysis(
    pll_exec: Parallel,
    zarr_paths: Dict[str, Path],
    person_joint_pairs: List[Tuple[str, str]],
    all_chunks: List[str],
    rqa_settings: RQASettings,
) -> pl.DataFrame:
    """Run individual joint level recurrence analysis"""
    msg.divider("Self Joint Level Recurrence Analysis")
    indiv_rqa_per_joint = (
        delayed(indiv_joint_level_recurrence)(
            read_zarr_into_dict(zarr_paths[person], chunk, joint),
            person,
            joint,
            chunk,
            rqa_settings.recurrence_rate,
        )
        for person, joint in person_joint_pairs
        for chunk in all_chunks
    )
    indiv_out = pll_exec(indiv_rqa_per_joint)
    indiv_out = merge_results(indiv_out)
    print(indiv_out.head())
    return indiv_out


def run_cross_person_analysis(
    pll_exec: Parallel,
    zarr_paths: Dict[str, Path],
    all_pairs: List[Tuple[str, str]],
    all_chunks: List[str],
    rqa_settings: RQASettings,
) -> pl.DataFrame:
    """Run joint level cross recurrence analysis"""
    msg.divider("Joint Level Cross Recurrence Analysis")
    cross_rqa = (
        delayed(cross_person_joint_level_recurrence)(
            read_zarr_into_dict(zarr_paths, chunk),
            person1,
            person2,
            chunk,
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


def run_beat_consistency_analysis(pll_exec: Parallel, df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    """Run beat consistency analysis for individuals"""
    msg.divider("Self Beat Consistency Scores")
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    bec_tables = (
        delayed(beat_consistency)(
            row["bvh"],
            row["audio"],
            row["person"],
            row["chunk_name"],
            plot=True,
            plot_path=output_dir / f"{row['person']}_{row['chunk_name']}.pdf",
        )
        for row in df.iter_rows(named=True)
    )
    bec_tables = pll_exec(bec_tables)
    bec_tables = merge_results(bec_tables)
    print(bec_tables.head())
    return bec_tables


def run_cross_beat_consistency_analysis(pll_exec: Parallel, df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    """Run beat consistency analysis across persons
    person1 vs person2
    where the motion used is for person1 and the audio for person2.
    """
    msg.divider("Beat Consistency Scores Cross Person")
    # Create a cross join of the dataframe with itself
    # to get all combinations of person pairs
    output_dir.mkdir(parents=True, exist_ok=True)
    bec_subset = (
        df.select(["person", "bvh", "audio", "chunk_name"])
        .join(df.select(["person", "bvh", "audio", "chunk_name"]), how="cross")
        .filter(pl.col("person") != pl.col("person_right"))  # Exclude self-comparisons
        .select(["person", "person_right", "bvh", "audio_right", "chunk_name"])  # Select relevant columns
        .unique(subset=["person", "person_right", "chunk_name"])  # Ensure unique pairs per chunk/thin slice
    )
    bec_tables_cross = (
        delayed(beat_consistency)(
            row["bvh"],
            row["audio_right"],
            # Combine person names for unique identifier, suboptimal for now but will do for now.
            row["person"] + "_" + row["person_right"],
            row["chunk_name"],
            plot=True,
            plot_path=output_dir / f"{row['person']}_{row['person_right']}_{row['chunk_name']}.pdf",
        )
        for row in bec_subset.iter_rows(named=True)
    )
    bec_tables_cross = pll_exec(bec_tables_cross)
    bec_tables_cross = merge_results(bec_tables_cross)
    print(bec_tables_cross.head())
    return bec_tables_cross


def cross_person_sdtw(
    pll_exec: Parallel,
    zarr_paths: Dict[str, Path],
    all_pairs: List[Tuple[str, str]],
    all_chunks: List[str],
    sdtw_settings: SDTWSettings,
) -> pl.DataFrame:
    msg.divider("SDTW Analysis - Cross Person")
    sdtw_results = (
        delayed(run_cross_person_sdtw)(
            read_zarr_into_dict(zarr_paths, chunk),
            person1,
            person2,
            chunk,
            sdtw_settings.gamma,
        )
        for person1, person2 in all_pairs
        for chunk in all_chunks
    )
    sdtw_out = pll_exec(sdtw_results)
    sdtw_out = [x for xs in sdtw_out for x in xs]
    sdtw_out = merge_results(sdtw_out)
    print(sdtw_out.head())
    return sdtw_out


def indiv_person_sdtw(
    pll_exec: Parallel,
    zarr_paths: Dict[str, Path],
    all_chunks: List[str],
):
    people = zarr_paths.keys()
    normal_zarrs = {person: zarr_paths[person]["normal"] for person in people}
    altered_zarrs = {person: zarr_paths[person]["zarr"] for person in people}
    indiv_sdtw_results = (
        delayed(run_indiv_person_sdtw)(
            read_zarr_into_dict(normal_zarrs, chunk),
            read_zarr_into_dict(altered_zarrs, chunk),
            person,
            chunk,
        )
        for person in people
        for chunk in all_chunks
    )
    indiv_sdtw_results = pll_exec(indiv_sdtw_results)
    indiv_sdtw_results = [x for xs in indiv_sdtw_results for x in xs]
    indiv_sdtw_results = merge_results(indiv_sdtw_results)
    print(indiv_sdtw_results.head())
    return indiv_sdtw_results


def run_pitch_var_analysis(
    pll_exec: Parallel,
    audio_data: Dict[str, Dict[str, Path]],
    gamma: float = 1.0,
):
    msg.divider("SDTW Analysis - Pitch Variability")
    people = audio_data.keys()
    sdtw_results = []
    for person in people:
        intervened_audio_path = audio_data[person]["audio"]
        normal_audio_path = audio_data[person]["normal_audio"]
        intervened_audio_files = intervened_audio_path.rglob("*.wav")
        normal_audio_files = normal_audio_path.rglob("*.wav")
        paired_files = zip(normal_audio_files, intervened_audio_files)

        sdtw_results.extend(
            delayed(run_pitch_var_sdtw)(
                normal,
                intervened,
                person,
                normal.stem[-3:],
                gamma=gamma,
            )
            for normal, intervened in paired_files
        )
    sdtw_out = pll_exec(sdtw_results)
    sdtw_out = merge_results(sdtw_out)
    print(sdtw_out.head())
    return sdtw_out


def main(cfg_path: Path, n_jobs: int = -1, output_dir: Path = Path(here() / "results")) -> int:
    """
    Analyze the synchronization metrics.
    Args:
        cfg_path: Path to the configuration file.
        n_jobs: Number of parallel jobs to run. Default is -1, which uses all available cores.
        output_dir: Directory to save the results. Default is "results" in the current directory.
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

    # # Run analyses in parallel
    with Parallel(n_jobs=n_jobs) as pll_exec:
        # Individual joint analysis
        if "indiv_rqa" in config.metrics_to_run:
            indiv_out = run_indiv_joint_analysis(pll_exec, zarr_paths, person_joint_pairs, all_chunks, rqa_settings)
            indiv_out.write_parquet(output_dir / "indiv_joint_recurrence.parquet")

        # Cross-person analysis
        if "crqa" in config.metrics_to_run:
            cross_out = run_cross_person_analysis(pll_exec, zarr_paths, all_pairs, all_chunks, rqa_settings)
            cross_out.write_parquet(output_dir / "cross_joint_recurrence.parquet")

        # Beat consistency analysis
        if "beat_consistency" in config.metrics_to_run:
            bec_tables = run_beat_consistency_analysis(pll_exec, df, output_dir / "bc_plots")
            bec_tables.write_parquet(output_dir / "beat_consistency.parquet")

        # Cross beat consistency analysis
        if "cross_beat_consistency" in config.metrics_to_run:
            bec_tables_cross = run_cross_beat_consistency_analysis(pll_exec, df, output_dir / "cross_bc_plots")
            bec_tables_cross.write_parquet(output_dir / "cross_beat_consistency.parquet")

        # SDTW analysis
        if "sdtw" in config.metrics_to_run:
            sdtw_out = cross_person_sdtw(pll_exec, zarr_paths, all_pairs, all_chunks, config.sdtw_settings)
            sdtw_out.write_parquet(output_dir / "sdtw_results.parquet")
        # Individual SDTW analysis
        if "indiv_motion_sdtw" in config.metrics_to_run:
            indiv_sdtw_out = indiv_person_sdtw(pll_exec, zarr_paths, all_chunks)
            indiv_sdtw_out.write_parquet(output_dir / "indiv_motion_sdtw_results.parquet")

        # Pitch variability analysis
        if "sdtw_pitch_shifted" in config.metrics_to_run:
            sdtw_out = run_pitch_var_analysis(pll_exec, config.bvh_audio_folder_paths, config.sdtw_settings.gamma)
            sdtw_out.write_parquet(output_dir / "sdtw_pitch_shifted_results.parquet")
    return 0


if __name__ == "__main__":
    tyro.cli(main)
