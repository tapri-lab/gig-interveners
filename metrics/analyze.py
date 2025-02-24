from pathlib import Path
from typing import Dict, List, Tuple

import einops
import numpy as np
import synchronization as sync
import tyro
from becemd import compute_becemd
from cmd_utils import Config, ResultsTable, read_zarr_into_dict
from numpy.typing import NDArray
from omegaconf import OmegaConf
from tqdm.rich import tqdm
from wasabi import msg
import itertools

# fmt: off
function_dict = {
    name: getattr(sync, name) for name in dir(sync)
    if callable(getattr(sync, name)) and not name.startswith("_")
}
function_dict["compute_becemd"] = compute_becemd
# fmt: on


class BasicRQA:
    def __init__(self, recurrence_radius: float):
        """
        A basic rqa wrapper class
        A basic class for calculating recurrence quantification analysis metrics.

        Parameters
        ----------
        recurrence_radius : float
            Threshold radius within which points are considered recurrent. This value determines
            whether two points in phase space are considered "close enough" to be recurrent.

        Notes
        -----
        The class provides methods to:
        - Calculate recurrence matrices from input data
        - Compute RQA metrics from recurrence matrices

        Args:
            recurrence_radius : float - The threshold value used to create the recurrence matrix
        """
        self.recurrence_radius = recurrence_radius

    def calculate_rec_matrix(self, data):
        return sync.recurrence_matrix(data, radius=self.recurrence_radius)

    def calculate_rqa_metrics(self, rec_matrix):
        return sync.rqa_metrics(rec_matrix)


def joint_level_self_recurrence(
    metrics: List[str],
    joints: List[str],
    data: Dict[str, Dict[str, NDArray]],  # person -> joint -> data
    recurrence_radius: float,
    frames_first: bool = True,
) -> List[ResultsTable]:
    results = []
    msg.divider("Joint Level Self Recurrence Analysis")
    msg.info("Calculating Recurrence Matrix")
    rqa = BasicRQA(recurrence_radius=recurrence_radius)

    for person in data:
        for joint in (pbar := tqdm(joints)):
            pbar.set_description(f"Processing joint: {joint}")
            res_table = ResultsTable(title=f"{person} - {joint}")
            d = data[person][joint]
            if frames_first:
                d = einops.rearrange(d, "f d -> d f")
            rec_matrix = rqa.calculate_rec_matrix(d)
            for metric in metrics:
                func = function_dict[metric]
                if metric == "rqa_metrics":
                    rec, det, mean_length, max_length = func(rec_matrix)
                    res_table.add_result("Recurrence Rate", rec)
                    res_table.add_result("Determinism", det)
                    res_table.add_result("Mean Length", mean_length)
                    res_table.add_result("Max Length", max_length)
                    continue
                res = func(rec_matrix)
                res_table.add_result(metric, res)
            results.append(res_table)
    return results


def joint_level_cross_recurrence(
    metrics: List[str],
    joints: List[str],
    data: Dict[str, Dict[str, NDArray]],  # person -> joint -> data
    recurrence_radius: float,
    frames_first: bool = True,
) -> List[ResultsTable]:
    results = []
    msg.divider("Joint Level Cross Recurrence Analysis")
    rqa = BasicRQA(recurrence_radius=recurrence_radius)
    persons = list(data.keys())
    persons = itertools.product(persons, persons)
    for person1, person2 in (pbar := tqdm(persons)):
        pbar.set_description(f"Processing person pair: {person1}, {person2}")
        res_table = ResultsTable(title=f"{person1} vs {person2}")
        for joint in joints:
            d1 = data[person1][joint]
            d2 = data[person2][joint]
            if frames_first:
                d1 = einops.rearrange(d1, "f d -> d f")
                d2 = einops.rearrange(d2, "f d -> d f")
            rec_matrix = rqa.calculate_rec_matrix(d1, d2)
            for metric in metrics:
                func = function_dict[metric]
                if metric == "rqa_metrics":
                    rec, det, mean_length, max_length = func(rec_matrix)
                    res_table.add_result("Recurrence Rate", rec)
                    res_table.add_result("Determinism", det)
                    res_table.add_result("Mean Length", mean_length)
                    res_table.add_result("Max Length", max_length)
                    continue
                res = func(rec_matrix)
                res_table.add_result(metric, res)
        results.append(res_table)
    return results


def beat_consistency(
    bvh_files: List[Path],
    audio_files: List[Path],
    full_pairwise: bool = False,
    plot: bool = False,
) -> Tuple[List[ResultsTable], Dict[str, float]]:
    tables = []
    if full_pairwise:
        pairings = list(itertools.product(bvh_files, audio_files))
        for p_motion, p_audio in (pbar := tqdm(pairings, total=len(pairings))):
            pbar.set_description(f"Processing pair {p_motion.stem}, {p_audio.stem}")
            table = ResultsTable(title=f"Beat Consistency - Pair {p_motion.stem} - {p_audio.stem}")
            _, res = compute_becemd(p_motion, p_audio, plot=plot)
            for k in res["scores"]:
                table.add_result(k, res["scores"][k])
            tables.append(table)
    else:
        # for single person only
        for p_motion, p_audio, idx in (
            pbar := tqdm(zip(bvh_files, audio_files, range(len(bvh_files))), total=len(bvh_files))
        ):
            pbar.set_description(f"Processing person {p_motion.stem}, {p_audio.stem}")
            table = ResultsTable(title=f"Beat Consistency - Person {p_motion.stem}")
            _, res = compute_becemd(p_motion, p_audio, plot=plot)
            for k in res["scores"]:
                table.add_result(k, res["scores"][k])
            tables.append(table)
    return tables, res


def main(cfg_path: Path, zarr_path: Path) -> int:
    """
    Analyze the synchronization metrics.
    Args:
        cfg_path: Path to the configuration file.
        npz_path: Path to the npz file with world coordinates.
    """
    schema = OmegaConf.structured(Config)
    config = OmegaConf.load(cfg_path)
    config = OmegaConf.merge(schema, config)
    bvh_file_path = config.bvh_files
    audio_file_path = config.audio_files
    bvh_files = list(bvh_file_path.glob("*.bvh"))
    audio_files = list(audio_file_path.glob("*.wav"))

    metrics = config.metrics
    individual_metrics = metrics["individual"]
    compute_pairwise_bec = metrics["pairwise"]["beat_consistency"] is not None
    data = read_zarr_into_dict(zarr_path)

    # Process individuals and then pairwise for the group

    results = joint_level_self_recurrence(
        individual_metrics["recurrence"], config.joints, data, config.recurrence_radius
    )
    for res_table in results:
        res_table.show()
    msg.divider("Beat Consistency Scores")
    bec_tables, becemd_results = beat_consistency(bvh_files, audio_files, compute_pairwise_bec, True)
    for table in bec_tables:
        table.show()

    return 0


if __name__ == "__main__":
    tyro.cli(main)
