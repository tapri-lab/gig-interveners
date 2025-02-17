from pathlib import Path
from typing import Dict, List

import einops
import numpy as np
import synchronization as sync
import tyro
import yaml
from beatconsistencyEMD import compute_becemd
from cmd_utils import Config, ResultsTable
from numpy.typing import NDArray
from tqdm.rich import tqdm
from wasabi import msg

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
    data: Dict[str, NDArray],
    recurrence_radius: float,
    frames_first: bool = True,
) -> List[ResultsTable]:
    results = []
    msg.divider("Joint Level Self Recurrence Analysis")
    msg.info("Calculating Recurrence Matrix")
    rqa = BasicRQA(recurrence_radius=recurrence_radius)

    for joint in (pbar := tqdm(joints)):
        pbar.set_description(f"Processing joint: {joint}")
        res_table = ResultsTable(title=joint)
        d = data[joint]
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


def beat_consistency(bvh_file: Path, audio_file: Path, full_pairwise: bool = False, plot: bool = False):
    pass


def main(cfg_path: Path, npz_path: Path) -> int:
    """
    Analyze the synchronization metrics.
    Args:
        cfg_path: Path to the configuration file.
    """
    config = yaml.load(cfg_path.read_text(), Loader=yaml.FullLoader)
    config = Config(**config)
    metrics = config.metrics
    individual_metrics = metrics["individual"]
    data = np.load(npz_path.expanduser())

    # Process individuals and then pairwise for the group

    results = joint_level_self_recurrence(
        individual_metrics["recurrence"], config.joints, data, config.recurrence_radius
    )
    for res_table in results:
        res_table.show()
    becemd_results = compute_becemd()
    return 0


if __name__ == "__main__":
    tyro.cli(main)
