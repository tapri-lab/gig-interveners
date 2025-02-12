from copy import deepcopy
from pathlib import Path
from typing import List, Dict

import einops
import numpy as np
import synchronization as sync
import tyro
import yaml
from tqdm.rich import trange, tqdm
from cmd_utils import Config, ResultsTable
from numpy.typing import NDArray
from wasabi import msg

# fmt: off
function_dict = {
    name: getattr(sync, name) for name in dir(sync)
    if callable(getattr(sync, name)) and not name.startswith("_")
}
# fmt: on


class BasicRQA:
    def __init__(self, recurrence_radius: float):
        self.reccurence_radius = recurrence_radius

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
    d = data
    msg.divider("Joint Level Self Recurrence Analysis")
    msg.info("Calculating Recurrence Matrix")

    for joint in (pbar := tqdm(joints)):
        pbar.set_description(f"Processing joint: {joint}")
        res_table = ResultsTable(title=joint)
        if frames_first:
            data = einops.rearrange(d[joint], "f d -> d f")
        rec_matrix = sync.recurrence_matrix(data, radius=recurrence_radius)
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


def main(cfg_path: Path, npz_path: Path) -> int:
    """
    Analyze the synchronization metrics.
    Args:
        cfg_path: Path to the configuration file.
    """
    config = yaml.load(cfg_path.read_text(), Loader=yaml.FullLoader)
    config = Config(**config)
    data = np.load(npz_path.expanduser())

    results = joint_level_self_recurrence(
        config.metrics["recurrence_indiv"], config.joints, data, config.recurrence_radius
    )
    for res_table in results:
        res_table.show()
    return 0


if __name__ == "__main__":
    tyro.cli(main)
