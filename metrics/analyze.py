from pathlib import Path
from typing import List

import einops
import numpy as np
import synchronization as sync
import tyro
import yaml
from cmd_utils import Config, ResultsTable
from numpy.typing import NDArray
from wasabi import msg

# fmt: off
function_dict = {
    name: getattr(sync, name) for name in dir(sync)
    if callable(getattr(sync, name)) and not name.startswith("_")
}
# fmt: on


def self_recurrence(
    res_table: ResultsTable, metrics: List[str], data: NDArray, recurrence_radius: float, frames_first: bool = True
) -> ResultsTable:
    msg.info("Calculating Recurrence Matrix")
    if frames_first:
        data = einops.rearrange(data, "f d -> d f")
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
    return res_table


def main(cfg_path: Path, npz_path: Path) -> int:
    """
    Analyze the synchronization metrics.
    Args:
        cfg_path: Path to the configuration file.
    """
    config = yaml.load(cfg_path.read_text(), Loader=yaml.FullLoader)
    config = Config(**config)
    data = np.load(npz_path.expanduser())
    res_table = ResultsTable()
    results = self_recurrence(res_table, config.metrics["recurrence_indiv"], data, config.recurrence_radius)
    results.show()
    return 0


if __name__ == "__main__":
    tyro.cli(main)
