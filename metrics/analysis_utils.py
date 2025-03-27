from pathlib import Path
from typing import Dict, List, Optional, Tuple, SupportsFloat, Any

import einops
import synchronization as sync
from becemd import compute_becemd
from pandas import DataFrame

from rich.console import Console
from rich.table import Table
from rich_tools import table_to_df
from numpy.typing import NDArray
from pyunicorn.timeseries.cross_recurrence_plot import CrossRecurrencePlot
import pandas as pd
import polars as pl


class ResultsTable:
    def __init__(self, title="Results"):
        self.table = Table(title=title)
        self.metadata = {}
        self.add_columns()

    def add_columns(self):
        self.table.add_column("Metric", justify="left", style="cyan")
        self.table.add_column("Value", justify="right", style="magenta")

    def add_result(self, metric: str, value: SupportsFloat):
        self.table.add_row(metric, str(value))

    def show(self):
        console = Console()
        console.print(self.table)

    def to_df(self) -> DataFrame:
        return table_to_df(self.table)

    def save_csv(self, path: Path):
        path = path.expanduser()
        df = self.to_df()
        df.to_csv(path)

    def add_metadata(self, key: Any, value: Any):
        self.metadata[key] = value

    def get_metadata(self) -> Dict[Any, Any]:
        return self.metadata


def merge_results(tables: List[ResultsTable]) -> pl.DataFrame:
    merged_table = []
    for table in tables:
        df = table.to_df()
        for key, value in table.get_metadata().items():
            df[key] = value
        merged_table.append(df)
    merged_table = pd.concat(merged_table, ignore_index=True)
    merged_table = pl.from_pandas(merged_table)
    return merged_table


class RQA:
    def __init__(self, recurrence_radius: float, recurrence_rate: Optional[float] = None):
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
        self.recurrence_rate = recurrence_rate

    def calculate_rec_matrix(self, data: NDArray) -> NDArray:
        return sync.recurrence_matrix(data, radius=self.recurrence_radius)

    def calculate_rqa_metrics(self, rec_matrix: NDArray) -> Tuple[float, float, float, float]:
        return sync.rqa_metrics(rec_matrix)

    def calculate_crqa_metrics(self, signal1: NDArray, signal2: NDArray) -> Tuple[float, float, float, float]:
        match self.recurrence_rate:
            case None:
                cr = CrossRecurrencePlot(signal1, signal2, threshold=self.recurrence_radius, metric="euclidean")
            case float():
                cr = CrossRecurrencePlot(signal1, signal2, recurrence_rate=self.recurrence_rate, metric="euclidean")
            case _:
                raise ValueError(f"Invalid recurrence rate value, got {self.recurrence_rate}")

        matrix = cr.recurrence_matrix()
        return sync.rqa_metrics(matrix)


def indiv_joint_level_recurrence(
    joint_data: NDArray,
    person: str,
    joint_name: str,
    chunk: str,
    recurrence_radius: Optional[float] = None,
    recurrence_rate: Optional[float] = None,
    frames_first: bool = True,
) -> ResultsTable:
    rqa = RQA(recurrence_radius=recurrence_radius, recurrence_rate=recurrence_rate)

    res_table = ResultsTable(title=f"{''.join(person)}-{joint_name}-{chunk}")
    res_table.add_metadata("person", person)
    res_table.add_metadata("joint", joint_name)
    res_table.add_metadata("chunk", chunk)
    if frames_first:
        joint_data = einops.rearrange(joint_data, "f d -> d f")

    rec_matrix = rqa.calculate_rec_matrix(joint_data)

    rec, det, mean_length, max_length = sync.rqa_metrics(rec_matrix)
    res_table.add_result("Recurrence Rate", rec)
    res_table.add_result("Determinism", det)
    res_table.add_result("Mean Length", mean_length)
    res_table.add_result("Max Length", max_length)

    return res_table


def cross_person_joint_level_recurrence(
    joint_data: Dict[str, Dict[str, NDArray]],
    person1: str,
    person2: str,
    chunk: str,
    recurrence_radius: Optional[float] = None,
    recurrence_rate: Optional[float] = None,
) -> List[ResultsTable]:
    results = []

    rqa = RQA(recurrence_radius=recurrence_radius, recurrence_rate=recurrence_rate)
    persons = list(joint_data.keys())
    joints = list(joint_data[persons[0]].keys())
    for joint in joints:
        res_table = ResultsTable(title=f"{person1} vs {person2} - {joint}- {chunk}")
        res_table.add_metadata("person1", person1)
        res_table.add_metadata("person2", person2)
        res_table.add_metadata("joint", joint)
        res_table.add_metadata("chunk", chunk)
        pj1 = joint_data[person1][joint]
        pj2 = joint_data[person2][joint]

        rec_rate, det, mean_length, max_length = rqa.calculate_crqa_metrics(pj1, pj2)

        res_table.add_result("Recurrence Rate", rec_rate)
        res_table.add_result("Determinism", det)
        res_table.add_result("Mean Length", mean_length)
        res_table.add_result("Max Length", max_length)

        results.append(res_table)
    return results


def beat_consistency(
    bvh_file: Path,
    audio_file: Path,
    person: str,
    chunk: str,
    plot: bool = False,
) -> Tuple[List[ResultsTable], Dict[str, float]]:
    table = ResultsTable(title=f"Beat Consistency-{person}-{chunk}")
    if (ps := person.split("_")) and len(ps) > 1:
        table.add_metadata("person1", ps[0])
        table.add_metadata("person2", ps[1])
    else:
        table.add_metadata("person", person)
        table.add_metadata("chunk", chunk)

    _, res = compute_becemd(bvh_file, str(audio_file), plot=plot)
    for k in res["scores"]:
        table.add_result(k, res["scores"][k])

    return table
