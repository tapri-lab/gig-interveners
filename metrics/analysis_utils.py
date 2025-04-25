from pathlib import Path
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
import jax
import pandas as pd
import polars as pl
import synchronization as sync
from becemd import compute_becemd
from numpy.typing import NDArray
from ott.geometry.costs import SoftDTW
from pandas import DataFrame
from pyunicorn.timeseries.cross_recurrence_plot import CrossRecurrencePlot, RecurrencePlot
from rich.console import Console
from rich.table import Table
from rich_tools import table_to_df
from sta import sdtw
from tslearn.metrics import soft_dtw_alignment
import librosa


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
    def __init__(self, recurrence_rate: float = 0.02):
        """
        A basic rqa wrapper class
        A basic class for calculating recurrence quantification analysis metrics.

        Parameters
        ----------
        recurrence_rate : float
            Recurrence rate for cross-recurrence analysis. If None, the default value is used.

        Notes
        -----
        The class provides methods to:
        - Calculate recurrence matrices from input data
        - Compute RQA metrics from recurrence matrices

        Args:
            recurrence_radius : float - The threshold value used to create the recurrence matrix
        """
        self.recurrence_rate = recurrence_rate

    def calculate_rqa_metrics(self, signal: NDArray) -> Tuple[Tuple[float, float, float, float], float]:
        r = RecurrencePlot(
            signal,
            metric="euclidean",
            recurrence_rate=self.recurrence_rate,
            normalize=True,
        )
        dist = r.distance_matrix("euclidean")
        rec_matrix = r.recurrence_matrix()
        threshold = r.threshold_from_recurrence_rate(dist, self.recurrence_rate)
        entr = r.diag_entropy()
        return sync.rqa_metrics(rec_matrix), threshold

    def calculate_crqa_metrics(
        self,
        signal1: NDArray,
        signal2: NDArray,
    ) -> Tuple[Tuple[float, float, float, float], float]:
        cr = CrossRecurrencePlot(
            signal1,
            signal2,
            recurrence_rate=self.recurrence_rate,
            metric="euclidean",
            normalize=True,
        )

        rec_matrix = cr.recurrence_matrix()
        dist = cr.distance_matrix("euclidean")
        threshold = cr.threshold_from_recurrence_rate(dist, self.recurrence_rate)
        return sync.rqa_metrics(rec_matrix), threshold


def indiv_joint_level_recurrence(
    joint_data: NDArray,
    person: str,
    joint_name: str,
    chunk: str,
    recurrence_rate: float = 0.02,
) -> ResultsTable:
    rqa = RQA(recurrence_rate=recurrence_rate)

    res_table = ResultsTable(title=f"{''.join(person)}-{joint_name}-{chunk}")
    res_table.add_metadata("person", person)
    res_table.add_metadata("joint", joint_name)
    res_table.add_metadata("chunk", chunk)

    (rec, det, mean_length, max_length), thr = rqa.calculate_rqa_metrics(joint_data)
    res_table.add_result("Recurrence Rate", rec)
    res_table.add_result("Determinism", det)
    res_table.add_result("Mean Length", mean_length)
    res_table.add_result("Max Length", max_length)
    res_table.add_result("Recurrence Radius", thr)

    return res_table


def cross_person_joint_level_recurrence(
    joint_data: Dict[str, Dict[str, NDArray]],
    person1: str,
    person2: str,
    chunk: str,
    recurrence_rate: float = 0.02,
) -> List[ResultsTable]:
    results = []

    rqa = RQA(recurrence_rate=recurrence_rate)
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

        (rec_rate, det, mean_length, max_length), thr = rqa.calculate_crqa_metrics(pj1, pj2)

        res_table.add_result("Recurrence Rate", rec_rate)
        res_table.add_result("Determinism", det)
        res_table.add_result("Mean Length", mean_length)
        res_table.add_result("Max Length", max_length)
        res_table.add_result("Recurrence Radius", thr)

        results.append(res_table)
    return results


def beat_consistency(
    bvh_file: Path,
    audio_file: Path,
    person: str,
    chunk: str,
    plot: bool = False,
    plot_path: Optional[Path] = None,
) -> Tuple[List[ResultsTable], Dict[str, float]]:
    table = ResultsTable(title=f"Beat Consistency-{person}-{chunk}")
    if (ps := person.split("_")) and len(ps) > 1:
        table.add_metadata("person1", ps[0])
        table.add_metadata("person2", ps[1])
        table.add_metadata("chunk", chunk)
    else:
        table.add_metadata("person", person)
        table.add_metadata("chunk", chunk)

    _, res = compute_becemd(bvh_file, str(audio_file), plot=plot, plot_save_path=plot_path)
    for k in res["scores"]:
        table.add_result(k, res["scores"][k])

    return table


def run_cross_person_sdtw(
    joint_data: Dict[str, Dict[str, NDArray]],
    person1: str,
    person2: str,
    chunk: str,
    gamma: float = 0.01,
) -> List[ResultsTable]:
    persons = list(joint_data.keys())
    joints = list(joint_data[persons[0]].keys())
    results = []
    s = jax.jit(SoftDTW(gamma=gamma))
    scaler = StandardScaler()
    for joint in joints:
        res_table = ResultsTable(title=f"{person1} vs {person2}-{joint}-{chunk}")
        res_table.add_metadata("person1", person1)
        res_table.add_metadata("person2", person2)
        res_table.add_metadata("joint", joint)
        res_table.add_metadata("chunk", chunk)
        pj1 = joint_data[person1][joint]
        pj2 = joint_data[person2][joint]
        pj1 = scaler.fit_transform(pj1)
        pj2 = scaler.fit_transform(pj2)

        dist = sdtw(s, pj1, pj2)
        res_table.add_result("Distance", dist)
        results.append(res_table)
    return results


def run_indiv_person_sdtw(
    joint_data_normal: Dict[str, NDArray],
    joint_data_altered: Dict[str, NDArray],
    person: str,
    chunk: str,
    gamma: float = 0.01,
):
    results = []
    s = jax.jit(SoftDTW(gamma=gamma))
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    for joint in joint_data_normal.keys():
        res_table = ResultsTable(title=f"{person}-{joint}-{chunk}")
        res_table.add_metadata("person", person)
        res_table.add_metadata("joint", joint)
        res_table.add_metadata("chunk", chunk)
        pj1 = joint_data_normal[joint]
        pj2 = joint_data_altered[joint]
        pj1 = scaler1.fit_transform(pj1)
        pj2 = scaler2.fit_transform(pj2)

        dist = sdtw(s, pj1, pj2)
        res_table.add_result("Distance", dist)
        results.append(res_table)
    return results


def run_pitch_var_sdtw(
    normal_audio_path: Path,
    altered_audio_path: Path,
    person: str,
    chunk: str,
    *,
    gamma: float = 1.0,
) -> ResultsTable:
    res_table = ResultsTable(title=f"{person}-{chunk}")
    res_table.add_metadata("person", person)
    res_table.add_metadata("chunk", chunk)
    res_table.add_metadata("normal_audio", str(normal_audio_path))
    res_table.add_metadata("altered_audio", str(altered_audio_path))
    y1, sr1 = librosa.load(str(normal_audio_path), sr=None)
    y2, sr2 = librosa.load(str(altered_audio_path), sr=None)

    f0_1 = librosa.yin(y1, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0_2 = librosa.yin(y2, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0_1 = f0_1[~np.isnan(f0_1)]
    f0_2 = f0_2[~np.isnan(f0_2)]

    path, sim = soft_dtw_alignment(f0_1, f0_2, gamma=gamma)
    path2, sim2 = soft_dtw_alignment(f0_1, f0_1, gamma=gamma)
    res_table.add_result("Distance_Intervened", sim.item())
    res_table.add_result("Distance_Non_Intervened", sim2.item())

    return res_table
