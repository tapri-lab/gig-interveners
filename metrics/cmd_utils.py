from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, SupportsFloat

from omegaconf import OmegaConf
from pandas import DataFrame
from pyprojroot import here
from rich.console import Console
from rich.table import Table
from rich_tools import table_to_df
import zarr
import zarr.storage

# Register the "here" resolver
OmegaConf.register_new_resolver("here", lambda: here())


class ResultsTable:
    def __init__(self, title="Results"):
        self.table = Table(title=title)
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


@dataclass
class Config:
    bvh_files: Path
    audio_files: Path
    joints: List[str]
    metrics: Dict[str, Any]
    recurrence_radius: float


def read_zarr_into_dict(zarr_path: Path):
    """
    Read a zarr file into a dictionary.
    Args:
        zarr_path: Path to the zarr file in zip format.
    Returns:
        Dict: Dictionary containing the zarr data.
    """
    store = zarr.storage.ZipStore(zarr_path, read_only=True)
    root = zarr.open_group(store=store, mode="r")
    res = {}
    for person in root.keys():
        res[person] = {}
        for joint in root[person].keys():
            res[person][joint] = root[person][joint][:]
    store.close()
    return res
