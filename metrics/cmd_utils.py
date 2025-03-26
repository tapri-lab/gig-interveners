from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, SupportsFloat

import polars as pl
import zarr
import zarr.storage
from numba.core.types.containers import Tuple
from omegaconf import OmegaConf
from pandas import DataFrame
from pyprojroot import here
from rich.console import Console
from rich.table import Table
from rich_tools import table_to_df

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
class RQASettings:
    threshold: Optional[float]
    recurrence_rate: Optional[float]
    joint_pair_rec: List[Tuple]


@dataclass
class EMDSettings:
    cross_person: bool = False


@dataclass
class Config:
    base_data_path: Path
    bvh_audio_folder_paths: Dict[str, Dict[str, Path]]  # {"person<a>": {"bvh": Path, "audio": Path}}
    rqa_settings: RQASettings
    emd_settings: EMDSettings


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


def load_file_paths(mapping: Dict):
    all_paths = []
    for person, ftype_map in mapping.items():
        bvh_files = []
        audio_files = []
        chunk_keys = []
        for ftype, path in ftype_map.items():
            match ftype:
                case "bvh":
                    bvh_files = list(path.glob("*.bvh"))
                    bvh_files.sort()
                case "zarr":
                    zarr_root = zarr.open_group(
                        store=zarr.storage.ZipStore(path.expanduser(), read_only=True), mode="r"
                    )
                    chunk_keys = list(zarr_root.keys())
                    chunk_keys.sort()
                case "audio":
                    audio_files = list(path.glob("*.wav"))
                    audio_files.sort()
        all_paths.extend(
            [
                (person, chunk_key, bvh_file, audio_file)
                for bvh_file, audio_file, chunk_key in zip(bvh_files, audio_files, chunk_keys)
            ]
        )

    return pl.DataFrame(all_paths, schema=["person", "chunk", "bvh", "audio"], orient="row")
