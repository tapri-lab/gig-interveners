from rich.table import Table
from rich.console import Console
from typing import List, SupportsFloat, Dict
from dataclasses import dataclass
from rich_tools import table_to_df
from pathlib import Path
from pandas import DataFrame


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
    joints: List[str]
    metrics: Dict[str, List[str]]
    recurrence_radius: float
