from rich.table import Table
from rich.console import Console
from typing import List, SupportsFloat, Dict
from dataclasses import dataclass


class ResultsTable:
    def __init__(self, title="Results"):
        self.table = Table(title=title)

    def add_columns(self):
        self.table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        self.table.add_column("Value", justify="right", style="magenta")

    def add_result(self, metric: str, value: SupportsFloat):
        self.table.add_row(metric, str(value))

    def show(self):
        console = Console()
        console.print(self.table)


@dataclass
class Config:
    metrics: Dict[str, List[str]]
    recurrence_radius: float
