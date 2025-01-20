from rich.table import Table
from typing import SupportsFloat

class ResultsTable():
    def __init__(self, results):
        self.table = Table(title="Results")

    def add_columns(self):
        self.table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        self.table.add_column("Value", justify="right", style="magenta")

    def add_result(self, metric: str, value: SupportsFloat):
        self.table.add_row(metric, str(value))
