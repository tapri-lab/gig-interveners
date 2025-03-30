from typing import Optional

from jaxtyping import Array
from ott.geometry.costs import CostFn, SoftDTW


def sdtw(s: SoftDTW, x: Array, y: Array, gamma: float, ground_cost: Optional[CostFn] = None) -> float:
    return s(x, y)
