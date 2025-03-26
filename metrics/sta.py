from typing import Optional

from jaxtyping import Array
from ott.geometry.costs import CostFn, SoftDTW


def sdtw(x: Array, y: Array, gamma: float, ground_cost: Optional[CostFn] = None) -> float:
    s = SoftDTW(gamma=gamma)
    return s(x, y)
