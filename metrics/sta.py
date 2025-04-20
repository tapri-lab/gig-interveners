from jaxtyping import Array
from ott.geometry.costs import SoftDTW


def sdtw(s: SoftDTW, x: Array, y: Array) -> float:
    return s(x, y)
