import jax
import jax.numpy as jnp
from jaxtyping import Array
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def sinkhorn_div(x: Array, y: Array, eps: float):
    geom = pointcloud.PointCloud(x, y, epsilon=1e-3)
    solver = sinkhorn.Sinkhorn()
    problem = linear_problem.LinearProblem(geom)
    res = solver(problem)
    return res.reg_ot_cost, res


def sta_distances(
    x,
    y,
    metric,
    beta=0.01,
    epsilon=0.01,
    gamma=1.0,
    return_grad=False,
    device="cpu",
    return_cost_mat=False,
    dtype=jnp.float64,
    **kwargs,
):
    """Compute STA distance matrix between spacial time series.

    Parameters:
    -----------

    x: tensor (n_timestamps_1, dimension_1, dimension_2, ...)
    y: tensor (n_time_series, n_timestamps_2, dimension_1, dimension_2, ...)
    metric: tensor (dimension, dimension)
        OT ground kernel
    beta: float
        hyperparameter of SoftDTW

    Returns:
    --------

    sta: float or array (n_time_series)
        distances between x and y
    """

    betas = jnp.asarray(beta)

    if (betas == 0).sum() and return_grad:
        raise ValueError("STA non differentiable with beta=0")
    divergence_vg = jax.jit(jax.value_and_grad(sinkhorn_div, has_aux=True))
    (div, res), grad = divergence_vg(x, y, epsilon)
    W = res.matrix

    if W.ndim == 2:
        W = W[:, None]
    n_timestamps_x, n_time_series, n_timestamps_y = W.shape
