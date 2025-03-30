import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import functools
    from typing import Any, Callable

    import jax
    import jax.numpy as jnp

    import matplotlib.pyplot as plt

    import ott
    from ott.geometry import pointcloud, geometry
    from ott.problems.linear import linear_problem
    from ott.solvers import linear
    from ott.solvers.linear import sinkhorn
    from ott.tools import plot, sinkhorn_divergence
    from ott.geometry.costs import SoftDTW
    return (
        Any,
        Callable,
        SoftDTW,
        functools,
        geometry,
        jax,
        jnp,
        linear,
        linear_problem,
        ott,
        plot,
        plt,
        pointcloud,
        sinkhorn,
        sinkhorn_divergence,
    )


@app.cell
def _(jax, jnp):
    key1, key2 = jax.random.split(jax.random.key(0), 2)

    x = 0.25 * jax.random.normal(key1, (25, 3))  # Source
    y = 0.5 * jax.random.normal(key2, (50, 3)) + jnp.array((6, 0, 0))  # Target
    return key1, key2, x, y


@app.cell
def _(plt, x, y):
    plt.scatter(x[:, 0], x[:, 1], edgecolors="k", marker="o", label="x", s=200)
    plt.scatter(y[:, 0], y[:, 1], edgecolors="k", marker="X", label="y", s=200)
    plt.legend(fontsize=15)
    plt.show()
    return


@app.cell
def _(pointcloud, x, y):
    geom = pointcloud.PointCloud(x, y, epsilon=1e-3)
    return (geom,)


@app.cell
def _(geom):
    cost = geom.cost_matrix
    cost
    return (cost,)


@app.cell
def _(cost):
    cost.shape
    return


@app.cell
def _(sinkhorn):
    solver = sinkhorn.Sinkhorn()
    return (solver,)


@app.cell
def _(geom, linear_problem):
    prob = linear_problem.LinearProblem(geom)
    return (prob,)


@app.cell
def _(jax, prob, solver):
    xs = jax.jit(solver)(prob)
    return (xs,)


@app.cell
def _(xs):
    xs.matrix.shape
    return


@app.cell
def _(xs):
    xs.reg_ot_cost
    return


@app.cell
def _(SoftDTW):
    sdtw = SoftDTW(gamma=0.2)
    return (sdtw,)


@app.cell
def _(jax, sdtw, x, y):
    sdtw(x, y), jax.jit(sdtw)
    return


@app.cell
def _(geometry, linear):
    def sink(a, b, cost, epsilon, min_iterations, max_iterations):
        return linear.solve(
            geometry.Geometry(cost_matrix=cost, epsilon=epsilon),
            a=a,
            b=b,
            lse_mode=False,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
        ).reg_ot_cost
    return (sink,)


@app.cell
def _(cost, jnp, xs):
    jnp.trace(xs.matrix.T @ cost)
    return


@app.cell
def _(xs):
    xs.matrix
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
