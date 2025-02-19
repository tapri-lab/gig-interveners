import marimo

__generated_with = "0.11.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import bvhio
    return bvhio, mo


@app.cell
def _(bvhio):
    root = bvhio.readAsHierarchy("/Users/ojas/projects/data/c-cut.bvh")
    return (root,)


@app.cell
def _(root):
    joint = root.filter("LeftShoulder")[0]
    return (joint,)


@app.cell
def _(bvhio, joint):
    bvhio.Euler.fromQuatTo(joint.getKeyframe(101).Rotation, "XYZ", False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
