import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import bvhio
    return bvhio, mo


@app.cell
def __(bvhio):
    bvh = bvhio.readAsBvh("/Users/ojas/projects/data/Session_1/c/c-cut.bvh")
    return (bvh,)


@app.cell
def __(bvh):
    bvh.Root.Channels
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
