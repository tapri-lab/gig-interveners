import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import parselmouth
    from parselmouth.praat import call
    return call, mo, np, parselmouth


@app.cell
def __(mo):
    mo.audio("/Users/ojas/projects/data/Session_1/c/c-clean.wav")
    return


@app.cell
def __(parselmouth):
    sound = parselmouth.Sound("/Users/ojas/projects/data/Session_1/c/c-clean.wav")
    return (sound,)


@app.cell
def __(call, sound):
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    return (manipulation,)


@app.cell
def __(manipulation):
    manipulation.class_name
    return


@app.cell
def __(call, manipulation):
    pitch_tier = call(manipulation, "Extract pitch tier")
    return (pitch_tier,)


@app.cell
def __(call, pitch_tier, sound):
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, 2)
    return


@app.cell
def __(call, manipulation, pitch_tier):
    call([pitch_tier, manipulation], "Replace pitch tier")
    return


@app.cell
def __(call, manipulation):
    sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")
    return (sound_octave_up,)


@app.cell
def __(sound_octave_up):
    type(sound_octave_up)
    return


@app.cell
def __(mo, sound_octave_up):
    mo.audio(sound_octave_up.values.squeeze().tobytes())
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
