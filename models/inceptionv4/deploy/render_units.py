from pathlib import Path

from nowcast_runtime.deployment import render_units

if __name__ == "__main__":
    render_units(
        Path(__file__).resolve().parent.parent,
        "inceptionv4_nowcast",
        "InceptionV4",
        "inceptionv4-now",
        "config/latam.json",
    )
