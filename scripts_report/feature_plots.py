"""Figures for the data description section."""

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILES = {
    "DS01": "N-CMAPSS_DS01.h5",
    "DS02": "N-CMAPSS_DS02.h5",
    "DS03": "N-CMAPSS_DS03.h5",
}

FLIGHT_CLASSES = {1: "class 1", 2: "class 2", 3: "class 3"}
COLOURS = ["#0072B2", "#D55E00", "#009E73"]

COLUMN = 3.5
TEXT = 7.16

STYLE = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
}


def load(path: Path, stride: int) -> pd.DataFrame:
    with h5py.File(path, "r") as file:
        columns = {}
        for group in ("W", "X_s"):
            block = file[f"{group}_dev"][::stride]
            names = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in np.asarray(file[f"{group}_var"]).ravel()
            ]
            for index, name in enumerate(names):
                columns[name] = block[:, index]
        aux = file["A_dev"][::stride]
        for index, name in enumerate(["unit", "cycle", "Fc", "hs"]):
            columns[name] = aux[:, index].astype(int)
        columns["RUL"] = file["Y_dev"][::stride].ravel()
    return pd.DataFrame(columns)


def flight_durations(path: Path) -> pd.DataFrame:
    # Length in samples. The sampling rate is not documented in the paper
    with h5py.File(path, "r") as file:
        aux = file["A_dev"][:, :3]
    frame = pd.DataFrame(aux, columns=["unit", "cycle", "Fc"])
    flights = frame.groupby(["unit", "cycle"]).agg(Fc=("Fc", "first"), n=("Fc", "size"))
    return flights.groupby("Fc")["n"].agg(["count", "min", "median", "max"])


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def features(frame: pd.DataFrame, out: Path) -> None:
    channels = [c for c in frame.columns if c not in ("unit", "cycle", "Fc", "hs", "RUL")]
    fig, axes = plt.subplots(5, 4, figsize=(TEXT, TEXT * 0.72))
    for axis, channel in zip(axes.ravel(), channels, strict=False):
        axis.hist(frame[channel], bins=40, color=COLOURS[0])
        axis.set_title(channel, fontsize=7)
        axis.set_yticks([])
        axis.tick_params(axis="x", labelsize=5.5)
    for axis in axes.ravel()[len(channels) :]:
        axis.axis("off")
    save(fig, out, "features")


def flight_and_classes(frame: pd.DataFrame, stride: int, out: Path) -> None:
    unit = frame["unit"].iloc[0]
    cycle = frame.loc[frame["unit"] == unit, "cycle"].min()
    flight = frame[(frame["unit"] == unit) & (frame["cycle"] == cycle)]
    threshold = flight["alt"].quantile(0.75)

    fig, axes = plt.subplots(1, 2, figsize=(TEXT, TEXT * 0.28))

    axes[0].plot(np.arange(len(flight)) * stride, flight["alt"] / 1000.0, color=COLOURS[0])
    axes[0].axhline(threshold / 1000.0, ls="--", lw=0.9, color=COLOURS[1])
    axes[0].text(
        len(flight) * stride // 4, threshold / 1000.0, "75th percentile",
        va="bottom", ha="center", fontsize=6.5, color=COLOURS[1],
    )
    axes[0].set_xlabel("sample within flight")
    axes[0].set_ylabel("altitude [kft]")
    axes[0].set_title("one flight", fontsize=8, loc="left")

    flights = frame.groupby(["unit", "cycle"], as_index=False).agg(
        Fc=("Fc", "first"), samples=("RUL", "size")
    )
    flights["length"] = flights["samples"] * stride
    for index, flight_class in enumerate(sorted(flights["Fc"].unique())):
        subset = flights[flights["Fc"] == flight_class]
        axes[1].hist(
            subset["length"] / 1000.0, bins=25, alpha=0.75,
            color=COLOURS[index % len(COLOURS)],
            label=FLIGHT_CLASSES.get(int(flight_class), str(flight_class)),
        )
    axes[1].set_xlabel("flight length [thousand samples]")
    axes[1].set_ylabel("flights")
    axes[1].set_title("flight classes", fontsize=8, loc="left")
    axes[1].legend()

    save(fig, out, "flight_and_classes")


def degradation(frame: pd.DataFrame, out: Path) -> None:
    # Only engines of one flight class; the class shifts the level far more
    # than wear does.
    classes = frame.groupby("unit")["Fc"].first()
    selected = classes[classes == classes.value_counts().idxmax()].index[:2]

    fig, axis = plt.subplots(figsize=(COLUMN, COLUMN * 0.6))
    for index, unit in enumerate(selected):
        per_flight = frame[frame["unit"] == unit].groupby("cycle")["T48"].mean()
        colour = COLOURS[index % len(COLOURS)]
        axis.plot(per_flight.index, per_flight, color=colour, alpha=0.3, lw=0.7)
        axis.plot(
            per_flight.index,
            per_flight.rolling(10, min_periods=1, center=True).mean(),
            color=colour, lw=1.4, label=f"engine {unit}",
        )
    axis.set_xlabel("flight number")
    axis.set_ylabel("mean T48 per flight [°R]")
    axis.legend()
    save(fig, out, "degradation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DS01", choices=sorted(FILES))
    parser.add_argument("--data", type=Path, default=Path("data/raw/NCMAPSS"))
    parser.add_argument("--out", type=Path, default=Path("report/figures"))
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    path = args.data / FILES[args.dataset]
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    args.out.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(STYLE)

    print(f"{args.dataset} flight length per class [samples]")
    print(flight_durations(path).round(2).to_string())

    frame = load(path, args.stride)
    features(frame, args.out)
    flight_and_classes(frame, args.stride, args.out)
    degradation(frame, args.out)

    print(f"\n{args.dataset}: 3 figures written to {args.out}")


if __name__ == "__main__":
    main()
