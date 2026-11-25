from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.kernel_ridge import KernelRidge
from cmapss.evaluate import summarise

def plot_sklearn_h5(
    model: KernelRidge,
    data: dict,
    name: str,
    split: str,
    features: str,
    params: dict,
    subdataset: str,
    cruise: bool = False
) -> mpl.figure.Figure:

    if split not in data:
        raise ValueError

    only_cruise = ""
    if cruise == True:
        only_cruise = "_onlycruise"

    x = data[split]["x"]
    y_true = data[split]["y"]
    y_pred = np.clip(model.predict(x), 0.0, None)

    fig, ax = plt.subplots()
    plt.subplots_adjust(right=0.7)

    ax.scatter(y_true, y_pred, alpha=0.4, s=10, label="Predictions")

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual RUL (flights)")
    ax.set_ylabel("Predicted RUL (flights)")
    ax.legend()

    scores = summarise(y_true ,y_pred)

    text = (
        f"split = {split}\n"
        f"flights = {len(y_true)}\n"
        f"features = {features}\n\n"
    )
    for key, value in (params or {}).items():
        text += f"{key} = {value}\n"
    text += (
        f"\nmae = {scores['mae']:.2f}\n"
        f"rmse = {scores['rmse']:.2f}\n"
        f"bias = {scores['bias']:+.2f}\n"
        f"R2 = {scores['R2']:+.2f}\n"
        f"phm08 = {scores['phm08']:.0f}"
    )
    fig.text(0.72, 0.5, text, va="center")

    #plots = Path("plots/phm08")

    plots = Path("plots/RMSE")
    plots.mkdir(exist_ok=True)
    fig.savefig(plots / f"{subdataset}_{name}_{split}_{features}{only_cruise}.png", bbox_inches="tight")

    return fig
