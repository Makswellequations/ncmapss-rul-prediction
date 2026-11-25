import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from cmapss.features import get_data
import pickle


# liest alle jsons ohne onlycruise ein
def read_results(folder: str) -> list:
    results = []
    for file in sorted(glob.glob(f"{folder}/*.json")):
        if "onlycruise" in file:
            continue
        parts = Path(file).stem.split("_")

        with open(file) as handle:
            content = json.load(handle)

        results.append({
            "subdataset": parts[0],
            "model": parts[1],
            "features": parts[2],
            "train_rmse": content["train"]["mse"] ** 0.5, # Wurzel ziehen da CV RMSE
            "cv_rmse": content["cv_rmse"],
            "test_rmse": content["test"]["mse"] ** 0.5, # analog
            "test_r2": content["test"]["r2"],
            "test_bias": content["test"]["bias"],
        })
    return results


# holt einen einzelnen wert, nan wenn der lauf fehlt
def value(results: list, subdataset: str, model: str, features: str, key: str) -> float:
    for row in results:
        if row["subdataset"] == subdataset and row["model"] == model and row["features"] == features:
            return row[key]
    return np.nan


if __name__ == "__main__":
    results = read_results("models/RMSE/json")

    models = ("Ridge", "KernelRidge", "SupportVectorRegression", "GaussianProcessRegressor", "RandomForestRegressor")
    labels = ("Ridge", "KRR", "SVR", "GPR", "RFR")
    subdatasets = ("DS01", "DS02", "DS03")
    all_vs_single_flightclass = ("DS01", "DS02")
    all_flight_class = ("DS01","DS03")

    position = np.arange(len(models))

    plots = Path("report/plots")
    tables = Path("report/tables")
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    ## Plot 1 eval - cv (legacy)
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    for ax, subdataset in zip(axes, subdatasets):
        observable_diff = [value(results, subdataset, m, "observable", "test_rmse") - value(results, subdataset, m, "observable", "cv_rmse") for m in models]
        oracle_diff =  [value(results, subdataset, m, "oracle", "test_rmse") - value(results, subdataset, m, "oracle", "cv_rmse") for m in models]

        ax.bar(position - 0.2, observable_diff, 0.4, label="observable")
        ax.bar(position + 0.2, oracle_diff, 0.4, label="oracle")
        ax.axhline(0, lw=0.8)
        ax.set_xticks(position)
        ax.set_xticklabels(labels, ha = "center")
        ax.set_title(subdataset)
        ax.set_ylim(-6, 60)

    
    axes[0].set_ylabel("eval RMSE - CV RMSE")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "eval RMSE - CV RMSE.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    ## Tabelle 1 - DS01/DSO3 observable vs oracle 
    rows = []
    for model, label in zip(models, labels):
        rows.append({
            "Model": label,
            "DS01 obs": value(results, "DS01", model, "observable", "test_rmse"),
            "DS01 oracle": value(results, "DS01", model, "oracle", "test_rmse"),
            "DS03 obs": value(results, "DS03", model, "observable", "test_rmse"),
            "DS03 oracle": value(results, "DS03", model, "oracle", "test_rmse"),
        })

    table_1 = pd.DataFrame(rows)
    with open(tables / "table_allclasses.tex", "w") as f:
        f.write(table_1.to_latex(index=False, float_format="%.2f"))

    ## Tabelle 2 - DS01/DS02 
    rows = []
    for model, label in zip(models, labels):
        rows.append({
            "Model": label,
            "DS01 obs": value(results, "DS01", model, "observable", "test_rmse"),
            "DS01 oracle": value(results, "DS01", model, "oracle", "test_rmse"),
            "DS02 obs": value(results, "DS02", model, "observable", "test_rmse"),
            "DS02 oracle": value(results, "DS02", model, "oracle", "test_rmse"),})
    
    table_1 = pd.DataFrame(rows)
    with open(tables / "table_allclassesvsoneclass.tex", "w") as f:
        f.write(table_1.to_latex(index=False, float_format="%.2f"))


    ## Tabelle Appendix - DS01/DS02/DSO3 observable vs oracle 
    rows = []
    for model, label in zip(models, labels):
        rows.append({
            "Model": label,
            "DS01 obs": value(results, "DS01", model, "observable", "test_rmse"),
            "DS01 oracle": value(results, "DS01", model, "oracle", "test_rmse"),
            "DS02 obs": value(results, "DS02", model, "observable", "test_rmse"),
            "DS02 oracle": value(results, "DS02", model, "oracle", "test_rmse"),
            "DS03 obs": value(results, "DS03", model, "observable", "test_rmse"),
            "DS03 oracle": value(results, "DS03", model, "oracle", "test_rmse"),
        })

    table_1 = pd.DataFrame(rows)
    with open(tables / "table_all.tex", "w") as f:
        f.write(table_1.to_latex(index=False, float_format="%.2f"))


    # Table DS02 - observable vs oracle 
    rows = []
    for model, label in zip(models, labels):
        rows.append({
            "Model": label,
            "DS02 obs": value(results, "DS02", model, "observable", "test_rmse"),
            "DS02 oracle": value(results, "DS02", model, "oracle", "test_rmse"),
        })

    table_1 = pd.DataFrame(rows)
    with open(tables / "table_oneflightclass.tex", "w") as f:
        f.write(table_1.to_latex(index=False, float_format="%.2f"))


    ## Tabelle A - R2 werte stadt CV (Appendix)
    rows = []
    for model, label in zip(models, labels):
        row = {"Model": label}
        for ds in subdatasets:
            for f in ("observable", "oracle"):
                row[f"{ds} {f[:3]}"] = value(results, ds, model, f, "test_r2")
        rows.append(row)

    df = pd.DataFrame(rows)
    with open(tables / "table_r2.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))

         
    ## Plot 2 - Ridge gegen SVR auf DS01 u. DS02 observable
    fig, axes = plt.subplots(2, 4, figsize=(7, 4.5), sharex="row")
    for row, subdataset in zip(axes, ("DS01", "DS02")):
        data = get_data(subdataset, features="observable")
        y_true = data["eval"]["y"]

        for ax, model, label in zip(row, ("RandomForestRegressor", "KernelRidge", "SupportVectorRegression","GaussianProcessRegressor"), ("RFR", "KRR", "SVR","GPR")):
            with open(f"models/RMSE/pickle/{subdataset}_{model}_observable.pickle", "rb") as handle:
                fitted = pickle.load(handle)
            y_pred = fitted.predict(data["eval"]["x"]).clip(0)

            ax.scatter(y_true, y_pred, alpha=0.4, s=8)
            ax.plot([0, y_true.max()], [0, y_true.max()], "r--", lw=1)
            ax.set_title(f"{subdataset} {label}")
            ax.set_ylim(0, 100)
            ax.set_xlim(0,75)

        row[0].set_ylabel("predicted RUL (flights)")

    for ax in axes[1]:
        ax.set_xlabel("actual RUL (flights)")

    fig.tight_layout()
    fig.savefig(plots / "scatter_ds01_ds02.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


    ## Plot 3 - DS01 oracle (oben) vs observable (unten)
    fig, axes = plt.subplots(2, 4, figsize=(7, 4.5), sharex=True, sharey=True)
    fig.suptitle("DS01")

    for row, features in zip(axes, ("oracle", "observable")):
        data = get_data("DS01", features=features)
        y_true = data["eval"]["y"]

        for ax, model, label in zip(row, ("RandomForestRegressor", "KernelRidge", "SupportVectorRegression", "GaussianProcessRegressor"), ("RFR", "KRR", "SVR", "GPR")):
            with open(f"models/RMSE/pickle/DS01_{model}_{features}.pickle", "rb") as handle:
                fitted = pickle.load(handle)
            y_pred = fitted.predict(data["eval"]["x"]).clip(0)

            ax.scatter(y_true, y_pred, alpha=0.4, s=8)
            ax.plot([0, y_true.max()], [0, y_true.max()], "r--", lw=1)
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 75)
            ax.grid(True, lw=0.3, alpha=0.5)

        twin = row[-1].twinx()
        twin.set_ylabel(features, rotation=270, labelpad=12)
        twin.set_yticks([])

        row[0].set_ylabel("predicted RUL (flights)")

    for ax, label in zip(axes[0], ("RFR", "KRR", "SVR", "GPR")):
        ax.set_title(label)

    for ax in axes[1]:
        ax.set_xlabel("actual RUL (flights)")

    fig.tight_layout()
    fig.savefig(plots / "scatter_ds01_or_vs_obs.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    ## Plot 4 - DS02 oracle (oben) vs observable (unten)
    fig, axes = plt.subplots(2, 4, figsize=(7, 4.5), sharex=True, sharey=True)
    fig.suptitle("DS02")

    for row, features in zip(axes, ("oracle", "observable")):
        data = get_data("DS02", features=features)
        y_true = data["eval"]["y"]

        for ax, model, label in zip(row, ("RandomForestRegressor", "KernelRidge", "SupportVectorRegression", "GaussianProcessRegressor"), ("RFR", "KRR", "SVR", "GPR")):
            with open(f"models/RMSE/pickle/DS02_{model}_{features}.pickle", "rb") as handle:
                fitted = pickle.load(handle)
            y_pred = fitted.predict(data["eval"]["x"]).clip(0)

            ax.scatter(y_true, y_pred, alpha=0.4, s=8)
            ax.plot([0, y_true.max()], [0, y_true.max()], "r--", lw=1)
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 75)
            ax.grid(True, lw=0.3, alpha=0.5)

        twin = row[-1].twinx()
        twin.set_ylabel(features, rotation=270, labelpad=12)
        twin.set_yticks([])

        row[0].set_ylabel("predicted RUL (flights)")

    for ax, label in zip(axes[0], ("RFR", "KRR", "SVR", "GPR")):
        ax.set_title(label)

    for ax in axes[1]:
        ax.set_xlabel("actual RUL (flights)")

    fig.tight_layout()
    fig.savefig(plots / "scatter_ds02_or_vs_obs.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    ## Plot 4A - DS03 oracle (oben) vs observable (unten)
    fig, axes = plt.subplots(2, 4, figsize=(7, 4.5), sharex=True, sharey=True)
    fig.suptitle("DS03")

    for row, features in zip(axes, ("oracle", "observable")):
        data = get_data("DS03", features=features)
        y_true = data["eval"]["y"]

        for ax, model, label in zip(row, ("RandomForestRegressor", "KernelRidge", "SupportVectorRegression", "GaussianProcessRegressor"), ("RFR", "KRR", "SVR", "GPR")):
            with open(f"models/RMSE/pickle/DS03_{model}_{features}.pickle", "rb") as handle:
                fitted = pickle.load(handle)
            y_pred = fitted.predict(data["eval"]["x"]).clip(0)

            ax.scatter(y_true, y_pred, alpha=0.4, s=8)
            ax.plot([0, y_true.max()], [0, y_true.max()], "r--", lw=1)
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 75)
            ax.grid(True, lw=0.3, alpha=0.5)

        twin = row[-1].twinx()
        twin.set_ylabel(features, rotation=270, labelpad=12)
        twin.set_yticks([])

        row[0].set_ylabel("predicted RUL (flights)")

    for ax, label in zip(axes[0], ("RFR", "KRR", "SVR", "GPR")):
        ax.set_title(label)

    for ax in axes[1]:
        ax.set_xlabel("actual RUL (flights)")

    fig.tight_layout()
    fig.savefig(plots / "scatter_ds03_or_vs_obs.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    ## Plot 4B - DS01 oracle (oben) vs observable (unten), nur reiseflug
    fig, axes = plt.subplots(2, 4, figsize=(7, 4.5), sharex=True, sharey=True)
    fig.suptitle("DS01 (cruise only)")

    for row, features in zip(axes, ("oracle", "observable")):
        data = get_data("DS01", features=features, cruise=True)
        y_true = data["eval"]["y"]

        for ax, model, label in zip(row, ("RandomForestRegressor", "KernelRidge", "SupportVectorRegression", "GaussianProcessRegressor"), ("RFR", "KRR", "SVR", "GPR")):
            with open(f"models/RMSE/pickle/DS01_{model}_{features}_onlycruise.pickle", "rb") as handle:
                fitted = pickle.load(handle)
            y_pred = fitted.predict(data["eval"]["x"]).clip(0)

            ax.scatter(y_true, y_pred, alpha=0.4, s=8)
            ax.plot([0, y_true.max()], [0, y_true.max()], "r--", lw=1)
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 75)
            ax.grid(True, lw=0.3, alpha=0.5)

        twin = row[-1].twinx()
        twin.set_ylabel(features, rotation=270, labelpad=12)
        twin.set_yticks([])

        row[0].set_ylabel("predicted RUL (flights)")

    for ax, label in zip(axes[0], ("RFR", "KRR", "SVR", "GPR")):
        ax.set_title(label)

    for ax in axes[1]:
        ax.set_xlabel("actual RUL (flights)")

    fig.tight_layout()
    fig.savefig(plots / "scatter_ds01_or_vs_obs_cruise.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    ## Plot 4C - DS02 oracle (oben) vs observable (unten), nur reiseflug
    fig, axes = plt.subplots(2, 4, figsize=(7, 4.5), sharex=True, sharey=True)
    fig.suptitle("DS02 (cruise only)")

    for row, features in zip(axes, ("oracle", "observable")):
        data = get_data("DS02", features=features, cruise=True)
        y_true = data["eval"]["y"]

        for ax, model, label in zip(row, ("RandomForestRegressor", "KernelRidge", "SupportVectorRegression", "GaussianProcessRegressor"), ("RFR", "KRR", "SVR", "GPR")):
            with open(f"models/RMSE/pickle/DS02_{model}_{features}_onlycruise.pickle", "rb") as handle:
                fitted = pickle.load(handle)
            y_pred = fitted.predict(data["eval"]["x"]).clip(0)

            ax.scatter(y_true, y_pred, alpha=0.4, s=8)
            ax.plot([0, y_true.max()], [0, y_true.max()], "r--", lw=1)
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 75)
            ax.grid(True, lw=0.3, alpha=0.5)

        twin = row[-1].twinx()
        twin.set_ylabel(features, rotation=270, labelpad=12)
        twin.set_yticks([])

        row[0].set_ylabel("predicted RUL (flights)")

    for ax, label in zip(axes[0], ("RFR", "KRR", "SVR", "GPR")):
        ax.set_title(label)

    for ax in axes[1]:
        ax.set_xlabel("actual RUL (flights)")

    fig.tight_layout()
    fig.savefig(plots / "scatter_ds02_or_vs_obs_cruise.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    ## Plot 5 - bias
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    for ax, subdataset in zip(axes, subdatasets):
        observable_bias = [value(results, subdataset, m, "observable", "test_bias") for m in models]
        oracle_bias = [value(results, subdataset, m, "oracle", "test_bias") for m in models]
        ax.bar(position - 0.2, observable_bias, 0.4, label="observable")
        ax.bar(position + 0.2, oracle_bias, 0.4, label="oracle")
        ax.axhline(0, lw=0.8, color="k")
        ax.set_xticks(position)
        ax.set_xticklabels(labels, ha="center")
        ax.set_title(subdataset)
        ax.set_ylim(-4, 12)

    axes[0].set_ylabel("bias (flights)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "bias.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    