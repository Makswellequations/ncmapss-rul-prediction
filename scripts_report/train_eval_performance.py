import pickle
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from cmapss.evaluate import summarise
from cmapss.features import get_data
 
models = ("Ridge", "KernelRidge", "SupportVectorRegression", "GaussianProcessRegressor", "RandomForestRegressor")
labels = ("Ridge", "KRR", "SVR", "GPR", "RFR")
subdatasets = ("DS01", "DS02", "DS03")
position = np.arange(len(models))
 
 
def train_performance() -> pd.DataFrame:
    rows = []
    for pickle_path in sorted(Path("models/RMSE/pickle").glob("*.pickle")):
        if "onlycruise" in pickle_path.stem:
            continue
 
        subdataset, name, features = pickle_path.stem.split("_")
 
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)
 
        data = get_data(subdataset, features=features)
        y_true = data["train"]["y"]
        y_pred = np.clip(model.predict(data["train"]["x"]), 0.0, None)
 
        scores = summarise(y_true, y_pred)
 
        rows.append({
            "Dataset": subdataset,
            "Model": name,
            "Features": features,
            "rmse": scores["rmse"],
            "bias": scores["bias"],
            "R2": scores["R2"],
        })
 
    df = pd.DataFrame(rows)
 
    # erst alle oracle-, dann alle observable-Zeilen, wie in der Appendix-Tabelle
    df["Features"] = pd.Categorical(df["Features"], categories=["oracle", "observable"], ordered=True)
    df = df.sort_values(["Features", "Dataset", "Model"]).reset_index(drop=True)
 
    Path("report/tables").mkdir(parents=True, exist_ok=True)
    with open("report/tables/train_performance.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))
    return df
 
 
# holt einen einzelnen wert, nan wenn der lauf fehlt
def value(dataframe: pd.DataFrame, subdataset: str, model: str, features: str, key: str) -> float:
    row = dataframe[
        (dataframe["Dataset"] == subdataset)
        & (dataframe["Model"] == model)
        & (dataframe["Features"] == features)
    ]
    return row[key].values[0] if len(row) else np.nan

def eval_performance() -> pd.DataFrame:
    rows = []
    for pickle_path in sorted(Path("models/RMSE/pickle").glob("*.pickle")):
        if "onlycruise" in pickle_path.stem:
            continue

        subdataset, name, features = pickle_path.stem.split("_")

        with open(pickle_path, "rb") as f:
            model = pickle.load(f)

        data = get_data(subdataset, features=features)
        y_true = data["eval"]["y"]
        y_pred = np.clip(model.predict(data["eval"]["x"]), 0.0, None)

        scores = summarise(y_true, y_pred)

        rows.append({
            "Dataset": subdataset,
            "Model": name,
            "Features": features,
            "rmse": scores["rmse"],
            "bias": scores["bias"],
            "R2": scores["R2"],
        })

    df = pd.DataFrame(rows)

    df["Features"] = pd.Categorical(df["Features"], categories=["oracle", "observable"], ordered=True)
    df = df.sort_values(["Features", "Dataset", "Model"]).reset_index(drop=True)

    Path("report/tables").mkdir(parents=True, exist_ok=True)
    with open("report/tables/eval_performance.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))
    return df
 
if __name__ == "__main__":
    train_performance()
    eval_performance()
