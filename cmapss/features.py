# Condensing flight recordings into Feature Vectors 

import numpy as np
import pandas as pd

from cmapss.ncmapss import NCmapssSplit
from cmapss.ncmapss import load


# Convert Split into one row per flight with reduced & formated timeseries data
def flight_features(
    split: NCmapssSplit,
    cruise: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    
    #  Values Sensoreadings of each flight get processed into 
    timeseries_compression = {
    "mean": lambda g: g.mean(),
    "std": lambda g: g.std(),
    "p10": lambda g: g.quantile(0.10),
    "p50": lambda g: g.quantile(0.50),
    "p90": lambda g: g.quantile(0.90),
}

    frame = split.frame
    # Maske für nur Reise
    if cruise:
        frame = frame[cruise_mask(frame)]

    classification = frame.groupby(["unit", "cycle"], sort=True)

    temp1 = []
    for sensor, compression in timeseries_compression.items():
        temp = compression(classification[split.feature_names])
        temp.columns = [f"{column}_{sensor}" for column in temp.columns]
        temp1.append(temp)
    features = pd.concat(temp1, axis =1)


    meta = classification.agg(Fc=("Fc", "first"), samples=("RUL", "size"))
    meta["duration_h"] = meta["samples"] * split.stride / 3600.0
    target = classification["RUL"].first()


    return (
        features.reset_index(drop=True),
        target.reset_index(drop=True),
        meta.reset_index(),
    )


#  maske mit flugdaten auf reiseflughöhe(=cruise)
def cruise_mask(frame: pd.DataFrame, quantile: float = 0.75) -> np.ndarray:
    threshold = frame.groupby(["unit", "cycle"])["alt"].transform(
        lambda a: a.quantile(quantile)
    )
    return (frame["alt"] >= threshold).to_numpy()

def get_data(dataset: str,downsample: int = 10,features: str = "observable",cruise: bool = False,) -> dict:
    dev_split = load(dataset, "dev" , stride= downsample,features=features)
    test_split = load(dataset, "test", stride = downsample,features=features)

    X_train, y_train, meta_train = flight_features(dev_split,cruise=cruise)
    X_test, y_test, meta_test = flight_features(test_split,cruise=cruise)

    return  {"train":{"x": X_train.to_numpy(dtype="float32"), "y":y_train.to_numpy(dtype="float32")},
            "eval":{"x":X_test.to_numpy(dtype="float32"),"y":y_test.to_numpy(dtype="float32")}, "meta": {"train": meta_train, "test": meta_test}}


    


    



