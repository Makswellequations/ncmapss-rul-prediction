
import json
import os
import pickle
import numpy as np


from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from cmapss.evaluate import phm08_score, bias

import warnings
from scipy.linalg import LinAlgWarning

# hardbound für RUL pred < 0 
def train(hyper_model: GridSearchCV, data: dict, name: str,units: dict,features: str,subdataset:str, cruise:bool = False) -> KernelRidge:
    x = data["train"]["x"]
    y = data["train"]["y"]

    only_cruise = ""
    if cruise == True:
        only_cruise = "_onlycruise"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        hyper_model.fit(x, y,groups=units)

    best_model = hyper_model.best_estimator_
    best_params = {
        k.split("__")[-1]: v
        for k, v in hyper_model.best_params_.items()
    }

    x_train = data["train"]["x"]
    y_train = data["train"]["y"]
    y_pred_train = np.clip(best_model.predict(x_train), 0.0, None)
    x_test = data["eval"]["x"]
    y_test = data["eval"]["y"]
    y_pred_test = np.clip(best_model.predict(x_test), 0.0, None)

    result = {
        "best_params": best_params,
        "cv_rmse": -hyper_model.best_score_,
        "train": {
            "mae": mean_absolute_error(y_train, y_pred_train),
            "mse": mean_squared_error(y_train, y_pred_train),
            "r2": r2_score(y_train, y_pred_train),
            "bias": bias(y_train, y_pred_train),
            "phm08": phm08_score(y_train, y_pred_train),
        },
        "test": {
            "mae": mean_absolute_error(y_test, y_pred_test),
            "mse": mean_squared_error(y_test, y_pred_test),
            "bias": bias(y_test, y_pred_test),
            "r2": r2_score(y_test, y_pred_test),
            "phm08": phm08_score(y_test,y_pred_test),
        },
    }
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/RMSE/pickle", exist_ok=True)
    os.makedirs("models/RMSE/json", exist_ok=True)
    with open(f"models/RMSE/json/{subdataset}_{name}_{features}{only_cruise}.json", "w") as file:
        json.dump(result, indent=2, fp=file)
    with open(f"models/RMSE/pickle/{subdataset}_{name}_{features}{only_cruise}.pickle", "wb") as file:
        pickle.dump(best_model, file=file)

    #os.makedirs("models/phm08/pickle", exist_ok=True)
    #os.makedirs("models/phm08/json", exist_ok=True)
    #with open(f"models/phm08/json/{subdataset}_{name}_{features}{only_cruise}.json", "w") as file:
    #    json.dump(result, indent=2, fp=file)
    #with open(f"models/phm08/pickle/{subdataset}_{name}_{features}{only_cruise}.pickle", "wb") as file:
    #    pickle.dump(best_model, file=file)

    return best_model