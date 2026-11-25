"""
Metrics for evaluating models
"""

import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # mean signed error; positive means the model overpredicts RUL, which is the dangerous direction here
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(y_pred - y_true))


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric score which punishes overprediction harder, which matters here because
    reporting more remaining flights than an engine has is the unsafe error.

    Formula: 
     d_i = y_pred_i - y_true_i        (signed error per flight, in flights)

               { exp(-d_i / 13) - 1   if d_i < 0   (early: predicted too low)
        s_i  = {
               { exp( d_i / 10) - 1   if d_i >= 0  (late: predicted too high)

        score = sum_i s_i

    Saxena et al., PHM 2008.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0), np.exp(d / 10.0)) - 1.0))


def summarise(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": bias(y_true, y_pred),
        "R2": r2_score(y_true,y_pred),
        "phm08": phm08_score(y_true, y_pred),
    }
