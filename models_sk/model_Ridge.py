import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from cmapss.features import get_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from models_sk.skplot_flights_x import plot_sklearn_h5
from models_sk.train_skmodels import train

def get_hyper_model_ridge(partitions: int = 8) -> GridSearchCV:
    model = make_pipeline(StandardScaler(), Ridge()) # Standard scaler für normalisierung der daten
    param_grid = {
        "ridge__alpha": np.logspace(-3, 4, 8),
    }
    hyper_model = GridSearchCV(
        model, param_grid=param_grid, cv= GroupKFold(n_splits=partitions), scoring="neg_root_mean_squared_error" #scoring=phm08_scorer
    )
    return hyper_model



if __name__ == "__main__":
    name = "Ridge"
    features = "observable"
    subdataset = "DS01"

    data = get_data(subdataset, features=features)
    units = data["meta"]["train"]["unit"].to_numpy()

    hypermodel = get_hyper_model_ridge(len(np.unique(units)))
    model = train(hypermodel, data, name, units, features, subdataset, False)

    params = {k.removeprefix("ridge__"): v for k, v in hypermodel.best_params_.items()}

    plot_sklearn_h5(model, data, name, "train", features, params, subdataset, False)
    plot_sklearn_h5(model, data, name, "eval", features, params, subdataset, False)