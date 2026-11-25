import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from cmapss.features import get_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from models_sk.skplot_flights_x import plot_sklearn_h5
from models_sk.train_skmodels import train
from sklearn.gaussian_process.kernels import RBF

def get_hyper_model_gpr(partitions: int) -> GridSearchCV:
    kernel = RBF(length_scale=10.0, length_scale_bounds=(1e-1, 1e3))
    model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0),
    )
    param_grid = {
        "gaussianprocessregressor__alpha": np.logspace(-3, 2, 6),
        "gaussianprocessregressor__n_restarts_optimizer": [3],
    }
    hyper_model = GridSearchCV(
        model, param_grid, cv=GroupKFold(n_splits=partitions),
        #scoring=phm08_scorer, n_jobs=-1,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    return hyper_model


if __name__ == "__main__":
    name = "GaussianProcessRecessor"
    features = "observable"
    subdataset = "DS01"
    only_cruise = False

    data = get_data(subdataset, features=features, cruise= only_cruise)
    units = data["meta"]["train"]["unit"].to_numpy()

    hypermodel = get_hyper_model_gpr(len(np.unique(units)))
    model = train(hypermodel, data, name, units, features,subdataset)

    params = {k.split("__")[-1]: v for k, v in hypermodel.best_params_.items()} # prefixe von hyperpar entfernen -> einfacher für allgemeine plot func!

    plot_sklearn_h5(model, data, name, "train", features, params, subdataset)
    plot_sklearn_h5(model, data, name, "eval", features, params, subdataset)