import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from cmapss.features import get_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from models_sk.skplot_flights_x import plot_sklearn_h5
from models_sk.train_skmodels import train


def get_hyper_model_svr(partitions: int) -> GridSearchCV:
    model = make_pipeline(StandardScaler(), SVR()) # Standard scaler für normalisierung der daten 
    param_grid = { 
        "svr__kernel": ["rbf"],
        "svr__C": np.logspace(0, 5, 6),       
        "svr__gamma": np.logspace(-6, -1, 6),  
        "svr__epsilon": [1.0, 3.0, 5.0],
    }
    hyper_model = GridSearchCV(
        model, param_grid, cv= GroupKFold(n_splits=partitions), scoring="neg_root_mean_squared_error", n_jobs=1
    #scoring=phm08_scorer, n_jobs=1
    )
    return hyper_model




if __name__ == "__main__":
    name = "SupportVectorRegression"
    features = "observable"
    subdataset = "DS01"
    only_cruise = False

    data = get_data(subdataset, features=features, cruise=only_cruise)
    units = data["meta"]["train"]["unit"].to_numpy()

    hypermodel = get_hyper_model_svr(len(np.unique(units)))
    model = train(hypermodel, data, name, units, features, subdataset)

    params = {k.split("__")[-1]: v for k, v in hypermodel.best_params_.items()} # prefixe von hyperpar entfernen -> einfacher für allgemeine plot func!

    plot_sklearn_h5(model, data, name, "train", features, params, subdataset)
    plot_sklearn_h5(model, data, name, "eval", features, params, subdataset)