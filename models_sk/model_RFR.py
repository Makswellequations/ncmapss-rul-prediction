import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from cmapss.features import get_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from models_sk.skplot_flights_x import plot_sklearn_h5
from models_sk.train_skmodels import train

def get_hyper_model_rfr(partitions: int = 8) -> GridSearchCV:
    model = make_pipeline(StandardScaler(), RandomForestRegressor()) # Standard scaler für normalisierung der daten 
    param_grid = {
        "randomforestregressor__max_features": ["sqrt", 0.3, 1.0],
        "randomforestregressor__min_samples_leaf": [1, 2, 5, 10],
        "randomforestregressor__max_depth": [None, 10, 20],
    }
    hyper_model = GridSearchCV(
        model, param_grid=param_grid, cv= GroupKFold(n_splits=partitions), scoring="neg_root_mean_squared_error"
    #scoring=phm08_scorer
    )
    return hyper_model


if __name__ == "__main__":
    name = "RandomForestRegressor"
    features = "oracle"
    subdataset = "DS01"
    only_cruise = False

    data = get_data(subdataset, features=features, cruise=only_cruise)
    units = data["meta"]["train"]["unit"].to_numpy()

    hypermodel = get_hyper_model_rfr(len(np.unique(units)))
    model = train(hypermodel, data, name, units, features, subdataset)

    params = {k.removeprefix("randomforestregressor__"): v for k, v in hypermodel.best_params_.items()}
    print("check\n")

    plot_sklearn_h5(model, data, name, "train", features, params, subdataset)
    plot_sklearn_h5(model, data, name, "eval", features, params, subdataset)