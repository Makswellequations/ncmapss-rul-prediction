import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold,KFold
from cmapss.features import get_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from models_sk.skplot_flights_x import plot_sklearn_h5
from models_sk.train_skmodels import train

def get_hyper_model_krr(partitions: int = 8) -> GridSearchCV:
    model = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")) # Standard scaler für normalisierung der daten 
    param_grid = {
        "kernelridge__alpha": np.logspace(-4, 2, 7),
        "kernelridge__gamma": np.logspace(-4, 2, 7),
    }
    hyper_model = GridSearchCV(
        model, param_grid=param_grid, cv= GroupKFold(n_splits=partitions),  scoring="neg_root_mean_squared_error" #scoring=phm08_scorer
    )
    return hyper_model




if __name__ == "__main__":
    name = "KernelRidge"
    features = "oracle"
    subdataset = "DS01"
    only_cruise = False

    data = get_data(subdataset, features=features,  cruise=  only_cruise)
    units = data["meta"]["train"]["unit"].to_numpy()

    hypermodel = get_hyper_model_krr(len(np.unique(units)))
    model = train(hypermodel, data, name, units, features, subdataset)

    params = {k.removeprefix("kernelridge__"): v for k, v in hypermodel.best_params_.items()}
    print("check\n")

    plot_sklearn_h5(model, data, name, "train", features, params, subdataset)
    plot_sklearn_h5(model, data, name, "eval", features, params, subdataset)