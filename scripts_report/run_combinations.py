import numpy as np

from cmapss.features import get_data
from models_sk.train_skmodels import train
from models_sk.skplot_flights_x import plot_sklearn_h5
from models_sk.model_KRR import get_hyper_model_krr
from models_sk.model_SVR import get_hyper_model_svr
from models_sk.model_RFR import get_hyper_model_rfr
from models_sk.model_GPR import get_hyper_model_gpr
from models_sk.model_Ridge import get_hyper_model_ridge


if __name__ == "__main__":
    subdatasetlist = ("DS02", "DS01", "DS03")
    featurelist = ("oracle", "observable")

    modellist = {
        "Ridge": get_hyper_model_ridge,
        "KernelRidge": get_hyper_model_krr,
        "SupportVectorRegression": get_hyper_model_svr,
        "RandomForestRegressor": get_hyper_model_rfr,
        "GaussianProcessRegressor": get_hyper_model_gpr,
    }
    cruiselist = (True, False)

    for subdataset in subdatasetlist:
        for features in featurelist:
            for cruise in cruiselist:
                data = get_data(subdataset, features=features, cruise=cruise)
                units = data["meta"]["train"]["unit"].to_numpy()

                for name, get_hyper_model in modellist.items():
                    print(f"  running {subdataset} {name} {features} {cruise} ...", flush=True)
                    hypermodel = get_hyper_model(len(np.unique(units)))
                    model = train(hypermodel, data, name, units, features, subdataset, cruise)

                    params = {k.split("__")[-1]: v for k, v in hypermodel.best_params_.items()}

                    plot_sklearn_h5(model, data, name, "train", features, params, subdataset, cruise)
                    plot_sklearn_h5(model, data, name, "eval", features, params, subdataset, cruise)