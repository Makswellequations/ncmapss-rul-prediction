# Setup

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m cmapss.fetch_ncmapss_DS010203
```

fetches DS01, DS02 and DS03 (about 9GB) into `data/raw/NCMAPSS`.

run scripts from root like this for proper imports

```
.venv/bin/python -m scripts_report.data_description
.venv/bin/python scripts_report/feature_plots.py --dataset DS01
.venv/bin/python -m scripts_report.train_eval_performance
.venv/bin/python -m scripts_report.figures_results_onlyjson
```

>

---
# cmapss
Scripts regarding data processing such as fetching, loading, feature aggregation and evaluation metrics

- fetch_ncmapss_DS010203.py: fetches the data from LRZ
- ncmapss.py: loading data in slices from the .h5 files into DataFrames
- features.py: aggregates dataframes into features
- evaluate.py: metrics to evaluate data

---
# models_sk
- model_GPR.py: GaussianProcessRegressor
- model_KRR.py: KernelRidgeRegressor
- model_RFR.py: RandomForestRegressor
- model_Ridge.py: Ridge
- model_SVR.py: SupportVectorRegression
- skplot_flights_x.py: Plots $y_{true}$ vs $y_{pred}$ for model
- train_skmodels.py: model agnostic train loop

---
# report
The report, the bibliography and the generated figures, plots and tables

---
# scripts_report
- data_description.py: Generation of Table 1
- feature_plots.py: Generation of Fig 1, 2 and 5
- figures_results_onlyjson.py: Generation of Fig 3, 4, 6, 7, 8 and Table 2 -> contains some extra plots that were not used in the report
- run_combinations.py: Generation of models
- train_eval_performance.py: Generation of Table 3 and 4
