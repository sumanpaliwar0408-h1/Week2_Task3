# Task 3 - Sales Prediction Model | WeIntern Week 2

## Submission overview

This submission completes **all 11 implementation steps** in the Task 3 handbook. It trains and evaluates **Linear Regression** and **Random Forest Regression**, selects the model family based on time-ordered training cross-validation, optimizes the selected family via **RandomizedSearchCV**, and reports an untouched chronological holdout with RMSE, MAE and R-squared. A seasonal naive baseline provides a meaningful time-series reference.

**Dataset:** Actual Australian monthly wine-sales series (`wineind`) from January 1980 through August 1994 (176 months). Original source: Time Series Data Library / R `forecast` package, distributed by `pmdarima`. Not synthetic. This is distinct from the CO2 dataset used in Tasks 1 and 2 because Task 3 specifically requires sales data. Values are *reported sales units*, not assumed dollars.

Dataset docs: https://pkg.robjhyndman.com/forecast/reference/wineind.html
Python docs: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.datasets.load_wineind.html

## Main results

- **Model-ready data:** 164 months (12 months reserved as lag-feature warm-up, with no invented data).
- **Time-ordered 80/20 split:** 131 train (Jan 1981-Nov 1991), 33 holdout (Dec 1991-Aug 1994). No shuffling.
- **Training selection:** Random Forest (CV RMSE 3,146.94 versus Linear Regression 3,784.69); tuned Random Forest CV RMSE 2,978.36.
- **Holdout Linear Regression:** RMSE 2,312.65; MAE 1,821.03; R2 0.8222.
- **Holdout Tuned Random Forest:** RMSE 2,412.51; MAE 1,815.81; R2 0.8065.
- **Holdout original Random Forest:** RMSE 2,516.81; MAE 1,835.02; R2 0.7894.
- **Seasonal naive benchmark:** RMSE 2,691.20, MAE 1,989.09; R2 0.7592.
- **Best test RMSE:** Linear Regression; **best test MAE:** tuned Random Forest. Tuned RF is the *preselected* deployment candidate because it won train-only CV.
- **Illustrative next-month forecast:** 1994-09-01, 24,482.86 reported sales units, from a newly refit model, not an observed result.
- **Leading Random Forest input:** numeric__sales_lag_12 (importance 52.3%); this indicates model reliance, not causal effects.

## Reproduce the analysis

Python 3.10+ recommended; verified with Python 3.13.5, scikit-learn 1.8.0.

```bash
cd Task3_Sales_Prediction_Model
pip install -r requirements.txt
jupyter lab Task3_Sales_Prediction_Model.ipynb
# In Jupyter: Kernel -> Restart & Run All
```

The notebook uses the **bundled raw CSV**, so a network connection or `pmdarima` is **not** required to re-run it. Re-running overwrites exported figures, data tables and serialized model under `charts/`, `tables/`, `data/` and `model/`. It may take a few minutes due to four-fold time-series CV and hyperparameter tuning. The notebook contains comments, narrative decisions, validations, displayed tables and plots.

To independently re-extract the 176 rows from the original Python package (not needed for notebook execution):

```python
from pmdarima.datasets import load_wineind
series = load_wineind(as_series=True)
# This package and dataset are shipped with pmdarima; raw snapshot already included.
```

## Files and deliverables

- `Task3_Sales_Prediction_Model.ipynb` - **executed Jupyter Notebook** and full machine-learning pipeline.
- `Task3_Sales_Prediction_Model_Notebook.pdf` - PDF print of the executed notebook (14 A3 landscape pages to preserve wide output tables).
- `Task3_Sales_Prediction_Report.pdf` - A4 written model-performance report with visuals and limitations.
- `Task3_Sales_Prediction_Model.html` - HTML backup of the executed notebook.
- `data/australian_wine_sales_raw.csv` - original unmodified sales-value series with parsed date index.
- `data/wine_sales_engineered_modeling.csv` - analysis-ready rows and 15 engineered features.
- `tables/model_evaluation_comparison.csv` - model comparison for RMSE, MAE and R2.
- `tables/baseline_cv_results.csv`, `tables/tuning_details.json`, `tables/tuning_candidates_top15.csv` - CV evidence and tuning choices.
- `tables/holdout_actual_vs_predicted.csv` - actual and model predictions by month.
- `tables/random_forest_feature_importance.csv` - complete ranked tree feature importance.
- `tables/chronological_split.csv` - verified split boundaries.
- `tables/next_month_example_forecast.csv` - hypothetical next-month prediction.
- `charts/*.png` - seven saved and labeled charts, including feature importance and actual/predicted.
- `model/tuned_sales_forecast_pipeline.joblib`, `model/feature_schema.json` - fitted pipeline and feature schema.
- `MODEL_PERFORMANCE_AND_LIMITATIONS.md` - concise interpretation, evaluation and business recommendations.
- `SOURCE_AND_METHODOLOGY.md` - data attribution, SHA256, protocol and leakage caveats.

## Evaluation interpretation and cautions

**One-step-ahead rolling**, not static 33-month forecasting: previous months' **actual** observed sales may be used as lag inputs for later test months. Predictions are conditional on sequential updating of actuals; a 33-month future forecast without those observations has not been evaluated. Training-only feature transformations and CV tuning prevent look-ahead fitting into the holdout. Absolute historical performance should not be extrapolated to current business operations.

The original source has no price, promotion, market context, or customer covariates; calendar and historical-sales features were engineered but **not invented as third-party data**. Sales units in the source are not a declared currency. No causal inference is made from feature importances.
