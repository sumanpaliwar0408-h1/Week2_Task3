# Task 3 - Model performance, business interpretation and limitations

## What was built
- Real historical Australian wine sales Jan 1980-Aug 1994; 176 source observations, 164 usable model-ready observations.
- 15 features: months, quarters, year/trend, calendar cyclical terms, sales lags and shifted rolling values.
- Leakage-aware chronological 131/33 split (80/20), four-fold time-series CV on train data, train-fitted imputers and encoders.
- Two required ML regressors: Linear Regression, Random Forest; tuned Random Forest via 18-configuration RandomizedSearchCV. Seasonal naive lag-12 benchmark added.

## Evaluation results (original sales units)

| Model | Training CV RMSE | Test RMSE | Test MAE | Test R2 |
|---|---:|---:|---:|---:|
| Linear Regression | 3,784.69 | 2,312.65 | 1,821.03 | 0.8222 |
| Random Forest | 3,146.94 | 2,516.81 | 1,835.02 | 0.7894 |
| Tuned Random Forest | 2,978.36 | 2,412.51 | 1,815.81 | 0.8065 |
| Seasonal naive (lag-12) | not computed | 2,691.20 | 1,989.09 | 0.7592 |

**Decision:** Random Forest had lower *training CV RMSE* than Linear Regression and was tuned, with best parameters `{'regressor__n_estimators': 160, 'regressor__min_samples_leaf': 1, 'regressor__max_features': 0.6, 'regressor__max_depth': None}`. Tuned RF was fixed as the deployment design without using holdout data. In the final test, ordinary Linear Regression was a little better on RMSE and R2; tuned RF won on MAE. This disagreement suggests model rankings can vary by metric and historical interval.

**Feature importance:** biggest impurity-based Random Forest feature was `numeric__sales_lag_12` (importance 0.523). Important features represent historical predictive relationships, not causes of wine sales.

**One-month illustrative forecast:** 1994-09-01: 24,482.86 sales units, produced by the tuned model refit on all history to 1994-08-01. This value is a model estimate with unknown future accuracy, not a recorded historical sale.

## Business interpretation and recommendations
1. Plan replenishment and distribution around observed calendar seasonality, while checking whether the seasonal pattern holds in later data before acting.
2. Operate as a rolling monthly forecasting system: update with the previous month's actual sales and re-evaluate forecasts over time.
3. Keep a seasonal-naive benchmark and monitor MAE/RMSE by month. Add permitted marketing, promotion, pricing and economic covariates for stronger future models if they become available.

## Limits and validity conditions
- The source is small, historical (1980–1994), and contains no original explanatory features beyond timestamp and sales. Feature engineering creates multiple predictors from past sales/calendar, not from invented covariates.
- The **holdout is a rolling one-step evaluation** using prior realized sales, including previously observed holdout months. These reported metrics are not a multi-month forecast accuracy estimate.
- Statistical patterns may not generalize to new regimes, demand shocks or contemporary pricing. Forecast uncertainty intervals were not estimated.
- Tree importance is not causal, and the source does not specify a monetary currency; reported sales units should not be interpreted as dollars.
- All hyperparameter selection happened on training folds before scoring the holdout. This is an educational prototype, not a production-validated forecasting service.
