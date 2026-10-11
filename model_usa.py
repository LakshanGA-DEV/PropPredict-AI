"""
Phase 3: USA Stacking Ensemble Pipeline + Quantile Regression + Optuna + Real Data + Encoding
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score, median_absolute_error
)
from sklearn.ensemble import (
    StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from feature_engineering import (
    engineer_us_features,
    load_us_data,
    compute_universal_features,
    apply_target_encoding
)

os.makedirs('data', exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgb_usa(X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 100,
            'n_jobs': -1,
            'tree_method': 'hist',   # M1/M2/M4 compatible
        }
        model = XGBRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=100)
        scores = []
        for t, v in kf.split(X_train):
            model.fit(X_train.iloc[t], y_train.iloc[t])
            preds = model.predict(X_train.iloc[v])
            scores.append(mean_squared_error(y_train.iloc[v], preds) ** 0.5)
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=8, show_progress_bar=False)
    return study.best_params


def tune_lgb_usa(X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'random_state': 100,
            'verbose': -1,
            'n_jobs': -1,
        }
        model = LGBMRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=100)
        scores = []
        for t, v in kf.split(X_train):
            model.fit(X_train.iloc[t], y_train.iloc[t])
            preds = model.predict(X_train.iloc[v])
            scores.append(mean_squared_error(y_train.iloc[v], preds) ** 0.5)
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=8, show_progress_bar=False)
    return study.best_params



from pathlib import Path
from sklearn.model_selection import train_test_split
from feature_engineering import US_CATS, apply_encoding_map, CatBoostAdapter, calibrated_interval

BASE_DIR = Path(__file__).resolve().parent

# Allowlisted predictors exclude targets and any unrequested dataset columns
# (e.g. price_per_sqft, whose value may itself be derived from the target).
BASE_FEATURES = ('area', 'lot_area', 'bedrooms', 'bathrooms', 'year_built', 'year_remodeled', 'school_rating', 'walk_score', 'transit_score', 'bike_score', 'flood_zone', 'fire_zone', 'hoa_fees', 'commute_time', 'bsmt_total_sqft', 'bsmt_finished_sqft', 'hvac_age')
ENGINEERED_FEATURES = ('school_multiplier', 'livability_index', 'climate_risk', 'renovation_gap', 'renovation_potential', 'commute_cost_idx', 'lot_to_home_ratio', 'bsmt_finish_ratio', 'energy_score', 'log_area', 'beds_baths_interaction')


def train_us_model():
    quick = os.getenv('PROPPREDICT_QUICK_TRAIN') == '1'  # smoke testing only
    print('Loading and engineering USA data...')
    raw = load_us_data()
    source = 'real CSV' if (BASE_DIR / 'data' / 'us_houses.csv').exists() else 'synthetic demo'
    if quick:
        raw = raw.sample(n=min(len(raw), 180), random_state=42).reset_index(drop=True)
    train_and_cal_raw, test_raw = train_test_split(raw, test_size=0.2, random_state=42)
    train_raw, calibration_raw = train_test_split(
        train_and_cal_raw, test_size=0.2, random_state=43)
    X_train_all = engineer_us_features(train_raw)
    X_test_all = engineer_us_features(test_raw)
    X_cal_all = engineer_us_features(calibration_raw)
    # Target encoding learned solely on TRAINING rows; holdout labels never accessed.
    X_train_all['price'] = train_raw['price'].to_numpy()
    X_train_all, encoding_map = apply_target_encoding(X_train_all, US_CATS, 'price')
    X_train_all = X_train_all.drop(columns=['price'])
    X_test_all = apply_encoding_map(X_test_all, encoding_map)
    X_cal_all = apply_encoding_map(X_cal_all, encoding_map)

    requested = BASE_FEATURES + ENGINEERED_FEATURES + tuple(f'{c}_encoded' for c in US_CATS)
    features = [col for col in requested if col in X_train_all]
    X_train = X_train_all.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
    medians = X_train.replace([np.inf, -np.inf], np.nan).median().fillna(0).to_dict()
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    X_test = X_test_all.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    X_cal = X_cal_all.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
    X_cal = X_cal.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    y_train = np.log1p(train_raw['price'].to_numpy())
    y_test = np.log1p(test_raw['price'].to_numpy())

    params_path = BASE_DIR / 'us_best_params.json'
    if quick:
        best_xgb = {'n_estimators': 30, 'max_depth': 3, 'learning_rate': .09}
        best_lgb = {'n_estimators': 30, 'max_depth': 4, 'learning_rate': .09, 'num_leaves': 15}
    elif params_path.exists() and not os.getenv('FORCE_TUNE'):
        with params_path.open() as f:
            saved = json.load(f)
        best_xgb, best_lgb = saved['xgb'], saved['lgb']
    else:
        print('Tuning XGBoost and LightGBM for USA...')
        best_xgb = tune_xgb_usa(X_train, pd.Series(y_train))
        best_lgb = tune_lgb_usa(X_train, pd.Series(y_train))
        with params_path.open('w') as f:
            json.dump({'xgb': best_xgb, 'lgb': best_lgb}, f, indent=2)

    best_xgb = {**best_xgb, 'tree_method': 'hist', 'n_jobs': 1, 'random_state': 42}
    best_lgb = {**best_lgb, 'verbosity': -1, 'n_jobs': 1, 'random_state': 42}
    level0 = [
        ('xgb', XGBRegressor(**best_xgb)),
        ('lgb', LGBMRegressor(**best_lgb)),
        ('cat', CatBoostAdapter(iterations=30 if quick else 200,
                                 learning_rate=.05, verbose=False,
                                 random_state=42, thread_count=1)),
        ('rf', RandomForestRegressor(n_estimators=25 if quick else 100,
                                     random_state=42, n_jobs=1)),
    ]
    
    model = StackingRegressor(estimators=level0, final_estimator=Lasso(alpha=.001),
                              cv=KFold(n_splits=3 if quick else 5, shuffle=True, random_state=42),
                              n_jobs=1)
    print('Training USA stacked ensemble...')
    model.fit(X_train, y_train)
    q10 = GradientBoostingRegressor(loss='quantile', alpha=.10,
                                     n_estimators=25 if quick else 100,
                                     max_depth=3 if quick else 5,
                                     learning_rate=.05, random_state=42).fit(X_train, y_train)
    q90 = GradientBoostingRegressor(loss='quantile', alpha=.90,
                                     n_estimators=25 if quick else 100,
                                     max_depth=3 if quick else 5,
                                     learning_rate=.05, random_state=42).fit(X_train, y_train)

    # Split conformal adjustment: calibration labels were not used for fitting.
    # Always include the point estimate within the displayed interval.
    cal_point = np.maximum(0, np.expm1(model.predict(X_cal)))
    cal_q10 = np.maximum(0, np.expm1(q10.predict(X_cal)))
    cal_q90 = np.maximum(0, np.expm1(q90.predict(X_cal)))
    cal_low, cal_high = calibrated_interval(cal_point, cal_q10, cal_q90)
    cal_truth = calibration_raw['price'].to_numpy()
    conformal_scores = np.maximum.reduce([
        cal_low - cal_truth, cal_truth - cal_high, np.zeros_like(cal_truth)])
    target_coverage = 0.80
    quantile_level = min(1.0, np.ceil((len(conformal_scores) + 1) * target_coverage)
                         / len(conformal_scores))
    interval_adjustment = float(np.quantile(conformal_scores, quantile_level, method='higher'))

    predictions = np.expm1(model.predict(X_test))
    test_q10 = np.maximum(0, np.expm1(q10.predict(X_test)))
    test_q90 = np.maximum(0, np.expm1(q90.predict(X_test)))
    test_low, test_high = calibrated_interval(predictions, test_q10, test_q90, interval_adjustment)
    coverage = float(np.mean((test_raw['price'].to_numpy() >= test_low) &
                             (test_raw['price'].to_numpy() <= test_high)))
    print(f'Independent holdout interval coverage: {coverage:.1%} (target {target_coverage:.0%})')
    actual = np.expm1(y_test)
    print('--- Held-out USA metrics ({} rows) ---'.format(len(actual)))
    print('MAE:  {:,.2f}'.format(mean_absolute_error(actual, predictions)))
    print('RMSE: {:,.2f}'.format(mean_squared_error(actual, predictions) ** .5))
    print('MAPE: {:.2%}'.format(mean_absolute_percentage_error(actual, predictions)))
    print('R²:   {:.4f}'.format(r2_score(actual, predictions)))

    path = BASE_DIR / 'us_model.joblib'
    joblib.dump({'model': model, 'q10_model': q10, 'q90_model': q90,
                 'features': features, 'encoding_map': encoding_map,
                 'feature_medians': medians, 'source': source,
                 'holdout_rows': len(actual), 'schema_version': 3,
                 'interval_adjustment': interval_adjustment,
                 'interval_target_coverage': target_coverage,
                 'holdout_interval_coverage': coverage}, path)
    print(f'USA model saved → {path} ({source})')


if __name__ == '__main__':
    train_us_model()
