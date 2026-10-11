"""Shared, price-independent features and leakage-safe category encoding."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

BASE_DIR = Path(__file__).resolve().parent
INDIA_CATS = ('city', 'builder', 'facing')
US_CATS = ('zipcode',)
INDIA_REQUIRED = ('price', 'area', 'bedrooms', 'bathrooms', 'floor', 'total_floors',
                  'dist_metro', 'dist_bus', 'dist_railway', 'dist_highway',
                  'dist_it_park', 'dist_school', 'age_years', 'rera_approved')
US_REQUIRED = ('price', 'area', 'lot_area', 'bedrooms', 'bathrooms', 'year_built',
               'year_remodeled', 'school_rating', 'walk_score', 'transit_score',
               'bike_score', 'flood_zone', 'fire_zone', 'hoa_fees', 'commute_time',
               'bsmt_total_sqft', 'bsmt_finished_sqft', 'hvac_age')


def calibrated_interval(point, q10, q90, adjustment=0.0):
    """Intervals contain the point estimate; optional conformal additive correction.

    The coverage level depends on how the adjustment was calibrated and on the
    similarity of future data to the held-out calibration population.
    """
    p = np.maximum(0, np.asarray(point, dtype=float))
    lower = np.maximum(0, np.asarray(q10, dtype=float))
    upper = np.maximum(0, np.asarray(q90, dtype=float))
    amount = float(adjustment)
    if not np.isfinite(amount) or amount < 0:
        raise ValueError('Calibration adjustment must be a finite nonnegative number')
    if not all(np.all(np.isfinite(a)) for a in (p, lower, upper)):
        raise ValueError('Non-finite prediction or interval bounds')
    return (np.maximum(0, np.minimum.reduce([p, lower, upper]) - amount),
            np.maximum.reduce([p, lower, upper]) + amount)


def apply_target_encoding(df, cat_cols, target_col='price', n_folds=5):
    """Return out-of-fold category features and full *training-only* inference map.

    Call ONLY on the training split. Labels must never enter the holdout split's map.
    Encoded scale is log1p(price), matching the regression target.
    """
    if target_col not in df:
        raise ValueError(f'Missing training target {target_col}')
    if len(df) < 2:
        raise ValueError('Target encoding requires at least two training rows')
    result = df.copy().reset_index(drop=True)
    y = np.log1p(pd.to_numeric(result[target_col], errors='raise'))
    maps = {}
    for col in cat_cols:
        if col not in result:
            continue
        values = result[col].fillna('__missing__').astype(str)
        encoded = np.empty(len(result), dtype=float)
        kf = KFold(n_splits=min(n_folds, len(result)), shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(result):
            means = pd.DataFrame({'category': values.iloc[train_idx],
                                  'target': y.iloc[train_idx]}).groupby('category')['target'].mean()
            fold_prior = float(y.iloc[train_idx].mean())
            encoded[val_idx] = values.iloc[val_idx].map(means).fillna(fold_prior).to_numpy()
        result[f'{col}_encoded'] = encoded
        mapping = pd.DataFrame({'category': values, 'target': y}).groupby('category')['target'].mean().to_dict()
        mapping['__global_mean__'] = float(y.mean())
        maps[col] = mapping
        result = result.drop(columns=[col])
    return result, maps


def apply_encoding_map(df, encoding_map):
    """Encode categories using only training-set statistics at prediction time."""
    result = df.copy()
    for col, mapping in encoding_map.items():
        if col not in result:
            result[col] = '__missing__'
        values = result[col].fillna('__missing__').astype(str)
        result[f'{col}_encoded'] = values.map(mapping).fillna(mapping['__global_mean__']).astype(float)
        result = result.drop(columns=[col])
    return result


def compute_universal_features(df, target_col='price'):
    result = df.copy()
    if 'area' in result:
        result['log_area'] = np.log1p(result['area'].clip(lower=0))
    if {'bedrooms', 'bathrooms'}.issubset(result):
        result['beds_baths_interaction'] = result['bedrooms'] * result['bathrooms']
    return result


def _drop_targets(df):
    # No derived feature may use sale price, listing price, or a placeholder price.
    return df.drop(columns=['price', 'log_price', 'listing_price', 'emi_band',
                             'hoa_value_penalty', 'price_rent_ratio'], errors='ignore')


def engineer_india_features(df):
    df = _drop_targets(df.copy())
    vastu = {'East': 10, 'North-East': 10, 'North': 9, 'West': 7,
             'South-East': 6, 'North-West': 6, 'South': 5, 'South-West': 4}
    if 'facing' in df:
        df['vastu_score'] = df['facing'].map(vastu).fillna(5)
    distances = ['dist_metro', 'dist_bus', 'dist_railway', 'dist_highway']
    if all(c in df for c in distances):
        df['connectivity_index'] = 1 / (df[distances].sum(axis=1).clip(lower=0) + 0.1)
    if 'builder' in df:
        tier1 = ('DLF', 'Godrej', 'Prestige', 'Sobha', 'Lodha', 'Oberoi', 'Macrotech')
        df['builder_score'] = np.where(df['builder'].isin(tier1), 3,
                                       np.where(df['builder'].notna(), 2, 1))
    if 'rera_approved' in df:
        df['rera_multiplier'] = np.where(df['rera_approved'] == 1, 1.05, 1.0)
    if {'floor', 'total_floors'}.issubset(df):
        ratio = df['floor'] / df['total_floors'].clip(lower=1)
        df['floor_premium_idx'] = np.where(df['floor'] == 0, 0.95,
                                           np.where(ratio > 0.9, 1.10, 1.05))
    if 'city' in df:
        df['monsoon_risk'] = np.where(df['city'].isin(['Mumbai', 'Chennai']), 0.8, 1.0)
    if 'dist_school' in df:
        df['school_zone_score'] = np.where(df['dist_school'] <= 2, 1.2, 1.0)
    if 'dist_it_park' in df:
        df['it_hub_prox_score'] = np.where(df['dist_it_park'] <= 5, 1.15, 1.0)
    if 'age_years' in df:
        df['age_curve'] = np.where(df['age_years'] < 3, 1.1,
                                   np.where(df['age_years'] > 50, 1.2, np.exp(-0.01 * df['age_years'])))
    return compute_universal_features(df)


def engineer_us_features(df):
    df = _drop_targets(df.copy())
    if 'school_rating' in df:
        df['school_multiplier'] = df['school_rating'] / 10
    scores = ['walk_score', 'transit_score', 'bike_score']
    if all(c in df for c in scores):
        df['livability_index'] = df[scores].mean(axis=1)
    if {'flood_zone', 'fire_zone'}.issubset(df):
        df['climate_risk'] = (df['flood_zone'] + df['fire_zone']) / 2
    if {'year_built', 'year_remodeled'}.issubset(df):
        df['renovation_gap'] = df['year_remodeled'] - df['year_built']
        df['renovation_potential'] = (df['renovation_gap'] == 0).astype(int)
    if 'commute_time' in df:
        df['commute_cost_idx'] = df['commute_time'] * 3.5
    if {'lot_area', 'area'}.issubset(df):
        # A batch/zipcode median is not stable for single-row inference.
        df['lot_to_home_ratio'] = df['lot_area'] / df['area'].clip(lower=1)
    if {'bsmt_finished_sqft', 'bsmt_total_sqft'}.issubset(df):
        df['bsmt_finish_ratio'] = df['bsmt_finished_sqft'] / (df['bsmt_total_sqft'].clip(lower=0) + 1e-5)
    if 'hvac_age' in df:
        df['energy_score'] = np.where(df['hvac_age'] < 5, 10,
                                      np.maximum(0, 10 - (df['hvac_age'] - 5) * 0.5))
    return compute_universal_features(df)


def _load_csv(path, required, rename_map, defaults, lower_price, upper_price):
    df = pd.read_csv(path).rename(columns=rename_map)
    if df.columns.duplicated().any():
        raise ValueError(f'Duplicate columns after renaming in {path.name}')
    for col, value in defaults.items():
        if col not in df:
            df[col] = value
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'{path.name} is missing required columns: {", ".join(missing)}')
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required).copy()
    df = df.loc[df['price'].between(lower_price, upper_price) & (df['area'] > 0)].copy()
    if len(df) < 30:
        raise ValueError(f'{path.name} contains fewer than 30 valid rows; cannot train')
    return df.reset_index(drop=True)


def load_india_data():
    path = BASE_DIR / 'data' / 'india_houses.csv'
    if not path.exists():
        print('DEMO ONLY: India CSV missing; training on synthetic data.')
        return _generate_mock_india_data()
    return _load_csv(path, INDIA_REQUIRED, {
        'Price': 'price', 'Area': 'area', 'No. of Bedrooms': 'bedrooms',
        'Bathrooms': 'bathrooms', 'Floor': 'floor', 'Total Floors': 'total_floors',
        'Facing': 'facing', 'City': 'city', 'Age of Property (years)': 'age_years',
        'RERA Approved': 'rera_approved', 'Builder': 'builder'
    }, {'dist_metro': 3., 'dist_bus': 1.5, 'dist_railway': 5., 'dist_highway': 4.,
        'dist_it_park': 8., 'dist_school': 2., 'city': 'Unknown',
        'builder': 'Unknown', 'facing': 'East'}, 100_000, 500_000_000)


def load_us_data():
    path = BASE_DIR / 'data' / 'us_houses.csv'
    if not path.exists():
        print('DEMO ONLY: US CSV missing; training on synthetic data.')
        return _generate_mock_us_data()
    return _load_csv(path, US_REQUIRED, {
        'Price': 'price', 'Area': 'area', 'beds': 'bedrooms', 'baths': 'bathrooms',
        'lotArea': 'lot_area'
    }, {'year_built': 1990, 'year_remodeled': 2000, 'school_rating': 7,
        'walk_score': 60, 'transit_score': 50, 'bike_score': 45,
        'flood_zone': 0, 'fire_zone': 0, 'hoa_fees': 150., 'commute_time': 30.,
        'bsmt_total_sqft': 800., 'bsmt_finished_sqft': 600., 'hvac_age': 8,
        'zipcode': 90210}, 50_000, 10_000_000)


def _generate_mock_india_data(n_samples=5000):
    rng = np.random.default_rng(42)
    area = rng.uniform(400, 3600, n_samples)
    city = rng.choice(['Mumbai', 'Bengaluru', 'Delhi', 'Pune'], n_samples)
    bdr = rng.integers(1, 5, n_samples)
    age = rng.integers(0, 50, n_samples)
    base_per_sqft = pd.Series(city).map({'Mumbai': 17000, 'Bengaluru': 7500, 'Delhi': 10000, 'Pune': 6800}).to_numpy()
    price = np.maximum(1_000_000, area * base_per_sqft * (1 + 0.05 * bdr) * np.exp(-age / 150) * rng.lognormal(0, 0.13, n_samples))
    total_floors = rng.integers(5, 30, n_samples)
    return pd.DataFrame({
        'city': city, 'area': area, 'bedrooms': bdr,
        'bathrooms': rng.integers(1, 5, n_samples),
        'floor': [rng.integers(0, t + 1) for t in total_floors],
        'total_floors': total_floors,
        'facing': rng.choice(['East', 'North', 'West', 'South'], n_samples),
        'dist_metro': rng.uniform(.1, 10, n_samples), 'dist_bus': rng.uniform(.1, 5, n_samples),
        'dist_railway': rng.uniform(1, 15, n_samples), 'dist_highway': rng.uniform(.5, 8, n_samples),
        'dist_it_park': rng.uniform(1, 20, n_samples), 'dist_school': rng.uniform(.2, 5, n_samples),
        'age_years': age, 'rera_approved': rng.integers(0, 2, n_samples),
        'builder': rng.choice(['DLF', 'Godrej', 'Unknown'], n_samples), 'price': price
    })


def _generate_mock_us_data(n_samples=5000):
    rng = np.random.default_rng(100)
    area = rng.uniform(600, 4700, n_samples)
    zipcodes = rng.choice([90210, 10001, 78701, 33101, 98101], n_samples)
    geo = pd.Series(zipcodes).map({90210: 600, 10001: 900, 78701: 270, 33101: 380, 98101: 470}).to_numpy()
    year_built = rng.integers(1950, 2023, n_samples)
    basement_total = rng.uniform(0, 1500, n_samples)
    price = np.maximum(50_000, area * geo * np.exp(-(2026 - year_built) / 200) * rng.lognormal(0, .18, n_samples))
    return pd.DataFrame({
        'zipcode': zipcodes, 'area': area, 'lot_area': rng.uniform(2000, 15000, n_samples),
        'bedrooms': rng.integers(2, 6, n_samples), 'bathrooms': rng.integers(1, 5, n_samples),
        'year_built': year_built,
        'year_remodeled': np.minimum(2026, year_built + rng.integers(0, 10, n_samples)),
        'school_rating': rng.integers(1, 11, n_samples),
        'walk_score': rng.integers(0, 101, n_samples),
        'transit_score': rng.integers(0, 101, n_samples), 'bike_score': rng.integers(0, 101, n_samples),
        'flood_zone': rng.integers(0, 2, n_samples), 'fire_zone': rng.integers(0, 2, n_samples),
        'hoa_fees': rng.uniform(0, 500, n_samples), 'commute_time': rng.uniform(10, 90, n_samples),
        'bsmt_total_sqft': basement_total,
        'bsmt_finished_sqft': basement_total * rng.uniform(0, 1, n_samples),
        'hvac_age': rng.integers(1, 20, n_samples), 'price': price
    })


# CatBoost 1.2.x doesn't expose the scikit-learn 1.8 estimator tags when used
# directly within StackingRegressor; this small adapter supplies sklearn tags.
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor


class CatBoostAdapter(RegressorMixin, BaseEstimator):
    def __init__(self, iterations=200, learning_rate=.05, verbose=False,
                 random_state=42, thread_count=1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.learning_rate,
            verbose=self.verbose, random_state=self.random_state,
            thread_count=self.thread_count)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
