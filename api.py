"""PropPredict FastAPI service. All predictions use training-time preprocessing."""
import io
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from feature_engineering import (apply_encoding_map, engineer_india_features,
                                 engineer_us_features, calibrated_interval)

BASE_DIR = Path(__file__).resolve().parent


def _load_bundle(mode):
    path = BASE_DIR / f'{mode}_model.joblib'
    if not path.exists():
        print(f'{path.name} missing; training {mode} model...')
        if mode == 'india':
            from model_india import train_india_model
            train_india_model()
        else:
            from model_usa import train_us_model
            train_us_model()
    bundle = joblib.load(path)  # Only load model files you trust.
    if bundle.get('schema_version') != 3:
        raise RuntimeError(f'{path.name} uses an obsolete pipeline; delete this .joblib file and restart to retrain')
    return bundle


india_data = _load_bundle('india')
us_data = _load_bundle('us')
BUNDLES = {'india': india_data, 'usa': us_data}

# SHAP on the fitted XGBoost *base learner*, not the stacked prediction.
# If initialization fails, DO NOT replace explanations with random numbers.
EXPLAINERS = {}
try:
    import shap
    for mode, bundle in BUNDLES.items():
        try:
            EXPLAINERS[mode] = shap.TreeExplainer(bundle['model'].named_estimators_['xgb'])
        except Exception as exc:
            print(f'SHAP {mode} unavailable: {exc}')
            EXPLAINERS[mode] = None
except ImportError:
    EXPLAINERS = {'india': None, 'usa': None}

app = FastAPI(title='Global House Price ML API', version='2.1')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['null', 'http://localhost:8000', 'http://127.0.0.1:8000'],
    allow_origin_regex=r'http://(localhost|127\.0\.0\.1)(:\d+)?',
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type'],
)


@app.get('/', include_in_schema=False)
async def serve_index():
    return FileResponse(BASE_DIR / 'index.html')


@app.get('/health')
async def health():
    return {'status': 'ok', 'models': {k: v['source'] for k, v in BUNDLES.items()}}


class IndiaHouseRequest(BaseModel):
    area: float = Field(gt=0, le=1000000)
    bedrooms: int = Field(ge=0, le=50)
    bathrooms: int = Field(ge=0, le=50)
    floor: int = Field(ge=0, le=300)
    total_floors: int = Field(ge=1, le=300)
    dist_metro: float = Field(ge=0)
    dist_bus: float = Field(ge=0)
    dist_railway: float = Field(ge=0)
    dist_highway: float = Field(ge=0)
    dist_it_park: float = Field(ge=0)
    dist_school: float = Field(ge=0)
    age_years: int = Field(ge=0, le=300)
    rera_approved: int = Field(ge=0, le=1)
    listing_price: float = Field(default=0, ge=0)
    city: str = Field(default='Bengaluru', min_length=1)
    facing: str = 'East'
    builder: str = 'Unknown'

    @model_validator(mode='after')
    def floor_not_above_building(self):
        if self.floor > self.total_floors:
            raise ValueError('Property floor cannot exceed total floors')
        return self


class USHouseRequest(BaseModel):
    area: float = Field(gt=0, le=1000000)
    lot_area: float = Field(ge=0)
    bedrooms: int = Field(ge=0, le=50)
    bathrooms: int = Field(ge=0, le=50)
    year_built: int = Field(ge=1700, le=2100)
    year_remodeled: int = Field(ge=1700, le=2100)
    school_rating: int = Field(ge=1, le=10)
    walk_score: int = Field(ge=0, le=100)
    transit_score: int = Field(ge=0, le=100)
    bike_score: int = Field(ge=0, le=100)
    flood_zone: int = Field(ge=0, le=1)
    fire_zone: int = Field(ge=0, le=1)
    hoa_fees: float = Field(ge=0)
    commute_time: float = Field(ge=0)
    bsmt_total_sqft: float = Field(ge=0)
    bsmt_finished_sqft: float = Field(ge=0)
    hvac_age: int = Field(ge=0, le=100)
    listing_price: float = Field(default=0, ge=0)
    zipcode: int = Field(default=90210, ge=501, le=99950)

    @model_validator(mode='after')
    def valid_building(self):
        if self.year_remodeled < self.year_built:
            raise ValueError('Remodel year cannot precede build year')
        if self.bsmt_finished_sqft > self.bsmt_total_sqft:
            raise ValueError('Finished basement exceeds total basement area')
        return self


class InvestmentScoreRequest(BaseModel):
    predicted_price: float = Field(gt=0)
    locality_cagr: float = Field(ge=0, le=0.30)
    rental_yield: float = Field(ge=0, le=0.20)
    infrastructure_score: float = Field(ge=0, le=1)
    price_vs_fair_value: float = Field(ge=-1, le=10)


class RentVsBuyRequest(BaseModel):
    property_price: float = Field(gt=0)
    monthly_rent: float = Field(ge=0)
    down_payment_pct: float = Field(ge=0, le=1)
    loan_interest_rate: float = Field(ge=0, le=1)
    loan_tenure_years: int = Field(ge=1, le=50)
    annual_price_growth: float = Field(gt=-1, le=1)
    annual_rent_growth: float = Field(gt=-1, le=1)
    currency: Literal['INR', 'USD']


class FuturePriceRequest(BaseModel):
    current_price: float = Field(gt=0)
    city_or_state: str = Field(min_length=1)
    years: int = Field(default=5, ge=1, le=50)


def _model_matrix(mode, rows):
    bundle = BUNDLES[mode]
    raw = pd.DataFrame(rows)
    engineered = (engineer_india_features if mode == 'india' else engineer_us_features)(raw)
    engineered = apply_encoding_map(engineered, bundle['encoding_map'])
    X = engineered.reindex(columns=bundle['features'])
    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return X.fillna(bundle['feature_medians']).fillna(0)


def _predict_matrix(mode, X):
    bundle = BUNDLES[mode]
    price = np.maximum(0, np.expm1(bundle['model'].predict(X)))
    q10 = np.maximum(0, np.expm1(bundle['q10_model'].predict(X)))
    q90 = np.maximum(0, np.expm1(bundle['q90_model'].predict(X)))
    if not all(np.all(np.isfinite(z)) for z in (price, q10, q90)):
        raise ValueError('Model produced a non-finite result')
    low, high = calibrated_interval(price, q10, q90, bundle['interval_adjustment'])
    return price, low, high


def _explain(mode, X):
    explainer = EXPLAINERS.get(mode)
    if explainer is None:
        return [], 'unavailable'
    try:
        shap_values = np.asarray(explainer.shap_values(X))
        if shap_values.ndim != 2 or shap_values.shape[1] != len(X.columns):
            raise ValueError('Unexpected SHAP shape')
        vals = sorted(({'feature': name, 'impact': float(value)}
                       for name, value in zip(X.columns, shap_values[0])),
                      key=lambda z: abs(z['impact']), reverse=True)
        return vals[:10], 'XGBoost base learner (log-price), not entire ensemble'
    except Exception as exc:
        print(f'SHAP explanation failed for {mode}: {exc}')
        return [], 'unavailable'


def _negotiation(listing, price):
    if listing <= 0 or price <= 0:
        return None
    difference = (listing - price) / price * 100
    if difference > 10:
        return {'status': 'Overpriced', 'diff_pct': round(difference, 1),
                'suggested_offer': round(price * 1.03),
                'advice': 'Above the model estimate; compare local listings before negotiating.'}
    if difference < -5:
        return {'status': 'Underpriced', 'diff_pct': round(difference, 1),
                'suggested_offer': round(listing * .98),
                'advice': 'Below the model estimate; verify condition and comparable sales.'}
    return {'status': 'Fairly Priced', 'diff_pct': round(difference, 1),
            'suggested_offer': round(listing * .97),
            'advice': 'Near the model estimate; verify with local comparable properties.'}


def _predict_one(mode, data):
    X = _model_matrix(mode, [data.model_dump()])
    predictions, lows, highs = _predict_matrix(mode, X)
    shap_values, scope = _explain(mode, X)
    estimate = float(predictions[0])
    return {
        'prediction': round(estimate, 2),
        'range_low': round(float(lows[0]), 2),
        'range_high': round(float(highs[0]), 2),
        'currency': 'INR' if mode == 'india' else 'USD',
        'shap_values': shap_values, 'explanation_scope': scope,
        'data_source': BUNDLES[mode]['source'],
        'negotiation': _negotiation(data.listing_price, estimate),
    }


@app.post('/predict/india')
async def predict_india(data: IndiaHouseRequest):
    try:
        return _predict_one('india', data)
    except Exception as exc:
        raise HTTPException(500, detail='Prediction failed; check server logs') from exc


@app.post('/predict/usa')
async def predict_usa(data: USHouseRequest):
    try:
        return _predict_one('usa', data)
    except Exception as exc:
        raise HTTPException(500, detail='Prediction failed; check server logs') from exc


@app.post('/score/investment')
async def calculate_investment_score(data: InvestmentScoreRequest):
    # These values are explicit user-supplied assumptions, NOT model-derived inputs.
    # Growth, yield and discounts are scored relative to illustrative caps.
    score = (min(data.locality_cagr / .15, 1) * 30 +
             min(data.rental_yield / .10, 1) * 25 +
             data.infrastructure_score * 25 +
             min(max(-data.price_vs_fair_value, 0) / .25, 1) * 20)
    score = round(min(100, max(0, score)), 1)
    verdict, color = (('Excellent', 'green') if score >= 80 else
                      ('Good', 'cyan') if score >= 60 else
                      ('Average', 'amber') if score >= 40 else ('Poor', 'red'))
    return {'investment_score': score, 'verdict': verdict, 'color': color,
            'disclaimer': 'Illustrative scenario score using user-entered assumptions; not an investment recommendation.'}


@app.post('/calc/rent-vs-buy')
async def rent_vs_buy(data: RentVsBuyRequest):
    principal = data.property_price * (1 - data.down_payment_pct)
    down_payment = data.property_price * data.down_payment_pct
    monthly_rate = data.loan_interest_rate / 12
    periods = data.loan_tenure_years * 12
    growth_factor = (1 + monthly_rate) ** periods
    emi = (principal * monthly_rate * growth_factor / (growth_factor - 1)
           if monthly_rate > 0 else principal / periods)
    rent_total = 0
    first_better_year = None
    yearly = []
    for year in range(1, 31):
        elapsed = min(12 * year, periods)
        rent_total += data.monthly_rent * 12 * (1 + data.annual_rent_growth) ** (year - 1)
        value = data.property_price * (1 + data.annual_price_growth) ** year
        if monthly_rate == 0:
            balance = max(0, principal - emi * elapsed)
        else:
            acc = (1 + monthly_rate) ** elapsed
            balance = max(0, principal * acc - emi * (acc - 1) / monthly_rate)
        equity = value - balance
        # Simplified ownership net outlay = cash spent minus property equity.
        # Excludes taxes, maintenance, insurance, transaction fees and rent-investment returns.
        buy_net = down_payment + emi * elapsed - equity
        yearly.append({'year': year, 'buy_net_cost': round(buy_net, 2),
                       'rent_cumulative': round(rent_total, 2)})
        if first_better_year is None and buy_net < rent_total:
            first_better_year = year
    verdict = (f'Buying has lower modeled net cost from year {first_better_year}.'
               if first_better_year is not None else
               'Renting has lower modeled net cost within 30 years.')
    return {'monthly_emi': round(emi, 2), 'breakeven_year': first_better_year,
            'verdict': verdict, 'yearly_data': yearly,
            'disclaimer': 'Illustrative only: taxes, maintenance, fees and investment returns excluded.'}


@app.post('/predict/future')
async def future_price(data: FuturePriceRequest):
    cagr_map = {'mumbai': .09, 'bengaluru': .11, 'delhi': .08,
                'pune': .10, 'hyderabad': .12, 'chennai': .09,
                'kolkata': .07, 'california': .06, 'new york': .05,
                'texas': .07, 'florida': .08, 'washington': .06}
    city = data.city_or_state.strip().lower()
    cagr = cagr_map.get(city, .06 if len(city) == 2 or city in ('usa', 'us') else .08)
    value = data.current_price
    steps = []
    for year in range(1, data.years + 1):
        value *= 1 + cagr
        steps.append({'year': year, 'price': round(value, 2)})
    return {'cagr_used': cagr, 'projections': steps,
            'final_value': round(value, 2),
            'total_growth_pct': round((value / data.current_price - 1) * 100, 1),
            'disclaimer': 'Illustrative assumed growth rate; not a learned forecast.'}


@app.post('/predict/bulk')
async def predict_bulk(file: UploadFile = File(...), mode: Literal['india', 'usa'] = 'india'):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, detail='Please provide a .csv file')
    contents = await file.read(5_000_001)
    if len(contents) > 5_000_000:
        raise HTTPException(413, detail='CSV exceeds 5 MB')
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise HTTPException(400, detail=f'Invalid CSV: {exc}') from exc
    if df.empty or len(df) > 5000:
        raise HTTPException(400, detail='CSV must contain between 1 and 5000 data rows')
    schema = IndiaHouseRequest if mode == 'india' else USHouseRequest
    rows = []
    from pydantic import ValidationError
    for index, record in enumerate(df.to_dict('records'), 1):
        try:
            rows.append(schema.model_validate(record).model_dump())
        except ValidationError as exc:
            raise HTTPException(422, detail=f'CSV row {index}: {exc.errors()[0]["msg"]}') from exc
    try:
        X = _model_matrix(mode, rows)
        preds, lows, highs = _predict_matrix(mode, X)
    except Exception as exc:
        raise HTTPException(500, detail='Bulk prediction failed; check server logs') from exc
    return {'total_rows': len(rows), 'currency': 'INR' if mode == 'india' else 'USD',
            'data_source': BUNDLES[mode]['source'],
            'predictions': [{'row_index': i + 1, 'prediction': round(float(preds[i]), 2),
                             'range_low': round(float(lows[i]), 2),
                             'range_high': round(float(highs[i]), 2)}
                            for i in range(len(rows))]}
