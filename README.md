# Global PropPredict AI — Corrected project

An India + US real-estate **estimation prototype** with a FastAPI backend, a stacked regressor, XGBoost-component SHAP explanations, held-out calibrated approximate 80% prediction intervals, and an HTML dashboard.

> **Important:** The ZIP contains **no real training CSVs or production-trained models**. On first launch, it trains with **synthetic demonstration data**. Synthetic results are not real-world valuations or demonstrated 95% accuracy. Supply and validate representative real datasets before relying on any market estimate.

## Original folder structure (preserved)

```text
proppredict/
├── model_usa.py
├── requirements.txt
├── README.md
├── index.html
├── model_india.py
├── api.py
├── feature_engineering.py
└── tests/
    ├── test_regressions.py     # 10 fast automated checks
    └── browser_smoke.py        # optional real-Chromium end-to-end checks
```

Files generated automatically when you run the project (not included in this ZIP): `india_model.joblib`, `us_model.joblib`, `*_best_params.json`, `data/`, `catboost_info/`, and Python caches.

## Install & launch (macOS M4 or other supported Python machine)

Use **Python 3.11–3.13** with a fresh virtual environment:

```bash
cd proppredict
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m uvicorn api:app --reload
```

Once training finishes, visit **http://127.0.0.1:8000/**. API documentation is at **http://127.0.0.1:8000/docs**. Serving the dashboard with FastAPI is recommended. If you use VS Code Live Server on `localhost:5500` (or another localhost port), the browser targets FastAPI at `http://127.0.0.1:8000` automatically. Production deployments use the current origin; optionally override with `window.PROPPREDICT_API_URL` before loading the page's main script.

### Faster smoke check (small synthetic models; NOT for valuation)

For development/testing only, instead of the last launch command run:

```bash
PROPPREDICT_QUICK_TRAIN=1 python -m uvicorn api:app
```

Quick mode only changes training when the `.joblib` files do not exist. For full training, remove the quick-mode artifacts first:

```bash
rm -f india_model.joblib us_model.joblib
python -m uvicorn api:app --reload
```

### Train with your own CSVs

Place datasets in `data/india_houses.csv` and `data/us_houses.csv`. Both require a `price` target (not a price-per-square-foot proxy) and numeric property characteristics. You may use the original `Price` / `Area` and the mapped headings supported in `feature_engineering.py`. Missing simple India distance columns are assigned documented defaults. Unsupported columns are not included in modeling.

**India required numeric fields** (or supported aliases): `price, area, bedrooms, bathrooms, floor, total_floors, dist_metro, dist_bus, dist_railway, dist_highway, dist_it_park, dist_school, age_years, rera_approved`. Also provide `city, facing, builder` if available (defaults otherwise).

**USA required numeric fields** (or supported aliases): `price, area, lot_area, bedrooms, bathrooms, year_built, year_remodeled, school_rating, walk_score, transit_score, bike_score, flood_zone, fire_zone, hoa_fees, commute_time, bsmt_total_sqft, bsmt_finished_sqft, hvac_age`. Add `zipcode` if available; default 90210 is used otherwise.

**Model artifact compatibility:** This release uses schema version **3**; existing version-2 models MUST be retrained, because prediction intervals now require calibration metadata. Remove old cached models after changing the dataset, feature code, or upgrading:

```bash
rm -f india_model.joblib us_model.joblib india_best_params.json us_best_params.json
python -m uvicorn api:app --reload
```

Optional retune: `FORCE_TUNE=1 python -m uvicorn api:app`. First **full** launch may take considerable time because tuning and two ensembles are trained; there is no guaranteed duration.

## Prediction intervals (important)

Three disjoint subsets are used: the model-fitting subset, a **separate calibration subset**, and a held-out test subset. The model predicts a price plus quantile-based candidate bounds, expands the bounds to include the point prediction, and applies a conformal adjustment calculated from calibration-only errors. The **nominal target is roughly 80% marginal coverage**, contingent on exchangeability with the calibration data. This is **not** a guarantee for any individual property, and independent real-market validation is necessary. Synthetic data cannot establish coverage or accuracy in real markets.

## Optional real browser test

After the API is running and the first models have trained, on your Mac install the optional browser-testing dependency and execute:

```bash
python -m pip install playwright
python -m playwright install chromium
python tests/browser_smoke.py
```

This opens Chromium automatically and tests the actual FastAPI-served website, with genuine Chart.js and Tailwind network loading where available. Add `--show-browser` to watch the test; some CDN visual features require an internet connection. A sandboxed functional browser test has already checked the interactions, but our hosted environment blocks direct browser navigation and CDN loading, so **full native visual/network testing still needs to be performed on the deployment machine**.

## What was fixed

Additionally verified and fixed: normal (non-quick) USA model training called `tune_xgb_us()` and `tune_lgb_us()`, but the implemented functions were named `tune_xgb_usa()` and `tune_lgb_usa()`. Both calls now use the correct names. A regression check covers this path; the normal training branch was also smoke-tested with a small synthetic sample and stubbed tuner results.

1. **Price/target leakage:** removed engineered features that directly used the sale price (including India `emi_band`, US `hoa_value_penalty` / `price_rent_ratio`). Only declared property inputs feed the model. Metrics are evaluated on a held-out test split, not the training set.
2. **Broken category encoding:** training-only out-of-fold encoding for India city/builder/facing and US ZIP; saved training maps are used during **both** individual and CSV inference. The holdout labels do not contribute to the encoding maps.
3. **SHAP extraction and fabricated values:** corrected the fitted XGBoost learner lookup via `model.named_estimators_['xgb']`. SHAP charts explicitly explain the XGBoost **base learner**, **not** the full ensemble. If explanations fail, the API returns an empty list and a clear status—never random numbers.
4. **Stacking / dependency compatibility:** added a scikit-learn-compatible CatBoost adapter (necessary on tested scikit-learn 1.8 / CatBoost 1.2.8 installation).
5. **Fake frontend results:** removed fabricated predictions, investment scores, chart projections, bulk CSV results, and comparisons on connection failures. Errors are now shown instead.
6. **Validation and consistent inference:** requests validate input ranges, inconsistent floor/years/basement values, CSV upload type/size/rows; single and bulk use the same feature pipeline. USA ZIP is now editable in the main form.
7. **Rent/buy calculation:** remaining loan principal and equity are included. The projection uses illustrative assumptions, not a trained forecast.

8. **Uncalibrated/contradictory prediction bounds:** added a disjoint conformal calibration split. Every displayed interval now includes the point estimate. The interval has an approximate 80% nominal marginal coverage goal, **not** an exact per-property guarantee.
9. **Investment scenario assumptions:** growth, rental yield, infrastructure and an asking price are provided by the user; the UI no longer invents a comparable investment score. Calculations are illustrative, not trained investor rankings.
10. **Editable property data:** exposed previously hardcoded Indian access/age/RERA and USA renovation/commute/risk/basement/HVAC values; the main form sends the entered values to the API.
11. **Live Server origin:** localhost frontend ports route API calls to 8000; deployed same-origin hosting continues to work.
12. **Graceful charts and validation:** if Chart.js is unavailable, text results still appear. HTTP 422 messages display the actual input error rather than an API-disconnected warning.

## API routes

- `GET /health` — server status and whether data was synthetic or real
- `POST /predict/india` — India prediction, calibrated approximate 80% interval and component SHAP
- `POST /predict/usa` — USA prediction, calibrated approximate 80% interval and component SHAP
- `POST /predict/bulk?mode=india` (or `usa`) — CSV upload with 1–5000 rows and required feature columns
- `POST /score/investment` — illustrative investment heuristic, not model-trained
- `POST /calc/rent-vs-buy` — illustrative simplified 30-year ownership/rent comparison
- `POST /predict/future` — user-region assumption-based growth, not a learned forecast

Model results are approximate. The calibrated interval aims for about 80% marginal coverage on similar data, but synthetic test coverage does not establish actual real-world coverage; verify on representative unseen real data. Investment score parameters and projected growth rates are illustrative assumptions, and rent/buy calculations exclude ownership taxes, maintenance, insurance, fees and investment returns. Do not treat this app as an authoritative appraisal or financial recommendation.
