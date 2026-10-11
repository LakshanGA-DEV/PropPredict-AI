# PropPredict AI — QA summary (2026-10-11)

## What was tested in this environment

- **Python source:** syntax compilation succeeded.
- **JavaScript:** Node.js syntax check succeeded.
- **Regression unit tests:** **10 / 10 passed**.
- **Fresh quick mode model training:** both India and USA models trained from scratch on generated **synthetic** data.
- **Backend HTTP API:** health, India and USA single predictions, investment scenario, rent-vs-buy, future projection, validation error handling, and both bulk prediction endpoints exercised.
- **Prediction interval consistency:** 180 synthetic India examples and 180 synthetic USA examples processed by the bulk API; **0 / 360 point predictions** were outside their displayed intervals. Sampled single predictions matched bulk predictions.
- **Browser functionality:** **9 / 9 automated Chromium UI test workflows passed**, covering dashboard/health, India inputs/prediction, scenario score, rent-vs-buy/future chart callbacks, USA inputs/prediction, bulk CSV upload/result/download, comparisons, and invalid input.
- **Chart-library failure:** tested in Chromium with Chart.js missing; the app displayed fallback messages without an uncaught JavaScript exception.
- **Development origins:** inspected and evaluated API URLs for localhost ports 5500/5501/8000, a deployed HTTPS domain, and `file://`.

## Important browser test limitation

The sandbox's Chromium was blocked from navigating to localhost and file URLs by administrator policy. To test interactions, the HTML was loaded into Chromium via `set_content`, API requests were forwarded to the **real locally running FastAPI server**, and external Tailwind/Chart.js assets were replaced with tiny test substitutes. The bulk-upload browser request had its multipart payload reconstructed in the test transport; the real bulk API was also independently tested using native multipart requests.

Therefore the checks validate **browser JavaScript, forms, event handlers, API integration, and return values**, but DO NOT certify actual third-party Chart.js rendering, Tailwind layout, Safari, mobile sizing, or live browser navigation in the deployment environment. For that, after starting the server on your machine run `python tests/browser_smoke.py` (installation documented in README).

## Synthetic quick-training metrics (NOT real-estate accuracy)

| Measure | India | USA |
|---|---:|---:|
| Synthetic holdout rows | 36 | 36 |
| MAE | ₹3,044,802 | $252,621 |
| MAPE | 13.19% | 20.18% |
| R² | 0.9021 | 0.7328 |
| Approx. 80% interval empirical coverage | 86.1% | 77.8% |

**These results are from synthetic data and cannot be used to claim live market accuracy.** Real-world price estimation requires clean representative housing sales data, retraining, held-out evaluation, suitable location features, and a market-specific analysis of coverage and bias.

## Remaining deployment work

1. Provide `data/india_houses.csv` and `data/us_houses.csv` with the columns documented in README, then retrain models. The ZIP intentionally does not include temporary synthetic-trained `.joblib` files.
2. Run first-party browser smoke tests on your Mac with a working internet connection for the externally hosted JS/CSS resources.
3. Inspect model metrics on genuinely unseen, realistic market listings, confirm interval coverage, verify mobile Safari and responsive visual layout, and test the full non-quick tuning pipeline on the intended deployment machine.

**Status:** functional, smoke-tested prototype; not certified bug-free or production market-ready.
