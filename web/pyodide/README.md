# Pyodide target for ForecastXGBOnly

This folder holds the browser-only build for the XGBoost-only forecaster. It intentionally avoids the rest of `gbnet` (especially torch-dependent code) so the web app only ships what Pyodide can run.

Layout:
- `app/`: Minimal HTML/JS that boots Pyodide and runs a demo forecast.
- `py/`: Python sources copied from `gbnet` that are safe for Pyodide (`forecasting_xgb_only.py` only).

Quick start:
1. Serve this folder: `python -m http.server 8000 --directory web/pyodide`.
2. Visit `http://localhost:8000/app/index.html`. The page loads Pyodide, installs numpy/pandas/scipy/scikit-learn/xgboost from the Pyodide index, fetches the Prophet air passengers example CSV, trains `ForecastXGBOnly`, and prints a 12-month forecast.

Notes:
- Keep the main `gbnet` pip package unchanged; this folder is browser-only.
- If you update `gbnet/models/forecasting_xgb_only.py`, re-copy it into `web/pyodide/py/` before rebuilding the web assets.
