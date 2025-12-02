# GBNet forecaster

This folder holds the browser-only build for the XGBoost-only forecaster. It intentionally avoids the rest of `gbnet` (especially torch-dependent code) so the web app only ships what Pyodide (a browser-based python environment) can run.

Layout:
- `app/`: Minimal HTML/JS that boots and runs a demo forecast.
- `py/`: Python sources copied from `gbnet` that are safe for in-browser running (`forecasting_xgb_only.py` only).

Quick start:
1. Visit `https://mthorrell.github.io/gbnet/web/app/`. Or serve on your own with `python -m http.server 8000 --directory web`.
