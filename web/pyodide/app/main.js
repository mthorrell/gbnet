import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.26.1/full/pyodide.mjs";

const PYODIDE_VERSION = "0.26.1";
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const DATASET_URL =
  "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv";

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const bootBtn = document.getElementById("boot");
const resultsEl = document.getElementById("results");

const setStatus = (msg) => {
  if (statusEl) statusEl.textContent = msg;
};

const appendLog = (msg) => {
  if (!logEl) return;
  const stamp = new Date().toISOString();
  logEl.textContent += `[${stamp}] ${msg}\n`;
};

const ensureDir = (fs, path) => {
  const segments = path.split("/").filter(Boolean);
  let current = "";
  for (let i = 0; i < segments.length - 1; i += 1) {
    current += `/${segments[i]}`;
    if (!fs.analyzePath(current).exists) {
      fs.mkdir(current);
    }
  }
};

const writeTextFile = async (pyodide, url, destPath) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  const text = await res.text();
  ensureDir(pyodide.FS, destPath);
  pyodide.FS.writeFile(destPath, text);
};

const smokeTest = async (pyodide) => {
  setStatus("Running smoke test...");
  await pyodide.runPythonAsync(`
from forecasting_xgb_only import ForecastXGBOnly
print("ForecastXGBOnly import OK:", ForecastXGBOnly)
`);
};

const runDemoForecast = async (pyodide) => {
  setStatus("Fetching sample data...");
  await writeTextFile(pyodide, DATASET_URL, "/data/air_passengers.csv");

  setStatus("Training and forecasting...");
  const result = await pyodide.runPythonAsync(`
import pandas as pd
from forecasting_xgb_only import ForecastXGBOnly

df = pd.read_csv("/data/air_passengers.csv")
df["ds"] = pd.to_datetime(df["ds"])

m = ForecastXGBOnly(nrounds=50)
m.fit(df, df["y"])

future = pd.DataFrame({
    "ds": pd.date_range(
        df["ds"].max() + pd.offsets.MonthBegin(1),
        periods=12,
        freq="MS",
    )
})

forecast = m.predict(future)
train_pred = m.predict(df)

out = {
    "train_tail": train_pred.tail(5)
        .assign(ds=lambda d: d["ds"].dt.strftime("%Y-%m-%d"))
        [["ds", "y", "yhat"]]
        .to_dict("records"),
    "forecast": forecast
        .assign(ds=lambda d: d["ds"].dt.strftime("%Y-%m-%d"))
        [["ds", "yhat"]]
        .to_dict("records"),
}
out
`);

  const jsResult = result.toJs({ dict: true });
  renderResults(jsResult);
  result.destroy?.();
};

const renderResults = (data) => {
  if (!resultsEl) return;
  const trainRows = (data?.train_tail || [])
    .map((row) => `${row.ds}: y=${row.y}, yhat=${row.yhat.toFixed(2)}`)
    .join("\n");
  const forecastRows = (data?.forecast || [])
    .map((row) => `${row.ds}: yhat=${row.yhat.toFixed(2)}`)
    .join("\n");

  resultsEl.textContent = [
    "Recent fit (tail of training):",
    trainRows || "(none)",
    "",
    "Forecast (next 12 months):",
    forecastRows || "(none)",
  ].join("\n");
};

const boot = async () => {
  bootBtn.disabled = true;
  try {
    setStatus("Loading Pyodide runtime...");
    const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });

    setStatus("Loading base packages...");
    await pyodide.loadPackage([
      "numpy",
      "pandas",
      "scipy",
      "scikit-learn",
      "xgboost",
    ]);

    setStatus("Staging forecasting_xgb_only.py...");
    await writeTextFile(
      pyodide,
      "../py/forecasting_xgb_only.py",
      "/py/forecasting_xgb_only.py",
    );
    pyodide.runPython(`import sys; sys.path.insert(0, "/py")`);

    await smokeTest(pyodide);
    await runDemoForecast(pyodide);
    setStatus("Ready. Demo forecast computed.");
  } catch (err) {
    console.error(err);
    setStatus("Error while loading");
    appendLog(err?.stack ?? String(err));
    bootBtn.disabled = false;
  }
};

if (bootBtn) {
  bootBtn.addEventListener("click", () => {
    boot();
  });
}
