import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.26.1/full/pyodide.mjs";

const PYODIDE_VERSION = "0.26.1";
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const DATASET_URL =
  "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv";

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const bootBtn = document.getElementById("boot");
const resultsMetaEl = document.getElementById("results-meta");
const forecastTableEl = document.getElementById("forecast-table");
const copyForecastBtn = document.getElementById("copy-forecast-btn");
const viewAllBtn = document.getElementById("view-all-btn");
const viewForecastBtn = document.getElementById("view-forecast-btn");
const fileInputEl = document.getElementById("file-input");
const demoBtnEl = document.getElementById("demo-data-btn");
const horizonInputEl = document.getElementById("horizon-input");
const runForecastBtn = document.getElementById("run-forecast-btn");
const dataSummaryEl = document.getElementById("data-summary");
const errorsEl = document.getElementById("errors");
const chartCanvas = document.getElementById("chart-forecast");

let pyodideInstance = null;
let currentDatasetPath = null;
let currentDatasetLabel = null;
let forecastChart = null;
let lastResult = null;
let chartViewMode = "forecast"; // "forecast" (zoomed) or "all"

const setStatus = (msg) => {
  if (statusEl) statusEl.textContent = msg;
};

const setError = (msg) => {
  if (!errorsEl) return;
  if (msg) {
    errorsEl.textContent = msg;
    errorsEl.style.display = "block";
  } else {
    errorsEl.textContent = "";
    errorsEl.style.display = "none";
  }
};

const appendLog = (msg) => {
  if (!logEl) return;
  const stamp = new Date().toISOString();
  logEl.textContent += `[${stamp}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
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

const renderMeta = (data) => {
  if (!resultsMetaEl) return;
  resultsMetaEl.textContent = "";
  const meta = data?.meta || {};
  const nObs = meta.n_obs;
  const horizon = meta.horizon;
  const dsMin = meta.ds_min;
  const dsMax = meta.ds_max;
  const fMin = meta.forecast_ds_min;
  const fMax = meta.forecast_ds_max;

  const bits = [];
  if (nObs != null) {
    bits.push(`obs: ${nObs}`);
  }
  if (horizon != null) {
    bits.push(`horizon: ${horizon}`);
  }
  if (dsMin && dsMax) {
    bits.push(`train: ${dsMin} → ${dsMax}`);
  }
  if (fMin && fMax) {
    bits.push(`forecast: ${fMin} → ${fMax}`);
  }
  if (currentDatasetLabel) {
    bits.push(`dataset: ${currentDatasetLabel}`);
  }

  if (!bits.length) return;

  bits.forEach((text) => {
    const span = document.createElement("span");
    span.innerHTML = `<strong>•</strong> ${text}`;
    resultsMetaEl.appendChild(span);
  });
};

const renderChart = (data) => {
  if (!chartCanvas || typeof Chart === "undefined") return;

  const ctx = chartCanvas.getContext("2d");
  const train = data?.train || [];
  const forecast = data?.forecast || [];

  const labels = [...train.map((d) => d.ds), ...forecast.map((d) => d.ds)];
  if (!labels.length) {
    if (forecastChart) {
      forecastChart.destroy();
      forecastChart = null;
    }
    return;
  }

  const actual = [
    ...train.map((d) => (d.y != null ? d.y : null)),
    ...forecast.map(() => null),
  ];
  const nTrain = train.length;
  const yhat = [
    ...train.map((d) => (d.yhat != null ? d.yhat : null)),
    ...forecast.map((d) => (d.yhat != null ? d.yhat : null)),
  ];

  const hasUncertainty = forecast.some(
    (d) => d.yhat_lower != null || d.yhat_upper != null,
  );

  const total = labels.length;
  const nForecast = forecast.length;

  const lowerBand = hasUncertainty
    ? [
        ...train.map(() => null),
        ...forecast.map((d) =>
          d.yhat_lower != null ? d.yhat_lower : d.yhat != null ? d.yhat : null,
        ),
      ]
    : null;

  const upperBand = hasUncertainty
    ? [
        ...train.map(() => null),
        ...forecast.map((d) =>
          d.yhat_upper != null ? d.yhat_upper : d.yhat != null ? d.yhat : null,
        ),
      ]
    : null;

  const dataConfig = {
    labels,
    datasets: [],
  };

  dataConfig.datasets.push({
    label: "Actual",
    data: actual,
    borderColor: "#4b5563",
    backgroundColor: "rgba(75, 85, 99, 0.15)",
    tension: 0.15,
    pointRadius: 0,
    spanGaps: true,
  });

  dataConfig.datasets.push({
    label: "Forecast",
    data: yhat,
    borderColor: "#0070f3",
    backgroundColor: "rgba(0, 112, 243, 0.12)",
    tension: 0.15,
    pointRadius: 0,
    spanGaps: true,
  });

  if (hasUncertainty && lowerBand && upperBand) {
    // Lower bound (used as the base for the filled band)
    dataConfig.datasets.push({
      label: "",
      data: lowerBand,
      borderColor: "rgba(37, 99, 235, 0)", // invisible line
      backgroundColor: "rgba(37, 99, 235, 0)", // no fill here
      tension: 0.15,
      pointRadius: 0,
      spanGaps: true,
    });

    // Upper bound, filled down to the previous dataset (lower bound)
    dataConfig.datasets.push({
      label: "95% interval",
      data: upperBand,
      borderColor: "rgba(37, 99, 235, 0)",
      backgroundColor: "rgba(59, 130, 246, 0.15)",
      fill: "-1",
      tension: 0.15,
      pointRadius: 0,
      spanGaps: true,
    });
  }

  // Configure the default x-axis window.
  // "forecast" mode: forecast span occupies roughly the last 25% of the plot.
  // "all" mode: show full history.
  let xMin = undefined;
  let xMax = undefined;
  if (chartViewMode === "forecast" && nForecast > 0 && total > 0) {
    // Choose a window where the forecast takes ~25% of the width:
    // visibleSpan ≈ 4 * nForecast, anchored at the end.
    const visibleSpan = Math.min(total, Math.max(nForecast * 4, 1));
    xMax = total - 1;
    xMin = Math.max(0, xMax - visibleSpan + 1);
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Time",
        },
        min: xMin,
        max: xMax,
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Value",
        },
      },
    },
    plugins: {
      legend: {
        display: true,
      },
      tooltip: {
        mode: "index",
        intersect: false,
      },
    },
  };

  if (!forecastChart) {
    forecastChart = new Chart(ctx, {
      type: "line",
      data: dataConfig,
      options,
    });
  } else {
    forecastChart.data = dataConfig;
    forecastChart.options = options;
    forecastChart.update();
  }
};

const buildForecastTable = (data) => {
  const forecast = data?.forecast || [];
  if (!forecast.length) {
    return "No forecast data. Run a forecast to populate this table.";
  }

  const columns = ["ds", "y", "yhat", "yhat_lower", "yhat_upper"];
  const header = columns.join("\t");

  const rows = forecast.map((row) =>
    columns
      .map((col) => {
        const v = row[col];
        if (v == null) return "";
        if (typeof v === "number") return String(v);
        return String(v);
      })
      .join("\t"),
  );

  return [header, ...rows].join("\n");
};

const renderForecastTable = (data) => {
  if (!forecastTableEl) return;
  const text = buildForecastTable(data);
  forecastTableEl.value = text;
  if (copyForecastBtn) {
    copyForecastBtn.disabled = !data?.forecast || !data.forecast.length;
  }
};

const renderDataSummary = (summary) => {
  if (!dataSummaryEl) return;
  if (!currentDatasetPath) {
    dataSummaryEl.textContent = "(load a dataset to see details)";
    return;
  }

  const lines = [];
  lines.push(`Dataset: ${currentDatasetLabel || currentDatasetPath}`);
  if (summary) {
    lines.push(`Rows: ${summary.n_rows}, columns: ${summary.n_cols}`);
    if (Array.isArray(summary.columns)) {
      lines.push(`Columns: ${summary.columns.join(", ")}`);
    }
    if (summary.has_ds) {
      lines.push(`Date range: ${summary.ds_min} → ${summary.ds_max}`);
    } else {
      lines.push("No 'ds' column detected.");
    }
  }

  dataSummaryEl.textContent = lines.join("\n");
};

const previewDatasetSeries = async () => {
  if (!pyodideInstance || !currentDatasetPath) return;

  try {
    pyodideInstance.globals.set("dataset_path", currentDatasetPath);
    const proxy = await pyodideInstance.runPythonAsync(`
import pandas as pd

df = pd.read_csv(dataset_path)
df = df.copy()
if "ds" not in df.columns or "y" not in df.columns:
    raise ValueError("CSV must contain 'ds' and 'y' columns.")

df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df = df.dropna(subset=["ds", "y"])
if df.shape[0] == 0:
    raise ValueError(
        "No rows remain after dropping rows with missing 'ds' or 'y' values."
    )

meta = {
    "n_obs": int(df.shape[0]),
    "ds_min": df["ds"].min().strftime("%Y-%m-%d %H:%M:%S"),
    "ds_max": df["ds"].max().strftime("%Y-%m-%d %H:%M:%S"),
}

out = {
    "meta": meta,
    "train": df[["ds", "y"]].assign(
        ds=lambda d: d["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ).to_dict("records"),
    "forecast": [],
}
out
`);
    const result = proxy.toJs({ dict: true });
    proxy.destroy?.();

    lastResult = result;
    renderMeta(result);
    renderChart(result);
  } catch (err) {
    console.error(err);
    appendLog(err?.stack ?? String(err));
    setError(
      "Failed to plot dataset. Ensure it has 'ds' and 'y' columns with valid values.",
    );
  }
};

const updateDataSummary = async () => {
  if (!pyodideInstance || !currentDatasetPath) {
    renderDataSummary(null);
    return;
  }

  try {
    pyodideInstance.globals.set("dataset_path", currentDatasetPath);
    const proxy = await pyodideInstance.runPythonAsync(`
import pandas as pd

df = pd.read_csv(dataset_path)
out = {
    "n_rows": int(df.shape[0]),
    "n_cols": int(df.shape[1]),
    "columns": [str(c) for c in df.columns],
}
if "ds" in df.columns:
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    # Only consider rows with a valid ds (and y when present)
    subset_cols = ["ds"]
    if "y" in df.columns:
        subset_cols.append("y")
    df = df.dropna(subset=subset_cols)
    out["n_rows"] = int(df.shape[0])
    out["has_ds"] = df.shape[0] > 0
    if df.shape[0] > 0:
        has_time = (df["ds"].dt.normalize() != df["ds"]).any()
        fmt = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"
        out["ds_min"] = df["ds"].min().strftime(fmt)
        out["ds_max"] = df["ds"].max().strftime(fmt)
    else:
        out["ds_min"] = None
        out["ds_max"] = None
else:
    out["has_ds"] = False
out
`);
    const summary = proxy.toJs({ dict: true });
    proxy.destroy?.();
    renderDataSummary(summary);
  } catch (err) {
    console.error(err);
    setError(
      "Failed to summarize dataset. Ensure it is a valid CSV with 'ds' and 'y' columns.",
    );
    appendLog(err?.stack ?? String(err));
  }
};

const boot = async () => {
  if (!bootBtn) return;
  bootBtn.disabled = true;
  setError("");
  try {
    setStatus("Loading Pyodide runtime...");
    const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
    pyodideInstance = pyodide;

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
    appendLog("Pyodide and ForecastXGBOnly loaded.");

    if (fileInputEl) fileInputEl.disabled = false;
    if (demoBtnEl) demoBtnEl.disabled = false;

    setStatus("Runtime ready. Load a dataset to begin.");
  } catch (err) {
    console.error(err);
    setStatus("Error while loading");
    appendLog(err?.stack ?? String(err));
    bootBtn.disabled = false;
  }
};

const getHorizon = () => {
  if (!horizonInputEl) return 12;
  const n = parseInt(horizonInputEl.value, 10);
  if (!Number.isFinite(n) || n <= 0) return 12;
  return n;
};

const setCurrentDataset = async (path, label) => {
  currentDatasetPath = path;
  currentDatasetLabel = label;
  appendLog(`Dataset selected: ${label || path}`);
  await updateDataSummary();
  await previewDatasetSeries();
  if (runForecastBtn) runForecastBtn.disabled = false;
};

const handleUserFileChange = async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  if (!pyodideInstance) {
    setError("Load the forecasting engine before uploading data.");
    return;
  }

  try {
    setError("");
    setStatus("Loading user CSV...");
    const text = await file.text();
    const destPath = "/data/user.csv";
    ensureDir(pyodideInstance.FS, destPath);
    pyodideInstance.FS.writeFile(destPath, text);
    appendLog(`User CSV loaded (${file.name}, ${text.length} bytes).`);
    await setCurrentDataset(destPath, file.name);
    setStatus("User dataset ready. Configure horizon and run forecast.");
  } catch (err) {
    console.error(err);
    setStatus("Failed to load user dataset.");
    setError("Could not load the uploaded CSV. See log for details.");
    appendLog(err?.stack ?? String(err));
  }
};

const handleDemoClick = async () => {
  if (!pyodideInstance) {
    setError("Load the forecasting engine first.");
    return;
  }

  try {
    setError("");
    setStatus("Fetching demo dataset...");
    const destPath = "/data/air_passengers.csv";
    await writeTextFile(pyodideInstance, DATASET_URL, destPath);
    appendLog("Demo air passengers dataset downloaded.");
    await setCurrentDataset(destPath, "Air passengers demo");
    setStatus("Demo dataset ready. Configure horizon and run forecast.");
  } catch (err) {
    console.error(err);
    setStatus("Failed to load demo dataset.");
    setError("Could not fetch the demo dataset. See log for details.");
    appendLog(err?.stack ?? String(err));
  }
};

const handleRunForecast = async () => {
  if (!pyodideInstance) {
    setError("Load the forecasting engine first.");
    return;
  }
  if (!currentDatasetPath) {
    setError("Load a dataset (user CSV or demo) before running the forecast.");
    return;
  }

  const horizon = getHorizon();
  try {
    setError("");
    setStatus("Running forecast...");
    appendLog(
      `Running forecast on ${
        currentDatasetLabel || currentDatasetPath
      } (horizon=${horizon}).`,
    );

    pyodideInstance.globals.set("dataset_path", currentDatasetPath);
    pyodideInstance.globals.set("forecast_horizon", horizon);

    const proxy = await pyodideInstance.runPythonAsync(`
import pandas as pd
import numpy as np
from forecasting_xgb_only import ForecastXGBOnly

df = pd.read_csv(dataset_path)
df = df.copy()
if "ds" not in df.columns:
    raise ValueError("CSV must contain a 'ds' column.")
if "y" not in df.columns:
    raise ValueError("CSV must contain a 'y' column.")

df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df = df.dropna(subset=["ds", "y"])
if df.shape[0] == 0:
    raise ValueError(
        "No rows remain after dropping rows with missing 'ds' or 'y' values."
    )

m = ForecastXGBOnly(nrounds=50)
m.fit(df, df["y"])

train_pred = m.predict(df)

if "var_est" in train_pred.columns:
    z = 1.96
    std = np.sqrt(train_pred["var_est"].to_numpy())
    train_pred["yhat_lower"] = train_pred["yhat"] - z * std
    train_pred["yhat_upper"] = train_pred["yhat"] + z * std

if df["ds"].shape[0] >= 2:
    diffs = df["ds"].sort_values().diff().dropna()
    if not diffs.empty:
        step = diffs.mode().iloc[0]
    else:
        step = pd.Timedelta(days=1)
else:
    step = pd.Timedelta(days=1)

last_ds = df["ds"].max()
future_ds = [last_ds + step * (i + 1) for i in range(int(forecast_horizon))]
future = pd.DataFrame({"ds": future_ds})

forecast = m.predict(future)
if "var_est" in forecast.columns:
    z = 1.96
    std_f = np.sqrt(forecast["var_est"].to_numpy())
    forecast["yhat_lower"] = forecast["yhat"] - z * std_f
    forecast["yhat_upper"] = forecast["yhat"] + z * std_f

forecast_ds_min = forecast["ds"].min().strftime("%Y-%m-%d")
forecast_ds_max = forecast["ds"].max().strftime("%Y-%m-%d")

# Decide whether to include time-of-day in outputs.
has_time = (df["ds"].dt.normalize() != df["ds"]).any()
fmt = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"

meta = {
    "horizon": int(forecast_horizon),
    "n_obs": int(df.shape[0]),
    "ds_min": df["ds"].min().strftime(fmt),
    "ds_max": df["ds"].max().strftime(fmt),
    "forecast_ds_min": forecast["ds"].min().strftime(fmt),
    "forecast_ds_max": forecast["ds"].max().strftime(fmt),
}

out = {
    "meta": meta,
    "train": train_pred.assign(
        ds=lambda d: d["ds"].dt.strftime(fmt)
    ).to_dict("records"),
    "forecast": forecast.assign(
        ds=lambda d: d["ds"].dt.strftime(fmt)
    ).to_dict("records"),
}
out
`);
    const result = proxy.toJs({ dict: true });
    proxy.destroy?.();

    lastResult = result;
    renderMeta(result);
    renderChart(result);
    renderForecastTable(result);
    setStatus("Forecast complete.");
  } catch (err) {
    console.error(err);
    setStatus("Forecast failed.");
    setError("Forecast failed. Check that your CSV has 'ds' and 'y' columns.");
    appendLog(err?.stack ?? String(err));
  }
};

if (bootBtn) {
  bootBtn.addEventListener("click", () => {
    boot();
  });
}

if (fileInputEl) {
  fileInputEl.addEventListener("change", (event) => {
    handleUserFileChange(event);
  });
}

if (demoBtnEl) {
  demoBtnEl.addEventListener("click", () => {
    handleDemoClick();
  });
}

if (runForecastBtn) {
  runForecastBtn.addEventListener("click", () => {
    handleRunForecast();
  });
}

if (viewAllBtn) {
  viewAllBtn.addEventListener("click", () => {
    chartViewMode = "all";
    if (lastResult) {
      renderChart(lastResult);
    }
  });
}

if (viewForecastBtn) {
  viewForecastBtn.addEventListener("click", () => {
    chartViewMode = "forecast";
    if (lastResult) {
      renderChart(lastResult);
    }
  });
}

const handleCopyForecast = async () => {
  if (!forecastTableEl) return;
  const text = forecastTableEl.value || "";
  if (!text.trim()) {
    setStatus("No forecast data to copy.");
    return;
  }

  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      forecastTableEl.focus();
      forecastTableEl.select();
      document.execCommand("copy");
    }
    setStatus("Forecast table copied to clipboard.");
  } catch (err) {
    console.error(err);
    setStatus("Failed to copy forecast table. Select and copy manually.");
    appendLog(err?.stack ?? String(err));
  }
};

if (copyForecastBtn) {
  copyForecastBtn.addEventListener("click", () => {
    handleCopyForecast();
  });
}
