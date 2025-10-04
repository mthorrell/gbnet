# Forecasting starter code

```python
from gbnet.models import forecasting
import pandas as pd

# Assuming df is your DataFrame with 'ds' and 'y' columns
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_pedestrians_covid.csv"
df = pd.read_csv(url)
df['ds'] = pd.to_datetime(df['ds'])


# Split your data into training and testing sets
cutoff = df['ds'].quantile(0.4)
train_df = df[df['ds'] < cutoff].copy()
test_df = df[df['ds'] >= cutoff].copy()

# Initialize and fit the estimator
estimator = forecasting.Forecast()
estimator.fit(train_df[['ds']], train_df['y'])

# Make predictions
predictions = estimator.predict(test_df[['ds']])
```

# Ordinal Regression

See [this notebook](https://github.com/mthorrell/gbnet/blob/main/examples/ordinal_regression_comparison.ipynb) for more examples.

```python
from gbnet.models import ordinal_regression
import pandas as pd

url = "https://raw.githubusercontent.com/gagolews/ordinal-regression-data/refs/heads/master/abalone.csv"
df = pd.read_csv(url)

xcols = [col for col in df.columns if not (col == 'response')]
ycol = 'response'
num_classes = df[ycol].nunique()

estimator = ordinal_regression.GBOrd(num_classes=num_classes)
estimator.fit(df[xcols], df[ycol])

# Get unthresholded continuous outputs
preds = estimator.predict(df[xcols])

# Get thresholded outputs
preds = estimator.predict(df[xcols], return_logits=False)

# Get class probabilities
probs = estimator.predict_proba(df[xcols])
```

# Continuous-Time Survival Analysis

The `HazardSurvivalModel` pairs gradient-boosted hazards with a `HazardIntegrator` that performs trapezoidal integration. It supports both static covariates (expanded to piecewise-constant trajectories) and time-varying covariates supplied directly. See [this notebook](https://github.com/mthorrell/gbnet/blob/main/examples/hazard_survival_example.ipynb) for an end-to-end example.

```python
import numpy as np

from gbnet.models.survival import hazard_survival

# X must include a 'unit_id' column and may contain time-varying covariates
# y must contain ['unit_id', 'time', 'event'] columns

model = hazard_survival.HazardSurvivalModel(module_type="XGBModule", nrounds=150)
model.fit(X, y)

# Survival curves over specific times
survival_df = model.predict_survival(X, times=np.array([0, 5, 10, 15]))

# Expected and median survival times per unit
summary_df = model.predict(X)
```

# Discrete Time Survival Analysis

See [this notebook](https://github.com/mthorrell/gbnet/blob/main/examples/discrete_survival_examples.ipynb) for example usage.

- BetaSurvivalModel: Discrete survival using a Beta distribution, mimicing a mixture of geometric distributions
- ThetaSurvivalModel: Discrete survival using a geometric distribution with a single parameter, theta
