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

# Add predictions to your test DataFrame
test_df['pred'] = predictions
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
