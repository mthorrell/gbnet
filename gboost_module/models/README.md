# Forecasting starter code

```python
from gboost_module.models import forecasting
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
