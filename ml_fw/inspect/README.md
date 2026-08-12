# Inspect — Model Result Diagnostics

A collection of tools for inspecting and diagnosing machine learning model results through binned statistics and rolling-window metrics. This module provides functions to compute box-and-whisker statistics for residuals or model outputs binned by feature values, and to compute rolling-window performance metrics over time.

---

## Features

- Binned residual statistics (`boxplot_vx`)
  - Bin y-data (e.g., residuals) by one or more x-variables
  - Compute Tukey box-and-whisker statistics per bin
  - Supports configurable bin count and range per variable
  
- Binned performance metrics (`boxplot_metvx`)
  - Bin a computed metric (accuracy, MSE, custom) by x-variables
  - Build distribution of metric values per bin via k-fold random subsampling
  - Auto-select metric based on data type (integer → accuracy, float → MSE)
  - Supports multiple custom metrics per analysis
  
- Rolling-window metrics (`rolling_met`)
  - Compute rolling-window metrics over time or index
  - Auto-detect window size (60-minute for datetime, 10-point for numeric)
  - Support for multiple metrics and callable-based metric definitions
  - Output DataFrame ready for plotting

---

## Importing

```python
from ml_fw.inspect import boxplot_vx, boxplot_metvx, rolling_met
```

Or import the package:

```python
import ml_fw.inspect as inspect

# Use: inspect.boxplot_vx(...), inspect.boxplot_metvx(...), inspect.rolling_met(...)
```

---

## Main API

### `boxplot_vx`

```python
boxplot_vx(
    x_dat,
    y_dat,
    box_dat=None,
    box_meth=True,
    bins=10,
    xrange=None,
    whisker=1.5
)
```

Calculate boxplot statistics of y-data (e.g., residuals) binned by x-variables.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `x_dat` | pd.DataFrame \| list | Data for binning. If pd.DataFrame, contains the x-variables (one or more columns). If list, contains column names to extract from `box_dat`. Each column is binned separately. |
| `y_dat` | pd.DataFrame \| list | Y-data for computing statistics. If pd.DataFrame, contains y-values. If list, contains column names to extract from `box_dat`. |
| `box_dat` | pd.DataFrame, optional | Combined DataFrame (required if `x_dat`, `y_dat` are lists). Default is None. |
| `box_meth` | bool \| dict, optional | Placeholder for future statistics method selection. Default is True. |
| `bins` | int \| list, optional | Number of bins or list of bin counts (one per x-column). Default is 10. |
| `xrange` | list[tuple[float, float]], optional | (min, max) range for bins (one tuple per x-column). If None, uses (x.min(), x.max()). Default is None. |
| `whisker` | float, optional | Whisker coefficient for box-and-whisker calculation. Default is 1.5 (Tukey's boxplot). |

#### Returns

| Type | Description |
|---|---|
| dict | Nested dictionary: `results[x_column_name]['residuals']` contains: `'box_stats'` (list of dicts with keys `'mean'`, `'med'`, `'q1'`, `'q3'`, `'whislo'`, `'whishi'`, `'fliers'`), `'x_edge'` (bin edges), `'x_centre'` (bin centers), `'x_width'` (bin width). |

#### Example

```python
import pandas as pd
import numpy as np
from ml_fw.inspect import boxplot_vx

# Create sample data
df = pd.DataFrame({
    'x1': np.random.uniform(0, 100, 1000),
    'x2': np.random.uniform(-10, 10, 1000),
    'residuals': np.random.normal(0, 1, 1000)
})

# Bin residuals by x1 and x2
results = boxplot_vx(
    x_dat=df[['x1', 'x2']],
    y_dat=df['residuals'],
    bins=10
)

print(results.keys())
# dict_keys(['x1', 'x2'])

print(results['x1']['residuals'].keys())
# dict_keys(['box_stats', 'x_edge', 'x_centre', 'x_width'])

# Access box-and-whisker stats for first bin
box_stats = results['x1']['residuals']['box_stats'][0]
print(box_stats)
# {'mean': ..., 'med': ..., 'q1': ..., 'q3': ..., 'whislo': ..., 'whishi': ..., 'fliers': [...]}
```

---

### `boxplot_metvx`

```python
boxplot_metvx(
    x_dat,
    y_true,
    y_mod,
    box_dat=None,
    box_metric=None,
    kfolds=100,
    kfrac=0.5,
    bins=10,
    xrange=None,
    whisker=1.5
)
```

Calculate boxplot statistics of a metric (accuracy, MSE, custom) computed between true and model values, binned by x-variables.

For each bin, the function randomly subsamples a fraction (`kfrac`) of the true/model data `kfolds` times, computes the metric for each subsample, and builds a distribution of metric values. This distribution is then used to derive box-and-whisker statistics.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `x_dat` | pd.DataFrame \| list | Data for binning. If pd.DataFrame, contains the x-variables. If list, contains column names to extract from `box_dat`. |
| `y_true` | pd.DataFrame \| list | True labels/values. If pd.DataFrame, contains true data. If list, contains column names to extract from `box_dat`. Should be a single column. |
| `y_mod` | pd.DataFrame \| list | Model predictions. If pd.DataFrame, contains model data. If list, contains column names to extract from `box_dat`. Should be a single column. |
| `box_dat` | pd.DataFrame, optional | Combined DataFrame (required if `x_dat`, `y_true`, `y_mod` are lists). Default is None. |
| `box_metric` | callable \| list \| dict \| None, optional | Metric function(s). Options: None (auto-select based on data type: accuracy for integers, MSE for floats), callable (single metric), list (multiple metrics, auto-named `'metric_0'`, `'metric_1'`, ...), dict (custom names → callables). Each callable should accept `(y_true, y_pred)` and return a scalar. Default is None. |
| `kfolds` | int, optional | Number of k-fold subsamples per bin. Default is 100. |
| `kfrac` | float, optional | Fraction of bin data to sample in each fold (0 < `kfrac` <= 1). Default is 0.5. |
| `bins` | int \| list, optional | Number of bins or list of bin counts (one per x-column). Default is 10. |
| `xrange` | list[tuple[float, float]], optional | (min, max) range for bins. Default is None. |
| `whisker` | float, optional | Whisker coefficient (default 1.5 for Tukey). |

#### Returns

| Type | Description |
|---|---|
| dict | Nested dictionary: `results[x_column_name][metric_name]` contains: `'box_stats'`, `'x_edge'`, `'x_centre'`, `'x_width'`. Supports multiple metrics and x-columns. |

#### Example: Auto Metric Selection

```python
from ml_fw.inspect import boxplot_metvx

# Binary classification data
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])

# Auto-selects accuracy_score for integer data
results = boxplot_metvx(
    x_dat=features,
    y_true=y_true,
    y_mod=y_pred,
    bins=5,
    kfolds=50
)

print(results['feature_name'].keys())
# dict_keys(['accuracy'])  # auto-named for integer data
```

#### Example: Custom Metrics

```python
from sklearn.metrics import f1_score, precision_score
from ml_fw.inspect import boxplot_metvx

# Multiple custom metrics
results = boxplot_metvx(
    x_dat=features,
    y_true=y_true,
    y_mod=y_pred,
    box_metric={
        'F1': lambda yt, yp: f1_score(yt, yp),
        'Precision': lambda yt, yp: precision_score(yt, yp)
    },
    bins=10,
    kfolds=100,
    kfrac=0.5
)

print(results['feature_name'].keys())
# dict_keys(['F1', 'Precision'])
```

---

### `rolling_met`

```python
rolling_met(
    met_dat,
    y_true='y_true',
    y_pred='y_pred',
    on='DateTime',
    roll_kwargs=None,
    roll_metric=None
)
```

Calculate a rolling-window metric over time or index.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `met_dat` | pd.DataFrame | DataFrame containing true values, predicted values, and time/index column. |
| `y_true` | str, optional | Column name containing true values. Default is `'y_true'`. |
| `y_pred` | str, optional | Column name containing predicted values. Default is `'y_pred'`. |
| `on` | str, optional | Column name or `'index'` to use as the rolling-window reference. If `'index'`, the DataFrame index is used. Default is `'DateTime'`. |
| `roll_kwargs` | dict, optional | Keyword arguments for `DataFrame.rolling()` (e.g., `window`, `center`, `min_periods`). If None, auto-selects based on data type in `on` column: `{'window': '60min', 'center': True}` for datetime, `{'window': 10, 'center': True}` for numeric. Default is None. |
| `roll_metric` | callable \| list \| dict \| None, optional | Metric function(s). Options: None (auto-select: accuracy for integer data, MSE for float), callable (single metric), list (auto-named `'Metric 00'`, `'Metric 01'`, ...), dict (custom names → callables). Each callable receives `(y_true_series, y_pred_series)` and returns a scalar. Default is None. |

#### Returns

| Type | Description |
|---|---|
| pd.DataFrame | DataFrame with rolling-metric values and the `on` column. Column names are metric names or auto-generated. |

#### Example: Auto Metric

```python
import pandas as pd
from ml_fw.inspect import rolling_met

# Time-series classification results
df = pd.DataFrame({
    'DateTime': pd.date_range('2024-01-01', periods=100, freq='1H'),
    'y_true': np.random.randint(0, 2, 100),
    'y_pred': np.random.randint(0, 2, 100)
})

# Auto-selects 60-minute rolling window and accuracy metric
rolling_acc = rolling_met(
    met_dat=df,
    y_true='y_true',
    y_pred='y_pred',
    on='DateTime'
)

print(rolling_acc.columns)
# Index(['Accuracy', 'DateTime'])

print(rolling_acc.head())
#    Accuracy   DateTime
# 0       NaN 2024-01-01
# 1       0.5 2024-01-01 01:00:00
# ...
```

#### Example: Custom Rolling Window and Metrics

```python
from sklearn.metrics import f1_score
from ml_fw.inspect import rolling_met

rolling_result = rolling_met(
    met_dat=df,
    y_true='y_true',
    y_pred='y_pred',
    on='DateTime',
    roll_kwargs={'window': '2H', 'center': True, 'min_periods': 1},
    roll_metric={
        'Accuracy': lambda yt, yp: (yt == yp).mean(),
        'F1': lambda yt, yp: f1_score(yt, yp, zero_division=0)
    }
)

print(rolling_result.columns)
# Index(['Accuracy', 'F1', 'DateTime'])
```

---

## Output and Integration

All three functions return dictionaries/DataFrames designed to integrate with the `ml_fw.plot` module:

- **`boxplot_vx` and `boxplot_metvx` output** → feed directly into `ml_fw.plot.plot_boxplot()` for visualization.
- **`rolling_met` output** → a simple DataFrame, easily plotted with matplotlib or pandas `.plot()`.

### Example Workflow

```python
from ml_fw.inspect import boxplot_vx, boxplot_metvx
from ml_fw.plot import plot_boxplot

# 1. Compute binned residual statistics
residual_stats = boxplot_vx(
    x_dat=features,
    y_dat=residuals,
    bins=10
)

# 2. Plot using plot_boxplot
plot_result = plot_boxplot(
    results=residual_stats,
    separate_by='both',
    figsize=(12, 6)
)

fig = plot_result['fig']
fig.show()
```

---

## Use Cases

### Model Validation and Diagnostics

Examine whether model performance or residual statistics vary systematically with feature values, indicating potential underfitting or data-dependent bias.

### Binned Performance Assessment

Compare model accuracy or custom metrics across different ranges of a variable (e.g., solar wind speed, geomagnetic activity) to identify weak performance regions.

### Time-Series Performance Trends

Use `rolling_met` to track model performance over time and identify performance degradation or concept drift.

### Scientific Analysis

For heliophysics and space-weather applications, bin results by solar wind parameters, geomagnetic indices, or local time to understand physical dependencies in model behavior.

---

## Notes

- **Box-and-whisker statistics**: Computed using `scipy.stats.binned_statistic` on quartiles and means. The whisker coefficient (default 1.5) follows Tukey's definition: whisker range is [Q1 - 1.5×IQR, Q3 + 1.5×IQR].
- **Empty bins**: If a bin contains no data or very few data points (< 2 for `boxplot_vx`, ≤ 1 for `boxplot_metvx`), statistics will be NaN or omitted.
- **K-fold metric estimation**: The k-fold subsampling in `boxplot_metvx` builds a distribution of metric values per bin, providing confidence estimates. Increase `kfolds` for smoother distributions; decrease `kfrac` for more aggressive subsampling.
- **Rolling windows**: Window size auto-detection in `rolling_met` uses the most common (modal) time/index resolution. For irregular data, pass explicit `roll_kwargs={'window': ...}`.
- **NaN handling**: Both functions handle NaN values gracefully (exclude from statistics). Entire columns of NaN will result in NaN statistics.

---

## See Also

- **`ml_fw.plot.plot_boxplot()`** — Visualize `boxplot_vx` and `boxplot_metvx` output.
- **`ml_fw.profile.cor_matrix()`** — Understand feature correlations before binned analysis.
- **`ml_fw.ml_mod.train()`** — Train models whose results can be inspected with these tools.
- **Scipy `binned_statistic` docs**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
- **Pandas rolling docs**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
