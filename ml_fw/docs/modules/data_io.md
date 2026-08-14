# Data I/O — Feature and Target Dataset Creation

A collection of utilities for preparing feature and target datasets for machine learning workflows. This module handles common preprocessing tasks including log transformation, cyclical variable encoding, and time-lagged feature generation.

The module is was designed for heliophysics and space-weather applications where variables often span multiple orders of magnitude (e.g., density, energy flux) or have strong periodic patterns (e.g., local time, longitude), though can be used with any data.

---

## Features

- Uses Pandas and DataFrames

- Dataset creation and cleaning (`create`)
  - Extract feature and target columns
  - Log10 transformation for wide-dynamic-range variables
  - Cyclical (sin/cos) encoding for periodic variables
  - Automatic detection of periodicity scale (24h vs 360°)
  - Removal of NaN and infinite values
  
- Time-lagged feature generation (`feat_shift`)
  - Build lagged versions of features for time-series modeling
  - Automatic tolerance calculation from time-series resolution
  - Support for time columns or DataFrame index
  - Optional retention or removal of original features

---

## Importing

```python
from ml_fw import data_io
```

Or import individual functions:

```python
from ml_fw.data_io import create, feat_shift
```

---

## Main API

### `create`

```python
create(
    dataframe,
    feature_columns,
    target_columns,
    log_columns=None,
    cyclical_columns=None,
    time_column=None
)
```

Create feature and target datasets from a raw DataFrame with optional log transformation and cyclical encoding.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `dataframe` | pd.DataFrame | Input DataFrame containing all raw data. |
| `feature_columns` | list[str] | Column names to use as features (required). |
| `target_columns` | list[str] | Column names to use as targets (required). |
| `log_columns` | list[str], optional | Columns to apply log10 transformation. Use for variables spanning multiple orders of magnitude (e.g., density, flux). Transformed columns are renamed `log10_{original_name}` and originals are removed. Default is None. |
| `cyclical_columns` | list[str], optional | Columns to convert to cyclical (sin/cos) representation. Use for periodic variables like local time or longitude. Auto-detects 24h scale if max value ≤ 24, else 360° scale. Cyclical columns are replaced by `cos_{name}` and `sin_{name}` pairs. Default is None. |
| `time_column` | list[str], optional | Time column(s) to retain in output (useful for later indexing or joining). Default is None. |

#### Returns

| Type | Description |
|---|---|
| tuple[pd.DataFrame, pd.DataFrame] | (features, targets) — cleaned and transformed feature and target DataFrames. |

#### Behavior

- Selects specified columns and drops initial NaN values.
- Applies log10 transformation to `log_columns` (with warnings if transformation fails for individual columns).
- Applies cyclical encoding to `cyclical_columns`.
- Removes rows containing infinite values (from log of zero or near-zero).
- Drops remaining NaN values.
- Extracts target columns and removes them from the feature set.
- Raises `TypeError` if `feature_columns` or `target_columns` are not lists.
- Raises `ValueError` if required columns don't exist in the DataFrame.

#### Example

```python
import pandas as pd
from ml_fw import data_io

# Create a sample dataset
df = pd.DataFrame({
    'density': [1e-5, 1e-4, 1e-3],
    'temperature': [10, 20, 30],
    'local_time': [6, 12, 18],
    'target': [0.5, 1.0, 1.5]
})

# Extract and transform
features, targets = data_io.create(
    dataframe=df,
    feature_columns=['density', 'temperature', 'local_time'],
    target_columns=['target'],
    log_columns=['density'],
    cyclical_columns=['local_time']
)

print(features.columns)
# Index(['temperature', 'cos_local_time', 'sin_local_time', 'log10_density'])

print(targets.columns)
# Index(['target'])
```

---

### `feat_shift`

```python
feat_shift(
    dataframe,
    time_column='DateTime',
    periods=None,
    unit='min',
    tolerance=None,
    drop_original=False,
    drop_na=True
)
```

Create time-lagged features for time-series data using time-based shifting and nearest-neighbor alignment.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `dataframe` | pd.DataFrame | Input DataFrame with features and time information. |
| `time_column` | str, optional | Column name for time reference, or `'index'` to use the DataFrame index. Default is `'DateTime'`. |
| `periods` | int \| list[int], optional | Time periods to shift. Single int or list of ints (e.g., `5` or `[5, 10]`). Default is `[5]` (shift by 5 units). |
| `unit` | str, optional | Time unit for periods: `'min'`, `'h'`, `'D'`, etc. Default is `'min'`. |
| `tolerance` | pd.Timedelta, optional | Maximum acceptable time difference for alignment in `merge_asof`. If None, calculated from the modal (most common) time resolution. Default is None. |
| `drop_original` | bool, optional | If True, drop original feature columns and keep only shifted versions. Default is False. |
| `drop_na` | bool, optional | If True, drop rows with NaN values created by shifting. Default is True. |

#### Returns

| Type | Description |
|---|---|
| pd.DataFrame | DataFrame with original and time-lagged feature columns. Lagged columns are named `{feature}_lag{period}{unit}`. |

#### Behavior

- Normalizes `periods` to a list.
- Handles `time_column='index'` by resetting the index and processing it as a column.
- Raises `TypeError` if `periods` contains non-integers.
- Raises `KeyError` if `time_column` doesn't exist.
- Sorts data by time for efficient `merge_asof`.
- Creates lagged DataFrames by shifting the time column and performing a nearest-neighbor merge.
- Calculates tolerance from the modal time resolution if not provided (half the mode + 1 second buffer).
- Restores the original index after processing.
- Returns rows with NaN values dropped if `drop_na=True`.

#### Example

```python
import pandas as pd
import numpy as np
from ml_fw import data_io

# Create time-series data
dates = pd.date_range('2024-01-01', periods=10, freq='5min')
df = pd.DataFrame({
    'DateTime': dates,
    'value': np.arange(10),
    'feature_x': np.random.randn(10)
})

# Generate lagged features
df_lagged = data_io.feat_shift(
    dataframe=df,
    time_column='DateTime',
    periods=[5, 10],  # Lags of 5 and 10 minutes
    unit='min',
    drop_na=True
)

print(df_lagged.columns)
# Index(['DateTime', 'value', 'feature_x', 'value_lag5min', 'feature_x_lag5min',
#        'value_lag10min', 'feature_x_lag10min'])

print(df_lagged.shape)
# (8, 7) — 2 rows dropped due to NaN from initial lags
```

#### Using DataFrame Index as Time

```python
# If your DataFrame already has a datetime index
df_indexed = df.set_index('DateTime')

df_lagged = data_io.feat_shift(
    dataframe=df_indexed,
    time_column='index',
    periods=[5, 10],
    unit='min'
)
```

---

## Use Cases

### Space Weather and Heliophysics

- **Log transformation** for solar wind density, magnetic field magnitude, particle flux (span many orders of magnitude).
- **Cyclical encoding** for local time (0–24h) and magnetic local time (MLT) to preserve circular periodicity without discontinuities at 0/24h or 0/360° boundaries.

### General Time-Series Modeling

- **Lagged features** for autoregressive and hybrid models (e.g., predicting solar wind speed using lagged density, temperature).
- **Separate reference data** for fitting ARIMA models in `perturbed_input` workflows.

### Correlation and Feature Importance

- Preprocessed features for `profile.cor_matrix` or `ml_mod.train` allow cleaner downstream analysis.

---

## Notes

- **NaN and infinite handling**: `create()` removes rows with NaN or infinite values after transformations. If this results in excessive data loss, consider filling or removing problematic columns before calling `create()`.
- **Log of zero**: Log transformation will warn and drop rows where `log10` fails (e.g., zero or negative values in log-transformed columns). Pre-check data if this is a concern.
- **Cyclical scale detection**: The scale (24h vs 360°) is inferred from `max(column)`. If your local time is in [0, 23] and longitude in [0, 360], the auto-detection works correctly. For custom scalings, manually compute sin/cos and skip `cyclical_columns`.
- **Time tolerance in `feat_shift`**: The auto-calculated tolerance is `(modal_resolution / 2) + 1 second`. This works well for regular time grids; for irregular time series, consider passing a custom `tolerance` value.
- **Memory and performance**: For large DataFrames, `feat_shift` is optimized to minimize copying; `create()` processes features and targets sequentially to avoid storing the full cross-product.

---

## See Also

- **`ml_mod.train()`** — Train models on features created by `create()`.
- **`profile.cor_matrix()`** — Compute correlations for features prepared by `create()`.
- **`perturbed_input.fit_model()`** — Fit ARIMA models to reference time series for ensemble perturbation.
