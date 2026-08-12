# Profile — Correlation Matrix Analysis

Utilities for computing and analyzing correlations between feature and target variables. This module provides an efficient alternative to computing the full correlation matrix when only feature–target correlations are needed, with built-in support for stratification by categorical or boolean conditions.

---

## Features

- Efficient feature–target correlation (`cor_matrix`)
  - Computes correlations via `DataFrame.corrwith` (faster than full `DataFrame.corr`)
  - Supports multiple target columns with minimal computation overhead
  - Flexible input: DataFrame/Series pairs or column-name lists
  - Automatic tolerance calculation for time-series alignment via `merge_asof`
  - Multiple correlation methods (Pearson, Spearman, Kendall)
  
- Categorical stratification
  - Binary categorical columns: compute separate correlations for `== 1` and `!= 1` subsets
  - Boolean filters (callables): compute correlations for arbitrary subsets
  - Dictionary specification: custom labels for stratified results
  - Combine multiple stratification criteria

---

## Importing

```python
from ml_fw import profile
```

Or import directly:

```python
from ml_fw.profile import cor_matrix
```

---

## Main API

### `cor_matrix`

```python
cor_matrix(
    feature_dat,
    target_dat,
    correlation_dat=None,
    correlation_index=None,
    categorical_dat=None,
    correlation_method='pearson',
    numeric_only=False
)
```

Derive correlation matrix of features with target variable(s), with optional categorical or filter-based stratification.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `feature_dat` | pd.DataFrame \| list | Feature data to correlate. If pd.DataFrame, contains feature columns directly. If list, contains column names to extract from `correlation_dat` (requires `correlation_dat`). |
| `target_dat` | pd.DataFrame \| pd.Series \| list | Target data to correlate with features. If pd.DataFrame or pd.Series, contains target columns directly. If list, contains column names to extract from `correlation_dat`. |
| `correlation_dat` | pd.DataFrame, optional | Combined DataFrame containing both feature and target columns. Required if both `feature_dat` and `target_dat` are lists. Default is None. |
| `correlation_index` | str, optional | Column name to use for joining feature and target data (for time-series alignment). If None, joins on DataFrame index. Default is None. |
| `categorical_dat` | list \| dict, optional | Categorical or filter specification for stratified correlations. See details below. Default is None. |
| `correlation_method` | str, optional | Correlation method: `'pearson'`, `'spearman'`, or `'kendall'`. Default is `'pearson'`. |
| `numeric_only` | bool, optional | If True, include only numeric (float, int, bool) columns. Default is False. |

#### Returns

| Type | Description |
|---|---|
| pd.DataFrame | Correlation results with features as rows. Columns include `'All'` (base correlations) plus additional columns for each categorical variable. For multiple targets, column names are `'{label}:{target}'`. |

#### Categorical Stratification

The `categorical_dat` parameter enables computing correlations for subsets of the data:

- **String elements** (list of strings): Column names containing binary (0/1) categorical data.
  - Correlations are computed separately for rows where the column == 1 and != 1.
  - Output columns are named `'{column} == 1'` and `'{column} != 1'`.

- **Callable elements** (list of functions): Boolean filter functions applied to the data.
  - Example: `lambda x: x['AE_index'] > 500` filters for high activity periods.
  - Correlations computed only for rows where the filter returns True.
  - Output columns are auto-named `'call00'`, `'call01'`, etc., unless a dict is used.

- **Dictionary** (dict): Keys are used as output labels; values are strings or callables.
  - Example: `{'high_activity': lambda x: x['AE'] > 500, 'is_dayside': 'dayside_flag'}`
  - Gives full control over both the filter logic and output column naming.

- **List** (mixed): Strings and callables can be mixed in a list. Auto-naming (`'call00'`, etc.) is applied to callables.

#### Example Behavior

```python
# Input data
df = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [10, 20, 30, 40, 50],
    'y': [2, 4, 5, 8, 10],
    'storm_flag': [0, 1, 1, 0, 1]
})

# Basic correlation
cor_matrix(feature_dat=df[['x1', 'x2']], target_dat=df['y'])
# Output columns: ['All']

# Stratified by binary column
cor_matrix(
    feature_dat=df[['x1', 'x2']],
    target_dat=df['y'],
    correlation_dat=df,
    categorical_dat=['storm_flag']
)
# Output columns: ['All', 'storm_flag == 1', 'storm_flag != 1']

# Stratified by custom filter
cor_matrix(
    feature_dat=df[['x1', 'x2']],
    target_dat=df['y'],
    correlation_dat=df,
    categorical_dat={'high_values': lambda x: x['x1'] > 2}
)
# Output columns: ['All', 'high_values']
```

---

## Detailed Examples

### Simple DataFrame Input

```python
import pandas as pd
from ml_fw import profile

# Create sample data
df = pd.DataFrame({
    'density': [1e-5, 1e-4, 1e-3, 2e-4, 5e-5],
    'temperature': [300, 400, 500, 350, 320],
    'velocity': [300, 350, 400, 320, 310],
    'dst_index': [-20, -50, -30, -10, -5]
})

# Correlate features with target
correlations = profile.cor_matrix(
    feature_dat=df[['density', 'temperature', 'velocity']],
    target_dat=df['dst_index']
)

print(correlations)
#                All
# density     0.9543
# temperature 0.9922
# velocity    0.8871
```

### Column-Name Input with Merged Data

```python
# Use column names instead of extracting manually
correlations = profile.cor_matrix(
    feature_dat=['density', 'temperature', 'velocity'],
    target_dat=['dst_index'],
    correlation_dat=df
)
```

### Stratified Correlation: Binary Categorical

```python
# Add a categorical column
df['is_storm'] = (df['dst_index'] < -30).astype(int)

# Compute separate correlations for storm and non-storm periods
correlations = profile.cor_matrix(
    feature_dat=df[['density', 'temperature', 'velocity']],
    target_dat=df['dst_index'],
    correlation_dat=df,
    categorical_dat=['is_storm']
)

print(correlations.columns)
# Index(['All', 'is_storm == 1', 'is_storm != 1'])
```

### Stratified Correlation: Custom Filters

```python
# Define custom activity filters
high_activity_filter = lambda x: x['density'] > 1e-4
high_temp_filter = lambda x: x['temperature'] > 400

correlations = profile.cor_matrix(
    feature_dat=df[['density', 'temperature', 'velocity']],
    target_dat=df['dst_index'],
    correlation_dat=df,
    categorical_dat={
        'high_activity': high_activity_filter,
        'high_temperature': high_temp_filter
    }
)

print(correlations.columns)
# Index(['All', 'high_activity', 'high_temperature'])
```

### Time-Series Alignment

```python
import pandas as pd
from ml_fw import profile

# Time-indexed data that may not align perfectly
times = pd.date_range('2024-01-01', periods=5, freq='1h')
features_df = pd.DataFrame({
    'DateTime': times,
    'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
    'x2': [10.0, 20.0, 30.0, 40.0, 50.0]
})

# Slightly offset target times (simulating real observational misalignment)
targets_df = pd.DataFrame({
    'DateTime': times + pd.Timedelta(minutes=1),  # 1 minute offset
    'y': [2.0, 4.0, 5.0, 8.0, 10.0]
})

# Merge and correlate on nearest time
correlations = profile.cor_matrix(
    feature_dat=['x1', 'x2'],
    target_dat=['y'],
    correlation_dat=pd.concat([features_df, targets_df], axis=1),
    correlation_index='DateTime'
)

# Rows are aligned by nearest 'DateTime' value within auto-calculated tolerance
```

### Multiple Target Variables

```python
# Correlate features with multiple targets simultaneously
df = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [10, 20, 30, 40, 50],
    'y1': [2, 4, 5, 8, 10],
    'y2': [20, 40, 50, 80, 100]
})

correlations = profile.cor_matrix(
    feature_dat=df[['x1', 'x2']],
    target_dat=df[['y1', 'y2']]
)

print(correlations.columns)
# Index(['y1', 'y2'])
```

### Multiple Targets with Stratification

```python
correlations = profile.cor_matrix(
    feature_dat=df[['x1', 'x2']],
    target_dat=df[['y1', 'y2']],
    correlation_dat=df,
    categorical_dat={'high_x1': lambda x: x['x1'] > 2}
)

print(correlations.columns)
# Index(['All:y1', 'All:y2', 'high_x1:y1', 'high_x1:y2'])
```

---

## Use Cases

### Feature Selection

Identify features with high correlation to the target as candidates for model training.

```python
# Rank features by absolute correlation
cor = profile.cor_matrix(features, targets)
top_features = cor['All'].abs().nlargest(5)
```

### Activity-Dependent Analysis

Examine how feature–target relationships change under different conditions (e.g., storms vs quiet times, day vs night).

```python
cor = profile.cor_matrix(
    features, targets,
    correlation_dat=data,
    categorical_dat={
        'quiet': lambda x: x['AE'] < 300,
        'moderate': lambda x: (x['AE'] >= 300) & (x['AE'] < 600),
        'storm': lambda x: x['AE'] >= 600
    }
)
# Compare correlation patterns across activity levels
```

### Time-Series Domains

Useful for space weather, climate, hydrology, and financial data where feature and target data may come from slightly misaligned observations or sensors.

---

## Notes

- **Efficiency**: Uses `DataFrame.corrwith` for each target, which is significantly faster than computing the full N×N correlation matrix with `DataFrame.corr()` when only a subset of correlations is needed.
- **Alignment tolerance**: When `correlation_index` is provided, auto-calculated tolerance is half the most common (modal) index spacing. For irregular time series, pass a custom `tolerance` or align data beforehand.
- **NaN handling**: Both correlation methods (Pearson via `corrwith`, Ljung-Box via `scipy.stats`) exclude NaN values in their calculations (pairwise deletion). Entire rows of NaN in a column will result in NaN correlation.
- **Stratified subset size**: When using categorical filters, ensure each subset has enough data for meaningful correlation estimates. Subsets with very few samples (< 3) may produce unreliable correlations.
- **Column naming convention**:
  - Single target: column name is `'{label}'` (e.g., `'All'`, `'high_activity'`).
  - Multiple targets: column names are `'{label}:{target_name}'` (e.g., `'All:y1'`, `'high_activity:y2'`).

---

## See Also

- **`data_io.create()`** — Prepare cleaned features for correlation analysis.
- **`ml_mod.train()`** — Use correlated features in model training.
- **`inspect.boxplot_vx()`** — Visualize residuals binned by feature values.
- **Pandas `corrwith` docs**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corrwith.html
