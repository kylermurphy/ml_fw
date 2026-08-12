# Plot — Visualization for Model Diagnostics

Plotting utilities for rendering box-and-whisker plots and other visualizations of model inspection results. This module is designed to work seamlessly with output from `ml_fw.inspect` functions, providing flexible subplot layouts for comparing results across multiple variables and metrics.

---

## Features

- Box-and-whisker plotting (`plot_boxplot`)
  - Render diagnostic box plots from `inspect.boxplot_vx` or `inspect.boxplot_metvx` output
  - Flexible subplot layout control:
    - `'both'`: Create a grid with x-variables as rows and metrics as columns
    - `'x_col'`: One subplot per x-variable, with metrics overlaid and colored
    - `'metric_name'`: One subplot per metric, with x-variables overlaid and colored
  - Customizable colors, transparency (alpha), and figure size
  - Filtering by metric name or x-column
  - Support for matplotlib's `bxp` (box plot) with full control over box appearance

---

## Installation & Import

```python
from ml_fw.plot import plot_boxplot
```

Or import the package:

```python
import ml_fw.plot as plot

# Use: plot.plot_boxplot(...)
```

---

## Main API

### `plot_boxplot`

```python
plot_boxplot(
    results,
    separate_by='both',
    metric_name=None,
    x_col=None,
    colors=None,
    alphas=None,
    figsize=(12, 6),
    ax=None,
    fig=None,
    showmeans=True,
    showfliers=True
)
```

Plot boxplot statistics from `inspect.boxplot_vx` or `inspect.boxplot_metvx` output using matplotlib's `bxp` function.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `results` | dict | Nested dictionary output from `inspect.boxplot_vx` or `inspect.boxplot_metvx`. Structure: `{x_col_name: {metric_name: {'box_stats': [...], 'x_edge': [...], 'x_centre': [...], 'x_width': ...}}}` (for `boxplot_metvx`), or `{x_col_name: {'residuals': {...}}}` (for `boxplot_vx`). |
| `separate_by` | str, optional | Layout strategy for subplots: `'both'` (grid of x_col × metric), `'x_col'` (one subplot per x_col, metrics overlaid), `'metric_name'` (one subplot per metric, x_cols overlaid). Default is `'both'`. |
| `metric_name` | str, optional | Filter to plot only a specific metric. If None, plot all metrics. Default is None. |
| `x_col` | str, optional | Filter to plot only a specific x-column. If None, plot all x-columns. Default is None. |
| `colors` | list, optional | List of color specifications (hex codes, color names, or RGB tuples) to cycle through boxes. If None, uses matplotlib's default Set1 colormap. Default is None. |
| `alphas` | list, optional | List of transparency values (0–1) to cycle through box fills. If None, uses fully opaque (alpha=1.0). Default is None. |
| `figsize` | tuple, optional | Figure size (width, height) if creating a new figure. Default is (12, 6). |
| `ax` | matplotlib.axes.Axes, optional | Existing axes to plot on. Used for single-subplot cases. Default is None. |
| `fig` | matplotlib.figure.Figure, optional | Existing figure to use. If provided with `ax=None`, creates subplots within this figure. Default is None. |
| `showmeans` | bool, optional | Show mean markers on box plots (diamond markers). Default is True. |
| `showfliers` | bool, optional | Show outlier points (fliers). Default is True. |

#### Returns

| Type | Description |
|---|---|
| dict | Dictionary with keys: `'fig'` (matplotlib Figure), `'axes'` (dict mapping subplot identifiers to Axes objects), `'axes_flat'` (list of all Axes created). Axes dict keys depend on `separate_by`: `('x_col', 'metric_name')` for `'both'`, just `'x_col'` for `'x_col'`, just `'metric_name'` for `'metric_name'`. |

#### Behavior

- **Layout**: Subplots are created based on `separate_by`:
  - `'both'`: Creates an n_rows × n_cols grid where rows = unique x-columns, cols = unique metrics.
  - `'x_col'`: Creates n_rows subplots (one per x-column), each showing all metrics with different colors and optional legend.
  - `'metric_name'`: Creates n_cols subplots (one per metric), each showing all x-columns with different colors and optional legend.
- **Color cycling**: Colors are cycled through across all boxes (or all subplots, depending on layout) in the order provided.
- **Alpha cycling**: Transparency values are similarly cycled.
- **Filtering**: If `metric_name` or `x_col` are specified, only matching data is plotted; mismatched keys are silently skipped.
- **Tight layout**: `plt.tight_layout()` is applied automatically to avoid label overlap.

#### Example: Basic Usage with `boxplot_vx`

```python
import pandas as pd
import numpy as np
from ml_fw.inspect import boxplot_vx
from ml_fw.plot import plot_boxplot

# Create sample data
df = pd.DataFrame({
    'x1': np.random.uniform(0, 100, 1000),
    'residuals': np.random.normal(0, 1, 1000)
})

# Compute binned residual statistics
box_stats = boxplot_vx(
    x_dat=df[['x1']],
    y_dat=df['residuals'],
    bins=10
)

# Plot
plot_result = plot_boxplot(results=box_stats, separate_by='both')

fig = plot_result['fig']
fig.show()
```

#### Example: Multiple Metrics with Custom Colors

```python
from ml_fw.inspect import boxplot_metvx
from ml_fw.plot import plot_boxplot

# Compute multiple performance metrics binned by two features
metrics_result = boxplot_metvx(
    x_dat=features[['x1', 'x2']],
    y_true=y_true,
    y_mod=y_pred,
    box_metric={
        'Accuracy': lambda yt, yp: (yt == yp).mean(),
        'F1': lambda yt, yp: f1_score(yt, yp, zero_division=0)
    },
    bins=8,
    kfolds=100
)

# Plot with custom colors and transparency
plot_result = plot_boxplot(
    results=metrics_result,
    separate_by='metric_name',  # One subplot per metric
    colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
    alphas=[0.8, 0.6],  # Cycle through two transparency levels
    figsize=(14, 6)
)

fig = plot_result['fig']
fig.show()
```

#### Example: Overlaid Metrics per X-Column

```python
# Plot one subplot per x-column, with metrics overlaid
plot_result = plot_boxplot(
    results=metrics_result,
    separate_by='x_col',  # One subplot per x_col
    colors=['#1f77b4', '#ff7f0e'],
    figsize=(12, 8)
)

# Access individual axes for further customization
axes = plot_result['axes']
for x_col_name, ax in axes.items():
    ax.set_ylabel('Metric Value')
    ax.grid(True, alpha=0.3)
```

#### Example: Filtering and Single Metric

```python
# Plot only one metric across all x-columns
plot_result = plot_boxplot(
    results=metrics_result,
    metric_name='Accuracy',
    separate_by='both',
    figsize=(10, 6)
)
```

---

## Workflow Integration

Typical workflow combining `inspect` and `plot` modules:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_fw.inspect import boxplot_metvx
from ml_fw.plot import plot_boxplot

# 1. Train a model
X, y = ...  # your data
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 2. Inspect performance binned by features
perf_binned = boxplot_metvx(
    x_dat=X_test[['feature_1', 'feature_2']],
    y_true=y_test,
    y_mod=y_pred,
    box_metric=None,  # auto-select accuracy for classification
    bins=10,
    kfolds=50
)

# 3. Visualize
plot_result = plot_boxplot(
    results=perf_binned,
    separate_by='x_col',
    colors=['#1f77b4', '#ff7f0e'],
    figsize=(14, 6)
)

plot_result['fig'].savefig('model_diagnostics.png', dpi=150, bbox_inches='tight')
```

---

## Customization

### Access Individual Axes

After plotting, retrieve individual axes for further customization:

```python
plot_result = plot_boxplot(results=box_stats, separate_by='both')

# Access a specific subplot
ax = plot_result['axes'][('x_col_name', 'metric_name')]

# Customize
ax.set_title('Custom Title', fontsize=14)
ax.set_ylabel('Custom Y-Label')
ax.grid(True, alpha=0.3)
```

### Add Annotations or Thresholds

```python
# Add a horizontal reference line to each subplot
for ax in plot_result['axes_flat']:
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()
```

### Save Figure

```python
plot_result['fig'].savefig('diagnostics.png', dpi=150, bbox_inches='tight')
# or
plot_result['fig'].savefig('diagnostics.pdf', bbox_inches='tight')
```

---

## Use Cases

### Model Validation

Visualize how a model's accuracy or custom performance metrics vary across bins of important features to identify performance disparities or underfitting regions.

### Space-Weather and Geophysics

Bin model performance by solar wind speed, geomagnetic activity, or local time to understand how model behavior depends on physical conditions.

### Residual Analysis

Plot binned residual statistics (`boxplot_vx`) to check for systematic bias or heteroscedasticity in model predictions.

### Multi-Metric Comparison

Use `separate_by='metric_name'` to compare how different performance metrics (accuracy, F1, precision) behave across feature bins.

---

## Notes

- **Box plot components**: 
  - Box: interquartile range (Q1–Q3)
  - Line inside box: median (Q2)
  - Diamond (if `showmeans=True`): mean
  - Whiskers: extend to 1.5 × IQR outside the box (Tukey's definition)
  - Circles/points (if `showfliers=True`): outliers beyond whiskers
  
- **Empty bins or small samples**: If a bin has very few data points or all NaN values, box statistics may be undefined (NaN). These bins will appear empty in the plot.
- **Color mapping**: Colors are applied in order as boxes/subplots are drawn. For predictable coloring, specify colors explicitly.
- **Performance**: For very large result dictionaries (many x-columns × many metrics), creating all subplots may be memory-intensive. Consider filtering via `metric_name` or `x_col` to reduce the number of subplots.

---

## See Also

- **`ml_fw.inspect.boxplot_vx()`** — Generate binned residual statistics.
- **`ml_fw.inspect.boxplot_metvx()`** — Generate binned performance metrics.
- **`ml_fw.inspect.rolling_met()`** — Generate rolling-window metrics (simple DataFrame plotting with matplotlib).
- **Matplotlib `bxp` docs**: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bxp.html
