# ml_fw

**ML framework for developing and testing ML models.**

A lightweight Python toolkit for building machine learning workflows with a focus on heliophysics, space-weather, and similar scientific domains. `ml_fw` combines pandas, scikit-learn, and statsmodels to provide end-to-end support for feature engineering, model training and tuning, correlation profiling, result inspection, visualization, and ensemble input perturbation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kylermurphy/ml_fw.git
cd ml_fw
```

Install in editable mode:

```bash
pip install -e .
```

For testing and documentation extras:

```bash
pip install -e .[test]      # flake8, pytest-cov
pip install -e .[doc]       # sphinx, sphinx-rtd-theme, m2r2
```

**Requirements:** Python >= 3.12

---

## Package Layout

```
ml_fw/
├── data_io.py              Feature/target creation, time-lagged features
├── ml_mod.py               Model training & parameter tuning wrapper
├── profile.py              Correlation matrix profiling
├── inspect/                Model result inspection (binned/rolling diagnostics)
├── plot/                   Plotting tools for inspection results
└── perturbed_input/        ARIMA residual ensemble perturbation
```

---

## Modules

### `data_io` — Feature and Target Preparation

Utilities for preparing feature and target datasets for ML models. Supports log transformation for wide-dynamic-range variables and cyclical encoding (sin/cos) for periodic variables like local time and longitude.

```python
from ml_fw import data_io

features, targets = data_io.create(
    dataframe=df,
    feature_columns=['x1', 'x2'],
    target_columns=['y'],
    log_columns=['x1'],
    cyclical_columns=['local_time']
)

df_lagged = data_io.feat_shift(
    dataframe=df,
    time_column='DateTime',
    periods=[5, 10],
    unit='min'
)
```

**Main functions:** `create()`, `feat_shift()`

**Full documentation:** [ml_fw/docs/modules/data_io.md](ml_fw/docs/modules/data_io.md)

---

### `ml_mod` — Model Training & Tuning

Scikit-learn training wrapper with optional hyperparameter grid search. Supports fractional grid-search subsampling for faster tuning and multi-scorer parameter selection.

```python
from ml_fw import ml_mod
from sklearn.ensemble import RandomForestRegressor

model = ml_mod.train(
    f_dat=features,
    y_dat=targets,
    estimator=RandomForestRegressor(),
    grid_params={'n_estimators': [50, 100], 'max_depth': [5, 10]},
    grid_ratio=0.3  # Use 30% of data for grid search
)
```

**Main functions:** `train()`, `tune()`

**Full documentation:** [ml_fw/docs/modules/ml_mod.md](ml_fw/docs/modules/ml_mod.md)

---

### `profile` — Correlation Profiling

Efficient correlation computation between features and targets, with optional stratification by categorical variables or boolean filters.

```python
from ml_fw import profile

cor = profile.cor_matrix(
    feature_dat=features,
    target_dat=targets,
    categorical_dat=['storm_flag']  # Separate correlations for storm/non-storm
)
```

**Main functions:** `cor_matrix()`

**Full documentation:** [ml_fw/docs/modules/profile.md](ml_fw/docs/modules/profile.md)

---

### `inspect` — Model Diagnostics

Tools for inspecting model results through binned statistics and rolling-window metrics. Compute box-and-whisker stats or custom metrics across bins of a feature, or compute rolling metrics over time.

```python
from ml_fw.inspect import boxplot_vx, boxplot_metvx, rolling_met

# Bin residuals by a feature
box_stats = boxplot_vx(
    x_dat=features[['x1']],
    y_dat=residuals,
    bins=10
)

# Bin a performance metric (e.g., accuracy) by a feature
met_stats = boxplot_metvx(
    x_dat=features[['x1']],
    y_true=y_true,
    y_mod=y_pred,
    box_metric=lambda yt, yp: (yt == yp).mean()
)

# Rolling metric over time
rolling_accuracy = rolling_met(
    met_dat=results_df,
    y_true='y_true',
    y_pred='y_pred',
    on='DateTime',
    roll_metric=lambda yt, yp: (yt == yp).mean()
)
```

**Main functions:** `boxplot_vx()`, `boxplot_metvx()`, `rolling_met()`

**Full documentation:** [ml_fw/inspect/README.md](ml_fw/inspect/README.md)

---

### `plot` — Visualization Helpers

Render box-and-whisker plots from `inspect` module output using matplotlib. Flexible subplot layouts for comparing results across multiple variables and metrics.

```python
from ml_fw.plot import plot_boxplot

fig_dict = plot_boxplot(
    results=box_stats,
    separate_by='both',  # Create subplot grid
    colors=['#1f77b4', '#ff7f0e'],
    figsize=(12, 8)
)
```

**Main functions:** `plot_boxplot()`

**Full documentation:** [ml_fw/plot/README.md](ml_fw/plot/README.md)

---

### `perturbed_input` — Ensemble Input Perturbation

Standalone module for generating perturbed time series ensembles via ARIMA residual resampling. Useful for uncertainty quantification and ensemble modeling workflows.

```python
from ml_fw.perturbed_input import generate_perturbations

ensemble = generate_perturbations(
    y=time_series,
    n_ensemble=100,
    method='auto',
    seed=42
)
```

**Main functions:** `generate_perturbations()`, `fit_model()`, `characterize_residuals()`, `plot_ensemble()`, `plot_residual_diagnostics()`

**Full documentation:** [ml_fw/perturbed_input/README.md](ml_fw/perturbed_input/README.md)

---

## Testing

Run the test suite with pytest:

```bash
pytest
```

For verbose output:

```bash
pytest -v
```

Note: The current test suite covers the `perturbed_input` module. Other modules are documented but not yet covered by automated tests.

---

## Contributing

We welcome bug reports, feature suggestions, and improvements. See [docs/Maintenance_Philosophy.md](docs/Maintenance_Philosophy.md) for guidelines on contributing and our review process.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
