# Perturbed Input Ensemble Modeling

A Python module for generating perturbed time series for ensemble modelling workflows using ARIMA residual resampling.

The module fits an ARIMA model to an input time series (single or multiple series at once), extracts the model residuals, characterizes those residuals statistically, and resamples them to generate an ensemble of statistically plausible perturbed signals. When generating ensembles for co-perturbed series (e.g., correlated measurements from multiple instruments), series can share block-bootstrap start positions to stay time-aligned.

This is useful for uncertainty quantification, sensitivity testing, and time-series ensemble modelling workflows where multiple realistic versions of an input signal are needed.

---

## Features

- Automatic ARIMA fitting using `pmdarima.auto_arima`
- Residual extraction from fitted ARIMA models
- Residual diagnostic characterization
- Automatic sampling method recommendation
- Gaussian residual sampling
- Empirical bootstrap sampling
- Kernel density estimation sampling
- Moving block bootstrap sampling
- Ensemble perturbation generation (single or multi-series, vectorized)
- Block-sharing for co-perturbed series (time-aligned ensembles)
- Residual diagnostic plotting
- Ensemble visualization
- Reproducible random seeding
- Input validation utilities
- Pytest-compatible structure

---

## Structure

    perturbed_input/
    │
    ├── __init__.py       ← public API exports
    ├── diagnostics.py
    ├── ensemble.py       ← generate_perturbations (main entry point)
    ├── fit.py
    ├── plot.py
    ├── sampling.py
    └── utils.py

    tests/
    examples/

---

## Method Overview

The perturbation workflow has four main steps:

1. Fit an ARIMA model to the input signal (or a separate reference signal).
2. Extract the residuals from the fitted model.
3. Analyze the residuals using statistical diagnostics.
4. Resample the residuals and add them to the input signal.

The residual is defined as:

    e_t = y_t - y_hat_t

where:

- `y_t` is the original input signal
- `y_hat_t` is the fitted ARIMA signal
- `e_t` is the residual error

**Note on ARIMA burn-in:** ARIMA differencing (non-seasonal order `d` plus seasonal order `D` at period `m`) consumes the first `d + D*m` observations before the model can produce a real in-sample prediction. During this burn-in period, `fitted` is zero and the corresponding residuals are inflated (essentially equal to `y`) rather than genuine model error. To prevent these outliers from skewing diagnostics and ensemble generation, the first `d + D*m` residuals are automatically replaced with the mean of the remaining residuals by `fit_model()`. In the returned fitted model these values are replace with `np.nan`.

Each perturbed ensemble member is generated as:

    y_i = y + e_i

where:

- `y` is the original input signal
- `e_i` is a newly sampled residual sequence

When `fit_ts` is provided, the residuals are derived from a separate reference
series rather than from `y` itself, but the sampled residuals are still added
to `y`.

---

## Basic Usage

### Single-Series Input

    from perturbed_input import generate_perturbations

    ensemble = generate_perturbations(
        y=data,
        n_ensemble=100,
        method="auto",
        seasonal=False,
        seed=42,
    )

The returned `ensemble` array has shape `(n_ensemble, len(data))`.

For example, if the input signal has 500 samples and `n_ensemble=100`, the output shape will be:

    (100, 500)

### Multi-Series Input

For multiple series (e.g., correlated time series from several instruments), pass a 2-D array:

    # y has shape (500, 3) — three series, each with 500 samples
    y = np.array([...]).reshape(500, 3)

    ensemble = generate_perturbations(
        y=y,
        n_ensemble=100,
        method="auto",
        seasonal=False,
        seed=42,
    )

The returned `ensemble` array has shape `(n_ensemble, npts, n_series)`:

    (100, 500, 3)

Each series is fit and diagnosed independently. When the method is `'block'`, series using block bootstrap share block-start positions per ensemble member, keeping co-perturbed series time-aligned.

---

## Input Data Requirements

The module accepts clean numeric time series data: either a single series (1-D) or multiple series (2-D, one per column).

Users are responsible for loading their own dataset, selecting the target variable(s), and preprocessing the data before calling the perturbation functions.

The input `y` should be:

- **1-D array** `(n,)` for a single series, or **2-D array** `(npts, n_series)` for multiple series.
- **numeric** (int or float)
- **at least 10 samples long** (`npts >= 10`)
- **free of missing values** (no NaN)
- **ordered in time**

When `y` is 2-D, each column is treated as an independent time series: fit, diagnosed, and resampled separately. The output ensemble then has shape `(n_ensemble, npts, n_series)`.

**Example using a CSV file (single series):**

    import pandas as pd

    from perturbed_input import generate_perturbations

    df = pd.read_csv("data.csv")

    y = df["target_column"].dropna().to_numpy()

    ensemble = generate_perturbations(
        y=y,
        n_ensemble=100,
        method="auto",
        seed=42,
    )

**Example using multiple series from a CSV:**

    # Load multiple columns from a CSV
    df = pd.read_csv("data.csv", usecols=["series_1", "series_2", "series_3"])
    y = df.to_numpy()  # shape (n, 3)

    ensemble = generate_perturbations(
        y=y,
        n_ensemble=100,
        method="auto",
        seed=42,
    )
    # ensemble.shape == (100, n, 3)

The package can be used with any field that produces ordered numerical observations over time, as long as the data is converted into a clean array (1-D or 2-D) before use.

---

## Main API

### generate_perturbations

    generate_perturbations(
        y,
        n_ensemble=100,
        method="auto",
        block_length=None,
        seasonal=False,
        m=1,
        seed=None,
        verbose=False,
        fit=None,
        auto_arima_kwargs=None,
        kde_bandwidth=None,
        fit_ts=None,
    )

Generates an ensemble of perturbed time series.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `y` | array-like | Input time series. 1-D array `(n,)` for a single series, or 2-D array `(npts, n_series)` for multiple series. |
| `n_ensemble` | int | Number of perturbed realizations to generate |
| `method` | str | Residual sampling method: one of `'auto'`, `'gaussian'`, `'empirical'`, `'kde'`, or `'block'`. |
| `block_length` | int or None | Block length for block bootstrap sampling. Auto-estimated per series when None. |
| `seasonal` | bool | Whether to allow seasonal ARIMA terms. Ignored when `fit` is provided. |
| `m` | int | Seasonal period (e.g. `m=24` for hourly data with a daily cycle). Ignored when `fit` is provided. |
| `seed` | int or None | Random seed for reproducible sampling. |
| `verbose` | bool | If True, print selected ARIMA order, sampling method, and noise source (fit_ts or y) for each series. |
| `fit` | dict, list of dict, or None | Pre-fitted result(s) from `fit_model()` — skips refitting. For single-series input (1-D `y`), pass a dict. For multi-series input (2-D `y`), pass a list of one dict per series (in column order). Cannot be used with `fit_ts`. |
| `auto_arima_kwargs` | dict or None | Extra keyword arguments for `pmdarima.auto_arima()`. Ignored when `fit` is provided. |
| `kde_bandwidth` | float, str, or None | Bandwidth for KDE sampling: `'scott'`, `'silverman'`, or a scalar float. Only used when method is `'kde'`. |
| `fit_ts` | array-like or None | Reference time series used to derive the residual noise model. 1-D array `(n,)` or 2-D array `(npts_fit, n_series)` with the same shape convention as `y`. ARIMA is fitted to `fit_ts` (per series) and its residuals are sampled and added to `y`. May have a different length than `y`, but must have the same number of series. Cannot be used with `fit`. |
| `return_fitted` | bool | If True, return both the ensemble and the ARIMA fitted values as a tuple `(ensemble, fitted)`. Default is False (return only the ensemble array). |

### Supported Methods

- `auto` — Automatically select the sampling method based on residual diagnostics (per series).
- `gaussian` — Assume residuals are normally distributed.
- `empirical` — Bootstrap directly from the observed residuals.
- `kde` — Kernel density estimation of the residual distribution.
- `block` — Moving block bootstrap (for autocorrelated residuals).

### Multi-Series Input

When `y` is 2-D with shape `(npts, n_series)`, the module fits and perturbs each series independently while optionally keeping them time-aligned:

- Each column of `y` is fit to an ARIMA model independently.
- Residuals from each series are diagnosed and a sampling method is selected per series.
- When the resolved method is `'block'`, series sharing the same block length will share block-start positions within each ensemble member. This keeps co-perturbed series (e.g., measurements from correlated instruments) time-aligned so that perturbations occur at the same times across columns.

Example:

```python
import numpy as np
from ml_fw.perturbed_input import generate_perturbations

# Three time series, each with 500 samples
y = np.random.randn(500, 3)

ensemble = generate_perturbations(
    y=y,
    n_ensemble=100,
    method="block",  # All series use block sampling, sharing block positions
    seed=42,
)

# ensemble.shape == (100, 500, 3)
# Access individual series: ensemble[:, :, 0]
```

### Getting the Fitted ARIMA Curve

To plot the original series alongside the ARIMA model's fitted curve, use `return_fitted=True`:

```python
import numpy as np
import matplotlib.pyplot as plt
from ml_fw.perturbed_input import generate_perturbations

# Generate data
t = np.arange(200)
y = np.sin(2 * np.pi * t / 50) + 0.1 * np.random.randn(200)

# Get both ensemble and fitted ARIMA curve
ensemble, fitted = generate_perturbations(
    y=y,
    n_ensemble=50,
    method="auto",
    seed=42,
    return_fitted=True,
)

# Plot original vs. ARIMA fit
plt.plot(t, y, 'k-', linewidth=2, label="Original")
plt.plot(t, fitted, 'r-', linewidth=2, alpha=0.7, label="ARIMA Fit")
plt.legend()
plt.show()

# Plot ensemble members too
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(min(30, ensemble.shape[0])):
    ax.plot(t, ensemble[i], 'steelblue', alpha=0.2, linewidth=0.8)
ax.plot(t, y, 'k-', linewidth=2, label="Original")
ax.plot(t, fitted, 'r-', linewidth=2, label="ARIMA Fit")
ax.legend()
plt.show()
```

**Note:** When `fit_ts` is provided (a separate reference series), the returned `fitted` curve matches `fit_ts`'s length, not `y`'s length. You must handle x-axis alignment when plotting them together.

### Vectorization

The ensemble generation is vectorized across the ensemble dimension to avoid a per-member Python loop:
- For Gaussian, empirical, and KDE methods, sampling is done in a single `rng` call with `n_samples = n_ensemble × npts`, then reshaped to `(n_ensemble, npts)`.
- For block bootstrap, all block start indices for all ensemble members (and all series sharing a block length) are drawn in a single `rng.choice` call via `_draw_block_indices`.

The per-series ARIMA fitting, diagnostics, and (when needed) KDE setup remain a Python loop because `pmdarima`, statistical tests, and `gaussian_kde` all operate on a single 1-D series at a time.

### Returns

By default (`return_fitted=False`):
- A NumPy array with shape `(n_ensemble, n)` when `y` is 1-D, or `(n_ensemble, npts, n_series)` when `y` is 2-D.
- Each ensemble member is `y + sampled_residuals`.

When `return_fitted=True`:
- A tuple `(ensemble, fitted)` where `ensemble` has the shape above and `fitted` is the in-sample ARIMA fitted values.
- `fitted` uses the same shape convention as `y`: 1-D `(npts_fit,)` or 2-D `(npts_fit, n_series)`.
- When `fit_ts` is provided, `fitted` has the length of `fit_ts`, not `y` (they can differ).

---

## ARIMA Model Fitting

The `fit_model` function fits an ARIMA model to the input signal.

    from perturbed_input import fit_model

    fit = fit_model(
        y=data,
        seasonal=False,
    )

The function returns a dictionary containing:

    {
        "model": model,
        "fitted": fitted_values,
        "residuals": residuals,
        "order": model.order,
        "seasonal_order": model.seasonal_order,
        "aic": model.aic(),
    }

The ARIMA model is selected automatically using:

    pmdarima.auto_arima()

**Burn-in Residual Correction:** The returned `"residuals"` have their first `d + D*m` values replaced with the mean of the remaining residuals, where `d` and `D*m` are the non-seasonal and seasonal differencing orders. This prevents inflated burn-in residuals from skewing diagnostics. If the differencing order consumes the entire series (burn-in length >= series length), `fit_model()` raises a `ValueError`.

---

## Residual Characterization

Residuals are analyzed using the `characterize_residuals` function.

    from perturbed_input import characterize_residuals

    stats = characterize_residuals(residuals)

The function returns:

    {
        "mean": mean_value,
        "std": std_value,
        "skewness": skewness,
        "kurtosis": kurtosis_value,
        "shapiro_pvalue": shapiro_pvalue,
        "ljungbox_pvalue": ljungbox_pvalue,
        "recommended_method": recommended_method,
        "block_length": block_length,
    }

These statistics are used to determine which residual sampling method is most appropriate.

---

## Automatic Sampling Method Selection

When `method="auto"` is used, the module recommends a residual sampling method based on residual diagnostics.

The automatic method selection system attempts to match the residual sampling method to the statistical structure of the residual distribution.

Different sampling methods preserve different statistical properties. The selection logic is designed to choose the method that best preserves the observed residual behavior while avoiding unnecessary assumptions about the data.

In general:

- Gaussian sampling is preferred when residuals are approximately normal and independent.
- Empirical bootstrap sampling is used when residuals are non-Gaussian but still approximately independent.
- KDE sampling is preferred when residuals are strongly skewed or heavy-tailed because it can better preserve asymmetric or non-Gaussian distribution structure.
- Moving block bootstrap sampling is used when residuals contain temporal autocorrelation because independent resampling would destroy time dependence.

| Residual Behavior | Selected Method |
|---|---|
| Residuals are autocorrelated | `block` |
| Residuals are approximately Gaussian | `gaussian` |
| Residuals are strongly skewed or heavy-tailed | `kde` |
| Otherwise | `empirical` |

Autocorrelation is checked first because temporal dependence is usually more important to preserve than the marginal distribution shape.

---

## Statistical Criteria

### Shapiro-Wilk Normality Test

The Shapiro-Wilk test is used to check whether the residuals are approximately normally distributed.

The null hypothesis is:

    Residuals are normally distributed.

The module treats residuals as approximately normal when:

    p-value > 0.05

If the residuals are normal, have low skewness, and have low excess kurtosis, Gaussian sampling is recommended.

---

### Ljung-Box Autocorrelation Test

The Ljung-Box test is used to check whether the residuals are independently distributed.

The null hypothesis is:

    Residuals are independently distributed.

The module treats residuals as independent when:

    p-value > 0.05

If the Ljung-Box p-value is less than or equal to `0.05`, the residuals are treated as autocorrelated and block bootstrap sampling is recommended.

---

### Skewness Thresholds

Skewness measures asymmetry in the residual distribution.

| Absolute Skewness | Interpretation |
|---|---|
| `< 0.5` | Low skewness / approximately symmetric |
| `0.5 to 1.0` | Moderate skewness |
| `>= 1.0` | High skewness |

High skewness suggests that Gaussian sampling may not represent the residual distribution well.

---

### Kurtosis Thresholds

The module uses excess kurtosis, as returned by:

    scipy.stats.kurtosis()

For excess kurtosis:

    0 means approximately Gaussian tails.

| Absolute Excess Kurtosis | Interpretation |
|---|---|
| `< 1` | Near-Gaussian tails |
| `1 to 3` | Moderately non-Gaussian tails |
| `>= 3` | Strongly non-Gaussian tails |

High excess kurtosis suggests heavy tails, outliers, or extreme residual behavior.

In that case, KDE sampling is preferred because it can better preserve non-Gaussian distribution shape.

---

## Sampling Methods

### Gaussian Sampling

Gaussian sampling assumes residuals follow a normal distribution:

    e ~ N(mean, standard deviation)

Implementation:

    rng.normal(
        mu,
        sigma,
        n_samples,
    )

Use this when residuals are approximately normal, symmetric, and independent.

Advantages:

- Fast
- Simple
- Smooth
- Good for near-Gaussian residuals

Limitations:

- Does not preserve skewness well
- Does not preserve heavy tails well
- May underestimate extreme events

---

### Empirical Bootstrap

Empirical bootstrap samples directly from the observed residuals with replacement.

Implementation:

    rng.choice(
        residuals,
        size=n_samples,
        replace=True,
    )

Use this when residuals are independent but not clearly Gaussian.

Advantages:

- Nonparametric
- Simple
- Preserves the observed residual distribution
- Does not assume normality

Limitations:

- Does not preserve temporal dependence
- Cannot generate residual values outside the observed sample
- Can be limited by small sample size

---

### Kernel Density Estimation Sampling

KDE sampling estimates a smooth probability distribution from the residuals and samples from that distribution.

Implementation:

    kde = gaussian_kde(residuals)

    samples = kde.resample(
        n_samples,
        seed=rng,
    ).flatten()

Use this when residuals are strongly skewed, heavy-tailed, or clearly non-Gaussian.

Advantages:

- Nonparametric
- Smooths the residual distribution
- Can represent skewed distributions
- Can better represent heavy-tailed residuals

Limitations:

- Sensitive to bandwidth selection
- More computationally expensive
- May oversmooth small datasets

---

### Moving Block Bootstrap

Moving block bootstrap samples contiguous blocks of residuals.

This is used when residuals are autocorrelated.

Implementation idea:

    1. Choose a random block start index.
    2. Copy a block of residuals.
    3. Repeat until enough residuals are sampled.
    4. Trim the sampled sequence to the required length.

Use this when the residuals still contain temporal dependence.

Advantages:

- Preserves local autocorrelation
- Useful for time series residuals
- Better for dependent residual structures

Limitations:

- Block length affects results
- Can introduce discontinuities between blocks
- May not fully preserve long-memory dependence

---

## Block Length

If the user does not provide a block length, the module estimates one using:

    block_length = int(len(residuals) ** (1 / 3))

This gives a simple default block length that increases with the size of the residual series.

| Residual Length | Estimated Block Length |
|---|---|
| 100 | 4 |
| 500 | 7 |
| 1000 | 9 |
| 10000 | 21 |


---

## Input Validation

The module uses two validation helpers:

### `_validate_array` (used by `generate_perturbations`)

Validates and coerces input to a 2-D float NumPy array of shape `(n, n_series)`. A 1-D input of shape `(n,)` is promoted to `(n, 1)`.

    _validate_array(array, name)

Checks:
- `ndim` is 1 or 2 (3-D and higher raise an error).
- `shape[0] >= 10` (at least 10 samples required).
- No NaN values.

Error messages:

    ValueError: {name} must be a 1D or 2D array.
    ValueError: {name} must contain at least 10 values.
    ValueError: {name} must not contain NaN values.

Used for: `y` and `fit_ts` in `generate_perturbations`.

### `_validate_1d_array` (used by `fit_model`, plotting, and stats functions)

Validates and coerces input to a 1-D float NumPy array.

    _validate_1d_array(array, name)

Checks:
- `ndim` is exactly 1 (raises error for any other shape).
- `len(array) >= 10` (at least 10 samples required).
- No NaN values.

Error messages:

    ValueError: {name} must be a 1D array.
    ValueError: {name} must contain at least 10 values.
    ValueError: {name} must not contain NaN values.

Used for: residuals in `characterize_residuals`; inputs to `plot_residual_diagnostics`, `compute_ensemble_stats`, `plot_ensemble`; and `y` in `fit_model`.

---

## Plotting Tools

The module includes plotting utilities for residual diagnostics and ensemble visualization.

### Residual Diagnostic Plot

    from perturbed_input import plot_residual_diagnostics

    plot_residual_diagnostics(
        x=None,
        residuals=residuals,
        title="Example",
    )

This creates a four-panel diagnostic figure:

1. Residual time series
2. Residual autocorrelation function
3. Residual histogram with Gaussian PDF overlay
4. Residual Q-Q plot

These plots help determine whether the residuals are:

- centered around zero
- independent
- approximately Gaussian
- skewed
- heavy-tailed
- still structurally patterned

---

### Ensemble Plot

    from perturbed_input import plot_ensemble
    import matplotlib.pyplot as plt

    ax = plot_ensemble(
        x=None,
        y=data,
        ensemble=ensemble,
        n_show=50,
        plot_mean=True,
        plot_median=True,
        colormap="plasma",
        figsize=(16, 6),
    )
    ax.set_title("Perturbed Ensemble")
    plt.show()

This plots the original signal together with selected ensemble members,
optionally with ensemble mean and/or median overlays, and returns the
`matplotlib.axes.Axes` for further customization.

Ensemble members are colored using a matplotlib colormap (default `"plasma"`);
the original series is highlighted in black. Use `plot_mean=True` and/or
`plot_median=True` to overlay red dashed (mean) and/or orange dash-dot (median)
lines.

**Note:** `plot_ensemble` requires a 2-D ensemble of shape `(n_ensemble, len(y))`
and a 1-D `y`. For multi-series ensembles from `generate_perturbations` (3-D output),
call it per series:

    for j in range(ensemble.shape[2]):
        ax = plot_ensemble(x, y[:, j], ensemble[:, :, j])
        ax.set_title(f"Series {j}")
        plt.show()

### Ensemble Statistics

    from perturbed_input import compute_ensemble_stats

    stats = compute_ensemble_stats(x=None, y=data, ensemble=ensemble)

Returns a dictionary with pointwise statistics at each time step:

    {
        "x": x,
        "mean": ...,
        "median": ...,
        "std": ...,
        "min": ...,
        "max": ...,
        "q05": ...,   # 5th percentile
        "q95": ...,   # 95th percentile
    }

**Note:** `compute_ensemble_stats` currently requires a 2-D ensemble of shape `(n_ensemble, len(y))`
and a 1-D `y`. For multi-series ensembles, call it per series:

    for j in range(ensemble.shape[2]):
        stats = compute_ensemble_stats(x, y[:, j], ensemble[:, :, j])

---

## Reproducibility

The module uses NumPy's random number generator:

    np.random.default_rng(seed)

Providing a seed makes the generated ensemble reproducible.

Example:

    ensemble_1 = generate_perturbations(
        y=data,
        seed=42,
    )

    ensemble_2 = generate_perturbations(
        y=data,
        seed=42,
    )

Both calls will generate the same perturbation ensemble.

---

## Example Workflow

    import numpy as np

    from perturbed_input import (
        fit_model,
        characterize_residuals,
        generate_perturbations,
        plot_residual_diagnostics,
        plot_ensemble,
    )


    rng = np.random.default_rng(42)

    t = np.arange(500)

    data = (
        np.sin(2 * np.pi * t / 50)
        + 0.01 * t
        + rng.normal(0, 0.2, size=len(t))
    )


    fit = fit_model(
        y=data,
        seasonal=False,
    )

    residuals = fit["residuals"]

    stats = characterize_residuals(
        residuals
    )

    print(stats)


    plot_residual_diagnostics(
        x=t,
        residuals=residuals,
        title="Example Signal",
    )


    ensemble = generate_perturbations(
        y=data,
        n_ensemble=100,
        method="auto",
        seasonal=False,
        seed=42,
    )


    ax = plot_ensemble(
        x=t,
        y=data,
        ensemble=ensemble,
        n_show=50,
        plot_mean=True,
        colormap="plasma",
        show_boxplot=False,
    )
    ax.set_title("Example Perturbed Ensemble")

---

## Testing

Run the test suite with:

    pytest

For verbose output:

    pytest -v

The test suite should verify:

- valid input handling
- invalid input handling
- ARIMA model fitting output structure
- residual characterization output structure
- supported sampling methods
- ensemble output shape
- reproducibility with fixed seeds

---

## Intended Use

This module is intended for time-series uncertainty modelling workflows where realistic perturbed versions of an input signal are required. Although the project was motivated by space weather and scientific ensemble modelling, the core workflow is general and can be applied to any clean univariate numerical time series.

Possible applications include:

- space weather ensemble modelling
- geomagnetic activity modelling
- solar wind input perturbation
- climate and atmospheric time series
- hydrological and rainfall time series
- engineering signal perturbation
- environmental sensor data
- temperature measurements
- uncertainty quantification studies

---

## References

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. Time Series Analysis: Forecasting and Control.
- Hyndman, R. J., Athanasopoulos, G. Forecasting: Principles and Practice.
- Perturbed Input Ensemble Modeling With the Space Weather Modeling Framework.
- The Importance of Ensemble Techniques for Operational Space Weather Forecasting.