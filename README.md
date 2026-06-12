# Perturbed Input Ensemble Modeling

A standalone Python module for generating perturbed time series for ensemble modelling workflows using ARIMA residual resampling.

The module fits an ARIMA model to a univariate input time series, extracts the model residuals, characterizes those residuals statistically, and resamples them to generate an ensemble of statistically plausible perturbed signals.

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
- Ensemble perturbation generation
- Residual diagnostic plotting
- Ensemble visualization
- Reproducible random seeding
- Input validation utilities
- Pytest-compatible structure

---

## Project Structure

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

1. Fit an ARIMA model to the input signal.
2. Extract the residuals from the fitted model.
3. Analyze the residuals using statistical diagnostics.
4. Resample the residuals and add them back to the fitted signal.

The residual is defined as:

    e_t = y_t - y_hat_t

where:

- `y_t` is the original input signal
- `y_hat_t` is the fitted ARIMA signal
- `e_t` is the residual error

Each perturbed ensemble member is generated as:

    y_i = y_hat + e_i

where:

- `y_hat` is the fitted ARIMA signal
- `e_i` is a newly sampled residual sequence

---

## Installation

Clone the repository:

    git clone <repository-url>
    cd perturbed-input

Install the package locally:

    pip install -e .

---

## Dependencies

- numpy
- scipy
- matplotlib
- statsmodels
- pmdarima
- pytest

---

## Basic Usage

    from perturbed_input import generate_perturbations

    ensemble = generate_perturbations(
        y=data,
        n_ensemble=100,
        method="auto",
        seasonal=False,
        seed=42,
    )

The returned `ensemble` array has shape:

    (n_ensemble, n_samples)

For example, if the input signal has 500 samples and `n_ensemble=100`, the output shape will be:

    (100, 500)

---

## Input Data Requirements

The module currently accepts a clean, univariate numeric time series.

Users are responsible for loading their own dataset, selecting the target variable, and preprocessing the data before calling the perturbation functions.

The input `y` should be:

- one-dimensional
- numeric
- at least 10 samples long
- free of missing values
- ordered in time

Example using a CSV file:

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

This means the package can be used with any field that produces ordered numerical observations over time, as long as the data is converted into a clean one-dimensional array before use.

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
    )

Generates an ensemble of perturbed time series.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `y` | array-like | Input univariate time series |
| `n_ensemble` | int | Number of perturbed realizations to generate |
| `method` | str | Residual sampling method |
| `block_length` | int or None | Block length for block bootstrap sampling |
| `seasonal` | bool | Whether to allow seasonal ARIMA terms |
| `m` | int | Seasonal period (e.g. `m=24` for hourly data with a daily cycle) |
| `seed` | int or None | Random seed for reproducible sampling |
| `verbose` | bool | Print selected ARIMA order and sampling method |
| `fit` | dict or None | Pre-fitted result from `fit_model()` — skips refitting |
| `auto_arima_kwargs` | dict or None | Extra keyword arguments for `pmdarima.auto_arima()` |
| `kde_bandwidth` | float, str, or None | KDE bandwidth (`'scott'`, `'silverman'`, or scalar) |

### Supported Methods

- `auto`
- `gaussian`
- `empirical`
- `kde`
- `block`

### Returns

Returns a NumPy array with shape:

    (n_ensemble, len(y))

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

The module validates that input arrays are:

- one-dimensional
- numeric
- at least 10 samples long
- free of NaN values

The validation helper is:

    _validate_1d_array(
        array,
        name,
    )

Invalid inputs raise clear exceptions.

Examples:

    ValueError: y must be a 1D array.
    ValueError: y must contain at least 10 values.
    ValueError: y must not contain NaN values.

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

    fig = plot_ensemble(
        x=None,
        y=data,
        ensemble=ensemble,
        n_show=50,
        title="Perturbed Ensemble",
        show_boxplot=False,
    )

This plots the original signal together with selected ensemble members and
returns the `matplotlib.figure.Figure` for further customisation.

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


    plot_ensemble(
        x=t,
        y=data,
        ensemble=ensemble,
        n_show=50,
        title="Example Perturbed Ensemble",
        show_stats=True,
        show_boxplot=False,
    )

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
- energy demand forecasting
- sales and demand data
- stock prices and financial indicators
- economic indicators
- medical or biological measurements over time
- uncertainty quantification studies

---

## References

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. Time Series Analysis: Forecasting and Control.
- Hyndman, R. J., Athanasopoulos, G. Forecasting: Principles and Practice.
- Perturbed Input Ensemble Modeling With the Space Weather Modeling Framework.
- The Importance of Ensemble Techniques for Operational Space Weather Forecasting.