import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm

from scipy.stats import (
    gaussian_kde,
    skew,
    kurtosis,
    shapiro,
    norm,
    probplot,
)

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf


# ============================================================
# Constants
# ============================================================

VALID_METHODS = {
    "auto",
    "gaussian",
    "empirical",
    "kde",
    "block",
}


# ============================================================
# Validation Helper
# ============================================================

def _validate_1d_array(array, name):
    """
    Validate that the input is a valid 1D NumPy array.

    Parameters
    ----------
    array : array-like
        Input array to validate.

    name : str
        Variable name used in error messages.

    Returns
    -------
    np.ndarray
        Validated 1D NumPy array converted to float type.

    Raises
    ------
    ValueError
        If the array is not 1D, contains NaN values,
        or contains fewer than 10 values.
    """

    array = np.asarray(array, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    if len(array) < 10:
        raise ValueError(f"{name} must contain at least 10 values.")

    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN values.")

    return array


# ============================================================
# Sampling Helper
# ============================================================

def _sample_block(
    residuals,
    n_samples,
    block_length,
    rng,
):
    """
    Perform moving block bootstrap sampling.

    Parameters
    ----------
    residuals : np.ndarray
        Residual series used for bootstrap sampling.

    n_samples : int
        Number of residual samples to generate.

    block_length : int
        Length of each bootstrap block.

    rng : np.random.Generator
        NumPy random number generator instance.

    Returns
    -------
    np.ndarray
        Bootstrap sampled residual array with shape (n_samples,).

    Raises
    ------
    ValueError
        If block_length is larger than the residual length.
    """

    n = len(residuals)

    if block_length > n:
        raise ValueError(
            "block_length cannot be larger than residual length."
        )

    sampled = []

    starts = np.arange(0, n - block_length + 1)

    while len(sampled) < n_samples:

        start = rng.choice(starts)

        sampled.extend(
            residuals[start:start + block_length]
        )

    return np.asarray(sampled[:n_samples])

# ============================================================
# Residual Sampling Functions
# ============================================================

def _sample_residuals(
    residuals,
    method,
    n_samples,
    block_length,
    rng,
):
    """
    Sample residuals using the specified sampling method.

    Parameters
    ----------
    residuals : np.ndarray
        Residual series used for sampling.

    method : str
        Residual sampling method.

    n_samples : int
        Number of samples to generate.

    block_length : int
        Block length used for block bootstrap sampling.

    rng : np.random.Generator
        NumPy random number generator instance.

    Returns
    -------
    np.ndarray
        Sampled residual array.

    Raises
    ------
    ValueError
        If an invalid sampling method is provided.
    """

    if method == "gaussian":

        mu = np.mean(residuals)
        sigma = np.std(residuals)

        return rng.normal(
            mu,
            sigma,
            n_samples,
        )

    elif method == "empirical":

        return rng.choice(
            residuals,
            size=n_samples,
            replace=True,
        )

    elif method == "kde":

        kde = gaussian_kde(residuals)

        return kde.resample(
            n_samples,
            seed=rng,
        ).flatten()

    elif method == "block":

        return _sample_block(
            residuals=residuals,
            n_samples=n_samples,
            block_length=block_length,
            rng=rng,
        )

    else:

        raise ValueError(
            f"Unknown sampling method: {method}"
        )

# ============================================================
# Residual Sampling Method Recommendation
# ============================================================

def _recommend_method(
    shapiro_pvalue,
    ljungbox_pvalue,
    skewness,
    kurtosis_value,
):
    """
    Recommend an appropriate residual sampling method
    based on residual diagnostic statistics.

    Parameters
    ----------
    shapiro_pvalue : float
        Shapiro-Wilk normality test p-value.

    ljungbox_pvalue : float
        Ljung-Box autocorrelation test p-value.

    skewness : float
        Residual skewness value.

    kurtosis_value : float
        Residual kurtosis value.

    Returns
    -------
    str
        Recommended sampling method.
    """

    normal = shapiro_pvalue > 0.05

    independent = ljungbox_pvalue > 0.05

    low_skew = abs(skewness) < 0.5

    low_kurtosis = abs(kurtosis_value) < 1

    high_skew = abs(skewness) >= 1

    high_kurtosis = abs(kurtosis_value) >= 3

    if not independent:

        return "block"

    if normal and low_skew and low_kurtosis:

        return "gaussian"

    if high_skew or high_kurtosis:

        return "kde"

    return "empirical"

# ============================================================
# Residual Diagnostic Plotting
# ============================================================

def plot_residual_diagnostics(
    residuals,
    title="",
):
    """
    Plot residual diagnostic figures including:

    - Residual time series
    - Residual autocorrelation function (ACF)
    - Residual histogram with Gaussian PDF
    - Residual Q-Q plot

    Parameters
    ----------
    residuals : np.ndarray
        Residual series to analyze.

    title : str, optional
        Additional plot title text.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the diagnostic plots.
    """
   
    residuals = _validate_1d_array(
        residuals,
        "residuals",
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    # Residual time series
    axes[0, 0].plot(
        residuals,
        alpha=0.9,
    )

    axes[0, 0].set_title(
        f"Residual Time Series {title}"
    )

    # Residual ACF
    plot_acf(
        residuals,
        ax=axes[0, 1],
    )

    axes[0, 1].set_title(
        f"Residual ACF {title}"
    )

    # Histogram
    axes[1, 0].hist(
        residuals,
        bins="auto",
        density=True,
        alpha=0.9,
    )

    mu = np.mean(residuals)

    sigma = np.std(residuals)

    x = np.linspace(
        np.min(residuals),
        np.max(residuals),
        100,
    )

    pdf = norm.pdf(
        x,
        mu,
        sigma,
    )

    axes[1, 0].plot(
        x,
        pdf,
        linewidth=2,
    )

    axes[1, 0].set_title(
        f"Residual Histogram {title}"
    )

    # Q-Q plot
    probplot(
        residuals,
        dist="norm",
        plot=axes[1, 1],
    )

    axes[1, 1].set_title(
        f"Residual Q-Q Plot {title}"
    )

    plt.tight_layout()

    plt.show()

    return fig

# ============================================================
# ARIMA Model Fitting
# ============================================================

def fit_model(
    y: np.ndarray,
    seasonal: bool = False,
    m: int = 1,
) -> dict:
    """
    Fit an ARIMA model to a univariate time series.

    Parameters
    ----------
    y : np.ndarray
        Input time series with shape (n,).

    seasonal : bool, optional
        Whether to allow seasonal ARIMA terms
        during model fitting.

    m : int, optional
    Seasonal period. Use m=7 for weekly seasonality in daily data.    

    Returns
    -------
    dict
        Dictionary containing:

        - fitted ARIMA model
        - fitted values
        - residuals
        - ARIMA order
        - seasonal ARIMA order
        - model AIC
    """

    y = _validate_1d_array(y, "y")

    model = pm.auto_arima(
        y,
        seasonal=seasonal,
        m=m,
        information_criterion="aic",
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
        trace=False,
    )
    fitted = model.predict_in_sample()

    residuals = y - fitted

    return {
        "model": model,
        "fitted": np.asarray(fitted),
        "residuals": np.asarray(residuals),
        "order": model.order,
        "seasonal_order": model.seasonal_order,
        "aic": float(model.aic()),
    }

# ============================================================
# Residual Characterization
# ============================================================

def characterize_residuals(
    residuals: np.ndarray,
) -> dict:
    """
    Compute residual statistics and diagnostic tests.

    Parameters
    ----------
    residuals : np.ndarray
        Residual series from fitted model.

    Returns
    -------
    dict
        Dictionary containing:

        - mean
        - standard deviation
        - skewness
        - kurtosis
        - Shapiro-Wilk p-value
        - Ljung-Box p-value
        - recommended sampling method
        - estimated block length
    """

    residuals = _validate_1d_array(
        residuals,
        "residuals",
    )

    mean_value = np.mean(residuals)

    std_value = np.std(residuals)

    skewness = skew(residuals)

    kurtosis_value = kurtosis(residuals)

    shapiro_pvalue = shapiro(
        residuals
    ).pvalue

    ljung_box_table = acorr_ljungbox(
        residuals,
        lags=[5,10,20],
        return_df=True,
    )

    ljungbox_pvalue = float(
        ljung_box_table["lb_pvalue"].min()
    )

    block_length = int(len(residuals) ** (1 / 3))
      

    recommended_method = _recommend_method(
        shapiro_pvalue=shapiro_pvalue,
        ljungbox_pvalue=ljungbox_pvalue,
        skewness=skewness,
        kurtosis_value=kurtosis_value,
    )

    return {
        "mean": float(mean_value),
        "std": float(std_value),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis_value),
        "shapiro_pvalue": float(shapiro_pvalue),
        "ljungbox_pvalue": float(ljungbox_pvalue),
        "recommended_method": recommended_method,
        "block_length": int(block_length),
    }

# ============================================================
# Ensemble Perturbation Generation
# ============================================================

def generate_perturbations(
    y: np.ndarray,
    n_ensemble: int = 100,
    method: str = "auto",
    block_length: int = None,
    seasonal: bool = False,
    seed: int = None,
) -> np.ndarray:
    """
    Generate an ensemble of perturbed time series.

    Parameters
    ----------
    y : np.ndarray
        Input time series with shape (n,).

    n_ensemble : int, optional
        Number of perturbed realizations to generate.

    method : str, optional
        Residual sampling method. Options are:

        - "auto"
        - "gaussian"
        - "empirical"
        - "kde"
        - "block"

    block_length : int or None, optional
        Block length used for block bootstrap
        sampling. If None, the block length is
        estimated automatically.

    seasonal : bool, optional
        Whether to allow seasonal ARIMA terms
        during model fitting.

    seed : int or None, optional
        Random seed for reproducible ensemble generation.

    Returns
    -------
    np.ndarray
        Ensemble array with shape (n_ensemble, n).
    """

    y = _validate_1d_array(y, "y")

    if n_ensemble <= 0:
        raise ValueError(
            "n_ensemble must be greater than 0."
        )

    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS}"
        )

    if block_length is not None:
        if not isinstance(block_length, int):
            raise TypeError(
                "block_length must be an integer or None."
            )

        if block_length <= 0:
            raise ValueError(
                "block_length must be greater than 0."
            )

        if block_length > len(y):
            raise ValueError(
                "block_length cannot be larger than len(y)."
            )

    rng = np.random.default_rng(seed)

    fit = fit_model(
        y,
        seasonal=seasonal,
    )

    residuals = fit["residuals"]

    characterization = characterize_residuals(
        residuals
    )

    if method == "auto":

        method = characterization[
            "recommended_method"
        ]

    if block_length is None:

        block_length = characterization[
            "block_length"
        ]

    ensemble = []

    for _ in range(n_ensemble):

        sampled_residuals = _sample_residuals(
            residuals=residuals,
            method=method,
            n_samples=len(y),
            block_length=block_length,
            rng=rng,
        )

        y_perturbed = fit["fitted"] + sampled_residuals

        ensemble.append(y_perturbed)

    return np.asarray(ensemble)

# ============================================================
# Ensemble Plotting
# ============================================================

def plot_ensemble(
    y: np.ndarray,
    ensemble: np.ndarray,
    n_show: int = 50,
    title: str = "",
) -> None:
    """
    Plot the original time series together with
    ensemble perturbation realizations.

    Parameters
    ----------
    y : np.ndarray
        Original input time series.

    ensemble : np.ndarray
        Ensemble array with shape (n_ensemble, n).

    n_show : int, optional
        Number of ensemble realizations to display.

    title : str, optional
        Plot title.

    Returns
    -------
    None
    """

    y = _validate_1d_array(y, "y")

    ensemble = np.asarray(ensemble)

    if ensemble.ndim != 2:
        raise ValueError(
            "ensemble must be a 2D array."
        )

    if ensemble.shape[1] != len(y):
        raise ValueError(
            "ensemble.shape[1] must match len(y)."
        )

    n_show = min(
        n_show,
        ensemble.shape[0],
    )

    plt.figure(figsize=(10, 5))

    for i in range(n_show):

        plt.plot(
            ensemble[i],
            alpha=0.3,
        )

    plt.plot(
        y,
        linewidth=2,
    )

    plt.title(title)

    plt.xlabel("Time Index")

    plt.ylabel("Value")

    plt.tight_layout()

    plt.show()