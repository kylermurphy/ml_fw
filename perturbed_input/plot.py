import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import (
    norm,
    probplot,
)

from statsmodels.graphics.tsaplots import plot_acf

from typing import Any, Optional

from .utils import _validate_1d_array


# ============================================================
# Residual Diagnostic Plotting
# ============================================================

def plot_residual_diagnostics(
    x: Optional[Any],
    residuals: np.ndarray,
    title: str = "",
) -> plt.Figure:
    """
    Plot residual diagnostic figures including:

    - Residual time series
    - Residual autocorrelation function (ACF)
    - Residual histogram with Gaussian PDF
    - Residual Q-Q plot

    Parameters
    ----------
    x : array-like or None
        Optional x-axis values, such as dates or time stamps.
        If None, the time index is used.

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

    if x is None:

        x = np.arange(len(residuals))

    else:

        x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError(
                "x must be a 1D array."
            )

        if len(x) != len(residuals):
            raise ValueError(
                "len(x) must match len(residuals)."
            )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    # Residual time series
    axes[0, 0].plot(
        x,
        residuals,
        alpha=0.9,
    )

    axes[0, 0].set_title(
        f"Residual Time Series {title}"
    )

    axes[0, 0].set_xlabel("Time")

    axes[0, 0].set_ylabel("Residual")

    axes[0, 0].tick_params(
        axis="x",
        rotation=45,
    )

    # Residual ACF
    plot_acf(
        residuals,
        ax=axes[0, 1],
    )

    axes[0, 1].set_title(
        f"Residual ACF {title}"
    )

    axes[0, 1].set_xlabel("Lag")

    axes[0, 1].set_ylabel("Autocorrelation")

    # Histogram
    axes[1, 0].hist(
        residuals,
        bins="auto",
        density=True,
        alpha=0.9,
    )

    mu = np.mean(residuals)

    sigma = np.std(residuals)

    hist_x = np.linspace(
        np.min(residuals),
        np.max(residuals),
        100,
    )

    pdf = norm.pdf(
        hist_x,
        mu,
        sigma,
    )

    axes[1, 0].plot(
        hist_x,
        pdf,
        linewidth=2,
    )

    axes[1, 0].set_title(
        f"Residual Histogram {title}"
    )

    axes[1, 0].set_xlabel("Residual Value")

    axes[1, 0].set_ylabel("Density")

    # Q-Q plot
    probplot(
        residuals,
        dist="norm",
        plot=axes[1, 1],
    )

    axes[1, 1].set_title(
        f"Residual Q-Q Plot {title}"
    )

    axes[1, 1].set_xlabel("Theoretical Quantiles")

    axes[1, 1].set_ylabel("Ordered Residual Values")

    plt.tight_layout()

    plt.show()

    return fig


# ============================================================
# Ensemble Plotting
# ============================================================

def plot_ensemble(
    x: Optional[Any],
    y: np.ndarray,
    ensemble: np.ndarray,
    n_show: int = 50,
    title: str = "",
    show_stats: bool = False,
    show_boxplot: bool = False,
) -> Optional[dict]:
    """
    Plot the original time series together with
    ensemble perturbation realizations.

    Parameters
    ----------
    y : np.ndarray
        Original input time series.

    ensemble : np.ndarray
        Ensemble array with shape (n_ensemble, n).

    x : array-like or None, optional
        Optional x-axis values, such as dates or time stamps.
        If None, the time index is used.

    n_show : int, optional
        Number of ensemble realizations to display.

    title : str, optional
        Plot title.

    show_stats : bool, optional
        Whether to return ensemble mean, median, minimum,
        and maximum at each time index.

    show_boxplot : bool, optional
        Whether to show a box plot of the ensemble distribution
        at each time index.

    Returns
    -------
    dict or None
        If show_stats=True, returns a dictionary containing:

        - x
        - mean
        - median
        - min
        - max

        If show_stats=False, returns None.
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

    if x is None:

        x = np.arange(len(y))

    else:

        x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError(
                "x must be a 1D array."
            )

        if len(x) != len(y):
            raise ValueError(
                "len(x) must match len(y)."
            )

    stats = None

    if show_stats:

        stats = {
            "x": x,
            "mean": np.mean(ensemble, axis=0),
            "median": np.median(ensemble, axis=0),
            "min": np.min(ensemble, axis=0),
            "max": np.max(ensemble, axis=0),
        }

    n_show = min(
        n_show,
        ensemble.shape[0],
    )

    plt.figure(figsize=(10, 5))

    for i in range(n_show):

        plt.plot(
            x,
            ensemble[i],
            alpha=0.3,
        )

    plt.plot(
        x,
        y,
        linewidth=2,
        label="Original Series",
    )

    plt.title(title)

    plt.xlabel("Time Index")

    plt.ylabel("Value")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

    if show_boxplot:

        plt.figure(figsize=(10, 5))

        plt.boxplot(
            ensemble.T,
            showfliers=False,
        )

        plt.xticks(
            ticks=np.arange(1, len(x) + 1),
            labels=x,
            rotation=45,
        )

        plt.title(
            f"Ensemble Box Plot {title}"
        )

        plt.xlabel("Time Index")

        plt.ylabel("Value")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.show()

    return stats