import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import norm, probplot
from statsmodels.graphics.tsaplots import plot_acf
from typing import Optional

from .utils import _validate_1d_array


def plot_residual_diagnostics(
    x: Optional[np.ndarray],
    residuals: np.ndarray,
    title: str = "",
) -> Figure:
    """
    Four-panel residual diagnostic figure: time series, ACF, histogram, Q-Q plot.

    Parameters
    ----------
    x : array-like or None
        Optional x-axis values (e.g. timestamps). Integer index used when None.
    residuals : np.ndarray
        Residual series to analyze.
    title : str, optional
        Text appended to each subplot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = _validate_1d_array(residuals, "residuals")

    if x is None:
        x = np.arange(len(residuals))
    else:
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be a 1D array.")
        if len(x) != len(residuals):
            raise ValueError("len(x) must match len(residuals).")

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    axes[0, 0].plot(x, residuals, alpha=0.9)
    axes[0, 0].set_title(f"Residual Time Series {title}")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].tick_params(axis="x", rotation=45)

    plot_acf(residuals, ax=axes[0, 1])
    axes[0, 1].set_title(f"Residual ACF {title}")
    axes[0, 1].set_xlabel("Lag")
    axes[0, 1].set_ylabel("Autocorrelation")

    axes[1, 0].hist(residuals, bins="auto", density=True, alpha=0.9)
    mu = np.mean(residuals)
    sigma = np.std(residuals, ddof=1)
    hist_x = np.linspace(np.min(residuals), np.max(residuals), 200)
    axes[1, 0].plot(hist_x, norm.pdf(hist_x, mu, sigma), linewidth=2)
    axes[1, 0].set_title(f"Residual Histogram {title}")
    axes[1, 0].set_xlabel("Residual Value")
    axes[1, 0].set_ylabel("Density")

    probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f"Residual Q-Q Plot {title}")
    axes[1, 1].set_xlabel("Theoretical Quantiles")
    axes[1, 1].set_ylabel("Ordered Residual Values")

    fig.tight_layout()
    plt.show()

    return fig


def compute_ensemble_stats(
    x: Optional[np.ndarray],
    y: np.ndarray,
    ensemble: np.ndarray,
) -> dict:
    """
    Compute pointwise statistics across ensemble members at each time step.

    Parameters
    ----------
    x : array-like or None
        Optional x-axis values. Integer index used when None.
    y : np.ndarray
        Original time series with shape (n,).
    ensemble : np.ndarray
        Ensemble array with shape (n_ensemble, n).

    Returns
    -------
    dict
        Keys: x, mean, median, std, min, max, q05, q95.
    """
    y = _validate_1d_array(y, "y")
    ensemble = np.asarray(ensemble)

    if ensemble.ndim != 2 or ensemble.shape[1] != len(y):
        raise ValueError("ensemble must have shape (n_ensemble, len(y)).")

    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)
        if x.ndim != 1 or len(x) != len(y):
            raise ValueError("x must be 1D with the same length as y.")

    return {
        "x": x,
        "mean": np.mean(ensemble, axis=0),
        "median": np.median(ensemble, axis=0),
        "std": np.std(ensemble, axis=0, ddof=1),
        "min": np.min(ensemble, axis=0),
        "max": np.max(ensemble, axis=0),
        "q05": np.percentile(ensemble, 5, axis=0),
        "q95": np.percentile(ensemble, 95, axis=0),
    }


def plot_ensemble(
    x: Optional[np.ndarray],
    y: np.ndarray,
    ensemble: np.ndarray,
    n_show: int = 50,
    title: str = "",
    show_boxplot: bool = False,
) -> Figure:
    """
    Plot the original time series with ensemble perturbation realizations.

    Parameters
    ----------
    x : array-like or None
        Optional x-axis values (e.g. timestamps). Integer index used when None.
    y : np.ndarray
        Original input time series.
    ensemble : np.ndarray
        Ensemble array with shape (n_ensemble, n).
    n_show : int, optional
        Number of ensemble members to display (capped at n_ensemble).
    title : str, optional
        Plot title.
    show_boxplot : bool, optional
        Whether to also display a boxplot of the ensemble distribution.

    Returns
    -------
    matplotlib.figure.Figure
        The ensemble line plot figure.
    """
    y = _validate_1d_array(y, "y")
    ensemble = np.asarray(ensemble)

    if ensemble.ndim != 2:
        raise ValueError("ensemble must be a 2D array.")
    if ensemble.shape[1] != len(y):
        raise ValueError("ensemble.shape[1] must match len(y).")

    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be a 1D array.")
        if len(x) != len(y):
            raise ValueError("len(x) must match len(y).")

    n_show = min(n_show, ensemble.shape[0])

    fig, ax = plt.subplots(figsize=(16, 6))

    for i in range(n_show):
        ax.plot(x, ensemble[i], color="steelblue", alpha=0.3, linewidth=0.8)

    ax.plot(x, y, color="black", linewidth=2, label="Original")
    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.show()

    if show_boxplot:
        fig_bp, ax_bp = plt.subplots(figsize=(16, 6))
        ax_bp.boxplot(ensemble.T, showfliers=False)
        ax_bp.set_title(f"Ensemble Box Plot {title}")
        ax_bp.set_xlabel("Time Index")
        ax_bp.set_ylabel("Value")
        ax_bp.tick_params(axis="x", rotation=45)
        fig_bp.tight_layout()
        plt.show()

    return fig
