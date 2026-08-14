import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
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
    plot_mean: bool = False,
    plot_median: bool = False,
    show_boxplot: bool = False,
    colormap: str = "plasma",
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
) -> Axes:
    """
    Plot the original time series with ensemble perturbation realizations.

    Ensemble members are drawn in colors from a matplotlib colormap; the
    original series is highlighted in black. Optionally overlay ensemble
    mean (red dashed) and/or median (orange dash-dot) lines.

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
        Default is 50.
    plot_mean : bool, optional
        Whether to overlay ensemble mean as a red dashed line.
        Default is False.
    plot_median : bool, optional
        Whether to overlay ensemble median as an orange dash-dot line.
        Default is False.
    show_boxplot : bool, optional
        Whether to also display a boxplot of the ensemble distribution
        in a separate figure. Default is False.
    colormap : str, optional
        Matplotlib colormap name to color ensemble member lines.
        Default is "plasma".
    ax : matplotlib.axes.Axes, optional
        Existing Axes to plot on. If None, a new figure is created.
    figsize : tuple, optional
        Figure size (width, height) in inches. Only used if ax is None.
        Default is (16, 6).
    xlabel : str, optional
        X-axis label. If None, defaults to "Time Index".
    ylabel : str, optional
        Y-axis label. If None, defaults to "Value".
    legend : bool, optional
        Whether to show the legend. Default is True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the ensemble plot.
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
        # Handle pandas Series/DatetimeIndex input
        if isinstance(x, (pd.Series, pd.DatetimeIndex)):
            x = x.values if isinstance(x, pd.Series) else x.to_numpy()
        else:
            x = np.asarray(x)

        if hasattr(x, 'ndim') and x.ndim != 1:
            raise ValueError("x must be a 1D array.")
        if len(x) != len(y):
            raise ValueError("len(x) must match len(y).")

    n_show = min(n_show, ensemble.shape[0])

    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (16, 6))


    # Get colormap for ensemble members
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, n_show))

    # Plot ensemble members with colormap colors
    for i in range(n_show):
        ax.plot(x, ensemble[i], color=colors[i], alpha=0.3, linewidth=0.8)

    # Plot original series
    ax.plot(x, y, color="black", linewidth=2, label="Original")

    # Optionally plot mean and/or median
    if plot_mean or plot_median:
        if plot_mean:
            ax.plot(x, ensemble.mean(axis=0),
                color="red", linestyle="--", linewidth=1,
                label="Ensemble Mean"
            )
        if plot_median:
            ax.plot(x, np.median(ensemble, axis=0),
                color="orange", linestyle="-.", linewidth=1,
                label="Ensemble Median"
            )

    # Set labels with defaults
    ax.set_xlabel(xlabel or "Time Index")
    ax.set_ylabel(ylabel or "Value")
    ax.tick_params(axis="x", rotation=45)

    # Add legend if requested
    if legend:
        ax.legend()

    # Handle boxplot if requested
    if show_boxplot:
        _, ax_bp = plt.subplots(figsize=figsize or (16, 6))
        ax_bp.boxplot(ensemble.T, showfliers=False)
        ax_bp.set_xlabel(xlabel or "Time Index")
        ax_bp.set_ylabel(ylabel or "Value")
        ax_bp.tick_params(axis="x", rotation=45)

    # Refresh canvas in interactive environments
    try:
        ax.figure.canvas.draw()
    except (AttributeError, RuntimeError):
        # In non-interactive environments, this may not be available
        pass

    return ax
