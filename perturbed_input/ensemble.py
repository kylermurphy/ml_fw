import numpy as np
from typing import Any, Optional
from scipy.stats import gaussian_kde

from .utils import VALID_METHODS, _validate_1d_array
from .fit import fit_model
from .diagnostics import characterize_residuals
from .sampling import _sample_residuals


def generate_perturbations(
    y: np.ndarray,
    n_ensemble: int = 100,
    method: str = "auto",
    block_length: Optional[int] = None,
    seasonal: bool = False,
    m: int = 1,
    seed: Optional[int] = None,
    verbose: bool = False,
    fit: Optional[dict] = None,
    auto_arima_kwargs: Optional[dict[str, Any]] = None,
    kde_bandwidth: "float | str | None" = None,
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
        Residual sampling method. One of 'auto', 'gaussian', 'empirical',
        'kde', or 'block'.
    block_length : int or None, optional
        Block length for block bootstrap. Auto-estimated when None.
    seasonal : bool, optional
        Whether to allow seasonal ARIMA terms. Ignored when fit is provided.
    m : int, optional
        Seasonal period (e.g. m=24 for hourly data with a daily cycle).
        Ignored when fit is provided.
    seed : int or None, optional
        Random seed for reproducible ensemble generation.
    verbose : bool, optional
        If True, print the selected ARIMA order and sampling method.
    fit : dict or None, optional
        Pre-fitted result from fit_model(). Skips ARIMA fitting when provided,
        which avoids the cost of refitting when comparing methods or seeds.
    auto_arima_kwargs : dict or None, optional
        Additional keyword arguments for pmdarima.auto_arima().
        Ignored when fit is provided.
    kde_bandwidth : float, str, or None, optional
        Bandwidth for KDE sampling ('scott', 'silverman', or a scalar float).
        Only used when method='kde' or auto-selection chooses 'kde'.

    Returns
    -------
    np.ndarray
        Ensemble array with shape (n_ensemble, n).
    """
    y = _validate_1d_array(y, "y")

    if n_ensemble <= 0:
        raise ValueError("n_ensemble must be greater than 0.")

    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {list(VALID_METHODS)}, got {method!r}."
        )

    if block_length is not None:
        if not isinstance(block_length, int):
            raise TypeError("block_length must be an integer or None.")
        if block_length <= 0:
            raise ValueError("block_length must be greater than 0.")
        if block_length > len(y):
            raise ValueError("block_length cannot be larger than len(y).")

    rng = np.random.default_rng(seed)

    if fit is not None:
        _required = {"model", "fitted", "residuals", "order", "seasonal_order", "aic"}
        missing = _required - set(fit)
        if missing:
            raise ValueError(f"fit dict is missing required keys: {missing}.")
        if len(fit["fitted"]) != len(y):
            raise ValueError("fit['fitted'] length must match len(y).")
    else:
        fit = fit_model(y, seasonal=seasonal, m=m, auto_arima_kwargs=auto_arima_kwargs)

    residuals = fit["residuals"]

    # Only run the full diagnostic suite when the results are actually needed.
    needs_diagnostics = method == "auto" or block_length is None
    characterization = characterize_residuals(residuals) if needs_diagnostics else None

    if method == "auto":
        method = characterization["recommended_method"]

    if block_length is None:
        block_length = characterization["block_length"]

    if verbose:
        print(f"  ARIMA order      : {fit['order']}  seasonal: {fit['seasonal_order']}")
        print(f"  Sampling method  : {method}")
        print(f"  Block length     : {block_length}")

    # Fit KDE once outside the loop — gaussian_kde involves bandwidth estimation
    # and matrix factorisation, so re-fitting per member is wasteful.
    fitted_kde = gaussian_kde(residuals, bw_method=kde_bandwidth) if method == "kde" else None

    ensemble = np.empty((n_ensemble, len(y)))
    for i in range(n_ensemble):
        sampled = _sample_residuals(
            residuals=residuals,
            method=method,
            n_samples=len(y),
            block_length=block_length,
            rng=rng,
            kde=fitted_kde,
            kde_bandwidth=kde_bandwidth,
        )
        ensemble[i] = fit["fitted"] + sampled

    return ensemble
