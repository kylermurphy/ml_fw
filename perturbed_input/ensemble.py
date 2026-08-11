import numpy as np
from typing import Any, Optional
from scipy.stats import gaussian_kde

from .utils import VALID_METHODS, _validate_array
from .fit import fit_model
from .diagnostics import characterize_residuals
from .sampling import _sample_residuals, _draw_block_indices


def generate_perturbations(
    y: np.ndarray,
    n_ensemble: int = 100,
    method: str = "auto",
    block_length: Optional[int] = None,
    seasonal: bool = False,
    m: int = 1,
    seed: Optional[int] = None,
    verbose: bool = False,
    fit: "Optional[dict | list[dict]]" = None,
    auto_arima_kwargs: Optional[dict[str, Any]] = None,
    kde_bandwidth: "float | str | None" = None,
    fit_ts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate an ensemble of perturbed time series.

    Parameters
    ----------
    y : np.ndarray
        Input time series. Either 1D with shape (n,) for a single series, or
        2D with shape (npts_y, n_series) to perturb several series at once.
        Each series is fit and diagnosed independently.
    n_ensemble : int, optional
        Number of perturbed realizations to generate.
    method : str, optional
        Residual sampling method. One of 'auto', 'gaussian', 'empirical',
        'kde', or 'block'. With 'auto' (or when unresolved per series), the
        method is chosen independently for each series from its own
        residual diagnostics.
    block_length : int or None, optional
        Block length for block bootstrap. Auto-estimated per series when
        None.
    seasonal : bool, optional
        Whether to allow seasonal ARIMA terms. Ignored when fit is provided.
    m : int, optional
        Seasonal period (e.g. m=24 for hourly data with a daily cycle).
        Ignored when fit is provided.
    seed : int or None, optional
        Random seed for reproducible ensemble generation.
    verbose : bool, optional
        If True, print the selected ARIMA order and sampling method for
        each series.
    fit : dict, list of dict, or None, optional
        Pre-fitted result(s) from fit_model(). Skips ARIMA fitting when
        provided, which avoids the cost of refitting when comparing methods
        or seeds. For multi-series input, pass one dict per series (in
        column order); a single dict is only accepted when y has one
        series. Cannot be used together with fit_ts.
    auto_arima_kwargs : dict or None, optional
        Additional keyword arguments for pmdarima.auto_arima().
        Ignored when fit is provided.
    kde_bandwidth : float, str, or None, optional
        Bandwidth for KDE sampling ('scott', 'silverman', or a scalar float).
        Only used when method='kde' or auto-selection chooses 'kde'.
    fit_ts : array-like or None, optional
        Reference time series used to derive the residual noise model, with
        the same shape convention as y: 1D (n,) or 2D (npts_fit, n_series).
        When provided, ARIMA is fitted to fit_ts (per series) and its
        residuals are sampled and added to y. Useful when the noise
        structure should come from a separate (e.g. longer or historical)
        dataset. fit_ts may have a different length than y, but must have
        the same number of series. Cannot be used together with fit.

    Returns
    -------
    np.ndarray
        Ensemble array. Shape (n_ensemble, n) when y is 1D, or
        (n_ensemble, npts_y, n_series) when y is 2D. Each member is
        y + sampled_residuals.

    Notes
    -----
    When a series uses method='block' (explicitly or via 'auto'), the block
    start positions for a given ensemble member are shared across every
    series using 'block', so co-perturbed series stay time-aligned. Series
    remain independent across ensemble members.
    """
    was_1d = np.asarray(y).ndim == 1
    y = _validate_array(y, "y")
    npts_y, n_series = y.shape

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
        if block_length > npts_y:
            raise ValueError("block_length cannot be larger than len(y).")

    if fit_ts is not None and fit is not None:
        raise ValueError("fit and fit_ts cannot both be provided.")

    if fit_ts is not None:
        fit_ts = _validate_array(fit_ts, "fit_ts")
        if fit_ts.shape[1] != n_series:
            raise ValueError("fit_ts must have the same number of series as y.")

    fit_list = None
    if fit is not None:
        fit_list = [fit] if isinstance(fit, dict) else list(fit)
        if len(fit_list) != n_series:
            raise ValueError(
                "fit must supply one fit dict per series in y "
                f"({n_series} required, got {len(fit_list)})."
            )

    rng = np.random.default_rng(seed)

    # Fit, diagnose, and select a sampling method/block length per series.
    # This part is unavoidably a Python loop: pmdarima, the diagnostic
    # tests, and gaussian_kde all operate on a single 1D series.
    series_info = []
    for j in range(n_series):
        y_j = y[:, j]

        if fit_ts is not None:
            _source_fit = fit_model(
                fit_ts[:, j], seasonal=seasonal, m=m, auto_arima_kwargs=auto_arima_kwargs
            )
        elif fit_list is not None:
            fit_j = fit_list[j]
            _required = {"model", "fitted", "residuals", "order", "seasonal_order", "aic"}
            missing = _required - set(fit_j)
            if missing:
                raise ValueError(f"fit dict is missing required keys: {missing}.")
            if len(fit_j["fitted"]) != npts_y:
                raise ValueError("fit['fitted'] length must match len(y).")
            _source_fit = fit_j
        else:
            _source_fit = fit_model(y_j, seasonal=seasonal, m=m, auto_arima_kwargs=auto_arima_kwargs)

        residuals = _source_fit["residuals"]

        needs_diagnostics = method == "auto" or block_length is None
        characterization = characterize_residuals(residuals) if needs_diagnostics else None

        series_method = characterization["recommended_method"] if method == "auto" else method
        series_block_length = block_length if block_length is not None else characterization["block_length"]

        if verbose:
            noise_source = "fit_ts" if fit_ts is not None else "y"
            print(f"Series {j}:")
            print(f"  Noise source     : {noise_source}")
            print(f"  ARIMA order      : {_source_fit['order']}  seasonal: {_source_fit['seasonal_order']}")
            print(f"  Sampling method  : {series_method}")
            print(f"  Block length     : {series_block_length}")

        # Fit KDE once per series — gaussian_kde involves bandwidth
        # estimation and matrix factorisation, so re-fitting per ensemble
        # member would be wasteful.
        fitted_kde = gaussian_kde(residuals, bw_method=kde_bandwidth) if series_method == "kde" else None

        series_info.append((y_j, residuals, series_method, series_block_length, fitted_kde))

    # Generate the ensemble, vectorized across n_ensemble. Series sharing
    # method='block' also share their block start indices per member.
    ensemble = np.empty((n_ensemble, npts_y, n_series))
    block_indices_cache: dict[int, np.ndarray] = {}

    for j, (y_j, residuals, series_method, series_block_length, fitted_kde) in enumerate(series_info):
        if series_method == "block":
            block_matrix = block_indices_cache.setdefault(
                series_block_length,
                _draw_block_indices(len(residuals), npts_y, series_block_length, rng, n_ensemble),
            )
            sampled = residuals[block_matrix]
        else:
            sampled = _sample_residuals(
                residuals=residuals,
                method=series_method,
                n_samples=n_ensemble * npts_y,
                block_length=series_block_length,
                rng=rng,
                kde=fitted_kde,
                kde_bandwidth=kde_bandwidth,
            ).reshape(n_ensemble, npts_y)

        ensemble[:, :, j] = y_j + sampled

    return ensemble[..., 0] if was_1d else ensemble
