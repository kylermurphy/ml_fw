import numpy as np

from .utils import (
    VALID_METHODS,
    _validate_1d_array,
)
from .fit import fit_model
from .diagnostics import characterize_residuals
from .sampling import _sample_residuals


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
