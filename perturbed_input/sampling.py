import numpy as np

from scipy.stats import gaussian_kde


# ============================================================
# Sampling Helper
# ============================================================

def _sample_block(
    residuals: np.ndarray,
    n_samples: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
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
    residuals: np.ndarray,
    method: str,
    n_samples: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
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
