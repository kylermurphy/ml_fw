import numpy as np
from scipy.stats import gaussian_kde


def _sample_block(
    residuals: np.ndarray,
    n_samples: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorized moving block bootstrap sampling.

    Draws all block start indices at once with a single rng call, then
    builds the sample array with numpy advanced indexing — no Python loop.

    Parameters
    ----------
    residuals : np.ndarray
        Residual series.
    n_samples : int
        Number of samples to generate.
    block_length : int
        Length of each contiguous bootstrap block.
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Sampled residuals with shape (n_samples,).
    """
    n = len(residuals)
    if block_length > n:
        raise ValueError("block_length cannot be larger than the residual length.")

    starts = np.arange(0, n - block_length + 1)
    n_blocks = int(np.ceil(n_samples / block_length))
    chosen_starts = rng.choice(starts, size=n_blocks, replace=True)

    # Shape (n_blocks, block_length) → ravel → trim to n_samples
    block_indices = chosen_starts[:, None] + np.arange(block_length)[None, :]
    return residuals[block_indices.ravel()][:n_samples]


def _sample_gaussian(residuals, n_samples, block_length, rng, kde, kde_bandwidth):
    mu = np.mean(residuals)
    sigma = np.std(residuals, ddof=1)
    return rng.normal(mu, sigma, n_samples)


def _sample_empirical(residuals, n_samples, block_length, rng, kde, kde_bandwidth):
    return rng.choice(residuals, size=n_samples, replace=True)


def _sample_kde_method(residuals, n_samples, block_length, rng, kde, kde_bandwidth):
    if kde is None:
        kde = gaussian_kde(residuals, bw_method=kde_bandwidth)
    return kde.resample(n_samples, seed=rng).flatten()


def _sample_block_method(residuals, n_samples, block_length, rng, kde, kde_bandwidth):
    return _sample_block(residuals, n_samples, block_length, rng)


_SAMPLERS = {
    "gaussian": _sample_gaussian,
    "empirical": _sample_empirical,
    "kde": _sample_kde_method,
    "block": _sample_block_method,
}


def _sample_residuals(
    residuals: np.ndarray,
    method: str,
    n_samples: int,
    block_length: int,
    rng: np.random.Generator,
    kde: "gaussian_kde | None" = None,
    kde_bandwidth: "float | str | None" = None,
) -> np.ndarray:
    """
    Sample residuals using the specified method.

    Parameters
    ----------
    residuals : np.ndarray
        Residual series.
    method : str
        One of 'gaussian', 'empirical', 'kde', 'block'.
    n_samples : int
        Number of samples to generate.
    block_length : int
        Block length for block bootstrap (ignored by other methods).
    rng : np.random.Generator
        NumPy random number generator.
    kde : gaussian_kde or None, optional
        Pre-fitted KDE object. Fitted from residuals on first call if None.
    kde_bandwidth : float, str, or None, optional
        Bandwidth passed to gaussian_kde when kde is None.

    Returns
    -------
    np.ndarray
        Sampled residuals with shape (n_samples,).
    """
    if method not in _SAMPLERS:
        raise ValueError(f"Unknown sampling method: {method!r}.")
    return _SAMPLERS[method](residuals, n_samples, block_length, rng, kde, kde_bandwidth)
