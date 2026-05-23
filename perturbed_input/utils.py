import numpy as np

from typing import Any


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

def _validate_1d_array(
    array: Any,
    name: str,
) -> np.ndarray:
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
