import numpy as np
from typing import Any

# Sorted tuple so error messages print in a consistent, readable order.
VALID_METHODS = ("auto", "block", "empirical", "gaussian", "kde")


def _validate_1d_array(array: Any, name: str) -> np.ndarray:
    """Validate and coerce input to a clean 1-D float NumPy array."""
    array = np.asarray(array, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if len(array) < 10:
        raise ValueError(f"{name} must contain at least 10 values.")
    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN values.")
    return array


def _validate_array(array: Any, name: str) -> np.ndarray:
    """
    Validate and coerce input to a clean 2-D float NumPy array of shape
    (n, n_series). A 1-D input of shape (n,) is promoted to (n, 1).
    """
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if array.shape[0] < 10:
        raise ValueError(f"{name} must contain at least 10 values.")
    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN values.")
    return array
