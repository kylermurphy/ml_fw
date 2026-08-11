import numpy as np
import pmdarima as pm
from typing import Any, Optional

from .utils import _validate_1d_array


def fit_model(
    y: np.ndarray,
    seasonal: bool = False,
    m: int = 1,
    auto_arima_kwargs: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Fit an ARIMA model to a univariate time series.

    Parameters
    ----------
    y : np.ndarray
        Input time series with shape (n,).
    seasonal : bool, optional
        Whether to allow seasonal ARIMA terms.
    m : int, optional
        Seasonal period (e.g. m=24 for hourly data with a daily cycle).
    auto_arima_kwargs : dict or None, optional
        Additional keyword arguments passed to pmdarima.auto_arima().
        Example: {"max_p": 5, "max_q": 5, "trace": True}

    Returns
    -------
    dict
        Keys: model, fitted, residuals, order, seasonal_order, aic.
    """
    y = _validate_1d_array(y, "y")

    if auto_arima_kwargs is None:
        auto_arima_kwargs = {}
    if not isinstance(auto_arima_kwargs, dict):
        raise TypeError("auto_arima_kwargs must be a dictionary or None.")

    model = pm.auto_arima(
        y,
        seasonal=seasonal,
        m=m,
        information_criterion="aic",
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
        trace=False,
        **auto_arima_kwargs,
    )

    fitted = model.predict_in_sample()
    residuals = y - fitted

    return {
        "model": model,
        "fitted": np.asarray(fitted),
        "residuals": np.asarray(residuals),
        "order": model.order,
        "seasonal_order": model.seasonal_order,
        "aic": float(model.aic()),
    }
