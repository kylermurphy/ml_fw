import numpy as np
import pmdarima as pm
from typing import Any, Optional

from .utils import _validate_1d_array


def _replace_burn_in(
    fit: np.ndarray, residuals: np.ndarray, order: tuple, seasonal_order: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace leading burn-in residuals and fit values. Residuals are replaced
    with the mean of the remaining residuals and fit values are replaced with
    nan.

    ARIMA differencing (d, and seasonal D at period m) consumes the first
    few observations before the model can produce a real in-sample
    prediction, so `fitted` is zero there and the corresponding residuals
    (`y - fitted`) are inflated rather than genuine model error.

    Parameters
    ----------
    fit : np.ndarray
        Fitted values from the ARIMA model.
    residuals : np.ndarray
        Raw residuals (y - fitted).
    order : tuple
        (p, d, q) from the fitted ARIMA model.
    seasonal_order : tuple
        (P, D, Q, m) from the fitted ARIMA model.

    Returns
    -------
    np.ndarray
        Fitted values with the first `d + D * m` values replaced by np.nan. 
        Unchanged if that count is zero.
    np.ndarray
        Residuals with the first `d + D * m` values replaced by the mean of
        the rest. Unchanged if that count is zero.

    Raises
    ------
    ValueError
        If the burn-in length is greater than or equal to the number of
        residuals (no genuine residuals remain to compute the mean from).
    """
    d = order[1]
    seasonal_d, m = seasonal_order[1], seasonal_order[3]
    n_burn = d + seasonal_d * m

    if n_burn == 0:
        return fit, residuals
    if n_burn >= len(residuals):
        raise ValueError(
            f"Burn-in length ({n_burn}) derived from order={order}, "
            f"seasonal_order={seasonal_order} is >= the number of residuals "
            f"({len(residuals)}); cannot compute a replacement mean."
        )

    residuals = residuals.copy()
    residuals[:n_burn] = np.mean(residuals[n_burn:])
    fit[:n_burn] = np.nan
    return fit, residuals


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

    Notes
    -----
    The first `d + D*m` residuals (where d and D are the differencing orders)
    are replaced with the mean of the remaining residuals. This is because ARIMA
    differencing consumes those initial observations, so the fitted values are
    zero there and the corresponding residuals are inflated outliers rather than
    genuine model error. This replacement ensures residual diagnostics and
    sampling are not skewed by these artificial burn-in points.

    Raises ValueError if the burn-in length is >= len(y), i.e., when the
    differencing order would consume the entire series.
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
    residuals = np.asarray(y - fitted)
    fitted, residuals = _replace_burn_in(fitted, residuals, model.order, model.seasonal_order)

    return {
        "model": model,
        "fitted": np.asarray(fitted),
        "residuals": residuals,
        "order": model.order,
        "seasonal_order": model.seasonal_order,
        "aic": float(model.aic()),
    }
