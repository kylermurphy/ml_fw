import warnings
import numpy as np
from scipy.stats import skew, kurtosis, shapiro, normaltest
from statsmodels.stats.diagnostic import acorr_ljungbox

from .utils import _validate_1d_array

_SHAPIRO_MAX_N = 5000


def _recommend_method(
    shapiro_pvalue: float,
    ljungbox_pvalue: float,
    skewness: float,
    kurtosis_value: float,
) -> str:
    """Recommend a residual sampling method based on diagnostic statistics."""
    normal = shapiro_pvalue > 0.05
    independent = ljungbox_pvalue > 0.05
    low_skew = abs(skewness) < 0.5
    low_kurtosis = abs(kurtosis_value) < 1
    high_skew = abs(skewness) >= 1
    high_kurtosis = abs(kurtosis_value) >= 3

    if not independent:
        return "block"
    if normal and low_skew and low_kurtosis:
        return "gaussian"
    if high_skew or high_kurtosis:
        return "kde"
    return "empirical"


def characterize_residuals(residuals: np.ndarray) -> dict:
    """
    Compute residual statistics and diagnostic tests.

    Parameters
    ----------
    residuals : np.ndarray
        Residual series from a fitted model.

    Returns
    -------
    dict
        Keys: mean, std, skewness, kurtosis, shapiro_pvalue, ljungbox_pvalue,
        recommended_method, block_length.

    Notes
    -----
    For series longer than 5000 samples, the D'Agostino-Pearson test replaces
    Shapiro-Wilk to avoid spurious normality rejection caused by excess
    statistical power at large n.
    """
    residuals = _validate_1d_array(residuals, "residuals")

    n = len(residuals)
    mean_value = float(np.mean(residuals))
    std_value = float(np.std(residuals, ddof=1))
    skewness = float(skew(residuals))
    kurtosis_value = float(kurtosis(residuals))

    if n > _SHAPIRO_MAX_N:
        warnings.warn(
            f"Residual length ({n}) exceeds {_SHAPIRO_MAX_N}. "
            "Switching from Shapiro-Wilk to D'Agostino-Pearson normality test "
            "to avoid spurious rejection for large samples.",
            UserWarning,
            stacklevel=2,
        )
        shapiro_pvalue = float(normaltest(residuals).pvalue)
    else:
        shapiro_pvalue = float(shapiro(residuals).pvalue)

    ljung_box_table = acorr_ljungbox(residuals, lags=[5, 10, 20], return_df=True)
    ljungbox_pvalue = float(ljung_box_table["lb_pvalue"].min())

    block_length = max(1, int(n ** (1 / 3)))

    recommended_method = _recommend_method(
        shapiro_pvalue=shapiro_pvalue,
        ljungbox_pvalue=ljungbox_pvalue,
        skewness=skewness,
        kurtosis_value=kurtosis_value,
    )

    return {
        "mean": mean_value,
        "std": std_value,
        "skewness": skewness,
        "kurtosis": kurtosis_value,
        "shapiro_pvalue": shapiro_pvalue,
        "ljungbox_pvalue": ljungbox_pvalue,
        "recommended_method": recommended_method,
        "block_length": block_length,
    }
