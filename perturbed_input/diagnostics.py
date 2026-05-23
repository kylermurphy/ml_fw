import numpy as np

from scipy.stats import (
    skew,
    kurtosis,
    shapiro,
)

from statsmodels.stats.diagnostic import acorr_ljungbox

from .utils import _validate_1d_array


# ============================================================
# Residual Sampling Method Recommendation
# ============================================================

def _recommend_method(
    shapiro_pvalue: float,
    ljungbox_pvalue: float,
    skewness: float,
    kurtosis_value: float,
) -> str:
    """
    Recommend an appropriate residual sampling method
    based on residual diagnostic statistics.

    Parameters
    ----------
    shapiro_pvalue : float
        Shapiro-Wilk normality test p-value.

    ljungbox_pvalue : float
        Ljung-Box autocorrelation test p-value.

    skewness : float
        Residual skewness value.

    kurtosis_value : float
        Residual kurtosis value.

    Returns
    -------
    str
        Recommended sampling method.
    """

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


# ============================================================
# Residual Characterization
# ============================================================

def characterize_residuals(
    residuals: np.ndarray,
) -> dict:
    """
    Compute residual statistics and diagnostic tests.

    Parameters
    ----------
    residuals : np.ndarray
        Residual series from fitted model.

    Returns
    -------
    dict
        Dictionary containing:

        - mean
        - standard deviation
        - skewness
        - kurtosis
        - Shapiro-Wilk p-value
        - Ljung-Box p-value
        - recommended sampling method
        - estimated block length
    """

    residuals = _validate_1d_array(
        residuals,
        "residuals",
    )

    mean_value = np.mean(residuals)

    std_value = np.std(residuals)

    skewness = skew(residuals)

    kurtosis_value = kurtosis(residuals)

    shapiro_pvalue = shapiro(
        residuals
    ).pvalue

    ljung_box_table = acorr_ljungbox(
        residuals,
        lags=[5,10,20],
        return_df=True,
    )

    ljungbox_pvalue = float(
        ljung_box_table["lb_pvalue"].min()
    )

    block_length = int(len(residuals) ** (1 / 3))
      

    recommended_method = _recommend_method(
        shapiro_pvalue=shapiro_pvalue,
        ljungbox_pvalue=ljungbox_pvalue,
        skewness=skewness,
        kurtosis_value=kurtosis_value,
    )

    return {
        "mean": float(mean_value),
        "std": float(std_value),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis_value),
        "shapiro_pvalue": float(shapiro_pvalue),
        "ljungbox_pvalue": float(ljungbox_pvalue),
        "recommended_method": recommended_method,
        "block_length": int(block_length),
    }
