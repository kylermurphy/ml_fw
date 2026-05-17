import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm

from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import skew, kurtosis, shapiro, norm, probplot
from statsmodels.stats.diagnostic import acorr_ljungbox


def fit_model(y, title="", plot=False, block_length=None):
    """
    Fit an ARIMA model, extract residuals, characterize them,
    and optionally plot diagnostics.
    """

    y = np.asarray(y)

    model = pm.auto_arima(
        y,
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
        trace=True,
    )

    residuals = model.resid()
    p, d, q = model.order

    characterization = characterize_residuals(
        residuals,
        model_df=p + q,
        block_length=block_length,
    )

    fig = None

    if plot:
        fig = plot_residual_diagnostics(residuals, title)

    return {
        "model": model,
        "residuals": residuals,
        "order": model.order,
        "aic": model.aic(),
        "characterization": characterization,
        "figure": fig,
    }


def characterize_residuals(residuals, model_df=0, block_length=None):
    """
    Compute residual statistics and recommend a sampling method.
    """

    residuals = np.asarray(residuals)

    mean_value = np.mean(residuals)
    std_value = np.std(residuals)
    skewness = skew(residuals)
    kurtosis_value = kurtosis(residuals)

    shapiro_pvalue = shapiro(residuals).pvalue

    ljung_box_table = acorr_ljungbox(
        residuals,
        lags=[1, 5, 10],
        model_df=model_df,
    )

    ljungbox_pvalue = ljung_box_table["lb_pvalue"].iloc[-1]
    ljung_box_pass = (ljung_box_table["lb_pvalue"] > 0.05).all()

    if block_length is None:
        block_length = int(len(residuals) ** (1 / 3))

    recommended_method = recommend_sampling_method(
        shapiro_pvalue,
        ljung_box_pass,
        skewness,
        kurtosis_value,
    )

    return {
        "sample_count": int(len(residuals)),
        "mean": float(mean_value),
        "std": float(std_value),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis_value),
        "shapiro_pvalue": float(shapiro_pvalue),
        "ljungbox_pvalue": float(ljungbox_pvalue),
        "ljung_box_pass": bool(ljung_box_pass),
        "block_length": int(block_length),
        "recommended_method": recommended_method,
    }


def recommend_sampling_method(
    shapiro_pvalue,
    ljung_box_pass,
    skewness,
    kurtosis_value,
):
    """
    Recommend residual sampling method based on normality,
    autocorrelation, skewness, and kurtosis.
    """

    normal = shapiro_pvalue > 0.05
    independent = ljung_box_pass

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


def plot_residual_diagnostics(residuals, title=""):
    """
    Plot residual time series, ACF, histogram, and Q-Q plot.
    """

    residuals = np.asarray(residuals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(residuals, alpha=0.9)
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].set_title(f"Residual Time Series {title}")

    plot_acf(residuals, ax=axes[0, 1])
    axes[0, 1].set_title(f"Residual ACF {title}")

    axes[1, 0].hist(residuals, bins="auto", density=True, alpha=0.9)

    mu = np.mean(residuals)
    sigma = np.std(residuals)

    x = np.linspace(np.min(residuals), np.max(residuals), 100)
    pdf = norm.pdf(x, mu, sigma)

    axes[1, 0].plot(x, pdf, linewidth=2)
    axes[1, 0].set_title(f"Residual Histogram {title}")

    probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f"Residual Q-Q Plot {title}")

    plt.tight_layout()
    plt.show()

    return fig