import numpy as np 
import matplotlib.pyplot as plt 
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import skew, kurtosis, shapiro, norm, probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.bootstrap import optimal_block_length

def fit_arima_and_characterize(y, title , plot = False, block_length = None):

    model = pm.auto_arima(y, seasonal=False, information_criterion='aic',
                       stepwise=True, error_action='ignore',
                       suppress_warnings=True, trace=True)
    
    residuals = model.resid() 

    p, d, q = model.order

    characterization = characterize_residuals(residuals, p, q, block_length = block_length)

    panel = None

    if plot:   
        panel = diagnostic_plot(residuals, title)
    

    return model, residuals, characterization, panel

def characterize_residuals(residuals, p, q, block_length = None):

    mean_result = np.mean(residuals)

    sd_result = np.std(residuals)

    kurtosis_result = kurtosis(residuals)

    skew_result = skew(residuals)

    shapiro_wilk_result = shapiro(residuals).pvalue

    ljung_box_table = acorr_ljungbox(residuals, lags = [1,5,10], model_df = p + q) # ** check with kyle 

    ljung_box_result = (ljung_box_table["lb_pvalue"] > 0.05).all()

# Shapiro Wilk tests if the data are normally distributed
# If it passes, the data is normal, and therefore it supports using a gaussian mode
# Ljung Box tests if the data has autocorrelation
# If it passes, the residuals look independant (ordinary residual resampling is likely ok)
# Low skew means that the results are roughly symmetric (near gaussian)
# Low kurtosis means low tail heaviness (outliers), not many deviations from the mean
# https://www.simplilearn.com/tutorials/statistics-tutorial/skewness-and-kurtosis?

    if shapiro_wilk_result > 0.05 and ljung_box_result:

        if abs(kurtosis_result) < 1 and abs(skew_result) < 0.5:

            recommended_method = "Gaussian Parametric is recommended"

        elif 1 < abs(kurtosis_result) < 3 and 0.5 < abs(skew_result) < 1:
             
             recommended_method = "Gaussian or Empirical Bootstrap are recommended"

        else:

            recommended_method = "Empirical Bootstrap is recommended"     

    elif shapiro_wilk_result < 0.05 and ljung_box_result:

        if abs(kurtosis_result) < 1 and abs(skew_result) < 0.5:

            recommended_method = "Empirical Bootstrap is recommended"

        elif 1 < abs(kurtosis_result) < 3 and 0.5 < abs(skew_result) < 1:
             
             recommended_method = "Empirical Bootstrap is recommended"

        else:

            recommended_method = "Empirical Bootstrap or KDE are recommended"

    elif shapiro_wilk_result > 0.05 and not ljung_box_result:

        recommended_method = "Block Bootstrap is recommended" 

    elif shapiro_wilk_result < 0.05 and not ljung_box_result: 

        recommended_method = "Block Bootstrap is recommended"  


    if block_length is None: 

        l = int(len(residuals) ** (1/3))

    else:

        l = block_length
    
    characterization = {
    "Sample count": float(len(residuals)),
    "mean": float(mean_result),
    "std": float(sd_result),
    "skew": float(skew_result),
    "kurtosis": float(kurtosis_result),
    "shapiro_pvalue": float(shapiro_wilk_result),
    "ljung_box_pass": bool(ljung_box_result),
    "optimal_block_length": int(l),
    "recommended_method": recommended_method
}
    
    return characterization

def diagnostic_plot(residuals, title):

    fig, axes = plt.subplots(2,2,figsize=(12,8))
    
    axes[0,0].plot(residuals, alpha = 0.9)
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].set_title(f"Residual Time Series of {title}")

    plot_acf(residuals, ax=axes[0,1])
    axes[0,1].set_title(f"Residual ACF of {title}")

    axes[1,0].hist(residuals, bins= 'auto', density = True, alpha = 0.9)

    mu = np.mean(residuals)
    sigma = np.std(residuals)

    x = np.linspace(min(residuals), max(residuals), 100)
    pdf = norm.pdf(x, mu, sigma)

    axes[1, 0].plot(x, pdf, 'r-', linewidth=2)
    axes[1, 0].set_title(f"Residual Histogram of {title}")

    probplot(residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title(f"Residual Q-Q Plot of {title}")

    plt.tight_layout()
    plt.show()

    return fig
