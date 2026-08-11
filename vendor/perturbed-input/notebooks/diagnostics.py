import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from scipy.stats import norm, ks_2samp

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

    stats.probplot(residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title(f"Residual Q-Q Plot of {title}")

    plt.tight_layout()
    plt.show()

  

    return fig