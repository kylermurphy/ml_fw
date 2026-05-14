import numpy as np
from scipy.stats import gaussian_kde
from arch.bootstrap import optimal_block_length
import matplotlib.pyplot as plt 

# Gaussian sampling method 
# Assumes samples follow a normal distribution 
def sample_gaussian(t, y, residuals, n_samples = None, plot = False, ensemble = False):

    mu, sigma = np.mean(residuals), np.std(residuals)

    if n_samples:

        n_samples = n_samples

    else: 
        
        n_samples = len(y)


    ensemble_gaussian = None
    gaussian_results = None
    fig = None


    if ensemble:

        ensemble_gaussian = []

        for i in range(20):
            
            gaussian_results = np.random.normal(mu, sigma, n_samples) 

            y_perturbed_gaussian = y + gaussian_results

            ensemble_gaussian.append(y_perturbed_gaussian)

    else: 

        gaussian_results = np.random.normal(mu, sigma, n_samples) 

    if plot:

        fig = plt.figure()

        if ensemble: 

            for i in range(20):

                plt.plot(t, ensemble_gaussian[i], alpha=0.4, color = 'red')

            plt.plot(t, y, color='blue')

            plt.title("Gaussian Ensemble")
            plt.show()

        else: 

            plt.plot(t, gaussian_results)
            plt.title("Gaussian Results")
            plt.show()
 
    return gaussian_results, ensemble_gaussian, fig


# Empirical bootstrap sampling method 
# Randomly resamples the residuals directly with replacement
def sample_empirical(t, y, residuals, n_samples=None, plot=False, ensemble=False):

    if n_samples:

        n_samples = n_samples

    else:

        n_samples = len(y)


    ensemble_empirical = None
    empirical_results = None
    fig = None


    if ensemble:

        ensemble_empirical = []

        for i in range(20):

            empirical_results = np.random.choice(residuals, size=n_samples, replace=True)

            y_perturbed_empirical = y + empirical_results

            ensemble_empirical.append(y_perturbed_empirical)

    else:

        empirical_results = np.random.choice(residuals, size=n_samples, replace=True)


    if plot:

        fig = plt.figure()

        if ensemble:

            for i in range(20):

                plt.plot(t, ensemble_empirical[i], alpha=0.4, color = 'red')

            plt.plot(t, y, color='blue')

            plt.title("Empirical Bootstrap Ensemble")
            plt.show()

        else:

            plt.plot(t, empirical_results)
            plt.title("Empirical Bootstrap Results")
            plt.show()


    return empirical_results, ensemble_empirical, fig


# KDE sampling method 
# Fits a probability density estimate to residuals 
def sample_kde(t, y, residuals, n_samples=None, plot=False, ensemble=False):

    kde = gaussian_kde(residuals)

    if n_samples:

        n_samples = n_samples

    else:

        n_samples = len(y)


    ensemble_kde = None
    kde_results = None
    fig = None


    if ensemble:

        ensemble_kde = []

        for i in range(20):

            kde_results = kde.resample(n_samples).flatten()

            y_perturbed_kde = y + kde_results

            ensemble_kde.append(y_perturbed_kde)

    else:

        kde_results = kde.resample(n_samples).flatten()


    if plot:

        fig = plt.figure()

        if ensemble:

            for i in range(20):

                plt.plot(t, ensemble_kde[i], alpha=0.4, color = 'red')

            plt.plot(t, y, color='blue')

            plt.title("KDE Ensemble")
            plt.show()

        else:

            plt.plot(t, kde_results)
            plt.title("KDE Results")
            plt.show()


    return kde_results, ensemble_kde, fig

# Moving block bootstrap sampling 
# Resamples blocks of residuals 
def sample_block(t, y, residuals, n_samples=None, block_length=None, plot=False, ensemble=False):

    residuals = np.asarray(residuals)
    n = len(residuals)

    if n_samples:

        n_samples = n_samples

    else:

        n_samples = len(y)


    if block_length is None:

        block_length = int(len(residuals) ** (1/3))


    if block_length > n:

        raise ValueError("block_length cannot be larger than number of residuals")


    ensemble_block = None
    block_results = None
    fig = None


    if ensemble:

        ensemble_block = []

        for i in range(20):

            sampled = []

            max_start = n - block_length

            starts = np.arange(0, max_start + 1)

            while len(sampled) < n_samples:

                start = np.random.choice(starts)

                block = residuals[start:start + block_length]

                sampled.extend(block)

            block_results = np.array(sampled[:n_samples])

            y_perturbed_block = y + block_results

            ensemble_block.append(y_perturbed_block)

    else:

        sampled = []

        max_start = n - block_length

        starts = np.arange(0, max_start + 1)

        while len(sampled) < n_samples:

            start = np.random.choice(starts)

            block = residuals[start:start + block_length]

            sampled.extend(block)

        block_results = np.array(sampled[:n_samples])


    if plot:

        fig = plt.figure()

        if ensemble:

            for i in range(20):

                plt.plot(t, ensemble_block[i], alpha=0.4, color = 'red')

            plt.plot(t, y, color='blue')

            plt.title("Block Bootstrap Ensemble")
            plt.show()

        else:

            plt.plot(t, block_results)
            plt.title("Block Bootstrap Results")
            plt.show()


    return block_results, ensemble_block, fig