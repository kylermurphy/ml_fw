import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def sample_residuals(t, y, residuals, method="gaussian",
                     n_samples=None, plot=False,
                     ensemble=False, ensemble_count=20,
                     block_length=None):

    residuals = np.asarray(residuals)
    y = np.asarray(y)

    if n_samples is None:
        n_samples = len(y)

    def draw_sample():

        if method == "gaussian":

            mu, sigma = np.mean(residuals), np.std(residuals)

            return np.random.normal(mu, sigma, n_samples)

        elif method == "empirical":

            return np.random.choice(residuals, size=n_samples, replace=True)

        elif method == "kde":

            kde = gaussian_kde(residuals)
            return kde.resample(n_samples).flatten()

        elif method == "block":

            n = len(residuals)
            L = block_length

            if L is None:

                L = int(n ** (1/3))

            if L > n:

                raise ValueError("block_length cannot be larger than residuals")

            sampled = []
            starts = np.arange(0, n - L + 1)

            while len(sampled) < n_samples:

                start = np.random.choice(starts)
                sampled.extend(residuals[start:start + L])

            return np.array(sampled[:n_samples])

        else:

            raise ValueError("method must be: gaussian, empirical, kde, or block")

    fig = None
    results = None
    ensemble_results = None

    if ensemble:

        ensemble_results = []

        for i in range(ensemble_count):

            sample = draw_sample()
            y_perturbed = y + sample
            ensemble_results.append(y_perturbed)

    else:

        results = draw_sample()

    if plot:

        fig = plt.figure()

        if ensemble:

            for series in ensemble_results:
                plt.plot(t, series, alpha=0.4)

            plt.plot(t, y)
            plt.title(f"{method.capitalize()} Ensemble")

        else:
            
            plt.plot(t, results)
            plt.title(f"{method.capitalize()} Results")

        plt.show()

    return results, ensemble_results, fig