import numpy as np
from scipy.stats import gaussian_kde
from arch.bootstrap import optimal_block_length

# Gaussian sampling method 
# Assumes samples follow a normal distribution 
def sample_gaussian(residuals, n_samples):

    mu, sigma = np.mean(residuals), np.std(residuals)

    return np.random.normal(mu, sigma, n_samples)


# Empirical bootstrap sampling method 
# Randomly resamples the residuals directly with replacement
def sample_empirical(residuals, n_samples):

    return np.random.choice(residuals, size=n_samples, replace=True)


# KDE sampling method 
# Fits a probability density estimate to residuals 
def sample_kde(residuals, n_samples):

    kde = gaussian_kde(residuals)
    return kde.resample(n_samples).flatten()


# Moving block bootstrap sampling 
# Resamples blocks of residuals 
def sample_block(residuals, n_samples, block_length):

    residuals = np.asarray(residuals)
    n = len(residuals)
    sampled = []

    if block_length > n:
        raise ValueError("block_length cannot be larger than number of residuals")

    max_start = n - block_length

    starts = np.arange(0, max_start + 1)

    while len(sampled) < n_samples:
        start = np.random.choice(starts)
        block = residuals[start:start + block_length]
        sampled.extend(block)

    return np.array(sampled[:n_samples])