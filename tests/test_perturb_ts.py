import numpy as np
import pytest

from statsmodels.tsa.arima_process import ArmaProcess

from perturb_ts import  (
    fit_model,
    characterize_residuals,
    generate_perturbations
)

def make_ar1 (
        n = 300,
        phi = 0.7,
        seed = 1
):
    rng = np.random.default_rng(seed)

    ar_coeffs = np.array([1,-phi])

    ma_coeffs = np.array([1])

    ar1_process = ArmaProcess(
        ar_coeffs,
        ma_coeffs
    )

    y = ar1_process.generate_sample(
        nsample=n,
        distrvs=rng.normal
    )

    return y   

def test_ar1_residual_distribution():

    y = make_ar1(n=300, phi=0.7)

    fit = fit_model(y)

    residuals = fit["residuals"]

    stats = characterize_residuals(residuals)

    assert abs(stats["mean"]) < 0.5

    assert stats["std"] > 0

    assert np.isfinite(stats["skewness"])

    assert np.isfinite(stats["kurtosis"])


@pytest.mark.parametrize(
    "method",
    [
        "gaussian",
        "empirical",
        "kde",
        "block",
    ],
)
def test_sampling_methods_shape(method):

    y = make_ar1(n=150, phi=0.7)

    n_ensemble = 10

    ensemble = generate_perturbations(
        y,
        n_ensemble=n_ensemble,
        method=method,
        seed=42,
    )

    assert ensemble.shape == (
        n_ensemble,
        len(y),
    )


def test_auto_on_white_noise():

    rng = np.random.default_rng(0)

    y = rng.normal(
        0,
        1,
        150,
    )

    n_ensemble = 5

    ensemble = generate_perturbations(
        y,
        n_ensemble=n_ensemble,
        method="auto",
        seed=42,
    )

    assert ensemble.shape == (
        n_ensemble,
        len(y),
    )


def test_auto_on_autocorrelated_residuals():

    y = make_ar1(n=150, phi=0.8)

    n_ensemble = 5

    ensemble = generate_perturbations(
        y,
        n_ensemble=n_ensemble,
        method="auto",
        seed=42,
    )

    assert ensemble.shape == (
        n_ensemble,
        len(y),
    )


def test_output_ensemble_shape():

    y = make_ar1(n=200, phi=0.7)

    n_ensemble = 20

    ensemble = generate_perturbations(
        y,
        n_ensemble=n_ensemble,
        method="gaussian",
        seed=42,
    )

    assert ensemble.shape == (
        n_ensemble,
        len(y),
    )


def test_invalid_method_produces_error():

    y = make_ar1(n=100)

    with pytest.raises(ValueError):

        generate_perturbations(
            y,
            method="jeffs_method",
        )


def test_invalid_n_ensemble_produces_error():

    y = make_ar1(n=100)

    with pytest.raises(ValueError):

        generate_perturbations(
            y,
            n_ensemble=0,
        )


def test_y_must_be_1d():

    y = np.ones((10, 2))

    with pytest.raises(ValueError):

        generate_perturbations(y)

