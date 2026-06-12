import numpy as np
import pytest
from statsmodels.tsa.arima_process import ArmaProcess

from perturbed_input import fit_model, characterize_residuals, generate_perturbations
from perturbed_input.sampling import _sample_block


def make_ar1(n=300, phi=0.7, seed=1):
    """Generate an AR(1) process for testing."""
    rng = np.random.default_rng(seed)
    ar1 = ArmaProcess(np.array([1, -phi]), np.array([1]))
    return ar1.generate_sample(nsample=n, distrvs=rng.normal)


# ---------------------------------------------------------------------------
# fit_model
# ---------------------------------------------------------------------------

def test_fit_model_output_structure():
    y = make_ar1(n=150)
    fit = fit_model(y)
    for key in ("model", "fitted", "residuals", "order", "seasonal_order", "aic"):
        assert key in fit
    assert fit["fitted"].shape == y.shape
    assert fit["residuals"].shape == y.shape
    assert isinstance(fit["aic"], float)


def test_fit_model_residuals_sum_to_near_zero():
    """ARIMA residuals should be approximately mean-zero."""
    y = make_ar1(n=300, phi=0.5)
    fit = fit_model(y)
    assert abs(np.mean(fit["residuals"])) < 0.5


# ---------------------------------------------------------------------------
# characterize_residuals
# ---------------------------------------------------------------------------

def test_characterize_residuals_output_structure():
    y = make_ar1(n=150)
    stats = characterize_residuals(fit_model(y)["residuals"])
    for key in ("mean", "std", "skewness", "kurtosis", "shapiro_pvalue",
                "ljungbox_pvalue", "recommended_method", "block_length"):
        assert key in stats
    assert stats["recommended_method"] in ("gaussian", "empirical", "kde", "block")
    assert stats["block_length"] >= 1


def test_ar1_residual_distribution():
    y = make_ar1(n=300, phi=0.7)
    stats = characterize_residuals(fit_model(y)["residuals"])
    assert abs(stats["mean"]) < 0.5
    assert stats["std"] > 0
    assert np.isfinite(stats["skewness"])
    assert np.isfinite(stats["kurtosis"])


def test_recommend_block_for_autocorrelated_residuals():
    """A random walk (strongly autocorrelated) should trigger block recommendation."""
    rng = np.random.default_rng(0)
    residuals = np.cumsum(rng.normal(0, 0.1, 300))
    assert characterize_residuals(residuals)["recommended_method"] == "block"


# ---------------------------------------------------------------------------
# generate_perturbations — shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["gaussian", "empirical", "kde", "block"])
def test_sampling_methods_shape(method):
    y = make_ar1(n=150, phi=0.7)
    ensemble = generate_perturbations(y, n_ensemble=10, method=method, seed=42)
    assert ensemble.shape == (10, len(y))


def test_auto_on_white_noise():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 150)
    ensemble = generate_perturbations(y, n_ensemble=5, method="auto", seed=42)
    assert ensemble.shape == (5, len(y))


def test_auto_on_autocorrelated_series():
    y = make_ar1(n=150, phi=0.8)
    ensemble = generate_perturbations(y, n_ensemble=5, method="auto", seed=42)
    assert ensemble.shape == (5, len(y))


def test_output_ensemble_shape():
    y = make_ar1(n=200, phi=0.7)
    ensemble = generate_perturbations(y, n_ensemble=20, method="gaussian", seed=42)
    assert ensemble.shape == (20, len(y))


# ---------------------------------------------------------------------------
# generate_perturbations — reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_ensemble():
    y = make_ar1(n=150)
    e1 = generate_perturbations(y, n_ensemble=5, seed=42)
    e2 = generate_perturbations(y, n_ensemble=5, seed=42)
    np.testing.assert_array_equal(e1, e2)


def test_different_seeds_produce_different_ensembles():
    y = make_ar1(n=150)
    e1 = generate_perturbations(y, n_ensemble=5, seed=42)
    e2 = generate_perturbations(y, n_ensemble=5, seed=99)
    assert not np.allclose(e1, e2)


# ---------------------------------------------------------------------------
# generate_perturbations — pre-fitted model
# ---------------------------------------------------------------------------

def test_prefitted_model_matches_auto_fit():
    """Passing a pre-fitted dict must give the same ensemble as auto-fitting."""
    y = make_ar1(n=150)
    fit = fit_model(y)
    e_prefitted = generate_perturbations(y, n_ensemble=5, seed=42, fit=fit)
    e_autofitted = generate_perturbations(y, n_ensemble=5, seed=42)
    np.testing.assert_array_equal(e_prefitted, e_autofitted)


def test_prefitted_model_with_wrong_length_raises():
    y = make_ar1(n=150)
    fit = fit_model(y)
    y_different = make_ar1(n=200)
    with pytest.raises(ValueError, match="length must match"):
        generate_perturbations(y_different, n_ensemble=5, fit=fit)


# ---------------------------------------------------------------------------
# generate_perturbations — fit_ts
# ---------------------------------------------------------------------------

def test_fit_ts_different_length_produces_correct_shape():
    """fit_ts may be a different length than y; output shape is (n_ensemble, len(y))."""
    y = make_ar1(n=100, phi=0.5)
    fit_ts = make_ar1(n=300, phi=0.5)
    ensemble = generate_perturbations(y, n_ensemble=10, fit_ts=fit_ts, seed=42)
    assert ensemble.shape == (10, len(y))


def test_fit_ts_and_fit_together_raises():
    y = make_ar1(n=100)
    with pytest.raises(ValueError, match="cannot both be provided"):
        generate_perturbations(y, fit=fit_model(y), fit_ts=make_ar1(n=200))


# ---------------------------------------------------------------------------
# generate_perturbations — value-level
# ---------------------------------------------------------------------------

def test_ensemble_grand_mean_close_to_original():
    """The ensemble grand mean should be within one std of the original mean."""
    y = make_ar1(n=200, phi=0.5)
    ensemble = generate_perturbations(y, n_ensemble=200, method="gaussian", seed=42)
    assert abs(np.mean(ensemble) - np.mean(y)) < np.std(y)


# ---------------------------------------------------------------------------
# generate_perturbations — error handling
# ---------------------------------------------------------------------------

def test_invalid_method_raises():
    with pytest.raises(ValueError):
        generate_perturbations(make_ar1(n=100), method="jeffs_method")


def test_invalid_n_ensemble_raises():
    with pytest.raises(ValueError):
        generate_perturbations(make_ar1(n=100), n_ensemble=0)


def test_non_1d_input_raises():
    with pytest.raises(ValueError):
        generate_perturbations(np.ones((10, 2)))


# ---------------------------------------------------------------------------
# block bootstrap — statistical property
# ---------------------------------------------------------------------------

def test_block_bootstrap_preserves_autocorrelation():
    """
    Block bootstrap should preserve lag-1 autocorrelation in strongly
    autocorrelated residuals better than iid resampling.
    """
    rng = np.random.default_rng(42)
    n = 300
    residuals = np.zeros(n)
    for i in range(1, n):
        residuals[i] = 0.95 * residuals[i - 1] + rng.normal()

    rng_b = np.random.default_rng(0)
    rng_e = np.random.default_rng(0)

    ac_block, ac_emp = [], []
    for _ in range(50):
        s_b = _sample_block(residuals, n, block_length=20, rng=rng_b)
        ac_block.append(np.corrcoef(s_b[:-1], s_b[1:])[0, 1])
        s_e = rng_e.choice(residuals, size=n, replace=True)
        ac_emp.append(np.corrcoef(s_e[:-1], s_e[1:])[0, 1])

    assert np.mean(ac_block) > np.mean(ac_emp)
