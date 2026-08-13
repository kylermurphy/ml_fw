import numpy as np
import pytest
from statsmodels.tsa.arima_process import ArmaProcess

from ml_fw.perturbed_input import fit_model, characterize_residuals, generate_perturbations
from ml_fw.perturbed_input.fit import _replace_burn_in
from ml_fw.perturbed_input.sampling import _sample_block


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
# Burn-in residual correction
# ---------------------------------------------------------------------------

def test_replace_burn_in_no_burn_in():
    """When n_burn=0, fitted and residuals should be unchanged."""
    fitted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    order = (1, 0, 0)  # d=0
    seasonal_order = (0, 0, 0, 0)  # D=0, m=0

    fitted_out, residuals_out = _replace_burn_in(fitted, residuals, order, seasonal_order)

    np.testing.assert_array_equal(fitted_out, fitted)
    np.testing.assert_array_equal(residuals_out, residuals)


def test_replace_burn_in_with_burn_in():
    """When n_burn > 0, the first n_burn values should be replaced."""
    fitted = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    residuals = np.array([10.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    order = (1, 1, 0)  # d=1, so n_burn=1
    seasonal_order = (0, 0, 0, 0)

    fitted_out, residuals_out = _replace_burn_in(fitted, residuals, order, seasonal_order)

    # First 1 fitted value should be replaced with nan
    assert np.isnan(fitted_out[0])
    # Rest of fitted should be unchanged
    np.testing.assert_array_equal(fitted_out[1:], fitted[1:])

    # First 1 residual value should be replaced with mean of [0.5,1,2,3,4]
    expected_mean = np.mean(residuals[1:])
    assert residuals_out[0] == expected_mean
    # Rest of residuals should be unchanged
    np.testing.assert_array_equal(residuals_out[1:], residuals[1:])


def test_replace_burn_in_seasonal():
    """Test with seasonal differencing: n_burn = d + D*m."""
    fitted = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    residuals = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0])
    order = (1, 1, 0)  # d=1
    seasonal_order = (0, 1, 0, 2)  # D=1, m=2, so n_burn=1+1*2=3

    fitted_out, residuals_out = _replace_burn_in(fitted, residuals, order, seasonal_order)

    # First 3 fitted values should be nan
    np.testing.assert_array_equal(np.isnan(fitted_out[:3]), [True, True, True])
    # Rest of fitted should be unchanged
    np.testing.assert_array_equal(fitted_out[3:], fitted[3:])

    # First 3 residuals should be replaced with mean
    expected_mean = np.mean(residuals[3:])
    np.testing.assert_array_almost_equal(residuals_out[:3], expected_mean)
    # Rest of residuals should be unchanged
    np.testing.assert_array_equal(residuals_out[3:], residuals[3:])


def test_replace_burn_in_too_large_raises():
    """When burn-in length >= len(residuals), should raise ValueError."""
    fitted = np.array([0.0, 0.0, 0.0, 0.0])
    residuals = np.array([1.0, 2.0, 3.0, 4.0])
    order = (1, 3, 0)  # d=3
    seasonal_order = (0, 0, 0, 0)  # n_burn = 3, but len=4

    with pytest.raises(ValueError, match="Burn-in length"):
        _replace_burn_in(fitted, residuals, order, seasonal_order)


def test_replace_burn_in_equal_length_raises():
    """When burn-in length == len(residuals), should raise ValueError."""
    fitted = np.array([0.0, 0.0, 0.0])
    residuals = np.array([1.0, 2.0, 3.0])
    order = (1, 3, 0)  # d=3, so n_burn=3
    seasonal_order = (0, 0, 0, 0)

    with pytest.raises(ValueError, match="Burn-in length"):
        _replace_burn_in(fitted, residuals, order, seasonal_order)


def test_fit_model_applies_burn_in_correction():
    """Integration test: fit_model should apply burn-in correction automatically."""
    # Create a non-stationary (differenced) series so auto_arima selects d=1
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0, 1, 200)
    y = np.cumsum(white_noise)  # Integrated white noise

    fit = fit_model(y)
    order = fit["order"]

    # If d >= 1, the first d fitted values should be nan and residuals replaced
    if order[1] > 0:
        n_burn = order[1]
        # The first n_burn fitted values should be nan
        assert np.all(np.isnan(fit["fitted"][:n_burn]))
        # The first n_burn residuals should all equal the mean of residuals[n_burn:]
        expected_mean = np.mean(fit["residuals"][n_burn:])
        np.testing.assert_array_almost_equal(
            fit["residuals"][:n_burn], expected_mean
        )


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


def test_invalid_ndim_raises():
    with pytest.raises(ValueError):
        generate_perturbations(np.ones((10, 2, 3)))


def test_mismatched_fit_ts_series_count_raises():
    y = make_ar1(n=100, phi=0.5)
    y2 = np.column_stack([y, make_ar1(n=100, phi=0.3, seed=2)])
    with pytest.raises(ValueError, match="same number of series"):
        generate_perturbations(y2, fit_ts=make_ar1(n=200))


# ---------------------------------------------------------------------------
# generate_perturbations — multi-series
# ---------------------------------------------------------------------------

def test_2d_input_produces_3d_ensemble():
    y = np.column_stack([make_ar1(n=150, phi=0.7, seed=1), make_ar1(n=150, phi=0.2, seed=2)])
    ensemble = generate_perturbations(y, n_ensemble=10, method="gaussian", seed=42)
    assert ensemble.shape == (10, y.shape[0], 2)


def test_2d_input_matches_1d_per_series():
    """Perturbing a series inside a 2D call must match perturbing it alone."""
    y1 = make_ar1(n=150, phi=0.7, seed=1)
    y2 = make_ar1(n=150, phi=0.2, seed=2)
    y = np.column_stack([y1, y2])
    ensemble_2d = generate_perturbations(y, n_ensemble=5, method="gaussian", seed=42)
    ensemble_1d = generate_perturbations(y1, n_ensemble=5, method="gaussian", seed=42)
    np.testing.assert_array_equal(ensemble_2d[:, :, 0], ensemble_1d)


def test_block_sampling_shared_across_series():
    """Series sampled with method='block' should reuse the same block start
    positions per ensemble member, so identical residuals produce identical
    sampled noise."""
    y1 = make_ar1(n=150, phi=0.7, seed=1)
    y = np.column_stack([y1, y1])  # two identical series
    ensemble = generate_perturbations(y, n_ensemble=5, method="block", seed=42)
    np.testing.assert_array_equal(ensemble[:, :, 0], ensemble[:, :, 1])


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


# ---------------------------------------------------------------------------
# return_fitted option
# ---------------------------------------------------------------------------


def test_return_fitted_false_by_default():
    """Without return_fitted=True, the return value should be a bare ndarray."""
    y = make_ar1(n=100, seed=42)
    result = generate_perturbations(y, n_ensemble=10, seed=42)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, len(y))


def test_return_fitted_shape_1d():
    """With return_fitted=True and 1D y, should return (ensemble, fitted) tuple."""
    y = make_ar1(n=100, seed=42)
    result = generate_perturbations(y, n_ensemble=10, return_fitted=True, seed=42)
    assert isinstance(result, tuple)
    assert len(result) == 2
    ensemble, fitted = result
    assert ensemble.shape == (10, len(y))
    assert fitted.shape == (len(y),)


def test_return_fitted_shape_2d():
    """With return_fitted=True and 2D y, fitted should match (npts, n_series)."""
    y1 = make_ar1(n=100, seed=1)
    y2 = make_ar1(n=100, seed=2)
    y = np.column_stack([y1, y2])
    result = generate_perturbations(y, n_ensemble=10, return_fitted=True, seed=42)
    assert isinstance(result, tuple)
    assert len(result) == 2
    ensemble, fitted = result
    assert ensemble.shape == (10, y.shape[0], y.shape[1])
    assert fitted.shape == (y.shape[0], y.shape[1])


def test_return_fitted_matches_fit_model():
    """Fitted values from generate_perturbations should match fit_model's output."""
    y = make_ar1(n=100, seed=42)
    fit = fit_model(y)
    ensemble, fitted = generate_perturbations(
        y, n_ensemble=5, fit=fit, return_fitted=True, seed=42
    )
    np.testing.assert_array_equal(fitted, fit["fitted"])


def test_return_fitted_with_fit_ts_uses_fit_ts_length():
    """When fit_ts is provided, fitted should match fit_ts's length, not y's."""
    y = make_ar1(n=100, seed=42)
    fit_ts = make_ar1(n=150, seed=100)
    ensemble, fitted = generate_perturbations(
        y, fit_ts=fit_ts, n_ensemble=5, return_fitted=True, seed=42
    )
    assert ensemble.shape == (5, len(y))
    assert fitted.shape == (len(fit_ts),)  # fitted matches fit_ts, not y
