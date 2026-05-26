# Perturbed-Input

This project develops a standalone Python module capable of generating **perturbed input time series** for use in ensemble modelling workflows. The approach uses ARIMA modelling to characterize a time series, extracts residual errors, and resamples from those errors to produce statistically consistent perturbations of the original signal. The project builds off of a recent paper [Perturbed Input Ensemble Modeling With the Space Weather Modeling Framework](https://doi.org/10.1029/2018SW002000) which used perturbed input to drive a global model of magnetospheric and space weather activity. In science ensemble model runs are important as they characterize the uncertainty in models, the variability and sensitivity when model parameters change, and can also help to quantify the liklihood of an event ocurring. See, [The Importance of Ensemble Techniques for Operational Space Weather Forecasting](https://doi.org/10.1029/2018SW001861) for more discussion and examples. 

This project is structured as a sequence of six self-contained smaller projects, each building directly on the last. By the end, we will have a well-documented, tested Python module suitable for inclusion in a larger modelling pipeline.

**Estimated Total Duration:** 7–8 weeks of core project work (full-time, ~35 hrs/week), leaving time for extensions, real-data validation and implementation in existing models, or a final technical report (which can lead to a full publication).

## Intro

### Git, GitHub and GitHub Desktop

We'll use Git for basic verstion control. We will mostly be using ```commit```, ```push``` and ```pull``` commands. As the project develops we may start using ```branches``` to build and test new features and ```merge``` to add those features and code changes to ```main``` branch. Below is some material for Git, GitHub, and GitHub Desktop.

#### Introductory Material
- [Git, GitHub, & GitHub Desktop for beginners](https://www.youtube.com/watch?v=8Dd7KRpKeaE)
- [git - the simple guide](https://rogerdudler.github.io/git-guide/)
- [GitHub Docs - Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world)
- [Git and GitHub for Beginners - Crash Course](https://www.youtube.com/watch?v=RGOj5yH7evk)

### Python

- [Kaggle intro to Python](https://www.kaggle.com/learn/python)
- [Intro To Python](https://www.datacamp.com/courses/intro-to-python-for-data-science)
- [Goodard Heliophysics Boot Camp](https://github.com/helio670/bootcamp)
- [Goddar Heliophysics Data Science](https://github.com/helio670/datascience)
- [An Introduction to Python Package Managers](https://www.jumpingrivers.com/blog/python-package-managers-pip-conda-poetry/)
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)
- [Quick Start Guide for Python in VS Code](https://code.visualstudio.com/docs/python/python-quick-start)
 - [Getting Started with Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
 - [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
 - [Google Colab](https://colab.research.google.com/)



## Project Timeline (Estimated/Suggested)

This is just an estimated timeline, somethings may take less or more time depending on availability etc. It should not be taken to **heart**.

| Week | Stage | Daily Focus | Milestone |
|------|-------|-------------|-----------|
| 1 | Stage 1 | Reading (3 days) + environment setup + concept summary (2 days) | ARIMA theory solid; written summary complete |
| 2 | Stage 2 | Sine wave notebook (2 days) + noisy signal + diagnostics (3 days) | Residual diagnostic 4-panel function done |
| 3–4 | Stage 3 | Reading (2 days) + four sampling methods (4 days) + comparison plots (2 days) | `sampling_methods.py` complete; ensemble plots generated |
| 5 | Stage 4 | `auto_arima` fitting (2 days) + `arima_utils.py` functions (3 days) | Automated pipeline working on four test signals |
| 6–7 | Stage 5 | API design + `perturb_ts.py` (3 days) + pytest suite (2 days) + README + examples (2 days) | `perturb_ts.py` v1.0 passing all tests |
| 8–9 | Stage 6 | VAR theory reading (2 days) + VAR fitting (2 days) + correlated sampling (2 days) + module extension (3 days) | Multivariate module complete; all tests passing |
| 10-11 | Validation | Apply module to real-world datasets (see below) | Real-data validation notebook complete |
| 12-13 | Wrap-up | Final technical report + code cleanup + documentation polish | Submission-ready module and report |
| 14-15 | Extensions | Choose 1–2 optional extensions (see below) | Selected extension implemented and tested |

### Validation
**Week 10-11: Real-Data Validation**

Apply the completed module to two or three real-world time series. Suggested datasets:

#### Required Tests
- **Solar Wind Observations:**

#### Optional Tests
- **Hydrology:** Daily streamflow data from the Water Survey of Canada (https://wateroffice.ec.gc.ca/). Streamflow is non-stationary and often heavy-tailed — a good stress test.
- **Meteorology:** Hourly temperature or wind speed from Environment and Climate Change Canada (https://climate.weather.gc.ca/). Useful for the multivariate extension (temperature + humidity as correlated pair).
- **Electrical load:** Ontario hourly electricity demand from the IESO (http://www.ieso.ca/power-data). Highly structured with daily and weekly seasonality — will motivate the SARIMA extension.
- **Finance (for methodology comparison):** Any stock closing price from Yahoo Finance via `yfinance`. Well-studied in bootstrapping literature, making it easy to cross-validate results.

For each dataset, produce: a fitted model summary, a residual characterization report, a 50-member ensemble plot, and a brief written assessment of whether the ensemble spread looks physically reasonable.

### Wrap-up

Generate a short report summarizing the work, explaining the methodology and functionality, and providing a few short examples. 

### Optional Extensions

Time permitting we can implement some additional functionality. 

1. **Seasonal ARIMA (SARIMA) — Recommended First Extension**
   Extend Stage 4 to support seasonal signals using `pm.auto_arima(seasonal=True, m=s)` where `s` is the seasonal period (e.g., `m=12` for monthly data, `m=24` for hourly). This is highly practical — most environmental and energy time series have seasonal cycles that plain ARIMA cannot capture. Update `fit_model()` to accept a `seasonal_period` parameter and dispatch to SARIMA when `seasonal_period > 1`.
   - Reference: Hyndman & Athanasopoulos, Chapter 9.9: https://otexts.com/fpp3/seasonal-arima.html

2. **Automatic Block Length Selection — Recommended Second Extension**
   Replace the `l ≈ n^(1/3)` heuristic with the data-driven method of Politis & White (2004) implemented in the `arch` library (`arch.bootstrap.optimal_block_length`). Integrate this into `characterize_residuals()` so the optimal block length is reported automatically.
   - Reference: Politis, D.N. & White, H. (2004). *Econometric Reviews*, 23(1), 53–70.
   - `arch` library docs: https://arch.readthedocs.io/en/latest/bootstrap/


3. **Copula-based Multivariate Sampling**
   Use copulas (e.g., Gaussian or t-copula) to model the joint residual distribution in the multivariate case, replacing the Cholesky-Gaussian approach. Copulas can capture non-linear and tail dependence that a multivariate Gaussian cannot. The `copulas` library (by DataCebo) provides a clean interface.
   - Reference: Sklar, A. (1959). "Fonctions de répartition à n dimensions et leurs marges." — The original copula theorem.
   - `copulas` library: https://sdv.dev/Copulas/

