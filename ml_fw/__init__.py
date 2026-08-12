# -*- coding: utf-8 -*-
"""Basic ML framework to help prototype ML models faster.

ml_fw is a lightweight Python toolkit for building machine learning workflows
with a focus on heliophysics, space-weather, and similar scientific domains.

Modules:
--------
- data_io : Feature and target dataset creation, feature engineering
- ml_mod : Model training and hyperparameter tuning
- profile : Correlation analysis and feature profiling
- inspect : Model diagnostics and result inspection
- plot : Visualization for inspection results
- perturbed_input : ARIMA residual ensemble perturbation for uncertainty quantification

Quick Start:
-----------
>>> from ml_fw import data_io, ml_mod
>>> features, targets = data_io.create(df, feature_columns=[...], target_columns=[...])
>>> model = ml_mod.train(features, targets, estimator=RandomForestRegressor())

See the README for detailed documentation and examples for each module.
"""

# Make submodules easily accessible
from . import data_io  # noqa: F401
from . import ml_mod  # noqa: F401
from . import profile  # noqa: F401
from . import inspect  # noqa: F401
from . import plot  # noqa: F401
from . import perturbed_input  # noqa: F401

__all__ = [
    "data_io",
    "ml_mod",
    "profile",
    "inspect",
    "plot",
    "perturbed_input",
]
