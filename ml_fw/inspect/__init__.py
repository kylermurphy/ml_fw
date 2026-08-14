# -*- coding: utf-8 -*-
"""A collection of tools for inspecting and diagnosing machine learning model results.

This module provides functions to:
- Compute box-and-whisker statistics for residuals binned by feature values (boxplot_vx)
- Compute performance metrics binned by features with k-fold uncertainty (boxplot_metvx)
- Compute rolling-window metrics over time or index (rolling_met)

Output from these functions integrates with ml_fw.plot.plot_boxplot for visualization.
"""

from ._boxplot_vx import boxplot_vx
from ._boxplot_metvx import boxplot_metvx
from ._rolling_met import rolling_met

__all__ = [
    "boxplot_vx",
    "boxplot_metvx",
    "rolling_met",
]
