# -*- coding: utf-8 -*-
"""Plotting utilities for model inspection and diagnostics.

This module provides visualization functions for box-and-whisker plots and other
diagnostic visualizations. Output from ml_fw.inspect functions (boxplot_vx,
boxplot_metvx) can be directly rendered using plot_boxplot.
"""

from ._boxplot import plot_boxplot

__all__ = [
    "plot_boxplot",
]