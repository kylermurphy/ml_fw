# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:28:59 2024.

@author: krmurph1

ivestigate residuals of a ml model
"""

import numpy as np
import pandas as pd
from scipy import stats

def _extract_data_vx(x_dat, y_dat, box_dat=None):
    """
    Extract and normalize input data.
    
    Returns
    -------
    x_data : DataFrame or Series
        X data for binning
    y_data : DataFrame or Series
        Y data for statistics
    x_cols : list
        Column names for x_data
    y_col : str or list
        Column name(s) for y_data
    """
    if isinstance(x_dat, list) and isinstance(y_dat, list) and isinstance(box_dat, pd.DataFrame):
        x_data = box_dat[x_dat]
        y_data = box_dat[y_dat]
        x_cols = x_dat
        y_col = y_dat
    elif isinstance(x_dat, (pd.DataFrame, pd.Series)) and isinstance(y_dat, (pd.DataFrame, pd.Series)):
        x_data = x_dat
        y_data = y_dat
        x_cols = list(x_dat.columns) if isinstance(x_dat, pd.DataFrame) else [x_dat.name or 'x']
        y_col = list(y_dat.columns) if isinstance(y_dat, pd.DataFrame) else [y_dat.name or 'y']
    else:
        raise ValueError("Invalid data input: either provide (x_dat, y_dat as DataFrames/Series) "
                         "or (x_dat, y_dat as lists with box_dat)")
    
    return x_data, y_data, x_cols, y_col


def _normalize_config(config_list, num_cols, default_value):
    """
    Normalize configuration list (bins or xrange) to match number of columns.
    
    Parameters
    ----------
    config_list : int, float, list, or None
        Configuration to normalize
    num_cols : int
        Number of columns to match
    default_value : int, float, or None
        Default value to use
    
    Returns
    -------
    list
        Normalized configuration list
    """
    if isinstance(config_list, list):
        # Special case for xrange with single column
        if default_value is None and len(config_list) == 2 and num_cols == 1:
            return [config_list]
        elif len(config_list) == num_cols:
            return config_list
    
    return [default_value] * num_cols


def _compute_bin_statistics(x, y, bins, range_tuple):
    """
    Compute all required statistics for binned data in a single pass.
    
    Parameters
    ----------
    x : ndarray
        X values for binning
    y : ndarray
        Y values to compute statistics on
    bins : int
        Number of bins
    range_tuple : tuple or None
        (min, max) range for bins
    
    Returns
    -------
    dict
        Dictionary with keys: mean, median, q1, q3, x_edge, x_indices
    """
    # Compute all statistics using binned_statistic with standard aggregations
    mean, x_edge, _ = stats.binned_statistic(x, y, bins=bins, range=range_tuple, statistic=np.nanmean)
    median, _, _ = stats.binned_statistic(x, y, bins=bins, range=range_tuple, statistic=np.nanmedian)
    q1, _, _ = stats.binned_statistic(x, y, bins=bins, range=range_tuple, 
                                      statistic=lambda vals: np.nanpercentile(vals, 25))
    q3, _, _ = stats.binned_statistic(x, y, bins=bins, range=range_tuple, 
                                      statistic=lambda vals: np.nanpercentile(vals, 75))
    
    return {
        'mean': mean,
        'median': median,
        'q1': q1,
        'q3': q3,
        'x_edge': x_edge
    }


def _create_box_stats_list(stats_dict, whisker):
    """
    Create list of box plot statistics dictionaries.
    
    Parameters
    ----------
    stats_dict : dict
        Dictionary with mean, median, q1, q3 arrays
    whisker : float
        Whisker coefficient (e.g., 1.5 for Tukey)
    
    Returns
    -------
    list
        List of box statistics dictionaries
    """
    box_stats = []
    for mean, med, q1, q3 in zip(stats_dict['mean'], stats_dict['median'], 
                                  stats_dict['q1'], stats_dict['q3']):
        iqr = q3 - q1
        box_stats.append({
            "mean": mean,
            "med": med,
            "q1": q1,
            "q3": q3,
            "whislo": q1 - whisker * iqr,
            "whishi": q3 + whisker * iqr,
            "fliers": []
        })
    return box_stats


def boxplot_vx(x_dat: pd.DataFrame | list,
                   y_dat: pd.DataFrame | list,
                   box_dat: pd.DataFrame = None,
                   box_meth: bool | dict = True,
                   bins: int | list = 10,
                   xrange: list[tuple[float, float]] | None = None,
                   whisker: float = 1.5) -> dict:
    """
    Calculate boxplot statistics of y as a function of x bins.

    Parameters
    ----------
    x_dat : pd.DataFrame | list
        Data for binning. If list, contains column names to extract from box_dat.
        
        The y-data within the binned data is then used to calculate statistics 
        and derive box and whisker values for each bin.

        The DataFrame can have more the one column. Each column will be binned
        (see bins) and box/whisker data returned.

    y_dat : pd.DataFrame | list
        Y data for computing statistics.

    box_dat : pd.DataFrame, optional
        Combined DataFrame (required if x_dat, y_dat are lists).

    box_meth : bool | dict, optional
        Placeholder for future statistics method selection.

    bins : int | list, default=10
        Number of bins or list of bin counts per x column.

        Used in call to scipy.stats.binned_statistic

    xrange : list[tuple[float, float]] | None, default=None
        (min, max) range for bins per x column.

        Used in call to scipy.stats.binned_statistic

    whisker : float, default=1.5
        Whisker coefficient (1.5 = Tukey's boxplot).

    Returns
    -------
    dict
        Nested dictionary mapping x column names to boxplot data (dict is nested for
        easier plotting).
            - results[x_col_name]['residuals'] contains:
            - 'box_stats': List of dicts with keys {mean, med, q1, q3, whislo, whishi, fliers}
            - 'x_edge': Bin edge positions
            - 'x_centre': Bin center positions
            - 'x_width': Width of bins
    """
    # Extract and validate data
    x_data, y_data, x_cols, y_col = _extract_data_vx(x_dat, y_dat, box_dat)
    
    # Normalize bins and xrange to match x columns
    bins_normalized = _normalize_config(bins, len(x_cols), 10)
    xrange_normalized = _normalize_config(xrange, len(x_cols), None)
    
    results = {}
    
    for col_name, num_bins, col_range in zip(x_cols, bins_normalized, xrange_normalized):
        # Extract x and y values, handling both DataFrame columns and Series
        try:
            x_vals = x_data[col_name].to_numpy().squeeze()
        except (KeyError, AttributeError):
            x_vals = x_data.to_numpy().squeeze()
        
        try:
            y_vals = y_data[y_col].to_numpy().squeeze() if isinstance(y_col, list) else y_data.to_numpy().squeeze()
        except (KeyError, AttributeError, TypeError):
            y_vals = y_data.to_numpy().squeeze()
        
        # Compute all statistics in one function call
        stats_result = _compute_bin_statistics(x_vals, y_vals, num_bins, col_range)
        
        # Calculate bin centers and widths (vectorized)
        x_edges = stats_result['x_edge']
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        x_width = x_edges[1] - x_edges[0]
        
        # Create box statistics list
        box_stats = _create_box_stats_list(stats_result, whisker)
        
        results[col_name] = {
            'residuals': {
                'box_stats': box_stats,
                'x_edge': x_edges,
                'x_centre': x_centers,
                'x_width': x_width
                }
            }
    
    return results
