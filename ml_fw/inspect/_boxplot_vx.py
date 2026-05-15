# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:28:59 2024.

@author: krmurph1

ivestigate residuals of a ml model
"""

import numpy as np
import pandas as pd
from scipy import stats

#TODO Check to make sure that y is single valued

def boxplot_vx(x_dat: pd.DataFrame | list,
               y_dat: pd.DataFrame | list,
               box_dat: pd.DataFrame = None,
               box_meth: bool | dict = True,
               bins: int | list = 10,
               xrange: list[tuple[float, float]] | None = None,
               whisker: float = 1.5) -> dict:
    """Calculate boxplot like statistics of y as a function of x.

    Parameters
    ----------
    x_dat : pd.DataFrame | list
        A pandas DataFrame containing the data for binning. The y-data within
        the binned is then used to calculate statistics and derive box and
        whisker values for each bin.
        The DataFrame can have more the one column. Each column will be binned
        (see bins) and box/whisker data returned.
        If x_dat is a list then the list contains the column names which
        correspond to the binning data in box_dat DataFrame.

    y_dat : pd.DataFrame | list
        A pandas DataFrame containing the y-data which will be use to calculate
        the box and whisker statistics for each bin from x_dat.
        If y_dat is a list it contains the column name for box_dat that
        contains the y-data with which box/whisker values are derived.
        y_dat should be a single columnd DataFrame or single valued list.

    box_dat : pd.DataFrame, optional
        The default is None.
        A pandas DataFrame continaing the x-data and y-data for which values
        are binned and subsequently used to derive box/whisker values,
        respectively.
        If provided, x_dat and y_dat should be lists specifying the column
        names of the x-data for binning and the y-data for deriving box/whisker
        values.

    box_meth : bool | dict, optional
        The default is True.
        Currently a place holder which may be used to change the statistics
        returned for the box/whisker values.

    bins : int | list, optional
        The default is 10.
        The number of bins with which the x-data is seprated into when deriving
        statistics. It defines the number of equal-width bins in the given
        range.
        If bins is a list it should have the same number of elements or
        columns as x_dat. Each value then specifies the bins for the
        corresponding x data.
        If the number of elements is not the same bins is set to None and the
        default value of 10 is used
        Used in call to scipy.stats.binned_statistic

    xrange : list[tuple[float, float]] | None, optional
        The default is None.
        The lower and upper range of the bins. If not provided, range is simply
        (x.min(), x.max()). Values outside the range are ignored.
        If xrange is a list each element should be a two element list which
        corresponds to the min and max for the corresponding element of x_col.
        In this case the length of xrange should be the same as x_col or the
        number of columns of x_col (depending on type)
        Used in call to scipy.stats.binned_statistic

    whisker : float, optional
        The default is 1.5.
        The position of the whiskers.
        If a float, the lower whisker is at the lowest datum above
        Q1 - whis*(Q3-Q1), and the upper whisker at the highest datum below
        Q3 + whis*(Q3-Q1), where Q1 and Q3 are the first and third quartiles.
        The default value of whis = 1.5 corresponds to Tukey's original
        definition of boxplots.

    Returns
    -------
    box_idx : Dictionary
        A dictionary for each of x_col specifing the values for
        the box/whisker plot.
        If x_dat is a DataFrame the keys are the column names else they
        are the values of the x_dat list.
        box_idx['key']
        Each value is a subsequent dictionary containing the box/whisker data
        for plotting.
        box_idx['key']['box_stats'] - a list of dictionaries containgin the
        box/whisker statistics for bin of the corresponding x_dat ('key').
        The required keys for a box plot (from matplotlib.axes.Axes.bxp) are:
            - med: Median (scalar).
            - q1, q3: First & third quartiles (scalars).
            - whislo, whishi: Lower & upper whisker positions (scalars).
        Optional keys which are used are:
            - mean: Mean (scalar). Needed if showmeans=True.
            - fliers: Data beyond the whiskers (array-like).
            Needed if showfliers=True. Always empty
        box_idx['key']['x_edge'] - an array of dtype float which returns
        the bin edges for the corresponding x_dat ('key'). Returned from
        scipy.stats.binned_statistic
        box_idx['key']['x_centre'] - an array of dtype float containing
        the centre x value for each bin from box_idx['key']['x_edge'].
        box_idx['key']['x_width'] - the width of a x-bin.
    """
    # get data for processing
    if isinstance(x_dat, list) \
            and isinstance(y_dat, list) \
            and isinstance(box_dat, pd.DataFrame):
        x_col = x_dat
        y_col = y_dat

        x_val = box_dat[x_col]
        y_val = box_dat[y_col]
    elif isinstance(x_dat, (pd.DataFrame, pd.Series)) \
            and isinstance(y_dat, (pd.DataFrame, pd.Series)):
        # get the column/series name
        if isinstance(x_dat, pd.DataFrame):
            x_col = x_dat.columns
        else:
            x_col = [x_dat.name] if x_dat.name else ['x_col']
        # get the columns/series name
        if isinstance(y_dat, pd.DataFrame):
            y_col = y_dat.columns
        else:
            y_col = y_dat.name if y_dat.name else 'y_col'

        # get the data
        x_val = x_dat
        y_val = y_dat

    # create a list for bins the same size as x_col
    if isinstance(bins,list) and len(bins) == len(x_col):
        bin_v = bins
    else:
        bin_v = np.zeros(len(x_col))
        bin_v[:] = bins

    # create a list for xrange the same size as x_col
    if isinstance(xrange,list) and len(xrange) == 2 and len(x_col) == 1:
        xran = [xrange]
    elif isinstance(xrange,list) and len(xrange) == len(x_col):
        xran = xrange
    else:
        xran = [None for x in x_col]

    # now we want to itterate over the x columns
    # and generate the box plot as binned statistic

    # define lambdas to calculate the upper and lower
    # quartiles
    lq_nan = lambda stat: np.nanpercentile(stat, 25)
    uq_nan = lambda stat: np.nanpercentile(stat, 75)

    box_idx = {}

    for idx, bn, xr in zip(x_col, bin_v, xran):
        # calculate the statistics as a function of idx
        xr = xr if isinstance(xr,list) and len(xr) == 2 else None

        # reshape the arrays
        try:
            x = x_val[idx].to_numpy().squeeze()
        except Exception:
            x = x_val.to_numpy().squeeze()

        try:
            y = y_val[y_col].to_numpy().squeeze()
        except Exception:
            y = y_val.to_numpy().squeeze()

        # calculate stats
        mean, x_edge, _ = stats.binned_statistic(x, y, bins=bn,
                                                 range=xr,
                                                 statistic=np.nanmean)
        median, _, _ = stats.binned_statistic(x, y,bins=bn,
                                              range=xr,
                                              statistic=np.nanmedian)
        low_q, _, _ = stats.binned_statistic(x, y, bins=bn,
                                             range=xr, statistic=lq_nan)
        up_q, _, _ = stats.binned_statistic(x, y, bins=bn,
                                            range=xr, statistic=uq_nan)

        # calculate x location of the stats
        x_cen = (x_edge[0:-1] + [x_edge[1:]]) / 2.
        x_cen = x_cen.squeeze()
        x_wid = x_edge[1] - x_edge[0]
        # create a list to hold the required
        # parameters to draw a box and whisker plot
        box_stats = []
        for mn, md, lq, uq, in zip(mean, median, low_q, up_q):
            val = {"mean": mn,  # not required
                   "med": md,
                   "q1": lq,
                   "q3": uq,
                   "whislo": lq - whisker * (uq - lq),  # required
                   "whishi": uq + whisker * (uq - lq),  # required
                   "fliers": []  # required if showfliers=True
                   }
            box_stats.append(val)

        # create a dictionary to store everything needed for plotting
        # and add it to the return dictionary
        box_idx[idx] = {'box_stats':box_stats, 'x_edge':x_edge,
                        'x_centre':x_cen, 'x_width':x_wid}

    return box_idx


# =============================================================================
# REFACTORED VERSION - boxplot_vx_ref
# =============================================================================

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


def boxplot_vx_ref(x_dat: pd.DataFrame | list,
                   y_dat: pd.DataFrame | list,
                   box_dat: pd.DataFrame = None,
                   box_meth: bool | dict = True,
                   bins: int | list = 10,
                   xrange: list[tuple[float, float]] | None = None,
                   whisker: float = 1.5) -> dict:
    """
    Calculate boxplot statistics of y as a function of x bins.
    
    This is a refactored version of boxplot_vx with improved readability,
    performance, and pythonic code patterns.

    Parameters
    ----------
    x_dat : pd.DataFrame | list
        Data for binning. If list, contains column names to extract from box_dat.
    y_dat : pd.DataFrame | list
        Y data for computing statistics.
    box_dat : pd.DataFrame, optional
        Combined DataFrame (required if x_dat, y_dat are lists).
    box_meth : bool | dict, optional
        Placeholder for future statistics method selection.
    bins : int | list, default=10
        Number of bins or list of bin counts per x column.
    xrange : list[tuple[float, float]] | None, default=None
        (min, max) range for bins per x column.
    whisker : float, default=1.5
        Whisker coefficient (1.5 = Tukey's boxplot).

    Returns
    -------
    dict
        Dictionary mapping x column names to boxplot data:
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
