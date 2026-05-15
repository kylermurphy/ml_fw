# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:44:50 2024.

@author: krmurph1
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from itertools import cycle


def boxplot_metvx_old(x_dat: pd.DataFrame | list,
                  y_true: pd.DataFrame | list,
                  y_mod: pd.DataFrame | list,
                  box_dat: pd.DataFrame = None,
                  box_metric=None,
                  kfolds: int = 100,
                  kfrac: int = 0.5,
                  bins: int | list = 10,
                  xrange: list[tuple[float, float]] | None = None,
                  whisker: float = 1.5):
    """Calculate boxplot like statistics of a metric (using y-t and y-m) vs x.

    The data is binned as a function of x. For each bin in x the a fraction of
    the true and model data are randomly sampled (kfrac). This sample is used
    to calculte a metric of the data. This is repeated kfolds time to generate
    a distribution of metric values for that bin.

    This distribution is then used to derive stats for a box and whisker plot.
    This is repeated for each bin of x.

    The metric can be a callable passed to the function, for example:

        met = lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred)
    or
        met = lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred,
                                                         normalize=False)

    and the callable can specify the parameters of the metric.

    More examples:
    - Regression score
        met = lambda y_true, y_pred: metrics.mean_absolute_error(y_true,
                                                                 y_pred)

    - Clustering score
        met = lambda y_true, y_pred: metrics.homogeneity_score(labels_true,
                                                               labels_pred)

    In these cases the callable would be passed using the box_metric keyword.

    Parameters
    ----------
    x_dat : pd.DataFrame | list
        A pandas DataFrame containing the data for binning. The metric
        (calculated from y-true and y-mod) is then binned and used to calculate
        statistics and derive box and whisker values for each bin.

        The DataFrame can have more the one column. Each column will be binned
        (see bins) and box/whisker data returned.

        If x_dat is a list then the list contains the column names which
        correspond to the binning data in box_dat DataFrame.

    y_true : pd.DataFrame | list
        A pandas DataFrame containing the true data which will be use to
        calculte the metric.

        If y_dat is a list it contains the column name for box_dat that
        contains the true data for calculating the metric.

        y_dat should be a single columnd DataFrame or single valued list.

    y_mod : pd.DataFrame | list
        A pandas DataFrame containing the model data which will be use to
        calculte the metric.

        If y_mod is a list it contains the column name for box_dat that
        contains the model data used to calculate the metric.

        y_mod should be a single columnd DataFrame or single valued list.

    box_dat : pd.DataFrame, optional
        The default is None.

        A pandas DataFrame continaing the x-data and y-true and y-mod which are
        used to calculate a metric. The metric is then binned as a fucntion of
        the x-data and subsequently used to derive box/whisker values..

        If provided, x_dat, y_true, and y_mod should be lists specifying the
        column names of the x-data for binning and the y-true and y-mod data
        for calculating metric values.

    box_metric : callable
        The default is None.

        A callable which can be used to calculate metric values from y-true and
        y-mod.

        If y-true and y-mod are integers then it is assumed the model is a
        classification type and the accuracy metric from scikit-kearn is used.

        If they are floats then the mean squared error from scikit-learn is
        used.

    kfolds : int, optional
        The default is 100.

        For each bin, randomly sample the metric kfolds time to create a
        distribution. Use this distribution to derive box and whisker stats.

    kfrac : int, optional
        The default is 0.5.

        The fraction of the data to use in each kfold.

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
    if isinstance(box_dat, pd.DataFrame) and \
            isinstance(x_dat, list) and \
            isinstance(y_true, list) and \
            isinstance(y_mod, list):
        x_d = box_dat[x_dat]
        x_c = x_dat.copy()
        y_t = box_dat[y_true].to_numpy().squeeze()
        y_p = box_dat[y_mod].to_numpy().squeeze()

    elif isinstance(x_dat, (pd.DataFrame, pd.Series)) and \
            isinstance(y_true, (pd.DataFrame, pd.Series)) and \
            isinstance(y_mod, (pd.DataFrame, pd.Series)):
        x_d = x_dat
        x_c = x_dat.columns
        y_t = y_true.to_numpy().squeeze()
        y_p = y_mod.to_numpy().squeeze()

    # define a metric to use if box_metric is None
    # if the true and modeled y-values are integers
    # assume we have a categorical model
    # else assume it is a regression model
    if np.issubdtype(y_t.dtype,np.integer) \
            and np.issubdtype(y_p.dtype,np.integer) \
            and not box_metric:
        print('Using Accuracy Metric')
        met = lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred)
    elif not box_metric:
        print('Using Mean Square Error Metric')
        met = lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred)
    else:
        print('Using passed metric')
        met = box_metric

    # create a list for bins the same size as x_col
    if isinstance(bins,list) and len(bins) == len(x_c):
        bin_v = bins
    else:
        bin_v = np.zeros(len(x_c))
        bin_v[:] = bins

    # create a list for xrange the same size as x_col
    if isinstance(xrange,list) and len(xrange) == 2 and len(x_c) == 1:
        xran = [xrange]
    elif isinstance(xrange,list) and len(xrange) == len(x_c):
        xran = xrange
    else:
        xran = [None for x in x_c]

    # put y-data into dataframe to simplify statistics calculations
    y_d = pd.DataFrame({"tr":y_t, "pr":y_p})

    box_idx = {}

    for idx, bn, xr in zip(x_c, bin_v, xran):
        # calculate the statistics as a function of idx
        xr = xr if isinstance(xr,list) and len(xr) == 2 else None

        # reshape the x arrays
        try:
            x = x_d[idx].to_numpy().squeeze()
        except Exception:
            x = x_d.to_numpy().squeeze()

        # use bin statistic to get the indices of the x data
        # for all the bins with which the data is binned into
        # this can then be used to subsquently bin the metric
        x_stat, x_edges, x_bnum = stats.binned_statistic(x, x,
                                                         bins=bn, range=xr)

        x_cen = (x_edges[0:-1] + [x_edges[1:]]) / 2.
        x_cen = x_cen.squeeze()
        x_wid = x_edges[1] - x_edges[0]

        # calculate the box stats for this x
        box_stats = []

        for i in np.arange(x_stat.size, dtype=int):
            # get the indices for values which lie between
            # bin[i] and bin[i+1]
            gd = x_bnum == i + 1
            # create an array of k-fold samples which
            # holds metric values from each sample which
            # box stats can be computed from
            if sum(gd) > 1:
                sval = np.array([
                                met(y_d.loc[gd,'tr'].sample(frac=kfrac,
                                                            random_state=x),
                                    y_d.loc[gd,'pr'].sample(frac=kfrac,
                                                            random_state=x))
                                for x in np.arange(kfolds)
                                ])

                lq = np.nanpercentile(sval,25)
                uq = np.nanpercentile(sval,75)
            else:
                sval = 0
                lq = 0
                uq = 0

            bval = {"mean": np.nanmean(sval),  # not required
                    "med": np.nanmedian(sval),
                    "q1": lq,
                    "q3": uq,
                    "whislo": lq - whisker * (uq - lq),  # required
                    "whishi": uq + whisker * (uq - lq),  # required
                    "fliers": []  # required if showfliers=True
                    }
            # append box to list
            box_stats.append(bval)

        # add box values to box dictionary
        box_idx[idx] = {'box_stats':box_stats, 'x_edge':x_edges,
                        'x_centre':x_cen, 'x_width':x_wid}

    return box_idx


# =============================================================================
# REFACTORED VERSION - boxplot_metvx_ref
# =============================================================================

def _extract_data_metvx(x_dat, y_true, y_mod, box_dat=None):
    """
    Extract and normalize input data.
    
    Returns
    -------
    x_data : DataFrame or Series
        X data for binning
    x_cols : list
        Column names for x_data
    y_true_arr : ndarray
        1D array of true values
    y_mod_arr : ndarray
        1D array of model predictions
    """
    if isinstance(box_dat, pd.DataFrame) and all(isinstance(d, list) for d in [x_dat, y_true, y_mod]):
        x_data = box_dat[x_dat]
        x_cols = x_dat
        y_true_arr = box_dat[y_true].to_numpy().squeeze()
        y_mod_arr = box_dat[y_mod].to_numpy().squeeze()
    elif all(isinstance(d, (pd.DataFrame, pd.Series)) for d in [x_dat, y_true, y_mod]):
        x_data = x_dat
        x_cols = list(x_dat.columns) if isinstance(x_dat, pd.DataFrame) else [x_dat.name or 'x']
        y_true_arr = y_true.to_numpy().squeeze()
        y_mod_arr = y_mod.to_numpy().squeeze()
    else:
        raise ValueError("Invalid data input: either provide (x_dat, y_true, y_mod as DataFrames) "
                         "or (x_dat, y_true, y_mod as lists with box_dat)")
    
    return x_data, x_cols, y_true_arr, y_mod_arr


def _select_metrics(y_true, y_mod, custom_metrics=None):
    """
    Select metric function(s) based on input or data types.
    
    Parameters
    ----------
    y_true : ndarray
        True values
    y_mod : ndarray
        Model predictions
    custom_metrics : callable, list of callables, dict, or None, optional
        Metric function(s) to use. Can be:
        - None: auto-select single metric based on data type
        - callable: single metric function
        - list: multiple callable functions (auto-named metric_0, metric_1, ...)
        - dict: {name: callable} for custom metric names
    
    Returns
    -------
    dict
        Dictionary mapping metric names to callable functions
    """
    if custom_metrics is None:
        # Auto-select single metric based on data type
        if np.issubdtype(y_true.dtype, np.integer) and np.issubdtype(y_mod.dtype, np.integer):
            return {'accuracy': metrics.accuracy_score}
        else:
            return {'mse': metrics.mean_squared_error}
    elif callable(custom_metrics):
        # Single callable
        return {'metric': custom_metrics}
    elif isinstance(custom_metrics, list):
        # List of callables
        if not all(callable(m) for m in custom_metrics):
            raise ValueError("All items in metric list must be callable")
        return {f'metric_{i}': func for i, func in enumerate(custom_metrics)}
    elif isinstance(custom_metrics, dict):
        # Dict of name -> callable
        if not all(callable(v) for v in custom_metrics.values()):
            raise ValueError("All values in metric dict must be callable")
        return custom_metrics
    else:
        raise ValueError("box_metric must be None, callable, list of callables, or dict of callables")


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


def _compute_bin_metrics(bin_mask, metric_func, y_data, kfolds, kfrac):
    """
    Compute metric values for a single bin via k-fold sampling.
    
    Parameters
    ----------
    bin_mask : ndarray
        Boolean mask for current bin
    metric_func : callable
        Metric function
    y_data : DataFrame
        DataFrame with 'tr' (true) and 'pr' (pred) columns
    kfolds : int
        Number of folds
    kfrac : float
        Fraction to sample each fold
    
    Returns
    -------
    ndarray
        Array of metric values from k-fold sampling
    """
    if bin_mask.sum() <= 1:
        return np.array([])
    
    # Get data for this bin
    y_bin = y_data.loc[bin_mask]
    
    # Compute metric for each k-fold sample
    metric_values = np.array([
        metric_func(
            y_bin['tr'].sample(frac=kfrac, random_state=seed).values,
            y_bin['pr'].sample(frac=kfrac, random_state=seed).values
        )
        for seed in range(kfolds)
    ])
    
    return metric_values


def _compute_bin_metrics_vectorized(bin_mask, metric_funcs, y_data, kfolds, kfrac):
    """
    Compute multiple metric values for a single bin via k-fold sampling.
    
    Parameters
    ----------
    bin_mask : ndarray
        Boolean mask for current bin
    metric_funcs : dict
        Dict mapping metric names to callable functions
    y_data : DataFrame
        DataFrame with 'tr' (true) and 'pr' (pred) columns
    kfolds : int
        Number of folds
    kfrac : float
        Fraction to sample each fold
    
    Returns
    -------
    dict
        Dict mapping metric names to arrays of metric values
    """
    if bin_mask.sum() <= 1:
        return {name: np.array([]) for name in metric_funcs}
    
    # Get data for this bin
    y_bin = y_data.loc[bin_mask]
    
    # Compute all metrics for each k-fold sample (vectorized)
    results = {}
    for metric_name, metric_func in metric_funcs.items():
        metric_values = np.array([
            metric_func(
                y_bin['tr'].sample(frac=kfrac, random_state=seed).values,
                y_bin['pr'].sample(frac=kfrac, random_state=seed).values
            )
            for seed in range(kfolds)
        ])
        results[metric_name] = metric_values
    
    return results


def _create_box_stats_dict(metric_values, whisker):
    """
    Create box plot statistics dictionary.
    
    Parameters
    ----------
    metric_values : ndarray
        Array of metric values
    whisker : float
        Whisker coefficient (e.g., 1.5 for Tukey)
    
    Returns
    -------
    dict
        Dictionary with box plot statistics
    """
    if len(metric_values) == 0:
        return {"mean": np.nan, "med": np.nan, "q1": np.nan, "q3": np.nan,
                "whislo": np.nan, "whishi": np.nan, "fliers": []}
    
    q1 = np.nanpercentile(metric_values, 25)
    q3 = np.nanpercentile(metric_values, 75)
    iqr = q3 - q1
    
    return {
        "mean": np.nanmean(metric_values),
        "med": np.nanmedian(metric_values),
        "q1": q1,
        "q3": q3,
        "whislo": q1 - whisker * iqr,
        "whishi": q3 + whisker * iqr,
        "fliers": []
    }


def boxplot_metvx(x_dat: pd.DataFrame | list,
                      y_true: pd.DataFrame | list,
                      y_mod: pd.DataFrame | list,
                      box_dat: pd.DataFrame = None,
                      box_metric=None,
                      kfolds: int = 100,
                      kfrac: float = 0.5,
                      bins: int | list = 10,
                      xrange: list[tuple[float, float]] | None = None,
                      whisker: float = 1.5) -> dict:
    """
    Calculate boxplot statistics of metric(s) (accuracy/error) across x bins.
    
    This is a refactored version of boxplot_metvx with improved readability,
    performance, and pythonic code patterns. Supports vectorized computation
    of multiple metrics simultaneously.

    Parameters
    ----------
    x_dat : pd.DataFrame | list
        Data for binning. If list, contains column names to extract from box_dat.
    y_true : pd.DataFrame | list
        True labels/values.
    y_mod : pd.DataFrame | list
        Model predictions.
    box_dat : pd.DataFrame, optional
        Combined DataFrame (required if x_dat, y_true, y_mod are lists).
    box_metric : callable, list of callables, dict, or None, optional
        Metric function(s) to use. Can be:
        - None: auto-select single metric based on data type
          (accuracy_score for integers, mean_squared_error for floats)
        - callable: single metric function, named 'metric'
        - list: multiple callable functions (auto-named metric_0, metric_1, ...)
        - dict: {name: callable} for custom metric names with custom callables
    kfolds : int, default=100
        Number of k-fold samples per bin.
    kfrac : float, default=0.5
        Fraction of data to sample in each fold.
    bins : int | list, default=10
        Number of bins or list of bin counts per x column.
    xrange : list[tuple[float, float]] | None, default=None
        (min, max) range for bins per x column.
    whisker : float, default=1.5
        Whisker coefficient (1.5 = Tukey's boxplot).

    Returns
    -------
    dict
        Nested dictionary with structure:
        - results[x_col_name][metric_name] contains:
          - 'box_stats': List of dicts with keys {mean, med, q1, q3, whislo, whishi, fliers}
          - 'x_edge': Bin edge positions
          - 'x_centre': Bin center positions
          - 'x_width': Width of bins
        
        Example with single metric (box_metric=None):
            results = {
                'x_col': {
                    'mse': {  # auto-named 'mse' for float data
                        'box_stats': [...],
                        'x_edge': array(...),
                        'x_centre': array(...),
                        'x_width': scalar
                    }
                }
            }
        
        Example with multiple metrics:
            results = {
                'x_col': {
                    'metric_0': {'box_stats': [...], ...},
                    'metric_1': {'box_stats': [...], ...}
                }
            }
    """
    # Extract and validate data
    x_data, x_cols, y_true_arr, y_mod_arr = _extract_data_metvx(x_dat, y_true, y_mod, box_dat)
    
    # Select and normalize metric function(s)
    metric_funcs = _select_metrics(y_true_arr, y_mod_arr, box_metric)
    
    # Normalize bins and xrange to match x columns
    bins_normalized = _normalize_config(bins, len(x_cols), 10)
    xrange_normalized = _normalize_config(xrange, len(x_cols), None)
    
    # Combine true and model data in DataFrame for efficient sampling
    y_combined = pd.DataFrame({"tr": y_true_arr, "pr": y_mod_arr})
    
    results = {}
    
    for col_name, num_bins, col_range in zip(x_cols, bins_normalized, xrange_normalized):
        # Extract x values, handling both DataFrame columns and Series
        try:
            x_vals = x_data[col_name].to_numpy().squeeze()
        except (KeyError, AttributeError):
            x_vals = x_data.to_numpy().squeeze()
        
        # Bin the x data
        _, x_edges, bin_indices = stats.binned_statistic(
            x_vals, x_vals, bins=num_bins, range=col_range
        )
        
        # Calculate bin centers and widths (vectorized)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        x_width = x_edges[1] - x_edges[0]
        
        # Compute box stats for each bin across all metrics
        col_results = {}
        for i in range(len(x_edges) - 1):
            # Get metric values for this bin across all metrics
            bin_mask = bin_indices == i + 1
            metric_values_dict = _compute_bin_metrics_vectorized(
                bin_mask, metric_funcs, y_combined, kfolds, kfrac
            )
            
            # Create box stats for each metric
            for metric_name, metric_values in metric_values_dict.items():
                if metric_name not in col_results:
                    col_results[metric_name] = []
                box_stat = _create_box_stats_dict(metric_values, whisker)
                col_results[metric_name].append(box_stat)
        
        # Wrap box_stats with metadata for each metric
        results[col_name] = {
            metric_name: {
                'box_stats': box_stats,
                'x_edge': x_edges,
                'x_centre': x_centers,
                'x_width': x_width
            }
            for metric_name, box_stats in col_results.items()
        }
    
    return results


# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def _format_x_axis(ax, x_centres, x_width):
    """
    Format x-axis tick marks and labels for better readability.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    x_centres : ndarray
        X position centers of bins
    x_width : float
        Width of each bin
    """
    # Set ticks at bin centers
    ax.set_xticks(x_centres)
    
    # Format tick labels - round to reasonable precision
    tick_labels = []
    for val in x_centres:
        if abs(val) < 0.001 or abs(val) > 10000:
            # Use scientific notation for very small/large numbers
            tick_labels.append(f'{val:.2e}')
        elif abs(val - int(val)) < 1e-10:
            # Integer values
            tick_labels.append(f'{int(val)}')
        else:
            # Decimal values - use 2 significant figures after decimal
            tick_labels.append(f'{val:.3f}')
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Add minor gridlines for better readability
    ax.grid(True, axis='x', alpha=0.2, linestyle='--', which='major')
    ax.grid(True, axis='y', alpha=0.1, linestyle=':', which='minor')


def plot_boxplot_metvx(results,
                       separate_by='both',
                       metric_name=None,
                       x_col=None,
                       colors=None,
                       alphas=None,
                       figsize=(12, 6),
                       ax=None,
                       fig=None,
                       showmeans=True,
                       showfliers=True):
    """
    Plot boxplot statistics from boxplot_metvx_ref output using matplotlib's bxp.
    
    Parameters
    ----------
    results : dict
        Output from boxplot_metvx_ref with structure:
        {x_col_name: {metric_name: {'box_stats': [...], 'x_edge': [...], ...}}}
    
    separate_by : {'both', 'x_col', 'metric_name'}, default='both'
        How to separate plots:
        - 'both': Create subplot grid with x_col as rows and metric_name as columns
        - 'x_col': Create separate plots for each x_col (all metrics in each plot)
        - 'metric_name': Create separate plots for each metric (all x_cols in each plot)
    
    metric_name : str, optional
        Filter to plot only a specific metric. If None, plot all metrics.
    
    x_col : str, optional
        Filter to plot only a specific x_col. If None, plot all x_cols.
    
    colors : list or None, default=None
        List of colors for box fill. Cycled through each box.
        If None, uses matplotlib default colors.
    
    alphas : list or None, default=None
        List of transparencies (0-1) for boxes. Cycled through each box.
        If None, uses opaque boxes (alpha=1).
    
    figsize : tuple, default=(12, 6)
        Figure size (width, height) if creating new figure.
    
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. Used only when separate_by='both' fails
        or for single subplot cases.
    
    fig : matplotlib.figure.Figure, optional
        Existing figure to use. If provided with ax=None, creates subplots within fig.
    
    showmeans : bool, default=True
        Show mean markers on boxplots (from box_stats if available).
    
    showfliers : bool, default=True
        Show outlier points (from box_stats if available).
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'fig': matplotlib Figure object
        - 'axes': dict mapping (x_col, metric_name) to Axes, or simplified dict if only one plot
        - 'axes_flat': list of all Axes objects created
    """
    # Validate separate_by
    if separate_by not in ['both', 'x_col', 'metric_name']:
        raise ValueError("separate_by must be 'both', 'x_col', or 'metric_name'")
    
    # Filter results based on metric_name and x_col
    filtered_results = {}
    for xcol, metrics_dict in results.items():
        if x_col is not None and xcol != x_col:
            continue
        filtered_metrics = {}
        for mname, data in metrics_dict.items():
            if metric_name is not None and mname != metric_name:
                continue
            filtered_metrics[mname] = data
        if filtered_metrics:
            filtered_results[xcol] = filtered_metrics
    
    if not filtered_results:
        raise ValueError("No data matches the provided filters (metric_name, x_col)")
    
    # Get unique x_cols and metric names
    unique_xcols = sorted(filtered_results.keys())
    unique_metrics = sorted(set(m for metrics_dict in filtered_results.values() for m in metrics_dict.keys()))
    
    # Setup colors and alphas cycles
    if colors is None:
        colors = plt.cm.Set1.colors
    color_cycle = cycle(colors)
    
    if alphas is None:
        alphas = [1.0]
    alpha_cycle = cycle(alphas)
    
    # Create figure and axes based on separate_by
    axes_dict = {}
    axes_list = []
    
    if separate_by == 'both':
        n_rows = len(unique_xcols)
        n_cols = len(unique_metrics)
        if fig is None:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows / 2))
        
        for i, xcol in enumerate(unique_xcols):
            for j, mname in enumerate(unique_metrics):
                ax_idx = i * n_cols + j + 1
                ax = fig.add_subplot(n_rows, n_cols, ax_idx)
                axes_list.append(ax)
                axes_dict[(xcol, mname)] = ax
                
                # Plot if data exists
                if xcol in filtered_results and mname in filtered_results[xcol]:
                    _plot_single_boxplot(
                        filtered_results[xcol][mname],
                        ax=ax,
                        colors=colors,
                        alphas=alphas,
                        showmeans=showmeans,
                        showfliers=showfliers,
                        title=f'{xcol} - {mname}'
                    )
                    # Format x-axis
                    data = filtered_results[xcol][mname]
                    _format_x_axis(ax, data['x_centre'], data['x_width'])
    
    elif separate_by == 'x_col':
        if fig is None:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(unique_xcols) / 2))
        
        for i, xcol in enumerate(unique_xcols):
            ax = fig.add_subplot(len(unique_xcols), 1, i + 1)
            axes_list.append(ax)
            axes_dict[xcol] = ax
            
            metric_colors = {}
            color_idx = 0
            alpha_idx = 0
            
            # Plot each metric separately on the same axes with different colors
            for mname in unique_metrics:
                if xcol in filtered_results and mname in filtered_results[xcol]:
                    data = filtered_results[xcol][mname]
                    box_stats = data['box_stats']
                    x_centres = data['x_centre']
                    x_width = data['x_width']
                    
                    # Get color and alpha for this metric
                    color = colors[color_idx % len(colors)]
                    alpha = alphas[alpha_idx % len(alphas)]
                    metric_colors[mname] = (color, alpha)
                    
                    # Convert color to RGBA tuple if it's a string or list
                    try:
                        rgba_color = list(to_rgba(color))
                    except (ValueError, AttributeError):
                        # Assume it's already a list like [r, g, b] or [r, g, b, a]
                        if len(color) == 3:
                            rgba_color = list(color) + [1.0]
                        else:
                            rgba_color = list(color)
                    
                    # Set box face and edge colors
                    box_fc = rgba_color.copy()
                    box_fc[-1] = alpha  # Set alpha for face color
                    box_ec = rgba_color.copy()
                    box_ec[-1] = 1.0  # Edge always opaque
                    
                    # Define properties for this metric
                    boxprops = {'fc': tuple(box_fc), 'ec': tuple(box_ec), 'lw': 1.5}
                    medianprops = {'c': tuple(rgba_color[:3] + [1.0]), 'lw': 2.0}
                    meanprops = {'marker': 'D', 'mfc': tuple(rgba_color[:3] + [1.0]), 
                                'mec': tuple(rgba_color[:3] + [1.0]), 'markersize': 5}
                    
                    # Plot this metric's boxes
                    ax.bxp(box_stats, positions=x_centres, widths=x_width * 0.6,
                          boxprops=boxprops, medianprops=medianprops, meanprops=meanprops,
                          showmeans=showmeans, showfliers=showfliers, patch_artist=True)
                    
                    color_idx += 1
                    alpha_idx += 1
            
            ax.set_title(f'{xcol}')
            ax.set_xlabel('Bin Center')
            ax.set_ylabel('Metric Value')
            
            # Format x-axis for readability
            _format_x_axis(ax, x_centres, x_width)
            
            # Add legend if multiple metrics
            if len(unique_metrics) > 1:
                legend_elements = [
                    plt.Line2D([0], [0], color=metric_colors[mname][0], lw=4, 
                             label=mname, alpha=metric_colors[mname][1])
                    for mname in unique_metrics if mname in metric_colors
                ]
                ax.legend(handles=legend_elements, loc='best')
    
    elif separate_by == 'metric_name':
        if fig is None:
            fig = plt.figure(figsize=(figsize[0] * len(unique_metrics) / 2, figsize[1]))
        
        for i, mname in enumerate(unique_metrics):
            ax = fig.add_subplot(1, len(unique_metrics), i + 1)
            axes_list.append(ax)
            axes_dict[mname] = ax
            
            xcol_colors = {}
            color_idx = 0
            alpha_idx = 0
            
            # Plot each x_col separately on the same axes with different colors
            for xcol in unique_xcols:
                if xcol in filtered_results and mname in filtered_results[xcol]:
                    data = filtered_results[xcol][mname]
                    box_stats = data['box_stats']
                    x_centres = data['x_centre']
                    x_width = data['x_width']
                    
                    # Get color and alpha for this x_col
                    color = colors[color_idx % len(colors)]
                    alpha = alphas[alpha_idx % len(alphas)]
                    xcol_colors[xcol] = (color, alpha)
                    
                    # Convert color to RGBA tuple if it's a string or list
                    try:
                        rgba_color = list(to_rgba(color))
                    except (ValueError, AttributeError):
                        # Assume it's already a list like [r, g, b] or [r, g, b, a]
                        if len(color) == 3:
                            rgba_color = list(color) + [1.0]
                        else:
                            rgba_color = list(color)
                    
                    # Set box face and edge colors
                    box_fc = rgba_color.copy()
                    box_fc[-1] = alpha  # Set alpha for face color
                    box_ec = rgba_color.copy()
                    box_ec[-1] = 1.0  # Edge always opaque
                    
                    # Define properties for this x_col
                    boxprops = {'fc': tuple(box_fc), 'ec': tuple(box_ec), 'lw': 1.5}
                    medianprops = {'c': tuple(rgba_color[:3] + [1.0]), 'lw': 2.0}
                    meanprops = {'marker': 'D', 'mfc': tuple(rgba_color[:3] + [1.0]), 
                                'mec': tuple(rgba_color[:3] + [1.0]), 'markersize': 5}
                    
                    # Plot this x_col's boxes
                    ax.bxp(box_stats, positions=x_centres, widths=x_width * 0.6,
                          boxprops=boxprops, medianprops=medianprops, meanprops=meanprops,
                          showmeans=showmeans, showfliers=showfliers, patch_artist=True)
                    
                    color_idx += 1
                    alpha_idx += 1
            
            ax.set_title(f'{mname}')
            ax.set_xlabel('Bin Center')
            ax.set_ylabel('Metric Value')
            
            # Format x-axis for readability (use data from last plotted x_col)
            if len(unique_xcols) > 0 and xcol in filtered_results and mname in filtered_results[xcol]:
                data = filtered_results[xcol][mname]
                _format_x_axis(ax, data['x_centre'], data['x_width'])
            
            # Add legend if multiple x_cols
            if len(unique_xcols) > 1:
                legend_elements = [
                    plt.Line2D([0], [0], color=xcol_colors[xcol][0], lw=4,
                             label=xcol, alpha=xcol_colors[xcol][1])
                    for xcol in unique_xcols if xcol in xcol_colors
                ]
                ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    return {
        'fig': fig,
        'axes': axes_dict,
        'axes_flat': axes_list
    }


def _plot_single_boxplot(data, ax, colors, alphas, showmeans, showfliers, title=None):
    """
    Plot a single boxplot on given axes.
    
    Parameters
    ----------
    data : dict
        Single metric data with keys 'box_stats', 'x_centre', 'x_width', 'x_edge'
    ax : matplotlib.axes.Axes
        Axes to plot on
    colors : list
        Colors to cycle through
    alphas : list
        Alphas to cycle through
    showmeans : bool
        Whether to show means
    showfliers : bool
        Whether to show fliers
    title : str, optional
        Title for the plot
    """
    box_stats = data['box_stats']
    x_centres = data['x_centre']
    x_width = data['x_width']
    
    # Add color and alpha to each box stat
    color_cycle = cycle(colors)
    alpha_cycle = cycle(alphas)
    
    for i, bstat in enumerate(box_stats):
        bstat_copy = bstat.copy()
        color = next(color_cycle)
        alpha = next(alpha_cycle)
        
        # Convert color to RGBA tuple
        try:
            rgba_color = list(to_rgba(color))
        except (ValueError, AttributeError):
            if len(color) == 3:
                rgba_color = list(color) + [1.0]
            else:
                rgba_color = list(color)
        
        # Set box face and edge colors
        box_fc = rgba_color.copy()
        box_fc[-1] = alpha  # Set alpha for face color
        box_ec = rgba_color.copy()
        box_ec[-1] = 1.0  # Edge always opaque
        
        bstat_copy['facecolor'] = tuple(box_fc)
        bstat_copy['edgecolor'] = tuple(box_ec)
        bstat_copy['linewidth'] = 1.5
        
        box_stats[i] = bstat_copy
    
    # Define properties using matplotlib boxprops/medianprops/meanprops
    boxprops = {'lw': 1.5}
    medianprops = {'lw': 2.0}
    meanprops = {'marker': 'D', 'markersize': 5}
    
    ax.bxp(box_stats, positions=x_centres, widths=x_width * 0.6,
           boxprops=boxprops, medianprops=medianprops, meanprops=meanprops,
           showmeans=showmeans, showfliers=showfliers, patch_artist=True)
    
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    if title:
        ax.set_title(title)
