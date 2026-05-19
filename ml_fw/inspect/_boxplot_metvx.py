# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:44:50 2024.

@author: krmurph1
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

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
    
    The data is binned as a function of x. For each bin in x a fraction of
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

    Parameters
    ----------
    x_dat : pd.DataFrame | list
        Data for binning. If list, contains column names to extract from box_dat.

        A pandas DataFrame containing the data for binning. The metric
        (calculated from y-true and y-mod) is then binned and used to calculate
        statistics and derive box and whisker values for each bin.

        The DataFrame can have more the one column. Each column will be binned
        (see bins) and box/whisker data returned.

        If x_dat is a list then the list contains the column names which
        correspond to the binning data in box_dat DataFrame.

    y_true : pd.DataFrame | list
        True labels/values.

        A pandas DataFrame containing the true data which will be use to
        calculte the metric.

        If y_dat is a list it contains the column name for box_dat that
        contains the true data for calculating the metric.

        y_dat should be a single columnd DataFrame or single valued list.

    y_mod : pd.DataFrame | list
        Model predictions.

        A pandas DataFrame containing the model data which will be use to
        calculte the metric.

        If y_mod is a list it contains the column name for box_dat that
        contains the model data used to calculate the metric.

        y_mod should be a single columnd DataFrame or single valued list.

    box_dat : pd.DataFrame, optional
        Combined DataFrame (required if x_dat, y_true, y_mod are lists).

        A pandas DataFrame continaing the x-data and y-true and y-mod which are
        used to calculate a metric. The metric is then binned as a fucntion of
        the x-data and subsequently used to derive box/whisker values..

        If provided, x_dat, y_true, and y_mod should be lists specifying the
        column names of the x-data for binning and the y-true and y-mod data
        for calculating metric values.

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

        The lower and upper range of the bins. If not provided, range is simply
        (x.min(), x.max()). Values outside the range are ignored.

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
