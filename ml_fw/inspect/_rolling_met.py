# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:16:51 2024.

@author: murph
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from collections.abc import Callable, Sequence

from pandas.api.types import is_datetime64_any_dtype as is_datetime


def _prepare_rolling_data(
        met_dat: pd.DataFrame,
        y_true: str,
        y_pred: str,
        on: str) -> tuple[pd.DataFrame, str]:
    """Validate inputs and return just the rolling columns.

    This helper supports on='index' by resetting the index before selecting
    the requested columns.
    """
    if not isinstance(met_dat, pd.DataFrame):
        raise TypeError('met_dat must be a pandas DataFrame')

    if not all(isinstance(label, str) for label in (y_true, y_pred, on)):
        raise TypeError('y_true, y_pred, and on must all be strings')

    if on.upper() == 'INDEX':
        if met_dat.index.name is None:
            on = 'index'
        else:
            on = met_dat.index.name
        met_dat = met_dat.reset_index()

    missing = [col for col in (on, y_true, y_pred) if col not in met_dat.columns]
    if missing:
        raise KeyError(f'Missing required columns in met_dat: {missing}')

    return met_dat[[on, y_true, y_pred]].copy(), on


def _infer_roll_kwargs(on_series: pd.Series, roll_kwargs: dict | None) -> dict:
    """Choose sensible rolling defaults and preserve user values."""
    if roll_kwargs is None:
        roll_kwargs = {
            'window': '60min' if is_datetime(on_series) else 10,
            'center': True,
        }
    else:
        roll_kwargs = dict(roll_kwargs)
        if 'window' not in roll_kwargs:
            roll_kwargs['window'] = '60min' if is_datetime(on_series) else 10

    return roll_kwargs


def _normalize_roll_metric(
        roll_metric: Sequence[Callable] | dict | Callable | None,
        y_true_series: pd.Series) -> dict[str, Callable[[pd.Series, pd.Series], float]]:
    """Normalize roll_metric into a consistent name -> callable mapping.
    
    If roll_metric is None, choose a default based on the type of y_true 
    (integer->classification, else regression). 
    
    If it's a single callable, wrap it in a dict. 
    If it's a list, create default names. 
    If it's already a dict, use it as is.
    """
    if roll_metric is None:
        if np.issubdtype(y_true_series.dtype, np.integer):
            return {'Accuracy': lambda tr, pr: metrics.accuracy_score(tr, pr)}
        return {'MSE': lambda tr, pr: metrics.mean_squared_error(tr, pr)}

    if isinstance(roll_metric, dict):
        return roll_metric

    if isinstance(roll_metric, (list, tuple)):
        return {f'Metric {i:02}': metric for i, metric in enumerate(roll_metric)}

    if callable(roll_metric):
        return {'Metric': roll_metric}

    raise TypeError('roll_metric must be a callable, list/tuple of callables, ' \
    'dict, or None')


def _compute_rolling_metrics(
        data: pd.DataFrame,
        on: str,
        y_true: str,
        y_pred: str,
        metrics_map: dict[str, Callable[[pd.Series, pd.Series], float]],
        roll_kwargs: dict) -> pd.DataFrame:
    """Compute rolling metrics efficiently using vectorized operations."""
    # Create rolling object once
    rolling_obj = data.set_index(on).rolling(**roll_kwargs)

    # Initialize results dictionary and compute metrics
    results = {
        metric_name: [
            metric_func(window[y_true], window[y_pred])
            for window in rolling_obj
        ]
        for metric_name, metric_func in metrics_map.items()
    }

    # Create result DataFrame
    result_df = pd.DataFrame(results)
    result_df[on] = rolling_obj.mean().index  # Use the same index as rolling windows

    return result_df


def rolling_met(
        met_dat: pd.DataFrame,
        y_true: str = 'y_true',
        y_pred: str = 'y_pred',
        on: str = 'DateTime',
        roll_kwargs: dict | None = None,
        roll_metric: Sequence[Callable] | dict | Callable | None = None) -> pd.DataFrame:
    """Calculate a rolling metric.

    Parameters
    ----------
    met_dat : pd.DataFrame
        Pandas DataFrame which contains the observed and predicted data as well
        as a index or time series which can be used to create a set of rolling
        windows which are used to derive metrics.

    y_true : str, optional
        The default is 'y_true'.
        The column name which contains the observerd or true values.

    y_pred : str, optional
        The default is 'y_pred'.
        The column name which contains the predicted or modeled values.

    on : str, optional
        The default is 'DateTime'.
        The used to define the rolling windows.
        If on='index' then the index of the met_data DataFrame is used.

    roll_kwargs : dict, optional
        The default is None.
        Keywords arguments to be passed to DataFrame.rolling() method.
        If nothing is passed the 'on' column is checked if it is datetime like
        and if true a 60 minute centered rolling window is used.
        Else a 10 point centred rolling window is used.

    roll_metric : list | dict, optional
        The default is None.
        A callable, list of callables, or dictionary of callables which can
        be used to derive a metric from the true and predicted values.

        The callables can be lambda functions utilizing metrics from
        sklearn.metrics. For example:
            met = lambda tr, pr: sklearn.metrics.accuracy_score(tr,pr)
        The labmdas can also be used to set keyword arguments. For example:
            met = lamba tr, pr: sklearn.metrics.log_loss(tr,pr, normalize=True)

        If roll_mtric is a list each list element should be metric callable.
        If it is a dictionary the key should be the name and value a metric
        callable.

    Returns
    -------
    rdf : pd.DataFrame
        A DataFrame containing the values of a rolling metric for each passed
        callable/metric and the index/DateTime values for each value.
        If roll_metric is a list the names of the column are 'Metric XX' where
        xx is the list element.
        If roll_metric is dictionary the names of the columns are the
        dictionary keys.
    """
    # validate and get data
    rolling_data, on_key = _prepare_rolling_data(met_dat, y_true, y_pred, on)
    # validate and get rolling kwargs
    roll_kwargs = _infer_roll_kwargs(rolling_data[on_key], roll_kwargs)
    # validate and normalize roll_metric
    metrics_map = _normalize_roll_metric(roll_metric, rolling_data[y_true])

    return _compute_rolling_metrics(
        data=rolling_data,
        on=on_key,
        y_true=y_true,
        y_pred=y_pred,
        metrics_map=metrics_map,
        roll_kwargs=roll_kwargs,
    )
