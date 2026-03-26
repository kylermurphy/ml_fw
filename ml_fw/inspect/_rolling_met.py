# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:16:51 2024.

@author: murph
"""

import numpy as np
import pandas as pd
from sklearn import metrics

from pandas.api.types import is_datetime64_any_dtype as is_datetime


def rolling_met(met_dat: pd.DataFrame,
                y_true: str = 'y_true',
                y_pred: str = 'y_pred',
                on: str = 'DateTime',
                roll_kwargs: dict = None,
                roll_metric: list | dict = None) -> pd.DataFrame:
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
        A DataFrame containg the values of a rolling metric for each passed
        callable/metric and the index/DateTime values for each value.
        If roll_metric is a list the names of the column are 'Metric XX' where
        xx is the list element.
        If roll_metric is dictionary the names of the columns are the
        dictionary keys.
    """
    # if strings are passed and met_dat DateFrame
    # then get only the data we need
    if isinstance(y_true, str) \
            and isinstance(y_pred,str) \
            and isinstance(on,str) \
            and isinstance(met_dat, pd.DataFrame):
        # check if the rolling is happening
        # on the index
        # if so reset the index and get the
        # index name
        if on.upper() == 'INDEX':
            if not met_dat.index.name:
                on = 'index'
            else:
                on = met_dat.index.name
            met_dat = met_dat.reset_index()
        # create a dataframe of only the
        # columns we need
        rdat = met_dat[[on,y_true,y_pred]].copy()
    # if 'on' is a DateTime like object then set some default values
    # for rolling, else assume rolling on int/float like value
    if is_datetime(rdat[on]) and not roll_kwargs:
        roll_kwargs = {'window':'60min', 'center':True}
    elif not roll_kwargs:
        roll_kwargs = {'window':10, 'center':True}
    # make sure window is in the roll_kwa
    if 'window' not in roll_kwargs:
        if is_datetime(rdat[on]):
            roll_kwargs['window'] = '60min'
        else:
            roll_kwargs['window'] = 10
    # define a metric to use if box_metric is None
    # if the true and modeled y-values are integers
    # assume we have a categorical model
    # else assume it is a regression model
    if roll_metric is None:
        if np.issubdtype(rdat[y_true].dtype, np.integer):
            met_d = {'Accuracy': lambda tr, pr: metrics.accuracy_score(tr, pr)}
        else:
            met_d = {'MSE': lambda tr, pr: metrics.mean_squared_error(tr, pr)}
    elif isinstance(roll_metric, list):
        met_d = {f'Metric {i:02}': m for i, m in enumerate(roll_metric)}
    elif isinstance(roll_metric, dict):
        met_d = roll_metric
    else:
        met_d = {'Metric': roll_metric}

    # create a dictionary to loop over
    # for calculating metrics
    if isinstance(met_d, list) and not \
       isinstance(met_d, dict):
        met_list = met_d
        met_d = dict()
        for i, lv in enumerate(met_list):
            met_d[f'Metric {i:02}'] = lv
    elif not isinstance(met_d, dict):
        met_d = {'Metric':met}
    # define the rolling window to compute the metric
    rdat_indexed = rdat.set_index(on)
    roll = rdat_indexed.rolling(**roll_kwargs)
    
    
    results = []

    for window in roll:
        idx = window.index

        y_t = rdat_indexed.loc[idx, y_true]
        y_p = rdat_indexed.loc[idx, y_pred]

        row = [metric(y_t, y_p) for metric in met_d.values()]
        results.append(row)

    rmet = np.array(results)

    # use the rolling to get and index for the returned
    # metric. this is needed in case step is used in the
    # rolling kwargs
    rind = rdat_indexed.rolling(**roll_kwargs).mean().index
    rdf = pd.DataFrame(data=rmet,columns=met_d.keys())
    rdf[on] = rind
    # rdf = pd.DataFrame({on:rind,'Metric':rmet})
    return rdf
