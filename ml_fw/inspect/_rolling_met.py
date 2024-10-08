# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:16:51 2024

@author: murph
"""

import pandas as pd
import numpy as np
from sklearn import metrics

from pandas.api.types import is_datetime64_any_dtype as is_datetime



def rolling_met(met_dat: pd.DataFrame,
                y_true: str = 'y_true',
                y_pred: str = 'y_pred',
                on: str = 'DateTime', 
                roll_kwargs: dict = None,
                roll_metric: list | dict = None):
    
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
                            on='index'
                        else:
                            on= met_dat.index.name
                        
                        rdat = met_dat.reset_index()
                    
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
    if np.issubdtype(rdat[y_true].dtype,np.integer) \
        and np.issubdtype(rdat[y_pred].dtype,np.integer) \
            and not roll_metric:
                print('Using Accuracy Metric')
                met = lambda true, pred: metrics.accuracy_score(true, pred)
                met_d = {'Accuracy':met}
    elif not roll_metric:
        print('Using Mean Square Error Metric')
        met = lambda true, pred: metrics.mean_squared_error(true, pred)
        met_d = {'MSE':met}
    else:
        print('Using passed metric')
        met_d = roll_metric
    
    # create a dictionary to loop over
    # for calculating metrics
    if isinstance(met_d, list) \
        and not isinstance(met_d, dict):
            met_d = dict()
            met_c = 0
            for lv in met_d:
                met_d[f'Metric {met_c:02}'] = lv
                met_c = met_c+1
    elif not isinstance(met_d, dict):
        met_d = {'Metric':met}
    
        
    # define the rolling window to compute the metric
    roll = rdat.set_index(on).rolling(**roll_kwargs)
    
    rmet = np.array([
        [
        mv(rdat.set_index(on).loc[l.index,y_true],
            rdat.set_index(on).loc[l.index,y_pred]) 
        for mk, mv in met_d.items()
        ] 
        for l in roll
        ])
     
    # use the rolling to get and index for the returned
    # metric. this is needed in case step is used in the
    # rolling kwargs
    rind = rdat.set_index(on).rolling(**roll_kwargs).mean().index
    
    rdf = pd.DataFrame(data=rmet,columns=met_d.keys())
    rdf[on] = rind
    
    #rdf = pd.DataFrame({on:rind,'Metric':rmet})
        
    return rdf
    
            
        
                