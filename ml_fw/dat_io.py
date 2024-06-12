# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:49:02 2024

@author: krmurph1
"""

import pandas as pd
import numpy as np

def create(dat: pd.DataFrame,
               feat_col: list = None,
               y_col: list = None,
               log_col: list = None,
               lt_col: list = None,
               t_col: list = None) -> pd.DataFrame:
    """
    
    Create a feature (x_dat) and dependent (y_dat) data set
    for use in SciKit Learn models

    Parameters
    ----------
    dat : pd.DataFrame
        A pandas dataframe containing all the data.
    feat_col : list, optional
        Columns to add to the feature data. 
        The default is None.
    y_col : list, optional
        Columns to add to the dependent.
        The default is None.
    log_col : list, optional
        Columns from the passed DataFrame that
        should be logged. In Heliophysics variables
        often span several order of magnitude and may
        require the data to be logged (semi-normalized).
        The default is None.
    lt_col : list, optional
        Columns from the passed DataFrame that 
        should be converted to cyclical arguments.
        For example longitude, local time, or
        magnetic local time.
        The default is None.
    t_col : list, optional
        If looking at time-series data the time
        column that the user would like to keep.
        The default is None.

    Raises
    ------
    TypeError
        If feat_col and y_col are not lists.
        feat_col and y_col are both required 
        keyword arguments

    Returns
    -------
    x_dat : pd.DataFrame
        The feature data.
    y_dat : pd.DataFrame
        The dependent data.

    """
    
    # create a list of feature and y data columns 
    # independent and dependent data
    if not isinstance(feat_col,list) or \
       not isinstance(y_col,list):
           raise TypeError('feat_col and y_col must both be lists')
           
    dat_col = feat_col+y_col
    
    # if a time column was passed add it to the list
    if isinstance(t_col,list):
        dat_col = dat_col+t_col
    
    # creaate a new dataframe from the columns of the
    # passed dataframe
    x_dat = dat[dat_col].dropna().copy()

    # from the feature columns log
    # a subset of passed columns and
    # drop the non-logged columns
    if isinstance(log_col,list):
       for i in log_col:
            try:
                x_dat[f'log10_{i}'] = np.log10(x_dat[i])
            except:
                print(f'Could not log column {i}')
    
    # from a subset of columns convert local time
    # or longitude columns to cyclical variables using
    # cos and sin so that there are no discontinuites
    # across 360->0 or 24h->0h
    if isinstance(lt_col,list):
        for i in lt_col:
            try:
                if dat[i].max() > 24:
                    x_dat[f'cos_{i}'] = np.cos(dat[i]*2*np.pi/360.)
                    x_dat[f'sin_{i}'] = np.sin(dat[i]*2*np.pi/360.)
                else:
                    x_dat[f'cos_{i}'] = np.cos(dat[i]*2*np.pi/24.)
                    x_dat[f'sin_{i}'] = np.sin(dat[i]*2*np.pi/24.)    
            except:
                print(f'Could not add {i} as a cos/sin time column')
    
    # drop values which are undefined
    # check for numpy undefinde
    # then use pandas dropna()
    x_dat = x_dat[~x_dat.isin([np.nan, np.inf, -np.inf]).any(axis=1)].dropna()
    
    # get y data from the final dataframe
    # and drop it from the feature data
    y_dat = x_dat[y_col].copy()
    x_dat = x_dat.drop(columns=y_col)    
    
    return x_dat, y_dat
    
    
def feat_shift(s_dat: pd.DataFrame, 
               t_col='DateTime',
               periods: list[int] = [5],
               unit: str = 'min',
               drop_orig: bool = False) -> pd.DataFrame:

    # if the time is the index
    # reset the index so time is a column
    if t_col == 'index' or t_col == 0:
        s_dat.reset_index('DateTime')
        t_col = 'DateTime'
    
    if isinstance(periods,str):
        periods = [periods]
    elif not isinstance(periods,list):
        raise TypeError('periods should be a string or a list')
        
    # check if our time column exists
    try:
        t_dat = s_dat[t_col]
    except:
        print(f'{t_col} cannot be found')
        raise KeyError(t_col)

    #get the data columns
    d_col = s_dat.columns.to_list().remove(t_col)
    # get the data
    r_dat = s_dat.copy().drop(axis=1,columns=t_col)
    # begin shifting the data
    for i in periods:
        r_dat[t_col] = t_dat+pd.Timedelta(i,unit=unit)
        r_suf = f' {i} {unit}'
        s_dat = pd.merge_asof(s_dat,r_dat, on=t_col,
                              direction='nearest',
                              suffixes = ('',r_suf)
                              )
        # drop the new time column as it's not needed
        #s_dat = s_dat.drop(axis=1,columns=t_col+r_suf)

    
    return s_dat
    
    
    
    
    
               