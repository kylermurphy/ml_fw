# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:44:50 2024.

@author: krmurph1
"""


import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics




def boxplot_metvx(x_dat: pd.DataFrame | list, 
                  y_true: pd.DataFrame | list,
                  y_mod: pd.DataFrame | list,
                  box_dat: pd.DataFrame = None, 
                  box_metric = None,
                  kfolds: int = 1000,
                  kfrac: int = 0.5, 
                  bins: int | list = 10,
                  xrange: list[tuple[float, float]] | None = None, 
                  whisker: float = 1.5):
    """Calculate boxplot like statistics of a metric (using y-t and y-p) vs x.
    
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
        The default is 1000.
        
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
    if isinstance(box_dat, pd.DataFrame) \
        and isinstance(x_dat, list) \
           and isinstance(y_true, list) \
               and isinstance(y_mod, list):
                   
                   x_d = box_dat[x_dat]
                   x_c = x_dat.copy()
                   y_t = box_dat[y_true].to_numpy().squeeze()
                   y_p = box_dat[y_mod].to_numpy().squeeze()
    
    elif isinstance(x_dat, (pd.DataFrame, pd.Series)) \
            and isinstance(y_true, (pd.DataFrame, pd.Series)) \
                and isinstance(y_mod, (pd.DataFrame, pd.Series)):

                    x_d = x_dat
                    x_c = x_dat.columns()                    
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
                met = lambda y_true, y_pred: metrics.accuracy_score(y_true, 
                                                                  y_pred)  
    elif not box_metric:
        print('Using Mean Square Error Metric')
        met = lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred)
    else:
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
        except:
            x = x_d.to_numpy().squeeze()
        
        # use bin statistic to get the indices of the x data
        # for all the bins with which the data is binned into
        # this can then be used to subsquently bin the metric
        x_stat, x_edges, x_bnum = stats.binned_statistic(x,x, bins=bn,range=xr)

        x_cen = (x_edges[0:-1]+[x_edges[1:]])/2.
        x_cen = x_cen.squeeze()
        x_wid = x_edges[1]-x_edges[0]
        
        # calculate the box stats for this x        
        box_stats = [ ]
        
        for i in np.arange(x_stat.size, dtype=int):
            # get the indices for values which lie between 
            # bin[i] and bin[i+1]
            gd = x_bnum == i+1
            # create an array of k-fold samples which
            # holds metric values from each sample which
            # box stats can be computed from
            sval = np.array([
                met(y_d.loc[gd,'tr'].sample(frac=kfrac,random_state=x),
                    y_d.loc[gd,'pr'].sample(frac=kfrac,random_state=x))
                for x in np.arange(kfolds)              
                ])
            
            lq = np.nanpercentile(sval,25)
            uq = np.nanpercentile(sval,75)
            
            bval = {
                "mean": np.nanmean(sval),  # not required
                "med": np.nanmedian(sval),
                "q1": lq,
                "q3": uq,
                "whislo": lq - whisker*(uq-lq),  # required
                "whishi": uq + whisker*(uq-lq),  # required
                "fliers": []  # required if showfliers=True
                }
            # append box to list
            box_stats.append(bval) 
        
        # add box values to box dictionary
        box_idx[idx] = {'box_stats':box_stats, 'x_edge':x_edges, 
                        'x_centre':x_cen, 'x_width':x_wid}
    
    
    return box_idx







  