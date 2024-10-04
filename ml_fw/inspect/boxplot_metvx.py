# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:44:50 2024

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
                  box_metric: str = None,
                  kfolds: int = 1000,
                  kfrac: int = 0.5, 
                  bins: int | list = 10,
                  xrange: list[tuple[float, float]] | None = None, 
                  whisker: float = 1.5):

        
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







  