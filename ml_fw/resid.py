# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:28:59 2024

@author: krmurph1


ivestigate residuals of a ml model

"""

import pandas as pd
import numpy as np
from scipy import stats



def boxplot_vx(x_dat: pd.DataFrame | list, 
               y_dat: pd.DataFrame | list,
               box_dat: pd.DataFrame = None,
               box_meth: bool | dict = True,
               bins: int | list = 10,
               xrange: list[tuple[float, float]] | None = None, 
               whisker: float = 1.5):
    
    # x dat - dataframe of x data to bin by or list for box_dat
    # y dat - dataframe of y data to calculate stats on or list for box_dat
    # box_dat - dataframe of data
    # box method - how the boxes will be calculated
    
    
    # if both f_dat and y_dat are lists then they contain
    # the column names of the data which we are generating
    # the box plot vx for
    
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
    
            

        
        
    # now we want to itterate over the x columns
    # and generate the box plot as binned statistic
    
    # define lambdas to calculate the upper and lower
    # quartiles
    lq_nan = lambda stat: np.nanpercentile(stat, 25)
    uq_nan = lambda stat: np.nanpercentile(stat, 75)
       
    box_idx = {}
    
    for idx in x_col:
        print(idx)
        # calculate the statistics as a function of idx
        try:
            x = x_val[idx].to_numpy().squeeze()
        except:
            x = x_val.to_numpy().squeeze()
            
        try:
            y = y_val[y_col].to_numpy().squeeze()
        except:
            x = x_val.to_numpy().squeeze()
        
        mean, x_edge, _ = stats.binned_statistic(x, y, bins=bins, 
                                                 range=xrange, 
                                                 statistic=np.nanmean)
        median, _, _ = stats.binned_statistic(x, y,bins=bins, 
                                              range=xrange, 
                                              statistic=np.nanmedian)
        low_q, _, _ = stats.binned_statistic(x, y, bins=bins, 
                                             range=xrange, statistic=lq_nan)
        up_q, _, _ = stats.binned_statistic(x, y, bins=bins, 
                                            range=xrange, statistic=uq_nan)
        
        x_cen = (x_edge[0:-1]+[x_edge[1:]])/2.
        x_cen = x_cen.squeeze()
        x_wid = x_edge[1]-x_edge[0]
        # create a list to hold the required
        # parameters to draw a box and whisker plot
        box_stats = [ ]
        for mn, md, lq, uq, in zip(mean, median, low_q, up_q):
            val = {
                "mean":  mn,  # not required
                "med": md,
                "q1": lq,
                "q3": uq,
                "whislo": lq - whisker*(uq-lq),  # required
                "whishi": uq + whisker*(uq-lq),  # required
                "fliers": []  # required if showfliers=True
                }
            box_stats.append(val)
        
        # create a dictionary to store everything needed for plotting
        # and add it to the return dictionary
        box_idx[idx] = {'box_stats':box_stats, 'x_edge':x_edge, 
                        'x_centre':x_cen, 'x_width':x_wid}
    
    return box_idx