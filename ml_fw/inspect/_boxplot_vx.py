# -*- coding: utf-8 -*-
"""

should rename this inspecting as it's not just residuals.

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
               whisker: float = 1.5) -> dict:
    """
    Calculate boxplot like statistics of y as a function of x.
    
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
        except:
            x = x_val.to_numpy().squeeze()
            
        try:
            y = y_val[y_col].to_numpy().squeeze()
        except:
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