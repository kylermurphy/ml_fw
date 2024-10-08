# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:16:09 2024

@author: murph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_fw import inspect as insp
import sklearn.metrics as metrics

fn = "C:/Users/murph/OneDrive/Phys4501/data/RF_FI_GEO_OOS_CHAMP.hdf5"
dat = pd.read_hdf(fn)

y_true = '400kmDensity'
y_pred = '400kmDensity_pred'
on = 'DateTime'
rkwargs = {'window':'90min','center':False}
met = {'hi':lambda y_true, y_pred: metrics.median_absolute_error(y_true, y_pred),
       'hizzle':lambda y_true, y_pred: metrics.mean_absolute_error(y_true, y_pred)}

sdate = '2005-08-21'
edate = '2005-08-30'

gd = (dat['DateTime'] > sdate) & (dat['DateTime'] <= edate)

rdat = dat[gd].copy()

#met_df = insp.rolling_met(rdat,y_true=y_true,y_pred=y_pred,on=on,
#                         roll_metric=met, roll_kwargs=rkwargs)


path_dat = "D:\data\Full_Model_Outputs.hdf5"
dd = pd.read_hdf(path_dat)
y_t = 'True Class'
y_p = 'Prediction'
on = 'index'
rkwargs = {'window':11,'center':True}

r_met = insp.rolling_met(dd,y_true=y_t,y_pred=y_p,on=on,
                        roll_kwargs=rkwargs)