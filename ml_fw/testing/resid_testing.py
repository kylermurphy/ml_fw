# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:44:19 2024

@author: krmurph1
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from ml_fw import resid as resid




fn = "E:\OneDrive\Phys4501\data\RF_FI_GEO_OOS_CHAMP.hdf5"
dat = pd.read_hdf(fn)
dat['resid'] = dat['400kmDensity']-dat['400kmDensity_pred']

x_dat = ['AE','SYM_H index']
y_dat = ['resid']

box_dat = resid.boxplot_vx(dat[x_dat], dat[y_dat])
