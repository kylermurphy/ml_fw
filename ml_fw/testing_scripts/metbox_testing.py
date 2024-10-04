# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:17:11 2024

@author: krmurph1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_fw import inspect as insp

dat = pd.read_hdf("C:\data\Full_Model_Outputs.hdf5")

# k folds parameters
k_folds = 100
k_size = 0.5

# whisker size
whisker = 1.5

# x data
x_col = ['SYM_H_mean']
x_ran = [-75,25]
x_sz = 25
x_bins = (x_ran[1]-x_ran[0])/x_sz

# y data
y_true = ['True Class']
y_pred = ['Prediction']

box_dat = insp.boxplot_metvx(x_col,y_true,y_pred,box_dat=dat,
                             bins=x_bins, xrange=x_ran, kfolds=k_folds)


# plot the data
showmean=True
fig, ax = plt.subplots(1, figsize=(8,6))

p_dat = box_dat[x_col[0]]
y_val = p_dat['box_stats'] # a list of dictionaries for each box/whisker instance
x_val = p_dat['x_centre'] # the center of each x bin
x_width = p_dat['x_width'] # the width of each x bin

# lets define some colors, alpha values (transparencies) and other properties to
# make the plot pretty 

cc = [1,0,0] # red box plot
bx_a = 0.25 # transparency level (alpha) for box
ln_a = 1.0 # transparency level for lines
ln_w = 2.0 # line width

b1 = ax.bxp(y_val, positions=x_val, widths=x_width, 
                  patch_artist=True, showmeans=showmean, 
                  shownotches=False, showcaps=False, 
                  boxprops={'ec':cc+[ln_a], 'fc':cc+[bx_a]}, # artist properties for boxes
                  medianprops={'c':cc, 'lw':ln_w}, # artist properties for medians
                  meanprops={'mec':cc, 'mfc':cc}) # artist propoerties for means


ax.set_ylabel('Accuracy')
ax.set_xlabel(x_col)