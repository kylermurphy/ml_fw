# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:44:19 2024

@author: krmurph1
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from ml_fw import resid as resid


showmean=True
whisker=0.0

srange = [-200,10]
sbins = 21
aerange = [0,2000]
aebins=20


fn = "E:\OneDrive\Phys4501\data\RF_FI_GEO_OOS_CHAMP.hdf5"
dat = pd.read_hdf(fn)
dat['resid'] = dat['400kmDensity']-dat['400kmDensity_pred']

x_dat = ['AE','SYM_H index']
y_dat = ['resid']

box_dat = resid.boxplot_vx(dat[x_dat], dat[y_dat],whisker=whisker, 
                           bins=[aebins,sbins], xrange=[aerange,srange])

fig, ax = plt.subplots(3,1, figsize=(8,9))

ae_box = box_dat['AE']
y_val = ae_box['box_stats']
x_val = ae_box['x_centre']
x_width = ae_box['x_width']

# use the matplotlib boxplot to plot

cc = [1,0,0]
bx_a = 0.25
ln_a = 1.0
ln_w = 2.0
 
b1 = ax[0].bxp(y_val, positions=x_val, widths=x_width, 
                  patch_artist=True, showmeans=showmean, 
                  shownotches=False, showcaps=False, 
                  boxprops={'ec':cc+[ln_a], 'fc':cc+[bx_a]},
                  medianprops={'c':cc, 'lw':ln_w}, 
                  meanprops={'mec':cc, 'mfc':cc})

ax[0].set_xticks(x_val,x_val.astype(int).astype(str),rotation=45)
ax[0].set_ylabel('Residuals')
ax[0].set_xlabel('AE')

# loop through the dictionary to plot all box_plots
pc = [cc, [0,0,1]]
for (key, value), ap, bc, in zip(box_dat.items(), ax[1:], pc):
    print(key)
    plt_box = box_dat[key]
    y_val = plt_box['box_stats']
    x_val = plt_box['x_centre']
    x_width = plt_box['x_width']
    
    b = ap.bxp(y_val, positions=x_val, widths=x_width, 
                      patch_artist=True, showmeans=showmean, 
                      shownotches=False, showcaps=False, 
                      boxprops={'ec':bc+[ln_a], 'fc':bc+[bx_a]},
                      medianprops={'c':bc, 'lw':ln_w}, 
                      meanprops={'mec':bc, 'mfc':bc})
    ap.set_xticks(x_val,x_val.astype(int).astype(str),rotation=45)
    ap.set_ylabel('Residuals')
    ap.set_xlabel(key)
