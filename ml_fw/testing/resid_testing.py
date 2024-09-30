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

srange = [-200,10, 4]
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

fig, ax = plt.subplots(1,1, figsize=(8,6))

ae_box = box_dat['AE']
y_val = ae_box['box_stats']
x_val = ae_box['x_centre']
x_width = ae_box['x_width']

# use the matplotlib boxplot to plot 
b1 = ax.bxp(y_val, positions=x_val, widths=x_width, 
                  patch_artist=True, showmeans=showmean, 
                  shownotches=False, showcaps=False)

ax.set_xticks(x_val,x_val.astype(int).astype(str),rotation=45)

# make the plot prettier
color = (1,0,0,0.5)
for patch, l_md, l_mn in zip(b1['boxes'], b1['medians'], b1['means']):
    # change patch color
    patch.set_facecolor([1,0,0,0.25])
    patch.set_edgecolor([1,0,0,1])
    #patch.set_alpha(0.25)
    # change median and mean color
    l_md.set_color(color)
    l_md.set_linewidth(2)
    l_md.set_linestyle('solid')
    l_mn.set_markerfacecolor(color)
    l_mn.set_markeredgecolor(color)


#box_stats, x_edge, x_cen, x_wid
