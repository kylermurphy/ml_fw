# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:03:38 2024

@author: krmurph1
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import data_io as dio
from ml_fw import profile as pro
import ml_mod as ml


from sklearn.ensemble import HistGradientBoostingRegressor as hgbr


# target data
td = "C:\data\SatDensities\satdrag_database_grace_B_reduced_feature_v3.hdf5"

# columns
df_cols = {
    'feat_col': ['2500_03', '43000_09', '85550_13','irr_1216',
           'B', 'alt', 'lat'],
    'y_col': ['dens_x'],
    't_col': ['DateTime'],
    'lt_col': ['lon'],
    }

df_cols = {
    'feat_col': ['AE','lat'],
    'y_col': ['dens_x'],
    't_col': ['DateTime'],
    'lt_col': None,
    }

f_df = pd.read_hdf(td)


f_dat, y_dat = dio.create(f_df,**df_cols)
y_dat = y_dat*10**12

###
# Test profiling
###

# x = pro.cor_matrix(f_dat.drop(columns='DateTime'),y_dat)

# y = pro.cor_matrix(f_dat=['B','AE','SYM_H index',], 
#                    y_dat=['dens_x','dens_mean'], cor_dat=f_df)

z = pro.cor_matrix(f_dat=['B','AE','SYM_H index','storm'], 
                    y_dat=['dens_x','dens_mean'],
                    cor_dat=f_df, 
                    cat_dat=['storm'])

ae_f = lambda x: x['AE'] > 500
sym_h = lambda x: x['SYM_H index'] < -50
storm = lambda x: x['storm'] == 1
w = pro.cor_matrix(f_dat=['B','AE','SYM_H index','storm'], 
                   y_dat=['dens_x','dens_mean'],
                   cor_dat=f_df, 
                   cat_dat={'storm walla walla':storm, 'storm':'storm'})

plt.figure(figsize=(8, 8))
sns.heatmap(w[0:-1].abs(),annot=False, fmt='.2f', cbar_kws={'label':'Abs Correlation - abs(r)'})
plt.yticks() 
plt.show()


###
# Test fitting
###
# grid_params = dict(
#     learning_rate=[0.05, 0.1, 0.2],
#     max_depth=[2, 5, 10, None],
#     max_iter=[100,300,500,750,1000],
#     min_samples_leaf=[1, 5, 10, 20])

# grid_params = dict(
#     learning_rate=[0.05,  0.2],
#     max_depth=[10, None],
#     max_iter=[750,1000],
#     min_samples_leaf=[10, 20])

# gridcv_k = dict(cv=3, 
#                 verbose=4,
#                 scoring=['neg_mean_absolute_error','r2'], 
#                 n_jobs=6, 
#                 return_train_score=True,
#                 refit=False)

# est = gbr_ls = hgbr(loss="squared_error")
                                             

# model = ml.train(f_dat.drop(columns='DateTime'),
#                  y_dat,
#                  est, 
#                  grid_params=grid_params, 
#                  grid_kwargs=gridcv_k)
