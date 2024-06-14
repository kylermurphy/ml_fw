# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:03:38 2024

@author: krmurph1
"""

import pandas as pd

import dat_io as dio
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

grid_params = dict(
    learning_rate=[0.05, 0.1, 0.2],
    max_depth=[2, 5, 10, None],
    max_iter=[100,300,500,750,1000],
    min_samples_leaf=[1, 5, 10, 20])

grid_params = dict(
    learning_rate=[0.05,  0.2],
    max_depth=[10, None],
    max_iter=[750,1000],
    min_samples_leaf=[10, 20])

gridcv_k = dict(cv=3, 
                verbose=4,
                scoring='neg_mean_absolute_error', 
                n_jobs=6, 
                return_train_score=True,
                refit=True)

est = gbr_ls = hgbr(loss="squared_error")
                                             

model = ml.train(f_dat.drop(columns='DateTime'),
                 y_dat,
                 est, 
                 grid_params=grid_params, 
                 grid_kwargs=gridcv_k)
