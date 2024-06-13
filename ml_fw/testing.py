# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:03:38 2024

@author: krmurph1
"""

import pandas as pd

import dat_io as dio



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

s_dat, d_col = dio.feat_shift(f_dat,t_col='DateTime',periods=[10],drop_orig=True)


