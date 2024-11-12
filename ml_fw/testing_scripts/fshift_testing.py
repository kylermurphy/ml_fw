# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from ml_fw import profile as pro
from ml_fw import data_io as dio

path_dat = f'D:\\data\\goes10_omni_1999_2010.hdf5'

dat = pd.read_hdf(path_dat)


# feat DataFrame
cor_col = ['>0.6 MeV']
f_df = dat[['B','Vsw','dynP']].copy()
y_df = dat[cor_col].copy()

# generate an array of shifts
# the units of the array are constant
# here we'll use minutes
ts1 = np.arange(5,60,5) # 5 minutes shifts to an hour
ts2 = np.arange(60,25*60,60) # hourly shifts to 24 hours 

# combine the shift arrays
tshift = np.concatenate([ts1,ts2]).tolist()

tshift = [5,10]

f_dfs = dio.feat_shift(f_df,t_col='index',periods=tshift, unit='min',
                       drop_na=False) 
