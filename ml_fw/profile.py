# -*- coding: utf-8 -*-



# correlation matrix 

# lagged correlation

# colinearity

import pandas as pd


def cor_matrix(f_dat: pd.DataFrame | list,
               y_dat: pd.DataFrame | pd.Series | list,
               c_dat: list = None,
               d_dat: pd.DataFrame = None ) -> pd.DataFrame:


    # if both f_dat and y_dat are lists then they
    # contain the columns names of the data to 
    # do the correlation matrix for
    # in this case d_dat must be a DataFrame
    if isinstance(f_dat, list) \
        and isinstance(y_dat, list) \
            and isinstance(d_dat, pd.dataframe):
                cor_col = f_dat
                cor_col.append(y_dat)
    # else if both 
    elif isinstance(f_dat, pd.DataFrame) \
        and isinstance(y_dat, (pd.DataFrame, pd.Series)):
            
        
                
            
    return x       