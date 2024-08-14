# -*- coding: utf-8 -*-



# correlation matrix 

# lagged correlation

# colinearity

import pandas as pd


def cor_matrix(f_dat: pd.DataFrame | list,
               y_dat: pd.DataFrame | pd.Series | list,
               cor_dat: pd.DataFrame = None,
               cor_ind: str = None,
               cat_dat: list | dict = None,
               cor_meth='pearson') -> pd.DataFrame:
   
    # if both f_dat and y_dat are lists then they
    # contain the columns names of the data to 
    # do the correlation matrix for
    # in this case cor_dat must be a DataFrame
    if isinstance(f_dat, list) \
        and isinstance(y_dat, list) \
            and isinstance(cor_dat, pd.DataFrame):
                f_col = f_dat
                y_col = y_dat
                cor_dat = cor_dat[f_col+y_col]
                
    # else if both f_dat and y_dat are pandas
    # combine them into one data frame to do the correlations
    # use the col names to do the correlations
    elif isinstance(f_dat, pd.DataFrame) \
        and isinstance(y_dat, (pd.DataFrame, pd.Series)):
            # if cor index is passed then join the 
            # arrays on cor_ind column, 
            # otherwise join them on index
            if cor_ind:
                f_dat = f_dat.set_index(cor_ind)
                y_dat = y_dat.set_index(cor_ind)
            
            
            # get the nominal resolution of the feature data
            # this is used for combining the matrices
            res = (pd.Series(f_dat.index[1:]) -
                           pd.Series(f_dat.index[:-1])).value_counts()
            res = res.index[0]
            
            
            # combine the DataFrames to get a single DataFrame
            cor_dat = pd.merge_asof(left=f_dat,right=y_dat,
                          right_index=True,left_index=True,
                          direction='nearest',tolerance=res)
        
            # get the columns that will be correlating
            f_col = list(f_dat.columns)
            y_col = list(y_dat.columns)

    
    
    # generate the initial correlations
    cor_plot = pd.DataFrame()
    cor_plot = cor_dat[f_col+y_col].corr(method=cor_meth,
                                         numeric_only=True)[y_col]
    
    if len(y_col) > 1:
        cor_plot = cor_plot.add_prefix('All:')
    else:
        cor_plot = cor_plot.rename(columns={y_col[0]:'All'})
    
    # parse the categorical data if it's passed
    # --
    # if it's a list the list contains the names
    # of the categorical variables, these are assumed to 
    # be binary and correlations are calculated for
    # column == 1 and column != 1
    # or the list contains a callable
    if isinstance(cat_dat, list):
        cat_dict = dict()
        cat_call = 0
        for lv in cat_dat:
            if isinstance(lv,str):
                cat_dict[lv] = lv
            else:
                cat_dict[f'call{cat_call:02}'] = lv
                cat_call = cat_call+1
    elif isinstance(cat_dat, dict):
        cat_dict = cat_dat  
        
                
    if cat_dat and isinstance(cat_dict,dict):  
        for ck, cv in cat_dict.items():
            if isinstance(cv,str):
                print(cv)
                cat_m = cor_dat[cv] == 1
                cor_1 = cor_dat[cat_m][f_col+y_col].corr(method=cor_meth,
                                                     numeric_only=False)[y_col]
                cor_2 = cor_dat[~cat_m][f_col+y_col].dropna().corr(method=cor_meth,
                                                     numeric_only=False)[y_col]
                if len(y_col) > 1:  
                    cor_1 = cor_1.add_prefix(f'{ck}==1:')
                    cor_2 = cor_2.add_prefix(f'{ck}!=1:')
                    print(cor_1)
                    print(cor_2)
                else:
                    cor_1 = cor_1.rename(columns={y_col[0]:f'{ck} == 1'})
                    cor_2 = cor_2.rename(columns={y_col[0]:f'{ck} != 1'})
                
                cor_plot = cor_plot.merge(cor_1,
                                          left_index=True,
                                          right_index=True)
                cor_plot = cor_plot.merge(cor_2,how='left',
                                          left_index=True,
                                          right_index=True)
            else:
                cor_1 = cor_dat.where(cv)[f_col+y_col].corr(method=cor_meth,
                                                        numeric_only=False)[y_col]
                if len(y_col) > 1:
                    print(cor_1)
                    cor_1 = cor_1.add_prefix(f'{ck}:')
                else:
                    cor_1 = cor_1.rename(columns={y_col[0]:f'{ck}'})
                
                cor_plot = cor_plot.merge(cor_1,
                                          left_index=True,
                                          right_index=True)


    
    return cor_plot