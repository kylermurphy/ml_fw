# -*- coding: utf-8 -*-

# callables are lambda functions 
# which can be used to filter the data

# correlation matrix 

# lagged correlation

# colinearity

import pandas as pd


def cor_matrix(f_dat: pd.DataFrame | list,
               y_dat: pd.DataFrame | pd.Series | list,
               cor_dat: pd.DataFrame = None,
               cor_ind: str = None,
               cat_dat: list | dict = None,
               cor_meth='pearson', 
               y_drop: bool = True) -> pd.DataFrame:
    """
    Derive correlation matrix of features with target variable.

    Parameters
    ----------
    f_dat : pd.DataFrame | list
        Feature data which is correlated with target data.
        
        A pd.DataFrame containing the feature data or a list containing the 
        column names of the feature data set.
        
    y_dat : pd.DataFrame | pd.Series | list
        Target data which is correlated with feature data
        
        A pd.DataFrame, pd.Series, or list which conatins the target dataset.
        
        If a list the column name of the target data.
        
    cor_dat : pd.DataFrame, optional
        The default is None.
        
        If both y_dat and f_dat are lists then the cor_dat pd.DataFrame
        contains the feature and target data where the columns correspond to 
        the list elements of y_dat and f_dat.
        
    cor_ind : str, optional
        The default is None.
        
        If cor_ind is passed and f_dat and y_dat are pd.DataFrames then cor_ind
        contains the column name which is used to join f_dat and y_dat. Else
        f_dat and y_dat are joined on index.
        
    cat_dat : list | dict, optional
        A list containing the column names which are categorical/binary data or
        callables that can be used to filter the data. 
        
        If the list element is a string then it contains the column name of 
        a binary categorical data and two sets of correlations are performed. 
        One on a subset of the data where the categorical variable is 0 and the
        other where the variable is 1. 
        
        If the list elements is a callable then the callable is a function that
        can be used to filter either the feature or target data. These can be 
        lambda functions used to filter a pd.DataFrame on a particular column.
        For example, the feature data has a columns 'AE' and 'SymH', then:
            
            ae_f = lambda x: x['AE'] > 500
            sym_f = lambda x: x['SymH'] < -50
            cat_dat = [ae_f, sym_f]
        
        Here the correlation pd.DataFrame will be filtered to look at the 
        correlations of the features with the target when the AE column is 
        greater then 500. Another set of correlations will be calculated when
        SymH is less then -50. 
        
        If cat_dat is a dictionary the values contain strings or callables 
        similar to if it was a list. The keys are used to name the columns of
        the returned correlation matrix. 
        
        If cat_dat is a list the correlations are returned with column names
        'call_xx' where xx is an integer.   
        
    cor_meth : TYPE, optional
        The default is 'pearson'.
        
        The type of correlation used in pd.DataFrame.corr
        
    y_drop : bool, optional
       The default is True.
       
       Drop the y_dat from row of the correlation matrix as it is always 1.

    Returns
    -------
    cor_plot : pd.DataFrame
        A pd.DataFrame whose rows are the correlations of the features with
        the target variable. 
        
        Additional columns are added to account for correlations provided via
        the cat_dat keyword.
    """
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
    # categorical variables can be strings of column names
    # - str elements
    # if a string assume the categorical variable is binary, 0 or 1
    # the column is separated into values that ==1 and !=1
    # and the correlations are calculted
    # - callables
    # use the DataFrame.where() function and the passed callable
    # to mask the data and calculate the correlations
    #
    # if a list is passed parse it into a dictionary 
    # - str elements
    # key is the str, value is the str
    #
    # - callable or non-str elemtents
    # key is an increasing integer or name of the
    # callable, value is the callable/element

    # create dictionary for categorical varialbes/filtering
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
        
    
    # calculate the correlations for categorical variables/filtering            
    if cat_dat and isinstance(cat_dict,dict):  
        for ck, cv in cat_dict.items():
            if isinstance(cv,str):
                cat_m = cor_dat[cv] == 1
                cor_1 = cor_dat[cat_m][f_col+y_col].corr(method=cor_meth,
                                                     numeric_only=False)[y_col]
                cor_2 = cor_dat[~cat_m][f_col+y_col].dropna().corr(method=cor_meth,
                                                     numeric_only=False)[y_col]
                if len(y_col) > 1:  
                    cor_1 = cor_1.add_prefix(f'{ck}==1:')
                    cor_2 = cor_2.add_prefix(f'{ck}!=1:')
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
                    cor_1 = cor_1.add_prefix(f'{ck}:')
                else:
                    cor_1 = cor_1.rename(columns={y_col[0]:f'{ck}'})
                
                cor_plot = cor_plot.merge(cor_1,
                                          left_index=True,
                                          right_index=True)


    
    if y_drop:
        cor_plot = cor_plot.drop(y_col)
    
    return cor_plot