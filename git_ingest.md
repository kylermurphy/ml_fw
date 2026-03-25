================================================
Directory structure
================================================

Directory structure:
└── kylermurphy-ml_fw/
    ├── README.md
    ├── LICENSE
    ├── pyproject.toml
    ├── requirements.txt
    ├── setup.cfg
    ├── test_requirements.txt
    ├── docs/
    │   └── faq.rst
    ├── ml_fw/
    │   ├── __init__.py
    │   ├── data_io.py
    │   ├── ml_mod.py
    │   ├── profile.py
    │   ├── inspect/
    │   │   ├── __init__.py
    │   │   ├── _boxplot_metvx.py
    │   │   ├── _boxplot_vx.py
    │   │   └── _rolling_met.py
    │   └── testing_scripts/
    │       ├── __init__.py
    │       ├── fshift_testing.py
    │       ├── metbox_testing.py
    │       ├── resid_testing.py
    │       ├── roll_testing.py
    │       └── testing.py
    └── .github/
        └── workflows/
            └── main.yml


================================================
FILE: README.md
================================================
# ml_fw
Basic ML framework to help prototype ML models faster



================================================
FILE: LICENSE
================================================
MIT License

Copyright (c) 2024 Kyle Murphy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



================================================
FILE: pyproject.toml
================================================
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = 'ml_fw'
version = "0.0.1"
description = "Basic ML framework to help prototype ML models faster"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
authors = [{name= "Kyle Murphy, et al.", email = "tbd@noplace.net"}]
classifiers = [
  "Development Status :: 3 - Alpha"
]
keywords = [
  "machine learning"
]
dependencies = [
  "numpy",
  "matplotlib",
  "pandas",
  "seaborn",
  "scikit-learn",
  "pytables"
]

[project.optional-dependencies]
test = [
  "flake8",
  "flake8-docstrings",
  "pytest-cov"
]
doc = [
  "sphinx"
]



================================================
FILE: requirements.txt
================================================
numpy
matplotlib
pandas
seaborn
scikit-learn
tables



================================================
FILE: setup.cfg
================================================
[flake8]
extend-ignore = E231, E731, W503
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist,ml_fw/testing_scripts



================================================
FILE: test_requirements.txt
================================================
flake8



================================================
FILE: docs/faq.rst
================================================
.. _faq:



================================================
FILE: ml_fw/__init__.py
================================================
# -*- coding: utf-8 -*-
"""Basic ML framework to help prototype ML models faster."""
import ml_fw  # noqa F401



================================================
FILE: ml_fw/data_io.py
================================================
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:49:02 2024.

@author: krmurph1
"""

import numpy as np
import pandas as pd


def create(dat: pd.DataFrame, feat_col: list = None, y_col: list = None,
           log_col: list = None, lt_col: list = None,
           t_col: list = None) -> pd.DataFrame:
    """Create a feature (x_dat) and dependent (y_dat) data set.

    For use in SciKit Learn models.

    Parameters
    ----------
    dat : pd.DataFrame
        A pandas dataframe containing all the data.
    feat_col : list, optional
        Columns to add to the feature data.
        The default is None.
    y_col : list, optional
        Columns to add to the dependent.
        The default is None.
    log_col : list, optional
        Columns from the passed DataFrame that
        should be logged. In Heliophysics variables
        often span several order of magnitude and may
        require the data to be logged (semi-normalized).
        The default is None.
    lt_col : list, optional
        Columns from the passed DataFrame that
        should be converted to cyclical arguments.
        For example longitude, local time, or
        magnetic local time.
        The default is None.
    t_col : list, optional
        If looking at time-series data the time
        column that the user would like to keep.
        The default is None.

    Raises
    ------
    TypeError
        If feat_col and y_col are not lists.
        feat_col and y_col are both required
        keyword arguments

    Returns
    -------
    x_dat : pd.DataFrame
        The feature data.
    y_dat : pd.DataFrame
        The dependent data.
    """
    # create a list of feature and y data columns
    # independent and dependent data
    if not isinstance(feat_col,list) or \
       not isinstance(y_col,list):
        raise TypeError('feat_col and y_col must both be lists')

    dat_col = feat_col + y_col

    # if a time column was passed add it to the list
    if isinstance(t_col,list):
        dat_col = dat_col + t_col

    # creaate a new dataframe from the columns of the
    # passed dataframe
    x_dat = dat[dat_col].dropna().copy()

    # from the feature columns log
    # a subset of passed columns and
    # drop the non-logged columns
    if isinstance(log_col,list):
        for i in log_col:
            try:
                x_dat[f'log10_{i}'] = np.log10(x_dat[i])
                x_dat = x_dat.drop(columns=i)
            except Exception:
                print(f'Could not log column {i}')

    # from a subset of columns convert local time
    # or longitude columns to cyclical variables using
    # cos and sin so that there are no discontinuites
    # across 360->0 or 24h->0h
    if isinstance(lt_col,list):
        for i in lt_col:
            try:
                if dat[i].max() > 24:
                    x_dat[f'cos_{i}'] = np.cos(dat[i] * 2 * np.pi / 360.)
                    x_dat[f'sin_{i}'] = np.sin(dat[i] * 2 * np.pi / 360.)
                else:
                    x_dat[f'cos_{i}'] = np.cos(dat[i] * 2 * np.pi / 24.)
                    x_dat[f'sin_{i}'] = np.sin(dat[i] * 2 * np.pi / 24.)
            except Exception:
                print(f'Could not add {i} as a cos/sin time column')

    # drop values which are undefined
    # check for numpy undefinde
    # then use pandas dropna()
    x_dat = x_dat[~x_dat.isin([np.nan, np.inf, -np.inf]).any(axis=1)].dropna()

    # get y data from the final dataframe
    # and drop it from the feature data
    y_dat = x_dat[y_col].copy()
    x_dat = x_dat.drop(columns=y_col)

    return x_dat, y_dat


def feat_shift(s_dat: pd.DataFrame,
               t_col='DateTime',
               periods: list[int] = [5],
               unit: str = 'min',
               tolerance: pd.Timedelta = None,
               drop_orig: bool = False,
               drop_na: bool = True) -> pd.DataFrame:
    """Time shift features.

    Function to shift feature data in time so
    that time lags can be applied to feature
    vectors.

    Parameters
    ----------
    s_dat : pd.DataFrame
        A dataframe with the features.
        The default is 'pd.DataFrame'
    t_col : TYPE, optional
        The column or index where time is located. This is used
        to shift the data.
        The default is 'DateTime'. 'index' or 0 will use the
        index as the time array
    periods : list[int], optional
        A list integer peiods to shift the data. The unit of the shift
        can be set using 'unit'
        The default is [5], which will shift the data 5 minutes.
        An arbitrary number of shifts can be suplied.
    unit : str, optional
        The unit of the time shift periods.
        Should be of similar types as pd.Timedelta.
        The default is 'min'.
    tolerance : pd.Timedelta, optional
        The max differnce between the orignal time index and time shifted
        index to rematch variables. If this is not set then the time column
        is used to determine a nominal resolution 'res' in seconds and
        tolerance is set as (res/2)+1.
        The default is None.
    drop_orig : bool, optional
        In some cases the orignal data is not required, only the time shifted
        data. This will drop the original data columns
        The default is False.
    drop_na : bool, optional
        The time shift creates a number of NaN at the begining of the
        new DataFrame, this drops those. The default is True.

    Raises
    ------
    TypeError
        If keywords are not correct type.
    KeyError
        If the time column can't be found in the DataFrame.

    Returns
    -------
    s_dat : TYPE
        A dataframe containing time shifted feature columns.
    """
    # get the index so we can
    # add it back at the end
    s_ind = s_dat.index.copy()

    # if the time is the index
    # reset the index so time is a column
    if t_col == 'index' or t_col == 0:
        s_dat = s_dat.reset_index(names='DateTime_idx9')
        t_col = 'DateTime_idx9'
        drop_dt = True

    if isinstance(periods,str):
        periods = [periods]
    elif not isinstance(periods,list):
        raise TypeError('periods should be a string or a list')

    # check if our time column exists
    try:
        t_dat = s_dat[t_col].copy()
    except Exception:
        print(f'{t_col} cannot be found')
        raise KeyError(t_col)

    # use the time column to get a nominal
    # resolution of the timeseries
    # if tolerance isn't defined this is
    # used as tolerance
    res = t_dat.reset_index(drop=True).diff().mode()[0].seconds
    if not tolerance:
        tolerance = pd.Timedelta(res / 2 + 1,unit='seconds')

    # get the data columns
    d_col = s_dat.columns.to_list()
    d_col.remove(t_col)
    # get the data
    r_dat = s_dat.copy().drop(columns=t_col)
    # begin shifting the data
    for i in periods:
        r_dat[t_col] = t_dat + pd.Timedelta(i,unit=unit)
        r_suf = f' {i} {unit}'
        s_dat = pd.merge_asof(s_dat,r_dat, on=t_col,
                              direction='nearest',
                              suffixes=('',r_suf),
                              tolerance=tolerance)

    # add the original index back in
    s_dat = s_dat.set_index(s_ind)
    if drop_dt:
        s_dat = s_dat.drop(columns=t_col)
    # drop the original columns
    if drop_orig:
        s_dat = s_dat.drop(columns=d_col)
    # return only indexes with
    if drop_na:
        s_dat = s_dat.dropna()
    return s_dat



================================================
FILE: ml_fw/ml_mod.py
================================================
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:54:54 2024.

@author: krmurph1
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def train(f_dat: pd.DataFrame, y_dat: pd.DataFrame, estimator,
          grid_params: dict = None, grid_kwargs: dict = {},
          grid_ratio: float = 0.3, random_state: int = 17):
    """Train the ML model.

    Parameters
    ==========
    f_dat : array-like. Shape (n_samples, n_features) or (n_samples, n_samples)
        Training vectors, where n_samples is the number of samples and
        n_features
        is the number of features. For precomputed kernel or distance matrix,
        the expected shape of X is (n_samples, n_samples).

    y_dat : array-like of shape (n_samples, n_output) or (n_samples,)
        Target relative to X for classification or regression;
        None for unsupervised learning.

    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a score function,
        or scoring must be passed.
    grid_params : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any
        sequence of parameter settings.

    grid_kwargs :
        GridSearchCV keyword arguments

    grid_ratio : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split. If int, represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.

    random_state
    Returns
    ==========
    est_fit : object
        Instance of fitted estimator.
    """
    # check the est parameters for random state
    # if it is none set it to random_state
    # else set random_state to the est.random_state
    est_p = estimator.get_params()
    if 'random_state' in est_p:
        if not estimator.random_state:
            estimator.random_state = random_state
        else:
            random_state = estimator.random_state

    # if a set of grid parameters
    # has been sent then do a grid
    # search to fine the best set of parameters

    # identify the best estimator or the nominal set of fit
    # parameters to set the best estimator
    if grid_params:
        print('''
              Performing grid search using grid_params and using
              GridSearchCV from scikit-learn.
              GridSearchCV parameteters can be set using the
              grid_kwargs variable.

              Important Note: this can take a substantial amount of
              time depending on the passed estimator.

        ''')

        if grid_ratio and grid_ratio < 1:
            print(
                f'Performing Grid Search with {grid_ratio * 100:.2f}% of data')
            x_grid, _, y_grid, _ = train_test_split(f_dat,y_dat,
                                                    train_size=grid_ratio,
                                                    random_state=random_state)
            est_tune = tune(estimator, grid_params,x_grid,y_grid,
                            **grid_kwargs)
        else:
            print('Performing Grid Search using all data')
            est_tune = tune(estimator, grid_params,f_dat,y_dat.values.ravel(),
                            **grid_kwargs)

        # check if the best estimator is present
        # if it is use that estimator to fit the model
        # if it isn't this is likely because multiple scores
        # have been passed.
        # if this is the case normalize the train and test
        # scores for each metric
        # combine them using root mean square
        # sqrt(average(sum(score**2)))
        if 'best_estimator_' in est_tune.__dict__:
            print('Using model determined best estimator.')
            est_fit = est_tune.best_estimator_
        else:
            print('''
                  Using linear combination of scorers to
                  determine best estimator
                  ''')
            dist = list()
            scorers = list(est_tune.scorer_.keys())
            average = ['mean_train_','mean_test_']
            for sc in scorers:
                for av in average:
                    dist.append(est_tune.cv_results_[av + sc])

            scaler = MinMaxScaler()
            scaler.fit(np.array(dist).transpose())
            dist2 = scaler.transform(np.array(dist).transpose())**2
            best_pos = np.sqrt(dist2.sum(axis=1) / dist2.shape[1]).argmax()

            # get the best parameters
            # and set the final estimator for fitting
            est_fit = estimator.set_params(
                **est_tune.cv_results_['params'][best_pos])

    # fit the model
    est_fit.fit(f_dat, y_dat.values.ravel())

    # return the model
    return est_fit


def tune(estimator, grid_param: dict, f_dat, y_dat, **kwargs):
    """Perform Grid Search.

    Parameters
    ==========
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a score function,
        or scoring must be passed.

    grid_param : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any
        sequence of parameter settings.

    f_dat : array-like. Shape (n_samples, n_features) or (n_samples, n_samples)
        Training vectors, where n_samples is the number of samples and
        n_features
        is the number of features. For precomputed kernel or distance matrix,
        the expected shape of X is (n_samples, n_samples).

    y_dat : array-like of shape (n_samples, n_output) or (n_samples,)
        Target relative to X for classification or regression;
        None for unsupervised learning.

    Returns
    ==========
    model_grid : object
        Instance of fitted estimator.
    """
    grid = GridSearchCV(estimator, param_grid=grid_param, **kwargs)
    model_grid = grid.fit(f_dat,y_dat.squeeze())

    return model_grid



================================================
FILE: ml_fw/profile.py
================================================
# -*- coding: utf-8 -*-
"""Correlation matrix with lagged correlation and colinearity."""
import pandas as pd


def cor_matrix(f_dat: pd.DataFrame | list,
               y_dat: pd.DataFrame | pd.Series | list,
               cor_dat: pd.DataFrame = None,
               cor_ind: str = None,
               cat_dat: list | dict = None,
               cor_meth='pearson',
               numeric_only: bool = False) -> pd.DataFrame:
    """Derive correlation matrix of features with target variable.

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

    numeric_only : bool, option
        The defauls is True

        Include only float, int or boolean data.

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
        cor_dat = cor_dat[f_col + y_col]

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
        res = (pd.Series(f_dat.index[1:])
               - pd.Series(f_dat.index[:-1])).value_counts()
        res = res.index[0]

        # combine the DataFrames to get a single DataFrame
        cor_dat = pd.merge_asof(left=f_dat,right=y_dat,
                                right_index=True,left_index=True,
                                direction='nearest',tolerance=res)

        # get the columns that will be correlating
        f_col = list(f_dat.columns)
        y_col = list(y_dat.columns)

    # get the data for correlations
    # don't keep computing the slices
    all_cols = f_col + y_col
    base_dat = cor_dat[all_cols]

    # generate the initial correlations
    cor_plot = pd.DataFrame()
    cor_plot = _corrwith(base_dat,f_col,y_col,cor_meth,numeric_only)

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
                cat_call = cat_call + 1
    elif isinstance(cat_dat, dict):
        cat_dict = cat_dat

    # calculate the correlations for categorical variables/filtering
    if cat_dat and isinstance(cat_dict,dict):
        for ck, cv in cat_dict.items():
            if isinstance(cv,str):
                cat_m = cor_dat[cv] == 1
                cat_cor = base_dat[cat_m]
                cat_not = base_dat[~cat_m]

                cor_1 = _corrwith(cat_cor,f_col,y_col,cor_meth,numeric_only)
                cor_2 = _corrwith(cat_not,f_col,y_col,cor_meth,numeric_only)

                #cor_1 = cat_cor.corr(method=cor_meth,
                #                     numeric_only=numeric_only)[y_col]
                #cor_2 = cat_not.dropna().corr(method=cor_meth,
                #                              numeric_only=numeric_only)[y_col]
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
                cat_cor = base_dat.where(cv)
                
                cor_1 = _corrwith(cat_cor,f_col,y_col,cor_meth,numeric_only)
                
                #cor_1 = cat_cor.corr(method=cor_meth,
                #                     numeric_only=numeric_only)[y_col]
                if len(y_col) > 1:
                    cor_1 = cor_1.add_prefix(f'{ck}:')
                else:
                    cor_1 = cor_1.rename(columns={y_col[0]:f'{ck}'})

                cor_plot = cor_plot.merge(cor_1,
                                          left_index=True,
                                          right_index=True)
    
    return cor_plot

def _corrwith(df, f_col,y_col, method, numeric_only):
    """Compute correlations between feature columns and target columns.

    A wrapper around pd.DataFrame.corrwith that handles single and multiple
    target columns, avoiding the full N×N correlation matrix computed by
    pd.DataFrame.corr.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both feature and target columns.

    f_col : list
        Column names of the feature data to correlate against the targets.

    y_col : list
        Column names of the target data to correlate against the features.

    method : str
        Correlation method passed to corrwith. One of 'pearson', 'spearman',
        or 'kendall'.

    numeric_only : bool
        If True, include only float, int, or boolean columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (len(f_col), len(y_col)) where each column
        contains the correlations of all features against one target variable.

    Notes
    -----
    For a single target column, corrwith is called once against the Series.
    For multiple target columns, corrwith is called once per target and the
    results are concatenated. Both cases are faster than pd.DataFrame.corr
    which computes the full N×N matrix before slicing.
    """    
    if len(y_col) == 1:
        return df[f_col].corrwith(df[y_col[0]], method=method,
                                  numeric_only=numeric_only).to_frame(y_col[0])
    else:
        # one corrwith per target — still much cheaper than full N×N corr
        return pd.concat(
            [df[f_col].corrwith(df[y], method=method,
                                numeric_only=numeric_only).rename(y)
             for y in y_col],
            axis=1
        )    



================================================
FILE: ml_fw/inspect/__init__.py
================================================
# -*- coding: utf-8 -*-
"""A collection of tools that are used to inspect model results."""


from ._boxplot_vx import *  # noqa: F401, F403
from ._boxplot_metvx import *  # noqa: F401, F403
from ._rolling_met import *  # noqa: F401, F403



================================================
FILE: ml_fw/inspect/_boxplot_metvx.py
================================================
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:44:50 2024.

@author: krmurph1
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


def boxplot_metvx(x_dat: pd.DataFrame | list,
                  y_true: pd.DataFrame | list,
                  y_mod: pd.DataFrame | list,
                  box_dat: pd.DataFrame = None,
                  box_metric=None,
                  kfolds: int = 100,
                  kfrac: int = 0.5,
                  bins: int | list = 10,
                  xrange: list[tuple[float, float]] | None = None,
                  whisker: float = 1.5):
    """Calculate boxplot like statistics of a metric (using y-t and y-m) vs x.

    The data is binned as a function of x. For each bin in x the a fraction of
    the true and model data are randomly sampled (kfrac). This sample is used
    to calculte a metric of the data. This is repeated kfolds time to generate
    a distribution of metric values for that bin.

    This distribution is then used to derive stats for a box and whisker plot.
    This is repeated for each bin of x.

    The metric can be a callable passed to the function, for example:

        met = lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred)
    or
        met = lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred,
                                                         normalize=False)

    and the callable can specify the parameters of the metric.

    More examples:
    - Regression score
        met = lambda y_true, y_pred: metrics.mean_absolute_error(y_true,
                                                                 y_pred)

    - Clustering score
        met = lambda y_true, y_pred: metrics.homogeneity_score(labels_true,
                                                               labels_pred)

    In these cases the callable would be passed using the box_metric keyword.

    Parameters
    ----------
    x_dat : pd.DataFrame | list
        A pandas DataFrame containing the data for binning. The metric
        (calculated from y-true and y-mod) is then binned and used to calculate
        statistics and derive box and whisker values for each bin.

        The DataFrame can have more the one column. Each column will be binned
        (see bins) and box/whisker data returned.

        If x_dat is a list then the list contains the column names which
        correspond to the binning data in box_dat DataFrame.

    y_true : pd.DataFrame | list
        A pandas DataFrame containing the true data which will be use to
        calculte the metric.

        If y_dat is a list it contains the column name for box_dat that
        contains the true data for calculating the metric.

        y_dat should be a single columnd DataFrame or single valued list.

    y_mod : pd.DataFrame | list
        A pandas DataFrame containing the model data which will be use to
        calculte the metric.

        If y_mod is a list it contains the column name for box_dat that
        contains the model data used to calculate the metric.

        y_mod should be a single columnd DataFrame or single valued list.

    box_dat : pd.DataFrame, optional
        The default is None.

        A pandas DataFrame continaing the x-data and y-true and y-mod which are
        used to calculate a metric. The metric is then binned as a fucntion of
        the x-data and subsequently used to derive box/whisker values..

        If provided, x_dat, y_true, and y_mod should be lists specifying the
        column names of the x-data for binning and the y-true and y-mod data
        for calculating metric values.

    box_metric : callable
        The default is None.

        A callable which can be used to calculate metric values from y-true and
        y-mod.

        If y-true and y-mod are integers then it is assumed the model is a
        classification type and the accuracy metric from scikit-kearn is used.

        If they are floats then the mean squared error from scikit-learn is
        used.

    kfolds : int, optional
        The default is 100.

        For each bin, randomly sample the metric kfolds time to create a
        distribution. Use this distribution to derive box and whisker stats.

    kfrac : int, optional
        The default is 0.5.

        The fraction of the data to use in each kfold.

    bins : int | list, optional
        The default is 10.

        The number of bins with which the x-data is seprated into when deriving
        statistics. It defines the number of equal-width bins in the given
        range.

        If bins is a list it should have the same number of elements or
        columns as x_dat. Each value then specifies the bins for the
        corresponding x data.

        If the number of elements is not the same bins is set to None and the
        default value of 10 is used

        Used in call to scipy.stats.binned_statistic

    xrange : list[tuple[float, float]] | None, optional
        The default is None.

        The lower and upper range of the bins. If not provided, range is simply
        (x.min(), x.max()). Values outside the range are ignored.

        If xrange is a list each element should be a two element list which
        corresponds to the min and max for the corresponding element of x_col.
        In this case the length of xrange should be the same as x_col or the
        number of columns of x_col (depending on type)

        Used in call to scipy.stats.binned_statistic

    whisker : float, optional
        The default is 1.5.

        The position of the whiskers.

        If a float, the lower whisker is at the lowest datum above
        Q1 - whis*(Q3-Q1), and the upper whisker at the highest datum below
        Q3 + whis*(Q3-Q1), where Q1 and Q3 are the first and third quartiles.
        The default value of whis = 1.5 corresponds to Tukey's original
        definition of boxplots.

    Returns
    -------
    box_idx : Dictionary
        A dictionary for each of x_col specifing the values for
        the box/whisker plot.

        If x_dat is a DataFrame the keys are the column names else they
        are the values of the x_dat list.

        box_idx['key']

        Each value is a subsequent dictionary containing the box/whisker data
        for plotting.

        box_idx['key']['box_stats'] - a list of dictionaries containgin the
        box/whisker statistics for bin of the corresponding x_dat ('key').

        The required keys for a box plot (from matplotlib.axes.Axes.bxp) are:
            - med: Median (scalar).
            - q1, q3: First & third quartiles (scalars).
            - whislo, whishi: Lower & upper whisker positions (scalars).
        Optional keys which are used are:
            - mean: Mean (scalar). Needed if showmeans=True.
            - fliers: Data beyond the whiskers (array-like).
            Needed if showfliers=True. Always empty

        box_idx['key']['x_edge'] - an array of dtype float which returns
        the bin edges for the corresponding x_dat ('key'). Returned from
        scipy.stats.binned_statistic

        box_idx['key']['x_centre'] - an array of dtype float containing
        the centre x value for each bin from box_idx['key']['x_edge'].

        box_idx['key']['x_width'] - the width of a x-bin.
    """
    # get data for processing
    if isinstance(box_dat, pd.DataFrame) and \
            isinstance(x_dat, list) and \
            isinstance(y_true, list) and \
            isinstance(y_mod, list):
        x_d = box_dat[x_dat]
        x_c = x_dat.copy()
        y_t = box_dat[y_true].to_numpy().squeeze()
        y_p = box_dat[y_mod].to_numpy().squeeze()

    elif isinstance(x_dat, (pd.DataFrame, pd.Series)) and \
            isinstance(y_true, (pd.DataFrame, pd.Series)) and \
            isinstance(y_mod, (pd.DataFrame, pd.Series)):
        x_d = x_dat
        x_c = x_dat.columns()
        y_t = y_true.to_numpy().squeeze()
        y_p = y_mod.to_numpy().squeeze()

    # define a metric to use if box_metric is None
    # if the true and modeled y-values are integers
    # assume we have a categorical model
    # else assume it is a regression model
    if np.issubdtype(y_t.dtype,np.integer) \
            and np.issubdtype(y_p.dtype,np.integer) \
            and not box_metric:
        print('Using Accuracy Metric')
        met = lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred)
    elif not box_metric:
        print('Using Mean Square Error Metric')
        met = lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred)
    else:
        print('Using passed metric')
        met = box_metric

    # create a list for bins the same size as x_col
    if isinstance(bins,list) and len(bins) == len(x_c):
        bin_v = bins
    else:
        bin_v = np.zeros(len(x_c))
        bin_v[:] = bins

    # create a list for xrange the same size as x_col
    if isinstance(xrange,list) and len(xrange) == 2 and len(x_c) == 1:
        xran = [xrange]
    elif isinstance(xrange,list) and len(xrange) == len(x_c):
        xran = xrange
    else:
        xran = [None for x in x_c]

    # put y-data into dataframe to simplify statistics calculations
    y_d = pd.DataFrame({"tr":y_t, "pr":y_p})

    box_idx = {}

    for idx, bn, xr in zip(x_c, bin_v, xran):
        # calculate the statistics as a function of idx
        xr = xr if isinstance(xr,list) and len(xr) == 2 else None

        # reshape the x arrays
        try:
            x = x_d[idx].to_numpy().squeeze()
        except Exception:
            x = x_d.to_numpy().squeeze()

        # use bin statistic to get the indices of the x data
        # for all the bins with which the data is binned into
        # this can then be used to subsquently bin the metric
        x_stat, x_edges, x_bnum = stats.binned_statistic(x, x,
                                                         bins=bn, range=xr)

        x_cen = (x_edges[0:-1] + [x_edges[1:]]) / 2.
        x_cen = x_cen.squeeze()
        x_wid = x_edges[1] - x_edges[0]

        # calculate the box stats for this x
        box_stats = []

        for i in np.arange(x_stat.size, dtype=int):
            # get the indices for values which lie between
            # bin[i] and bin[i+1]
            gd = x_bnum == i + 1
            # create an array of k-fold samples which
            # holds metric values from each sample which
            # box stats can be computed from
            if sum(gd) > 1:
                sval = np.array([
                                met(y_d.loc[gd,'tr'].sample(frac=kfrac,
                                                            random_state=x),
                                    y_d.loc[gd,'pr'].sample(frac=kfrac,
                                                            random_state=x))
                                for x in np.arange(kfolds)
                                ])

                lq = np.nanpercentile(sval,25)
                uq = np.nanpercentile(sval,75)
            else:
                sval = 0
                lq = 0
                uq = 0

            bval = {"mean": np.nanmean(sval),  # not required
                    "med": np.nanmedian(sval),
                    "q1": lq,
                    "q3": uq,
                    "whislo": lq - whisker * (uq - lq),  # required
                    "whishi": uq + whisker * (uq - lq),  # required
                    "fliers": []  # required if showfliers=True
                    }
            # append box to list
            box_stats.append(bval)

        # add box values to box dictionary
        box_idx[idx] = {'box_stats':box_stats, 'x_edge':x_edges,
                        'x_centre':x_cen, 'x_width':x_wid}

    return box_idx



================================================
FILE: ml_fw/inspect/_boxplot_vx.py
================================================
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:28:59 2024.

@author: krmurph1

ivestigate residuals of a ml model
"""

import numpy as np
import pandas as pd
from scipy import stats

#TODO Check to make sure that y is single valued

def boxplot_vx(x_dat: pd.DataFrame | list,
               y_dat: pd.DataFrame | list,
               box_dat: pd.DataFrame = None,
               box_meth: bool | dict = True,
               bins: int | list = 10,
               xrange: list[tuple[float, float]] | None = None,
               whisker: float = 1.5) -> dict:
    """Calculate boxplot like statistics of y as a function of x.

    Parameters
    ----------
    x_dat : pd.DataFrame | list
        A pandas DataFrame containing the data for binning. The y-data within
        the binned is then used to calculate statistics and derive box and
        whisker values for each bin.
        The DataFrame can have more the one column. Each column will be binned
        (see bins) and box/whisker data returned.
        If x_dat is a list then the list contains the column names which
        correspond to the binning data in box_dat DataFrame.

    y_dat : pd.DataFrame | list
        A pandas DataFrame containing the y-data which will be use to calculate
        the box and whisker statistics for each bin from x_dat.
        If y_dat is a list it contains the column name for box_dat that
        contains the y-data with which box/whisker values are derived.
        y_dat should be a single columnd DataFrame or single valued list.

    box_dat : pd.DataFrame, optional
        The default is None.
        A pandas DataFrame continaing the x-data and y-data for which values
        are binned and subsequently used to derive box/whisker values,
        respectively.
        If provided, x_dat and y_dat should be lists specifying the column
        names of the x-data for binning and the y-data for deriving box/whisker
        values.

    box_meth : bool | dict, optional
        The default is True.
        Currently a place holder which may be used to change the statistics
        returned for the box/whisker values.

    bins : int | list, optional
        The default is 10.
        The number of bins with which the x-data is seprated into when deriving
        statistics. It defines the number of equal-width bins in the given
        range.
        If bins is a list it should have the same number of elements or
        columns as x_dat. Each value then specifies the bins for the
        corresponding x data.
        If the number of elements is not the same bins is set to None and the
        default value of 10 is used
        Used in call to scipy.stats.binned_statistic

    xrange : list[tuple[float, float]] | None, optional
        The default is None.
        The lower and upper range of the bins. If not provided, range is simply
        (x.min(), x.max()). Values outside the range are ignored.
        If xrange is a list each element should be a two element list which
        corresponds to the min and max for the corresponding element of x_col.
        In this case the length of xrange should be the same as x_col or the
        number of columns of x_col (depending on type)
        Used in call to scipy.stats.binned_statistic

    whisker : float, optional
        The default is 1.5.
        The position of the whiskers.
        If a float, the lower whisker is at the lowest datum above
        Q1 - whis*(Q3-Q1), and the upper whisker at the highest datum below
        Q3 + whis*(Q3-Q1), where Q1 and Q3 are the first and third quartiles.
        The default value of whis = 1.5 corresponds to Tukey's original
        definition of boxplots.

    Returns
    -------
    box_idx : Dictionary
        A dictionary for each of x_col specifing the values for
        the box/whisker plot.
        If x_dat is a DataFrame the keys are the column names else they
        are the values of the x_dat list.
        box_idx['key']
        Each value is a subsequent dictionary containing the box/whisker data
        for plotting.
        box_idx['key']['box_stats'] - a list of dictionaries containgin the
        box/whisker statistics for bin of the corresponding x_dat ('key').
        The required keys for a box plot (from matplotlib.axes.Axes.bxp) are:
            - med: Median (scalar).
            - q1, q3: First & third quartiles (scalars).
            - whislo, whishi: Lower & upper whisker positions (scalars).
        Optional keys which are used are:
            - mean: Mean (scalar). Needed if showmeans=True.
            - fliers: Data beyond the whiskers (array-like).
            Needed if showfliers=True. Always empty
        box_idx['key']['x_edge'] - an array of dtype float which returns
        the bin edges for the corresponding x_dat ('key'). Returned from
        scipy.stats.binned_statistic
        box_idx['key']['x_centre'] - an array of dtype float containing
        the centre x value for each bin from box_idx['key']['x_edge'].
        box_idx['key']['x_width'] - the width of a x-bin.
    """
    # get data for processing
    if isinstance(x_dat, list) \
            and isinstance(y_dat, list) \
            and isinstance(box_dat, pd.DataFrame):
        x_col = x_dat
        y_col = y_dat

        x_val = box_dat[x_col]
        y_val = box_dat[y_col]
    elif isinstance(x_dat, (pd.DataFrame, pd.Series)) \
            and isinstance(y_dat, (pd.DataFrame, pd.Series)):
        # get the column/series name
        if isinstance(x_dat, pd.DataFrame):
            x_col = x_dat.columns
        else:
            x_col = [x_dat.name] if x_dat.name else ['x_col']
        # get the columns/series name
        if isinstance(y_dat, pd.DataFrame):
            y_col = y_dat.columns
        else:
            y_col = y_dat.name if y_dat.name else 'y_col'

        # get the data
        x_val = x_dat
        y_val = y_dat

    # create a list for bins the same size as x_col
    if isinstance(bins,list) and len(bins) == len(x_col):
        bin_v = bins
    else:
        bin_v = np.zeros(len(x_col))
        bin_v[:] = bins

    # create a list for xrange the same size as x_col
    if isinstance(xrange,list) and len(xrange) == 2 and len(x_col) == 1:
        xran = [xrange]
    elif isinstance(xrange,list) and len(xrange) == len(x_col):
        xran = xrange
    else:
        xran = [None for x in x_col]

    # now we want to itterate over the x columns
    # and generate the box plot as binned statistic

    # define lambdas to calculate the upper and lower
    # quartiles
    lq_nan = lambda stat: np.nanpercentile(stat, 25)
    uq_nan = lambda stat: np.nanpercentile(stat, 75)

    box_idx = {}

    for idx, bn, xr in zip(x_col, bin_v, xran):
        # calculate the statistics as a function of idx
        xr = xr if isinstance(xr,list) and len(xr) == 2 else None

        # reshape the arrays
        try:
            x = x_val[idx].to_numpy().squeeze()
        except Exception:
            x = x_val.to_numpy().squeeze()

        try:
            y = y_val[y_col].to_numpy().squeeze()
        except Exception:
            y = y_val.to_numpy().squeeze()

        # calculate stats
        mean, x_edge, _ = stats.binned_statistic(x, y, bins=bn,
                                                 range=xr,
                                                 statistic=np.nanmean)
        median, _, _ = stats.binned_statistic(x, y,bins=bn,
                                              range=xr,
                                              statistic=np.nanmedian)
        low_q, _, _ = stats.binned_statistic(x, y, bins=bn,
                                             range=xr, statistic=lq_nan)
        up_q, _, _ = stats.binned_statistic(x, y, bins=bn,
                                            range=xr, statistic=uq_nan)

        # calculate x location of the stats
        x_cen = (x_edge[0:-1] + [x_edge[1:]]) / 2.
        x_cen = x_cen.squeeze()
        x_wid = x_edge[1] - x_edge[0]
        # create a list to hold the required
        # parameters to draw a box and whisker plot
        box_stats = []
        for mn, md, lq, uq, in zip(mean, median, low_q, up_q):
            val = {"mean": mn,  # not required
                   "med": md,
                   "q1": lq,
                   "q3": uq,
                   "whislo": lq - whisker * (uq - lq),  # required
                   "whishi": uq + whisker * (uq - lq),  # required
                   "fliers": []  # required if showfliers=True
                   }
            box_stats.append(val)

        # create a dictionary to store everything needed for plotting
        # and add it to the return dictionary
        box_idx[idx] = {'box_stats':box_stats, 'x_edge':x_edge,
                        'x_centre':x_cen, 'x_width':x_wid}

    return box_idx



================================================
FILE: ml_fw/inspect/_rolling_met.py
================================================
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:16:51 2024.

@author: murph
"""

import numpy as np
import pandas as pd
from sklearn import metrics

from pandas.api.types import is_datetime64_any_dtype as is_datetime


def rolling_met(met_dat: pd.DataFrame,
                y_true: str = 'y_true',
                y_pred: str = 'y_pred',
                on: str = 'DateTime',
                roll_kwargs: dict = None,
                roll_metric: list | dict = None) -> pd.DataFrame:
    """Calculate a rolling metric.

    Parameters
    ----------
    met_dat : pd.DataFrame
        Pandas DataFrame which contains the observed and predicted data as well
        as a index or time series which can be used to create a set of rolling
        windows which are used to derive metrics.

    y_true : str, optional
        The default is 'y_true'.
        The column name which contains the observerd or true values.

    y_pred : str, optional
        The default is 'y_pred'.
        The column name which contains the predicted or modeled values.

    on : str, optional
        The default is 'DateTime'.
        The used to define the rolling windows.
        If on='index' then the index of the met_data DataFrame is used.

    roll_kwargs : dict, optional
        The default is None.
        Keywords arguments to be passed to DataFrame.rolling() method.
        If nothing is passed the 'on' column is checked if it is datetime like
        and if true a 60 minute centered rolling window is used.
        Else a 10 point centred rolling window is used.

    roll_metric : list | dict, optional
        The default is None.
        A callable, list of callables, or dictionary of callables which can
        be used to derive a metric from the true and predicted values.

        The callables can be lambda functions utilizing metrics from
        sklearn.metrics. For example:
            met = lambda tr, pr: sklearn.metrics.accuracy_score(tr,pr)
        The labmdas can also be used to set keyword arguments. For example:
            met = lamba tr, pr: sklearn.metrics.log_loss(tr,pr, normalize=True)

        If roll_mtric is a list each list element should be metric callable.
        If it is a dictionary the key should be the name and value a metric
        callable.

    Returns
    -------
    rdf : pd.DataFrame
        A DataFrame containg the values of a rolling metric for each passed
        callable/metric and the index/DateTime values for each value.
        If roll_metric is a list the names of the column are 'Metric XX' where
        xx is the list element.
        If roll_metric is dictionary the names of the columns are the
        dictionary keys.
    """
    # if strings are passed and met_dat DateFrame
    # then get only the data we need
    if isinstance(y_true, str) \
            and isinstance(y_pred,str) \
            and isinstance(on,str) \
            and isinstance(met_dat, pd.DataFrame):
        # check if the rolling is happening
        # on the index
        # if so reset the index and get the
        # index name
        if on.upper() == 'INDEX':
            if not met_dat.index.name:
                on = 'index'
            else:
                on = met_dat.index.name
            met_dat = met_dat.reset_index()
        # create a dataframe of only the
        # columns we need
        rdat = met_dat[[on,y_true,y_pred]].copy()
    # if 'on' is a DateTime like object then set some default values
    # for rolling, else assume rolling on int/float like value
    if is_datetime(rdat[on]) and not roll_kwargs:
        roll_kwargs = {'window':'60min', 'center':True}
    elif not roll_kwargs:
        roll_kwargs = {'window':10, 'center':True}
    # make sure window is in the roll_kwa
    if 'window' not in roll_kwargs:
        if is_datetime(rdat[on]):
            roll_kwargs['window'] = '60min'
        else:
            roll_kwargs['window'] = 10
    # define a metric to use if box_metric is None
    # if the true and modeled y-values are integers
    # assume we have a categorical model
    # else assume it is a regression model
    if np.issubdtype(rdat[y_true].dtype,np.integer) \
            and np.issubdtype(rdat[y_pred].dtype,np.integer) \
            and not roll_metric:
        print('Using Accuracy Metric')
        met = lambda true, pred: metrics.accuracy_score(true, pred)
        met_d = {'Accuracy':met}
    elif not roll_metric:
        print('Using Mean Square Error Metric')
        met = lambda true, pred: metrics.mean_squared_error(true, pred)
        met_d = {'MSE':met}
    else:
        print('Using passed metric')
        met_d = roll_metric
    # create a dictionary to loop over
    # for calculating metrics
    if isinstance(met_d, list) and not \
       isinstance(met_d, dict):
        met_d = dict()
        met_c = 0
        for lv in met_d:
            met_d[f'Metric {met_c:02}'] = lv
            met_c = met_c + 1
    elif not isinstance(met_d, dict):
        met_d = {'Metric':met}
    # define the rolling window to compute the metric
    roll = rdat.set_index(on).rolling(**roll_kwargs)
    rmet = np.array([
        [mv(rdat.set_index(on).loc[l.index,y_true],
            rdat.set_index(on).loc[l.index,y_pred])
         for mk, mv in met_d.items()]
         for l in roll]) # noqa E741

    # use the rolling to get and index for the returned
    # metric. this is needed in case step is used in the
    # rolling kwargs
    rind = rdat.set_index(on).rolling(**roll_kwargs).mean().index
    rdf = pd.DataFrame(data=rmet,columns=met_d.keys())
    rdf[on] = rind
    # rdf = pd.DataFrame({on:rind,'Metric':rmet})
    return rdf



================================================
FILE: ml_fw/testing_scripts/__init__.py
================================================
# -*- coding: utf-8 -*-


================================================
FILE: ml_fw/testing_scripts/fshift_testing.py
================================================
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

tshift = [5,10,15,20,25]

f_dfs = dio.feat_shift(f_df,t_col='index',periods=tshift, unit='min',
                       drop_na=False) 



================================================
FILE: ml_fw/testing_scripts/metbox_testing.py
================================================
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


================================================
FILE: ml_fw/testing_scripts/resid_testing.py
================================================
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:44:19 2024

@author: krmurph1
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from ml_fw import inspect as insp


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

box_dat = insp.boxplot_vx(dat[x_dat], dat[y_dat],whisker=whisker, 
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
    print(f'Plotting {key}')
    
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

    
box2 = insp.boxplot_vx(['AE'], ['resid'], dat,bins=20, xrange=aerange, whisker=0)  

fig2, ax2 = plt.subplots(1,1, figsize=(8,9))

ae_box = box2['AE']
y_val = ae_box['box_stats']
x_val = ae_box['x_centre']
x_width = ae_box['x_width']

# use the matplotlib boxplot to plot

cc = [1,0,0]
bx_a = 0.25
ln_a = 1.0
ln_w = 2.0
 
b1 = ax2.bxp(y_val, positions=x_val, widths=x_width, 
                  patch_artist=True, showmeans=showmean, 
                  shownotches=False, showcaps=False, 
                  boxprops={'ec':cc+[ln_a], 'fc':cc+[bx_a]},
                  medianprops={'c':cc, 'lw':ln_w}, 
                  meanprops={'mec':cc, 'mfc':cc})   



================================================
FILE: ml_fw/testing_scripts/roll_testing.py
================================================
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:16:09 2024

@author: murph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_fw import inspect as insp
import sklearn.metrics as metrics

fn = "C:/Users/murph/OneDrive/Phys4501/data/RF_FI_GEO_OOS_CHAMP.hdf5"
dat = pd.read_hdf(fn)

y_true = '400kmDensity'
y_pred = '400kmDensity_pred'
on = 'DateTime'
rkwargs = {'window':'90min','center':False}
met = {'hi':lambda y_true, y_pred: metrics.median_absolute_error(y_true, y_pred),
       'hizzle':lambda y_true, y_pred: metrics.mean_absolute_error(y_true, y_pred)}

sdate = '2005-08-21'
edate = '2005-08-30'

gd = (dat['DateTime'] > sdate) & (dat['DateTime'] <= edate)

rdat = dat[gd].copy()

#met_df = insp.rolling_met(rdat,y_true=y_true,y_pred=y_pred,on=on,
#                         roll_metric=met, roll_kwargs=rkwargs)


path_dat = "D:\data\Full_Model_Outputs.hdf5"
dd = pd.read_hdf(path_dat)
y_t = 'True Class'
y_p = 'Prediction'
on = 'index'
rkwargs = {'window':11,'center':True}

r_met = insp.rolling_met(dd,y_true=y_t,y_pred=y_p,on=on,
                        roll_kwargs=rkwargs)


================================================
FILE: ml_fw/testing_scripts/testing.py
================================================
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:03:38 2024

@author: krmurph1
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ml_fw import data_io as dio
from ml_fw import profile as pro

#import ml_mod as ml


#from sklearn.ensemble import HistGradientBoostingRegressor as hgbr


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

#df_cols = {
#    'feat_col': ['AE','lat'],
#    'y_col': ['dens_x'],
#    't_col': ['DateTime'],
#    'lt_col': None,
#    }

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



================================================
FILE: .github/workflows/main.yml
================================================
# This workflow will install Python dependencies, run tests and lint with a
# variety of Python versions. For more information see:
# https://help.github.com/actions/language-and-framework-guides/using-python-with-github-actions

name: Pytest with Flake8

on: [push, pull_request]

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        os: ["ubuntu-latest", "windows-latest", "macos-latest"]
        python-version: ["3.12"]
        
    name: Python ${{ matrix.python-version }} on ${{ matrix.os }} with numpy ${{ matrix.numpy_ver }}
    runs-on: ${{ matrix.os }}
    steps:
    - uses: actions/checkout@v4
    - name: Set up Pyton ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: pip install .[test]

    - name: Setup ml_fw
      run: |
        python -c "import ml_fw"

    - name: Test PEP8 compliance
      run: flake8 . --count --select=D,E,F,H,W --show-source --statistics


