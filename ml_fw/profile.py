# -*- coding: utf-8 -*-
"""Correlation matrix with lagged correlation and colinearity."""
import pandas as pd
from typing import Optional, Union, Callable


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


# ============================================================================
# REFACTORED VERSION
# ============================================================================

def _prepare_correlation_data(
    feature_dat: Union[pd.DataFrame, list],
    target_dat: Union[pd.DataFrame, pd.Series, list],
    correlation_dat: Optional[pd.DataFrame] = None,
    correlation_index: Optional[str] = None
) -> tuple[list, list, pd.DataFrame]:
    """Prepare and normalize correlation data inputs.

    Handles multiple input formats and returns feature columns, target columns,
    and a merged DataFrame ready for correlation calculation.

    Parameters
    ----------
    feature_dat : pd.DataFrame | list
        Feature data or list of feature column names.
    target_dat : pd.DataFrame | pd.Series | list
        Target data or list of target column names.
    correlation_dat : pd.DataFrame, optional
        DataFrame containing both feature and target data when both feature_dat
        and target_dat are lists.
    correlation_index : str, optional
        Column name to join feature_dat and target_dat on. If None, joins on index.

    Returns
    -------
    tuple[list, list, pd.DataFrame]
        (feature_columns, target_columns, merged_data)
    """
    # Case 1: Both inputs are lists pointing to columns in correlation_dat
    if isinstance(feature_dat, list) and isinstance(target_dat, list):
        if not isinstance(correlation_dat, pd.DataFrame):
            raise ValueError("correlation_dat must be a DataFrame when both "
                           "feature_dat and target_dat are lists")
        feature_cols = feature_dat
        target_cols = target_dat
        merged_data = correlation_dat[feature_cols + target_cols].copy()

    # Case 2: Both inputs are DataFrames/Series
    elif isinstance(feature_dat, pd.DataFrame) and \
         isinstance(target_dat, (pd.DataFrame, pd.Series)):
        feature_cols = list(feature_dat.columns)
        target_cols = list(target_dat.columns) if isinstance(target_dat, pd.DataFrame) \
                      else [target_dat.name or 0]

        # Set index for joining
        feature_data = feature_dat.copy()
        target_data = target_dat.copy()

        if correlation_index:
            feature_data = feature_data.set_index(correlation_index)
            target_data = target_data.set_index(correlation_index)

        # Calculate most common index spacing for tolerance
        index_diffs = (pd.Series(feature_data.index[1:]) -
                      pd.Series(feature_data.index[:-1]))
        index_resolution = index_diffs.value_counts().index[0]

        # Merge with nearest-neighbor based on tolerance
        merged_data = pd.merge_asof(
            left=feature_data,
            right=target_data,
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=index_resolution
        )
    else:
        raise ValueError("feature_dat and target_dat must be both DataFrames or "
                       "both lists, not mixed types")

    return feature_cols, target_cols, merged_data


def _normalize_categorical_dict(
    cat_dat: Union[list, dict]
) -> dict:
    """Convert categorical data list to normalized dictionary.

    Parameters
    ----------
    cat_dat : list | dict
        Categorical data specification.

    Returns
    -------
    dict
        Dictionary with string keys and string or callable values.
    """
    if isinstance(cat_dat, dict):
        return cat_dat

    if not isinstance(cat_dat, list):
        raise ValueError("cat_dat must be a list or dict")

    cat_dict = {}
    callable_count = 0

    for item in cat_dat:
        if isinstance(item, str):
            cat_dict[item] = item
        else:
            cat_dict[f'call{callable_count:02d}'] = item
            callable_count += 1

    return cat_dict


def _rename_correlations(
    correlation_df: pd.DataFrame,
    target_cols: list,
    label: str
) -> pd.DataFrame:
    """Apply consistent column naming to correlation results.

    Parameters
    ----------
    correlation_df : pd.DataFrame
        Correlation results to rename.
    target_cols : list
        Target column names.
    label : str
        Label to apply to column names.

    Returns
    -------
    pd.DataFrame
        Renamed correlation DataFrame.
    """
    if len(target_cols) > 1:
        return correlation_df.add_prefix(f'{label}:')
    else:
        return correlation_df.rename(columns={target_cols[0]: label})


def _add_categorical_correlations(
    cor_result: pd.DataFrame,
    base_data: pd.DataFrame,
    correlation_data: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    cat_dict: dict,
    correlation_method: str,
    numeric_only: bool
) -> pd.DataFrame:
    """Calculate and merge categorical subset correlations.

    Parameters
    ----------
    cor_result : pd.DataFrame
        Current correlation results to extend.
    base_data : pd.DataFrame
        Combined feature and target data.
    correlation_data : pd.DataFrame
        Original correlation data (may have categorical columns).
    feature_cols : list
        Feature column names.
    target_cols : list
        Target column names.
    cat_dict : dict
        Categorical variable specifications.
    correlation_method : str
        Correlation method (pearson, spearman, kendall).
    numeric_only : bool
        Include only numeric columns.

    Returns
    -------
    pd.DataFrame
        Updated correlation results with categorical columns added.
    """
    result = cor_result.copy()

    for cat_key, cat_value in cat_dict.items():
        if isinstance(cat_value, str):
            # Binary categorical variable: split on category == 1 and != 1
            mask_category_one = correlation_data[cat_value] == 1

            category_subset = base_data[mask_category_one]
            non_category_subset = base_data[~mask_category_one]

            cor_category = _corrwith(
                category_subset, feature_cols, target_cols,
                correlation_method, numeric_only
            )
            cor_non_category = _corrwith(
                non_category_subset, feature_cols, target_cols,
                correlation_method, numeric_only
            )

            # Rename with category labels
            cor_category = _rename_correlations(
                cor_category, target_cols, f'{cat_key} == 1'
            )
            cor_non_category = _rename_correlations(
                cor_non_category, target_cols, f'{cat_key} != 1'
            )

            result = result.merge(cor_category, left_index=True, right_index=True)
            result = result.merge(
                cor_non_category, left_index=True, right_index=True, how='left'
            )
        else:
            # Callable filter: apply to base data
            filtered_subset = base_data.where(cat_value)

            cor_filtered = _corrwith(
                filtered_subset, feature_cols, target_cols,
                correlation_method, numeric_only
            )

            cor_filtered = _rename_correlations(cor_filtered, target_cols, cat_key)

            result = result.merge(cor_filtered, left_index=True, right_index=True)

    return result


def cor_matrix_ref(
    feature_dat: Union[pd.DataFrame, list],
    target_dat: Union[pd.DataFrame, pd.Series, list],
    correlation_dat: Optional[pd.DataFrame] = None,
    correlation_index: Optional[str] = None,
    categorical_dat: Optional[Union[list, dict]] = None,
    correlation_method: str = 'pearson',
    numeric_only: bool = False
) -> pd.DataFrame:
    """Refactored version: Derive correlation matrix of features with target.

    Computes correlations between feature variables and target variable(s),
    with optional categorical/binary stratification.

    Parameters
    ----------
    feature_dat : pd.DataFrame | list
        Feature data which is correlated with target data.
        - pd.DataFrame: contains the feature data directly.
        - list: column names to extract from correlation_dat.

    target_dat : pd.DataFrame | pd.Series | list
        Target data to correlate with feature data.
        - pd.DataFrame or pd.Series: contains the target data directly.
        - list: column name(s) to extract from correlation_dat.

    correlation_dat : pd.DataFrame, optional
        Data source when feature_dat and target_dat are lists.
        Required if both feature_dat and target_dat are lists.
        Default is None.

    correlation_index : str, optional
        Column name to join feature_dat and target_dat on.
        If None, joins on index. Default is None.

    categorical_dat : list | dict, optional
        Categorical or filtering specification for stratified correlations.

        - str elements: Column name containing binary (0/1) categorical data.
          Correlations computed separately for category == 1 and != 1.

        - callable elements: Filter function (e.g., lambda) to apply to data.
          Correlations computed for filtered subset. Examples::

              ae_filter = lambda x: x['AE'] > 500
              sym_filter = lambda x: x['SymH'] < -50
              categorical_dat = [ae_filter, sym_filter]

        - dict: Keys are used as column labels in output. Values are strings
          or callables as described above.

        - list: Generated labels use 'call00', 'call01', etc. Default is None.

    correlation_method : str, optional
        Correlation method for pd.DataFrame.corrwith.
        One of {'pearson', 'spearman', 'kendall'}. Default is 'pearson'.

    numeric_only : bool, optional
        If True, include only float, int, or boolean columns. Default is False.

    Returns
    -------
    pd.DataFrame
        Correlation results with features as rows.
        Columns include 'All' (base correlations) plus additional columns
        for each categorical variable.

    Examples
    --------
    >>> features = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
    >>> target = pd.DataFrame({'Z': [2, 4, 5]})
    >>> cor_matrix_ref(features, target)
           All
    X  0.981981
    Y  0.992770

    With categorical stratification:
    >>> category = pd.DataFrame({'group': [0, 1, 0]}, index=features.index)
    >>> combined = pd.concat([features, target, category], axis=1)
    >>> cor_matrix_ref(
    ...     ['X', 'Y'], ['Z'], combined,
    ...     categorical_dat=['group']
    ... )
    """
    # Input validation and normalization
    feature_cols, target_cols, merged_data = _prepare_correlation_data(
        feature_dat, target_dat, correlation_dat, correlation_index
    )

    # Extract relevant columns only
    all_cols = feature_cols + target_cols
    base_data = merged_data[all_cols]

    # Base correlations
    correlation_result = _corrwith(
        base_data, feature_cols, target_cols,
        correlation_method, numeric_only
    )

    # Rename base correlation columns
    if len(target_cols) > 1:
        correlation_result = correlation_result.add_prefix('All:')
    else:
        correlation_result = correlation_result.rename(
            columns={target_cols[0]: 'All'}
        )

    # Add categorical/stratified correlations if requested
    if categorical_dat:
        cat_dict = _normalize_categorical_dict(categorical_dat)
        correlation_result = _add_categorical_correlations(
            correlation_result,
            base_data,
            merged_data,
            feature_cols,
            target_cols,
            cat_dict,
            correlation_method,
            numeric_only
        )

    return correlation_result

