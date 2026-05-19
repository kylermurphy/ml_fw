# -*- coding: utf-8 -*-
"""Correlation matrix with lagged correlation and colinearity."""
import pandas as pd
from typing import Optional, Union, Callable


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


def cor_matrix(
    feature_dat: Union[pd.DataFrame, list],
    target_dat: Union[pd.DataFrame, pd.Series, list],
    correlation_dat: Optional[pd.DataFrame] = None,
    correlation_index: Optional[str] = None,
    categorical_dat: Optional[Union[list, dict]] = None,
    correlation_method: str = 'pearson',
    numeric_only: bool = False
) -> pd.DataFrame:
    """Derive correlation matrix of features with target variable.

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

