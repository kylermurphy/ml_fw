# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:49:02 2024.

@author: krmurph1
"""

import numpy as np
import pandas as pd
import warnings


def _apply_log_transformation(dataframe: pd.DataFrame, 
                              log_columns: list[str]) -> pd.DataFrame:
    """Apply log10 transformation to specified columns.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe to transform.
    log_columns : list[str]
        Columns to apply log10 transformation.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with log-transformed columns added and originals removed.
    """
    result = dataframe.copy()
    for column in log_columns:
        try:
            result[f'log10_{column}'] = np.log10(result[column])
            result = result.drop(columns=column)
        except (ValueError, KeyError, TypeError) as e:
            warnings.warn(f'Could not log column {column}: {type(e).__name__}')
    return result


def _apply_cyclical_transformation(dataframe: pd.DataFrame, 
                                   cyclical_columns: list[str]) -> pd.DataFrame:
    """Convert cyclical columns (local time, longitude) to cos/sin components.
    
    Prevents discontinuities at 360°→0° or 24h→0h boundaries.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe to transform.
    cyclical_columns : list[str]
        Columns to convert to cyclical representation.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with cyclical columns replaced by cos/sin pairs.
    """
    result = dataframe.copy()
    for column in cyclical_columns:
        try:
            max_value = result[column].max()
            # Determine scale: 360° for angles, 24h for time
            scale = 360.0 if max_value > 24 else 24.0
            result[f'cos_{column}'] = np.cos(result[column] * 2 * np.pi / scale)
            result[f'sin_{column}'] = np.sin(result[column] * 2 * np.pi / scale)
            result = result.drop(columns=column)
        except (ValueError, KeyError, TypeError) as e:
            warnings.warn(f'Could not convert {column} to cyclical: {type(e).__name__}')
    return result


def _validate_required_columns(dataframe: pd.DataFrame, 
                               feature_columns: list[str],
                               target_columns: list[str]) -> None:
    """Validate that required columns exist in the dataframe.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to validate.
    feature_columns : list[str]
        Required feature columns.
    target_columns : list[str]
        Required target columns.
        
    Raises
    ------
    TypeError
        If columns are not lists.
    ValueError
        If required columns don't exist in dataframe.
    """
    if not isinstance(feature_columns, list) or not isinstance(target_columns, list):
        raise TypeError('feature_columns and target_columns must both be lists')
    
    all_required = set(feature_columns + target_columns)
    missing = all_required - set(dataframe.columns)
    if missing:
        raise ValueError(f'Missing columns in dataframe: {missing}')


def create(dataframe: pd.DataFrame, 
               feature_columns: list[str],
               target_columns: list[str],
               log_columns: list[str] | None = None, 
               cyclical_columns: list[str] | None = None,
               time_column: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create feature and target datasets for ML models.
    
    Handles feature engineering including log transformation and cyclical
    variable encoding (for local time and longitude), commonly used in heliophysics 
    and similar domains.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing all raw data.
    feature_columns : list[str]
        Column names to use as features (required).
    target_columns : list[str]
        Column names to use as targets (required).
    log_columns : list[str], optional
        Columns to apply log10 transformation (for variables spanning
        multiple orders of magnitude). Default is None.
    cyclical_columns : list[str], optional
        Columns to convert to cyclical representation using cos/sin
        (e.g., local time, longitude). Default is None.
    time_column : list[str], optional
        Time column to retain in output. Default is None.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - features: Processed feature data
        - targets: Processed target data

    Raises
    ------
    TypeError
        If feature_columns or target_columns are not lists.
    ValueError
        If required columns don't exist in dataframe.
    """
    # Validate inputs upfront
    _validate_required_columns(dataframe, feature_columns, target_columns)
    
    # Build column list for initial selection
    selected_columns = feature_columns + target_columns
    if time_column is not None:
        selected_columns.extend(time_column)
    if cyclical_columns is not None:
        selected_columns.extend(cyclical_columns)
    
    # Select and clean data (drop initial NaNs)
    result = dataframe[selected_columns].dropna().copy()
    
    # Apply transformations
    if log_columns is not None:
        result = _apply_log_transformation(result, log_columns)
    
    if cyclical_columns is not None:
        result = _apply_cyclical_transformation(result, cyclical_columns)
    
    # Remove infinite and remaining NaN values (more efficient than original)
    result = result[~np.isinf(result.select_dtypes(include=[np.number])).any(axis=1)]
    result = result.dropna()
    
    # Extract targets and remove from features
    targets = result[target_columns].copy()
    features = result.drop(columns=target_columns)
    
    return features, targets


def _calculate_timeseries_tolerance(time_series: pd.Series) -> pd.Timedelta:
    """Calculate default tolerance for time-series alignment.
    
    Uses the mode (most common) time resolution to estimate tolerance.
    
    Parameters
    ----------
    time_series : pd.Series
        Time series to analyze.
        
    Returns
    -------
    pd.Timedelta
        Calculated tolerance (half mode resolution + 1 second buffer).
    """
    resolution = time_series.reset_index(drop=True).diff().mode()
    if resolution.empty:
        return pd.Timedelta(1, unit='seconds')
    
    resolution_seconds = resolution.iloc[0].total_seconds()
    return pd.Timedelta(resolution_seconds / 2 + 1, unit='seconds')


def feat_shift(dataframe: pd.DataFrame,
                   time_column: str = 'DateTime',
                   periods: list[int] | int | None = None,
                   unit: str = 'min',
                   tolerance: pd.Timedelta | None = None,
                   drop_original: bool = False,
                   drop_na: bool = True) -> pd.DataFrame:
    """Create time-lagged features for time-series data.
    
    Shifts feature data in time to create lagged versions for use in
    predictive models. Optimized with early validation and minimal copying.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe with features and time information.
    time_column : str, optional
        Column name or 'index' to use as time reference. Default is 'DateTime'.
    periods : list[int] | int, optional
        Time periods to shift. Can be a single int or list of ints.
        Default is [5] (shift by 5 units).
    unit : str, optional
        Time unit for periods (e.g., 'min', 'h', 'D'). Default is 'min'.
    tolerance : pd.Timedelta, optional
        Maximum acceptable time difference for alignment. If None, calculated
        from time-series resolution. Default is None.
    drop_original : bool, optional
        If True, drop original feature columns and keep only shifted versions.
        Default is False.
    drop_na : bool, optional
        If True, drop rows with NaN values created by shifting. Default is True.

    Returns
    -------
    pd.DataFrame
        Dataframe with original and time-lagged feature columns.

    Raises
    ------
    TypeError
        If periods is not an int or list of ints.
    KeyError
        If time_column doesn't exist in dataframe.
    """
    # EARLY VALIDATION - fail fast before any dataframe operations
    # Normalize and validate periods upfront
    if periods is None:
        periods = [5]
    elif isinstance(periods, int):
        periods = [periods]
    elif isinstance(periods, list):
        if not all(isinstance(p, int) for p in periods):
            raise TypeError('All periods must be integers')
    else:
        raise TypeError('periods must be an int or list of ints')
    
    # Handle time column as index BEFORE validation
    # Save original index for restoration
    reset_time_index = False
    actual_time_col = time_column
    original_index = dataframe.index.copy()
    if time_column == 'index':
        working_df = dataframe.reset_index(names='DateTime_idx9')
        actual_time_col = 'DateTime_idx9'
        reset_time_index = True
    else:
        working_df = dataframe
    
    # Validate time column exists NOW (after index handling)
    if actual_time_col not in working_df.columns:
        raise KeyError(f'Time column "{actual_time_col}" not found in dataframe')
    
    # Only copy working_df once, after validation
    working_df = working_df.copy()
    
    # Get time series (no copy needed - not modifying original)
    time_series = working_df[actual_time_col]
    
    # Get feature columns (all except time)
    feature_columns = [col for col in working_df.columns if col != actual_time_col]
    
    # Calculate tolerance once
    if tolerance is None:
        tolerance = _calculate_timeseries_tolerance(time_series)
    
    # Sort by time for efficient merge_asof (this is critical for performance)
    working_df = working_df.sort_values(actual_time_col).reset_index(drop=True)
    time_series = working_df[actual_time_col]
    
    # Create lagged versions - minimize copying
    result = working_df.copy()
    for period in periods:
        # Create minimal lagged dataframe (only time + features, no unnecessary copies)
        lagged_df = working_df[[actual_time_col] + feature_columns].copy()
        lagged_df[actual_time_col] = time_series + pd.Timedelta(period, unit=unit)
        
        # Better suffix naming (database/naming-friendly)
        suffix = f'_lag{period}{unit}'
        result = pd.merge_asof(
            result, lagged_df,
            on=actual_time_col,
            direction='nearest',
            suffixes=('', suffix),
            tolerance=tolerance
        )
    
    # Restore original index
    result.index = original_index
    
    # Cleanup columns
    if reset_time_index:
        result = result.drop(columns=actual_time_col)
    if drop_original:
        result = result.drop(columns=feature_columns)
    if drop_na:
        result = result.dropna()
    
    return result
