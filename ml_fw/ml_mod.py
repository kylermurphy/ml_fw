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
                f'Performing Grid Search using {grid_ratio * 100:.2f}% of data')
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
            best_pos = np.sqrt(dist2.sum(axis=1)/dist2.shape[1]).argmax()

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
