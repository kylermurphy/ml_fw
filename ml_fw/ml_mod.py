# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:54:54 2024

@author: krmurph1
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def train(f_dat: pd.DataFrame,
          y_dat: pd.DataFrame, 
          estimator, 
          est_params: dict = None,
          est_kwargs: dict = None,
          grid_params: dict = None,
          grid_kwargs: dict = {}, 
          grid_ratio: float = 0.3, 
          random_state: int = 17):

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
                f'Performing Grid Search using {grid_ratio*100:.2f}% of data')
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
                    dist.append(est_tune.cv_results_[av+sc]) 
            
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

def tune(estimator, 
         grid_param: dict,
         f_dat,
         y_dat,
         **kwargs):
    

    grid = GridSearchCV(estimator, param_grid=grid_param, **kwargs)
    model_grid = grid.fit(f_dat,y_dat.squeeze())
    
    return model_grid



