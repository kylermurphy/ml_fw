# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:54:54 2024

@author: krmurph1
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV


def train(f_dat: pd.DataFrame,
          y_dat: pd.DataFrame, 
          estimator, 
          est_params: dict = None,
          est_kwargs: dict = None,
          grid_params: dict = None,
          grid_kwargs: dict = {}):

    # if a set of grid parameters
    # has been sent then do a grid 
    # search to fine the best set of parameters
    
### FIXME: need to add a size parameter/partition here

    if grid_params:
        print('''
              Performing grid search using grid_params and using
              GridSearchCV from scikit-learn.
              GridSearchCV parameteters can be set using the 
              grid_kwargs variable. 
              
              Important Note: this can take a substantial amount of
              time depending on the passed estimator              
        ''')
        print('Performing Grid Search:')
        
        est_tune = tune(estimator, grid_params,f_dat,y_dat, **grid_kwargs)
        # check if the best estimator is present
        if 'best_estimator_' in est_tune.__dict__:
            best_est = est_tune.best_estimator_
        else:
            # FIXME: Need to find the best estimator.
            best_est = None
            
    
    # decide if we're tuning the model
    
    # fit the model
    
    # return the model
    
    return est_tune

def tune(estimator, 
         grid_param: dict,
         f_dat,
         y_dat,
         **kwargs):
    

    grid = GridSearchCV(estimator, param_grid=grid_param, **kwargs)
    model_grid = grid.fit(f_dat,y_dat.squeeze())
    
    return model_grid

def residuals():
    
    
    return

