import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pmdarima as pm
from characterize_residuals import characterize_residuals 
from diagnostics import diagnostic_plot


def fit_arima_and_characterize(y, title , plot = False, block_length = None):

    model = pm.auto_arima(y, seasonal=False, information_criterion='aic',
                       stepwise=True, error_action='ignore',
                       suppress_warnings=True, trace=True)
    
    residuals = model.resid() 

    characterization = characterize_residuals(residuals, block_length = block_length)

    panel = None

    if plot:   
        panel = diagnostic_plot(residuals, title)

    return model, residuals, characterization, panel  

