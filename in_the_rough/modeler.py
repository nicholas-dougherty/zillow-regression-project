
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def baseline(y_train, y_validate):
    '''Function that will create a baseline prediction'''
    # Turn y into a dataframe
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # Predict mean:
        #Train
    pred_mean_train = y_train.tax_value.mean()
    y_train['pred_mean'] = pred_mean_train
        #Validate
    pred_mean_validate = y_validate.tax_value.mean()
    y_validate['pred_mean'] = pred_mean_validate
    # Find the Root Mean Squared Error
        #Train
    rmse_y_train = mean_squared_error(y_train.tax_value, y_train.pred_mean) ** .5
        #Validate
    rmse_y_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_mean) ** .5

    print("Baseline RMSE using Mean\nTrain/In-Sample: ", round(rmse_y_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_y_validate, 2))

    return y_train, y_validate

def linear_regression(X_train, y_train, X_validate, y_validate):
    '''Function to perform a linear regression on our data'''
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # Create
    lm = LinearRegression(normalize=True)
    # Fit
    lm.fit(X_train, y_train.tax_value)
    # Predict
    y_train['pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lm) ** (1/2)

    # predict validate
    y_validate['pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lm) ** (1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    return rmse_validate, y_validate