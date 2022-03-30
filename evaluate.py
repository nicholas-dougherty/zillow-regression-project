import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import math
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
rom sklearn.metrics import r2_score
import sklearn.metrics
import math

def scale_data(X_train, X_validate, X_test, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.
    
    Target is not scaled.
    
    columns_to_scale was originally used to check whether los_angeles and orange would cause trouble
    '''
    columns_to_scale = ['beds', 'baths', 'area', 'age', 'los_angeles', 'orange']
    
    X_train_scaled = X_train.copy()
    X_validate_scaled = X_validate.copy()
    X_test_scaled = X_test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(X_train_scaled[columns_to_scale])
    
    X_train_scaled[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
    X_validate_scaled[columns_to_scale] = scaler.transform(X_validate[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    if return_scaler:
        return scaler, X_train_scaled, X_validate_scaled, X_test_scaled
    else:
        return X_train_scaled, X_validate_scaled, X_test_scaled

    
def select_rfe(X, y, k, return_rankings=False, model=LinearRegression()):
    # Use the passed model, LinearRegression by default
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    features = X.columns[rfe.support_].tolist()
    if return_rankings:
        rankings = pd.Series(dict(zip(X.columns, rfe.ranking_)))
        return features, rankings
    else:
        return features


def plot_residuals(df, yhat, y):
    '''
    This function takes in a dataframe, the actual target variable 
    and model predictions then creates columns for residuals
    and baseline residuals. It returns a graph of both residual columns.
    '''

    # create a residual column
    df['residual'] = (yhat - y)

    # create a residual baseline column
    df['residual_baseline'] = (y.mean() - y)
    
    fig, ax = plt.subplots(figsize=(13,7))

    ax.hist(df.residual_baseline, label='baseline residuals', alpha=.6)
    ax.hist(df.residual, label='model residuals', alpha=.6)
    ax.legend()
    
    residuals = yhat - y
    plt.scatter(y, residuals)
    plt.axhline(y=0, color='black')
    plt.show()

    plt.show()

def regression_errors(df, yhat, y):
    '''
    
    '''
    
    SSE = mean_squared_error(yhat, y)*len(df)
    MSE = mean_squared_error(yhat, y)
    RMSE = sqrt(mean_squared_error(yhat, y))
    ESS = sum((yhat - y.mean())**2)
    TSS = sum((y - y.mean())**2)

    # compute explained variance
    R2 = ESS / TSS
    
    print('SSE is:', SSE)
    print('ESS is:', ESS)
    print('TSS is:', TSS)
    print('R2 is:', R2)
    print('MSE is:', MSE)
    print('RMSE is:', RMSE)
    
def baseline_mean_errors(y):
    import sklearn.metrics
    import math
    baseline = y.mean()
    residuals = baseline - y
    residuals_squared = sum(residuals**2)
    SSE = residuals_squared
    print(f'SSE baseline is {SSE}')
    
    MSE = SSE/len(y)
    print(f'MSE baseline is {MSE}')
    
    RMSE = sqrt(MSE)
    print(f'RMSE baseline {RMSE}')


def better_than_baseline(yhat, y):
    baseline = y.mean()
    residuals_baseline = baseline - y
    residuals_squared_baseline = sum(residuals_baseline**2)
    SSE_baseline = residuals_squared_baseline
    
    MSE_baseline = SSE_baseline/len(y)
    
    RMSE_baseline = sqrt(MSE_baseline)
    
    residuals = yhat - y
    residuals_squared = sum(residuals**2)
    SSE = residuals_squared
    
    MSE = sklearn.metrics.mean_squared_error(y,yhat)
    
    RMSE = sqrt(sklearn.metrics.mean_squared_error(y,yhat))
    
    if RMSE < RMSE_baseline:
        return True
    else: 
        return False
    
