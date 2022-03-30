import pandas as pd
import numpy as np
import env
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# custom module imports
from acquire import acquire_zillow_data
from prepare import remove_outliers, prepare_zillow
from wrangle import wrangle_zillow, scale_data
from explore import *

# feature selection imports
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# import scaling methods
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
# import modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

from mitosheet import *
import plotly.express as px
# import to remove warnings
import warnings
warnings.filterwarnings("ignore")

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

def scaling_impact(X_train, X_train_scaled):
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(X_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(X_train_scaled, bins=25, ec='black')
    plt.title('Scaled')

def select_rfe(X_train_scaled, y_train, k, return_rankings=False, model=LinearRegression()):
    # Use the passed model, LinearRegression by default
    rfe = RFE(model, n_features_to_select=k)
     # fit the data using RFE
    rfe.fit(X_train_scaled, y_train)
    # get mask of columns selected as list
    feature_mask = X_train_scaled.columns[rfe.support_].tolist()
    if return_rankings:
        rankings = pd.Series(dict(zip(X_train_scaled.columns, rfe.ranking_)))
        return feature_mask, rankings
    else:
        return feature_mask

# def create_baseline(y_train, y_validate):
#     # 1. Predict mean tax value 
# tv_pred_mean = y_train.tax_value.mean()
# y_train['tv_pred_mean'] = tv_pred_mean
# y_validate['tv_pred_mean'] = tv_pred_mean
# 
# # 2. Predict median tax value 
# tv_pred_median = y_train.tax_value.median()
# y_train['tv_pred_median'] = tv_pred_median
# y_validate['tv_pred_median'] = tv_pred_median
# 
# # 3. RMSE of tv_pred_mean
# rmse_train = mean_squared_error(y_train.tax_value, y_train.tv_pred_mean) ** 0.5
# rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tv_pred_mean) ** 0.5
# 
# print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
#       "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
# 
# # 4. RMSE of tv_pred_median
# rmse_train1 = mean_squared_error(y_train.tax_value, y_train.tv_pred_median) ** .5
# rmse_validate1 = mean_squared_error(y_validate.tax_value, y_validate.tv_pred_median) ** .5
# print('-----------')
# print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train1, 2), 
#       "\nValidate/Out-of-Sample: ", round(rmse_validate1, 2))
# 
# return 
# metric_df = pd.DataFrame(data=[{
#     'model': 'mean_baseline',
#     'rmse_outofsample': rmse_validate,
#     'r^2_outofsample': explained_variance_score(y_validate.tax_value, y_validate.tv_pred_mean)}])