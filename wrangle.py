from acquire import acquire_zillow_data, describe_data
from prepare import histogram, univariate 
from prepare import remove_outliers, prepare_zillow 
from env import host, username, password, get_db_url
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# import scaling methods
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
# import modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score




def wrangle_zillow(target):
    ''' 
    '''
    train, validate, test = prepare_zillow(acquire_zillow_data())
    
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    # Change series into data frame for y 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test

def get_db_url(database, username=username, host=host, password=password):
    return f"mysql+pymysql://{username}:{password}@{host}/{database}"

def acquire_zillow_data(use_cache=True):
    '''
    This function returns a snippet of zillow's database as a Pandas DataFrame. 
    When this SQL data is cached and extant in the os directory path, return the data as read into a df. 
    If csv is unavailable, aquisition proceeds regardless,
    reading the queried database elements into a dataframe, creating a cached csv file
    and lastly returning the dataframe for some sweet data science perusal.
    '''

    # If the cached parameter is True, read the csv file on disk in the same folder as this file 
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('zillow.csv')

    # When there's no cached csv, read the following query from Codeup's SQL database.
    print('CSV not detected.')
    print('Acquiring data from SQL database instead.')
    df = pd.read_sql('''
                        SELECT bedroomcnt AS beds, 
                               bathroomcnt AS baths, 
                               calculatedfinishedsquarefeet AS area, 
                               taxvaluedollarcnt AS tax_value, 
                               yearbuilt AS year_built, 
                               fips AS county         
                        FROM properties_2017
                            JOIN propertylandusetype USING (propertylandusetypeid)
                            JOIN predictions_2017 USING(parcelid)
                        WHERE propertylandusedesc IN ('Single Family Residential', 
                                                        'Inferred Single Family Residential')
                            AND transactiondate LIKE '2017%%';
                    
                     '''
                    , get_db_url('zillow'))

    # create a csv of the dataframe for the sake of efficiency. 
    df.to_csv('zillow.csv', index=False)
    
    return df

def remove_outliers(df, k, col_list):
    for col in col_list:
        
        q1, q3 = df[col].quantile([.25, .75])
        
        iqr = q3 - q1
        
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


    
def prepare_zillow(df):
    '''

    '''
    print('Undergoing preparatory stage: \n')
    #just in case there are blanks
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    # drop all nulls, for an affect of .00586 on data
    df.dropna(axis=0, how='any', inplace=True)
    print('199 null values removed. \n 52,319 rows remain.\n ')
    df = df.drop_duplicates(keep=False)
    
    df['year_built'] = df['year_built'].astype(int)
    df.year_built = df.year_built.astype(object) 
    df['age'] = 2017-df['year_built']
    df = df.drop(columns='year_built')
    df['age'] = df['age'].astype('int')
    print('Yearbuilt converted to age. \n')

     # modify two columns
    df['county'] = df.county.apply(lambda fips: '0' + str(int(fips)))
    df['county'].replace({'06037': 'los_angeles', '06059': 'orange', '06111': 'ventura'}, inplace=True)
    print(' Federal Information Processing Standard Publication (FIPS) \n reveals this Zillow data deals with three counties: \n Los Angeles (06037), Orange (06059), and Ventura (06111). \n Hence, data pertains only to Southern California. \n')

    
    orange = df[df['county'] == 'orange']
    ventura = df[df['county'] == 'ventura']
    los_angeles = df[df['county'] == 'los_angeles']
    orange = remove_outliers(orange, 1.5, ['beds', 'baths', 'area', 'tax_value', 'age'])
    ventura = remove_outliers(ventura, 1.5, ['beds', 'baths', 'area', 'tax_value', 'age'])
    los_angeles = remove_outliers(los_angeles, 1.5, ['beds', 'baths', 'area', 'tax_value', 'age'])
    df2 = pd.concat([orange, ventura], axis=0)
    df = pd.concat([df2, los_angeles], axis=0)
    # this row was discovered as having area as 152 square feet. YET it has five bed and two columns. Get it out. 
    df = df.drop(labels=51075, axis=0)
    
    print('Outliers removed individually on a county-by-county basis. \n County data recombined accordingly. \n DF row-based percentage loss after outlier eradication: 19.13%. \n')
    
    dummy_df = pd.get_dummies(df['county'],
                                 drop_first=False)
       # add the dummies as new columns to the original dataframe
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns=['ventura', 'county'])
    # may drop county later, might just opt to not use it. 
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    print(train.shape, validate.shape, test.shape)
    
    
    return train, validate, test

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


