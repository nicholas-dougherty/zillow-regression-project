from env import host, username, password, get_db_url
import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

def remove_outliers(df, k, col_list):
    for col in col_list:
        
        q1, q3 = df[col].quantile([.25, .75])
        
        iqr = q3 - q1
        
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def acquire_zillow_data(use_cache=True):
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('zillow.csv')
    print('Acquiring data from SQL database')
    df = pd.read_sql('''
                        SELECT bedroomcnt, 
                               bathroomcnt, 
                               calculatedfinishedsquarefeet, 
                               taxvaluedollarcnt, 
                               yearbuilt, 
                               fips          
                        FROM properties_2017
                            JOIN propertylandusetype USING (propertylandusetypeid)
                            JOIN predictions_2017 USING(parcelid)
                        WHERE propertylandusedesc IN ('Single Family Residential', 
                                                        'Inferred Single Family Residential')
                            AND transactiondate LIKE '2017%%';
                    
                     '''
                    , get_db_url('zillow'))
    df.to_csv('zillow.csv', index=False)
    
    return df

def prepare_zillow(df):
        #just in case there are blanks
    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # drop all nulls, for an affect of .00586 on data
    df.dropna(axis=0, how='any', inplace=True)

     # modify two columns
    df['fips'] = df.county_name.apply(lambda fips: '0' + str(int(fips)))
    df['fips'].replace({'06037': 'los_angeles', '06059': 'orange', '06111': 'ventura'}, inplace=True)
    
    df['yearbuilt'] = df['yearbuilt'].astype(int)
    df.yearbuilt = df.yearbuilt.astype(object) 

    df['age'] = 2017-df['yearbuilt']
    
    df = df.drop(columns='yearbuilt')

    df = df.rename(columns={
                        'calculatedfinishedsquarefeet': 'area',
                       'bathroomcnt': 'baths',
                        'bedroomcnt': 'beds',
                        'taxvaluedollarcnt':'tax_value',
                        'fips': 'county_name'}
              )

    df = remove_outliers(df, 1.5, ['beds', 'baths', 'area', 'tax_value', 'age'])
    
    df['age'] = df['age'].astype('int')
    
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test
#############

def scale_data(train, validate, test, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.
    '''
    columns_to_scale = ['beds', 'baths', 'tax_value', 'area', 'age']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
#############

def wrangled_zillow(train, validate, test):
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = scale_data(prepare_zillow(acquire_zillow_data()))
    
    return train, validate, test


def describe_data(df):
    '''
    This function takes in a pandas dataframe and prints out the shape, datatypes, number of missing values, 
    columns and their data types, summary statistics of numeric columns in the dataframe, as well as the value counts for categorical variables.
    '''
    # Print out the "shape" of our dataframe - rows and columns
    print(f'This dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')
    print('')
    print('--------------------------------------')
    print('--------------------------------------')
    
    # print the datatypes and column names with non-null counts
    print(df.info())
    print('')
    print('--------------------------------------')
    print('--------------------------------------')

    # print the number of missing values per column and the total
    print('Null Values: ')
    missing_total = df.isnull().sum().sum()
    missing_count = df.isnull().sum() # the count of missing values
    value_count = df.isnull().count() # the count of all values
    missing_percentage = round(missing_count / value_count * 100, 2) # percentage of missing values
    missing_df = pd.DataFrame({'count': missing_count, 'percentage': missing_percentage}) # create df
    print(missing_df)
    print(f' \n Total Number of Missing Values: {missing_total} \n')
    df_total = df[df.columns[:]].count().sum()
    proportion_of_nulls = round((missing_total / df_total), 4)
    print(f' Proportion of Nulls in Dataframe: {proportion_of_nulls}\n') 
    print('--------------------------------------')
    print('--------------------------------------')

    # print out summary stats for our dataset
    print('Here are the summary statistics of our dataset')
    print(df.describe().applymap(lambda x: f"{x:0.3f}"))
    print('')
    print('--------------------------------------')
    print('--------------------------------------')

    print('Relative Frequencies: \n')
    # Display top 5 values of each variable within reasonable limit
    limit = 25
    for col in df.columns:
        if df[col].nunique() < limit:
            print(f'Column: {col} \n {round(df[col].value_counts(normalize=True).nlargest(5), 3)} \n')
        else: 
            print(f'Column: {col} \n')
            print(f'Range of Values: [{df[col].min()} - {df[col].max()}] \n')
        print('------------------------------------------')
        print('--------------------------------------')
    