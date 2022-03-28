from env import host, username, password, get_db_url
import os
import pandas as pd 
import numpy as np

def acquire_zillow_data(cached=True):
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