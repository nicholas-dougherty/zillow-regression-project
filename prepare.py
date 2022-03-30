import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def histogram(df, var):
    '''
    Feed a DF and a variable within it and to generate a histogram for the var.
    '''
    print ('Distribution of ' + var)
    df[var].hist()
    plt.grid(False)
    plt.xlabel(var)
    plt.ylabel('Number of Properties')
    plt.show()

def univariate(df, cats, quants):
    '''
    Inputs categorical kitty :3 and quantitative variables from df.
    Outputs bar plot(s) per categorical variable(s)
    and histogram(s) and boxplot(s) for each continuous variable(s).
    Best accomplished by creating a list of each as variables.
    i.e.  cats = ['county'] # bed and bath can be considered as well as
    quants = ['area', 'tax_value']
    '''
    # plot frequencies for each categorical variable
    for var in cats: 
        print('Bar Plot of ' + var)
        bp = df[var].hist()
        plt.xlabel(var)
        plt.ylabel('count')
        bp.grid(False)
        plt.show()
        
    # print histogram for each continuous variable
    for var in quants:
        histogram(df, var)
        # creating boxplot for each variable
        plt.figure(figsize=(10,5))
        sns.boxplot(x=var, data=df,  palette="twilight_shifted")
        plt.title('Distribution of ' + var)
        plt.show()

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
          
          

