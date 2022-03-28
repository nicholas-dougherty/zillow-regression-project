import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


def plot_variable_pairs(train, cols, hue=None):
    '''
    This function takes in a df, a list of cols to plot, and default hue=None 
    and displays a pairplot with a red regression line.
    '''
    plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.7}}
    sns.pairplot(train[cols], hue=hue, kind="reg",plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.show()


def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
    """
    Accepts a string name of a categorical variable, 
    a string name from a continuous variable and their dataframe
    then it displays 4 different plots.
    """
   # fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 12))
    
    #plt.subplot(131)
    plt.suptitle(f'{continuous_var} by {categorical_var}', fontsize=18)
    
    sns.lineplot(x=categorical_var, y=continuous_var, data=df)
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    
   # plt.subplot(132)
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind='box', palette='deep')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    
   # plt.subplot(133)
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind="swarm", palette='muted')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    
   # plt.subplot(134)
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind="bar", palette='dark')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
