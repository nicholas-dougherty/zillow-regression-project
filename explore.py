from mitosheet import *
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import scipy.stats as stats

def plot_categorical_and_continuous_vars(train, categ_vars, cont_vars):
    cats = ['los_angeles', 'orange', 'beds', 'baths']
    quant_dogs = ['tax_value', 'age', 'area']
    for dog in quant_dogs:
        for cat in cats:

            plt.figure(figsize=(40,5))
            
            # barplot of average values
            plt.subplot(131)
            sns.barplot(data=train,
                        x=cat,
                        y=dog)
            plt.axhline(train[dog].mean(), 
                        ls='--', 
                        color='black')
            plt.title(f'{dog} by {cat}', fontsize=14)
            
            # box plot of distributions
            plt.subplot(132)
            sns.boxplot(data=train,
                          x=cat,
                          y=dog)
            
            plt.show()
            


def tax_correlations(train):
    '''
    Receives the zillow train sample, then uses pandas creates a
    heatmap of the correlations among quantitative features and target. 
    Is also used for X_train_scaled. 
    '''
    # establish figure size
    plt.figure(figsize=(6,6))
    # create the heatmap using the correlation dataframe created above
    heatmap = sns.heatmap(train.corr(), cmap='Purples', annot=True)
    # establish a plot title
    plt.title('Correlation Before Considering Scalers')
    # display the plot
    plt.show()

    # establish figure size
    plt.figure(figsize=(6,6))
    # creat the heatmap using the correlation dataframe created above
    heatmap = sns.heatmap(train.corr()[['tax_value']].sort_values(by='tax_value', ascending=False), vmin=-.5, vmax=.5, annot=True,cmap='flare')
    # establish a plot title
    plt.title('Features Correlated with Tax Value')
    # display the plot
    plt.show()



def plot_variable_pairs(train, hue=None):
    '''
    This function takes in a df, a list of cols to plot, and default hue=None 
    and displays a pairplot with a red regression line.
    '''
    cols = ['tax_value', 'age', 'area', 'beds', 'baths']
    plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.7}}
    sns.pairplot(train[cols], hue=hue, kind="reg",plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.show()
    
def explore_multivariate(train):
    '''
    This function takes in takes in a dataframe and generates boxplots showing
    the target variable for each class of the categorical variables 
    against the quantitative variables.
    '''
    cat_vars = ['beds', 'baths', 'orange']
    quant_dogs = ['area', 'los_angeles']
    for cat in cat_vars:
        for dog in quant_dogs:
            sns.lmplot(x=dog, y='age', data=train, scatter=True, hue=cat, palette ='muted')
            plt.xlabel(dog)
            plt.ylabel('age')
            plt.title(dog + ' vs ' + 'age' + ' by ' + cat)
            plt.show()  
    
def show_maps_tv_age(train):
    # Filter the dataframe so that it does not crash the browser
    train_filtered = train.copy()
    
    # Construct the graph and style it. Further customize your graph by editing this code.
        # See Plotly Documentation for help: https://plotly.com/python/plotly-express/
    fig = px.density_contour(train_filtered, x='tax_value', y='age')
    fig.update_layout(
        title='Homes from the 1960s are densely pocketed under 200k assessed tax value.',
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=.05
            )
        )
    )
    fig.show(renderer="iframe")
    
def show_maps_age_area(train):  
    # Filter the dataframe so that it does not crash the browser
    train_filtered = train.copy()
    
    # Construct the graph and style it. Further customize your graph by editing this code.
    # See Plotly Documentation for help: https://plotly.com/python/plotly-express/
    # Construct the graph and style it. Further customize your graph by editing this code.
    # See Plotly Documentation for help: https://plotly.com/python/plotly-express/
    fig = px.density_heatmap(train, x='area', y='age')
    fig.update_layout(
        title='Homes from the 60s have noticably less square-footage among the training sample',
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=.05
            )
        )
    )
    fig.show(renderer="iframe")
    
def baths_and_age(train):

    # Construct the graph and style it. Further customize your graph by editing this code.
    # See Plotly Documentation for help: https://plotly.com/python/plotly-express/
    fig = px.violin(train, x='baths', y='age')
    fig.update_layout(
        title='Older Homes have fewer bathrooms, and few before the 60s have half-baths',
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=.05
            )
        )
    )
    fig.show(renderer="iframe")
    
    
                    
def testing_area(train):
    
    # t test for area and value, I used 1679 sqft because it is the median area of all selected properties.
    # The results suggest they are related to each other.
    null_hypothesis = "Median home value 1,679 sq ft, whether increased or decreased, is independent of assessed value."
    alternative_hypothesis = "Median home value 1,679 sq ft, whether increased or decreased, is not independent of assessed value."
    a = 0.01 #a for alpha 

    bigger_houses = train[train.area>=1679]
    smaller_houses = train[train.area<1679]
    t, p = stats.ttest_ind(bigger_houses.tax_value, smaller_houses.tax_value)

    if p < a:
        print(f'Reject null hypothesis that: {null_hypothesis}')
    else:
        print(f'Fail to reject null hypothesis that: {null_hypothesis} There is not sufficient evidence to reject it.')
                    
def test_age(train): 
    alpha = 0.01
    r, p = stats.spearmanr(train.age, train.tax_value)
    r, p, alpha

    if p < alpha:
        print('''Having successfully rejected the null, a home's value in some way depends on its age ''')
    else:
        print('We cannot reject the null hypothesis')
                    
def test_baths(train):
    alpha = 0.01
     # conduct the stats test and store values for p and r
    r, p = stats.spearmanr(train.baths, train.age)
    # display the p and r values
    print('\nr = ', round(r, 2))
    print('p = ', round(p, 3))
    # compare p to alpha, display whether to reject the null hypothesis
    if p < alpha:
        print('\nReject H0')
        print("Thus, number of baths and age have a linear relationship.")
    else:
        print('\nFail to Reject H0')
                    
                    
                    