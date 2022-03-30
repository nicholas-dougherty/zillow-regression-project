1.) Project Overview

In this project, I exploit a Zillow dataset and create a predictive machine learning model. The best model predicts single-unit property values in three Southern California counties, pre-dominantly Los Angeles, alongside Orange, and 8% of it is Ventura.  

***
2.) Project Description

Predict tax value based on public assessment, using feature engineered variablls that have been thoroughly cleaned and interpreted.
I was selective with my columns, opting to avoid the excess of incomplete data within MySql. More on that in Acquire. 
***
3.) Goals

Delivery of a jupyter notebook contained herein going; step-by-step journey into the data science pipeline
Present to zillow-themed audience my report. 
***

4.) Deliverables

Actualization of the aforementioned goals. Ipynb for the report. Function modules for prepare and acquire.
A README.md with executive summary, contents, data dictionary, conclusion and next steps, and how to recreate this project.
Presentation slide deck
***

5.) Hypothesis

a.) Greater area, greater home value. 

b.) More baths, less aged since the property was built. 

c.) Older property, reduced value.

d.) The older the property, the more likely it will be under the mean area sq. footage.

***

Area is the best feature among my query for predicting home value based on tax assessments, followed up by bathrooms and county (orange or ventura).
Age could be more useful if binned appropriately, was incorporated in our our model regardless of its negative correlation, but there may be other features we could look into next time.
Location would be far more valuable if Los Angeles wasn't predominantly represented in the sample. 

Next steps would be:

Randomly sample to even out the county distribution into thirds.
Deepen my understanding of machine learning, I failed to understand the mathematical implications of any of this. 
***
***
***

# 8.) The Pipeline:

8a.) Planning 

Goal: First, organize myself on Trello. Get more familiarized with group by, sorting, and more deeply diving into the data frame via filtering. 

I also want to look into other features, like age and FIPS code, and see if that will also correlate to property value. A lot of these features could play hand in hand and help my model make better predictions.

Hypotheses: See number 5 above. 

***
8b.) Acquire 

Goal: In this stage, I used a connection URL to access the CodeUp database. Using a SQL query, I brought in the Zillow dataset with only properties set for single use, and were sold in between May-August 2017. I turned it into a pandas dataframe and created a .csv in order to use it for the rest of the pipeline. This process can be repeated, through the use of your own env and access to the credentials. Alternatively, there is data on kaggle, though it may not match this, and filtering it may be difficult. 

***
8c.) Prep 

Goal: Have Zillow dataset that is split into train, validate, test, and ready to be analyzed by the end. Assure appropriate datatypes are at hand, and that missing values/duplicates/outliers are addressed. If you want to mimic me directly, remove outliers on a county by county basis and then concat those three dataframes back into one. Put this in a prep.py. Counties were encoded and ventura was dropped as a dummy. A small number of nulls were detected and removed. I wrangled the data into train, validate, test, X_train, y_train, X_validate, y_validate, X_test, and y_test. 

For the next step: run statistical testing and visualize data to find relationships between variables. 
***
8d.) Explore 

Visualizations. Mito will be an essential import for you to do it as I did but plotly express will yield the same interactive maps. Find answers. Use the visuals and statistics tests to help answer your questions. I plotted distributions, made sure nothing was out of the ordinary after cleaning the dataset.

Conducted pearsons r, spearmans are, independent t-tests


For the next step: Select features to use to build a regression model that predicts property value 
***
8e.) Modeling and Evaluation 📈

Goal: develop a regression model that performs better than the baseline.

Take all of this part with a grain of salt as far as conclusions go. 

8f.) Delivery

Soon will be a presentation on zoom, this is an element of delivery too. 
***
9.) Conclusion

If you're still reading this, and have truly read it, I heart you. 
***
10.) Data Dictionary

Column Name	Renamed	Info
bathroomcnt	baths	number of bathrooms
bedroomcnt	beds	number of bedrooms
calculatedfinishedsquarefeet	area	number of square feet
fips	county	FIPS code (for county)
propertylandusetypeid	N/A	Type of property used just for joining
yearbuilt	N/A	The year the property was built Converted to age
taxvaluedollarcnt	tax_value	Property's tax value in dollars
transactiondate	N/A	Day the property was purchased just for joining
age	N/A	2017-yearbuilt (to see the age of the property)
***

11.) How to Recreate Project

You'll need your own username/pass/host credentials in order to use the get_connection function in my acquire.py to access the Zillow database

Have a copy of my acquire, prep, explore .py files. You can adjust the features to use, how to handle outliers, etc. or keep it as it is. 

Install mito if you want. Lux is awesome too. 
