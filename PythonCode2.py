# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:02:33 2021

@author: Group 7
"""

''' Change this file_path to your file location '''
file_path = 'C:/Centennial notes/Fall2021/309-Data Warehousing/Grp Project/Main/Bicycle_Thefts.csv';

#%% 1. Data Exploration
import pandas as pd
data_Bicycle=pd.read_csv(file_path)
######### get first five records
data_Bicycle.head(5)

######### get the shape of data
data_Bicycle.shape
########  get the column values
data_Bicycle.columns.values
# or
print(data_Bicycle.columns.values)
# or
for col in data_Bicycle.columns: 
    print(col) 

##### get the types of columns
data_Bicycle.dtypes

data_Bicycle.info()
from pandas_profiling import ProfileReport
profile = ProfileReport(data_Bicycle, title="Pandas Profiling Report", explorative=True)
profile.to_file("profiling_report.html")

###### create summaries of data
pd.set_option("display.max.columns", None)
pd.set_option("display.precision", 2)
pd.set_option('display.max_columns', None)
data_Bicycle.describe()
print(data_Bicycle.describe())

data_Bicycle.describe(include=object)

data_Bicycle["Bike_Type"].value_counts()

data_Bicycle["Occurrence_Month"].value_counts()


rows_without_missing_data = data_Bicycle.dropna()

rows_without_missing_data.shape

# Check missing values
data_Bicycle.isnull()
data_Bicycle.isnull().sum()

"---------------------------------------------------------------------------------------------------------------------------------------------------"

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data1_Bicycle = pd.read_csv(file_path)
print(data1_Bicycle.columns.values)


plt.figure(figsize=(12,4)) 
sns.countplot(x='Occurrence_Month', data=data1_Bicycle)

sns.countplot(x='Occurrence_Year', data=data1_Bicycle)

plt.figure(figsize=(7,4)) 
sns.countplot(x='Occurrence_DayOfWeek', data=data1_Bicycle)

plt.figure(figsize=(11,4)) 
sns.countplot(x='Report_Month', data=data1_Bicycle)

plt.figure(figsize=(7,4)) 
sns.countplot(x='Report_DayOfWeek', data=data1_Bicycle)

plt.figure(figsize=(10,4))
sns.countplot(x='Division', data=data1_Bicycle)

sns.countplot(x='City', data=data1_Bicycle)

plt.figure(figsize=(10,4))
sns.countplot(x='Premises_Type', data=data1_Bicycle)

sns.countplot(x='Bike_Type', data=data1_Bicycle)

sns.countplot(x='Status', data=data1_Bicycle)

plt.show();

#%% 2. Data modelling
import pandas as pd
data_Bicycle=pd.read_csv(file_path)

original_columns = data_Bicycle.columns #35 columns
# Remove columns that are not included in the Attributes documented in the dataset documentation - X, Y, ObjectId2
extra_columns = ['X', 'Y', 'ObjectId2']
data_Bicycle = data_Bicycle.drop(extra_columns, axis=1)
# Remove identifier columns
identifier_columns = ['OBJECTID', 'event_unique_id', 'Hood_ID']
data_Bicycle = data_Bicycle.drop(identifier_columns, axis=1)
data_Bicycle.shape # 29 columns

# Handle missing values
# Check the percentage of missing values -> if more than 25%, it's better to remove that feature
columns_have_missing_values = []
for i in data_Bicycle.columns:
    percentage = data_Bicycle[i].isnull().sum()/len(data_Bicycle)*100
    if (percentage > 0):
        columns_have_missing_values.append([i, percentage])
    
print(columns_have_missing_values)

# Drop Bike_Model column because the percentage of missing values is 37.73% > 25%
data_Bicycle = data_Bicycle.drop(['Bike_Model'], axis=1)

# Use the most frequent category to fill in the categorical variables
data_Bicycle['Bike_Make'].describe()
data_Bicycle['Bike_Make'].fillna(data_Bicycle['Bike_Make'].mode()[0],inplace=True)
data_Bicycle['Bike_Colour'].describe()
data_Bicycle['Bike_Colour'].fillna(data_Bicycle['Bike_Colour'].mode()[0],inplace=True)

# Use the average to fill in the missing Cost_of_Bike
data_Bicycle['Cost_of_Bike'].describe()
data_Bicycle['Cost_of_Bike'].fillna(data_Bicycle['Cost_of_Bike'].mean(),inplace=True)

# Check missing values again - at this point, all missing data is handled
data_Bicycle.isnull().sum()
data_Bicycle.shape #28


# Remove duplicate columns - Occurrence_Date, Report_Date
data_Bicycle = data_Bicycle.drop(['Occurrence_Date', 'Report_Date'], axis = 1)
# Remove less varient column - City (Value are almost the Toronto)
data_Bicycle = data_Bicycle.drop(['City'], axis = 1)
# Remove less related column - Division
data_Bicycle = data_Bicycle.drop(['Division'], axis = 1)
data_Bicycle.shape #24



# Categorical Data Management
# Change the target column - Status to integer
data_Bicycle['Status'].unique()
# Reduce the categories of the 'Status' column from three to two (binary)
# Assumption: Consider 'UNKNOWN' as 'STOLEN'
import numpy as np
data_Bicycle['Status']=np.where(data_Bicycle['Status'] =='UNKNOWN', 'STOLEN', data_Bicycle['Status'])
print(data_Bicycle['Status'].value_counts())

# Change the Status column from object to integer
data_Bicycle['Status']=(data_Bicycle['Status']=='RECOVERED').astype(int) # 1: Recovered 0: Stolen
print(data_Bicycle['Status'].value_counts())


import pandas as pd
categorical_columns=['Primary_Offence','Occurrence_Month','Occurrence_DayOfWeek', 'Report_Month', 
                     'Report_DayOfWeek','NeighbourhoodName','Location_Type','Premises_Type',
                     'Bike_Make','Bike_Type','Bike_Colour']

# Perform chi-squre test to test the relationship
from scipy.stats import chi2_contingency
cols_to_be_dropped_chi = []
for cat in categorical_columns:
    cross_tab_result = pd.crosstab(data_Bicycle[cat], data_Bicycle['Status'])
    chi_square_result = chi2_contingency(cross_tab_result)
    if (chi_square_result[1] > 0.05): # means the relationship is not significant
        cols_to_be_dropped_chi.append(cat)
print(cols_to_be_dropped_chi)
data_Bicycle = data_Bicycle.drop(cols_to_be_dropped_chi, axis=1)
data_Bicycle.shape #19

print(categorical_columns) #11
for col in cols_to_be_dropped_chi:
    categorical_columns.remove(col)
print(categorical_columns) #6

# Create the dummy variables
for col in categorical_columns:
    cat_list='col'+'_'+col
    print(cat_list)
    cat_list = pd.get_dummies(data_Bicycle[col], prefix=col)
    data_Bicycle1=data_Bicycle.join(cat_list)
    data_Bicycle=data_Bicycle1
data_Bicycle.head(2)
data_Bicycle.shape #1223

# Remove the original columns
data_Bicycle_columns=data_Bicycle.columns.values.tolist()
to_keep=[i for i in data_Bicycle_columns if i not in categorical_columns]
data_Bicycle_final=data_Bicycle[to_keep]
final_columns = data_Bicycle_final.columns.values
data_Bicycle_final.head(2)
data_Bicycle_final.shape #1217


# Feature Scaling - Standardization
from sklearn.preprocessing import StandardScaler
numeric_columns = ['Occurrence_Year', 'Occurrence_DayOfMonth', 'Occurrence_DayOfYear', 'Occurrence_Hour',
                  'Report_Year', 'Report_DayOfMonth', 'Report_DayOfYear', 'Report_Hour', 'Bike_Speed', 'Cost_of_Bike',
                  'Longitude', 'Latitude']

# apply standardization on numerical features
#create and fit scaler
scaler = StandardScaler()
scaler.fit(data_Bicycle_final[numeric_columns])

#scale selected data
data_Bicycle_final[numeric_columns] = scaler.transform(data_Bicycle_final[numeric_columns])


# Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
data_Bicycle_final_columns=data_Bicycle_final.columns.values.tolist()
Y=['Status']
X=[i for i in data_Bicycle_final_columns if i not in Y ]
len(Y)
len(X)


# Carry out feature selection
# We have many features so let us carryout feature selection from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 30)
rfe = rfe.fit(data_Bicycle_final[X],data_Bicycle_final[Y])
print(rfe.support_)
print(rfe.ranking_)
ranks = rfe.ranking_

feature_selection_result = list(zip(X, ranks))
feature_selected = []
for i in feature_selection_result:
    if (i[1] == 1):
        feature_selected.append(i[0])


print(feature_selected)
len(X)
len(Y)

# Update X and Y with selected features
X=data_Bicycle_final[feature_selected]
Y=data_Bicycle_final['Status']
X.shape
Y.shape


# split the data into 70% training and 30% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape


# Check if there's imbalance issue in data_Bicycle_final
import matplotlib.pyplot as plt
data_Bicycle_final.value_counts(data_Bicycle_final['Status']).plot.bar()
plt.title('Status class histogram')
plt.xlabel('Status')
plt.ylabel('Frequency')
data_Bicycle_final['Status'].value_counts()


# Handle imbalanced dataset
print("Before SMOTE, counts of label '1' ('RECOVERED'): {}".format(sum(Y_train==1)))
print("Before SMOTE, counts of label '0' ('STOLEN'): {} \n".format(sum(Y_train==0)))

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train_smote, Y_train_smote = sm.fit_resample(X_train, Y_train.ravel())

print('After SMOTE, the shape of X_train: {}'.format(X_train_smote.shape))
print('After SMOTE, the shape of Y_train: {} \n'.format(Y_train_smote.shape))

print("After SMOTE, counts of label '1' ('RECOVERED'): {}".format(sum(Y_train_smote==1)))
print("After SMOTE, counts of label '0' ('STOLEN'): {}".format(sum(Y_train_smote==0)))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_smote, Y_train_smote)

#predicting
y_pred = logreg.predict(X_test)
print(y_pred)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))

#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))

#

