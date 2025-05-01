# 8 April 2025
# Logistic Regression (aka logit, MaxEnt) classifier.

import pandas as pd # ctrl+i--> for help
import numpy as np
import matplotlib.pyplot as plt


# Importing the dataset
# final1 is validation dataset
dataset = pd.read_csv(r'C:\Users\GauravKunal\Desktop\DS\Machine Learning\#2 Classification\#1 Logistic Regression\logit classification.csv')


# depedent & Indepedent
x = dataset.iloc[ : , [2,3]].values
y = dataset.iloc[: , 4].values


# Train Test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
'''
random_state = 0   ac= 92.5%
random_state = 100  ac-82.5%
random_state = 51   ac= 88.7%
random_state = 41   ac= 78.7%
random_state = 92   ac= 82.5%

random state must should be 0. If you want to high accuracy
'''

# Feature Scaling - when we apply scaling values in the dataset
# adjust between ranges
# z-score  
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

'''
# with Normalizer - range is 0-1
from sklearn.preprocessing import Normalizer
sc = Normalizer()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

With this
ac- 0.725 | bias-0.62 | variance-0.725
'''


# model building
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
# Compare y_pred vs y_test


# Confusion Matrix - in which we find out accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
ac

# bias & variance
bias = classifier.score(x_train, y_train)   
bias

variance = classifier.score(x_test, y_test)
variance


# Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)
'''              precision    recall  f1-score   support

           0       0.72      1.00      0.84        58
           1       0.00      0.00      0.00        22

    accuracy                           0.72        80
   macro avg       0.36      0.50      0.42        80
weighted avg       0.53      0.72      0.61        80
'''


'''
This model has 
ac-92.50 | bias-82 | variance-92.50  --> this is good model

ac-92.50 | bias-45 | variance-92.50  --> Overfit
ac-92.50 | bias-82 | variance-45  --> Underfit

for overcome overfitting we apply cross validation technique
'''


'''
case1: testing 20%, std scaler, ac-92.5,bias-82, var-92.5
case2: testing 20%, normlizer, ac-72.5,b-62, var-72.5
case3: testing 25%, std scaler, ac-89,bias-82, var-89
case4: testing 25%, normlizer, ac-68,bias-63, var-68
case5: testing 20%, without scaling, ac-91,bias-81, var-91 

In case1 we get high accuracy  
'''

# Note: Always remember onething every ml model must require scaling


# =========================================================
# 9 April 2025

'''
Usecase - we trained the model with 2 attribute(age,salary)

now we need to apply validation data on the model.
validation data present in companies database we need to fetch
that data.
Now we have final1.csv for validation or future data.

We are trying to predict customer will purchase house/vehicle or not.
using out ml model classifier.
'''

'''
# My code

# Validation data prediction code

future_data = pd.read_csv(r"C:\Users\GauravKunal\Desktop\DS\Machine Learning\#2 Classification\#1 Logistic Regression\final1.csv")
future_data

# while training we use 2 attribute so for valiation we also need 2 attribute
# otherwise we get dimensionality error
x_future = future_data.iloc[:, [3,4]].values
# validation also scaling is required
x_future = sc.fit_transform(x_future)
x_future

y_future_pred = classifier.predict(x_future)
y_future_pred
'''

# Kodi's code

future_data = pd.read_csv(r"C:\Users\GauravKunal\Desktop\DS\Machine Learning\#2 Classification\#1 Logistic Regression\final1.csv")

# Copying the dataset
fut_data_copy = future_data

x_future = future_data.iloc[:, [3,4]].values
x_future = sc.fit_transform(x_future)

# Creating empty dataframe
y_future_pred = pd.DataFrame()

 
fut_data_copy['y_future_pred'] = classifier.predict(x_future)

fut_data_copy.to_csv('pred_model_by_logit.csv')

# To get the path
import os
os.getcwd()


'''
We have predictions(forcasting) y_future_pred we need to test this
with original data 

predicted_data vs actual_data

actual_data - it generates when customer purchase the vehicle or not

january month customer data
testing phase 1 --> ac - 80%

feburary month customer data
testing phase 2 --> ac - 85%

march month customer data
testing phase 3 --> ac - 92%

avg accuracy - 85% -- now we deploy the model on a website
'''


# We can use this code in use-case like user purchase house/vehicle or not.










