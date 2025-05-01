# 10 April 2025
# Support Vector Machine(Classification Algorithm)

import pandas as pd # ctrl+i--> for help
import numpy as np
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv(r"C:\Users\GauravKunal\Desktop\DS\Machine Learning\#2 Classification\#2 Support Vector Machine\logit classification.csv")

# depedent & Indepedent
x = dataset.iloc[ : , [2,3]].values
y = dataset.iloc[: , 4].values


# Train Test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# Feature Scaling - when we apply scaling values in the dataset
# adjust between ranges
# z-score  
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# model building
from sklearn.svm import SVC
classifier = SVC()
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

'''
Usecase - we trained the model with 2 attribute(age,salary)

now we need to apply validation data on the model.
validation data present in companies database we need to fetch
that data.
Now we have final1.csv for validation or future data.

We are trying to predict customer will purchase house/vehicle or not.
using out ml model classifier.
'''

# Validation data 

future_data = pd.read_csv(r"C:\Users\GauravKunal\Desktop\DS\Machine Learning\#2 Classification\#2 Support Vector Machine\final1.csv")

# Copying the dataset
fut_data_copy = future_data

x_future = future_data.iloc[:, [3,4]].values
x_future = sc.fit_transform(x_future)

# Creating empty dataframe
y_future_pred = pd.DataFrame()

 
fut_data_copy['y_future_pred'] = classifier.predict(x_future)

fut_data_copy.to_csv('pred_model_by_svm.csv')

# To get the path
import os
os.getcwd()




