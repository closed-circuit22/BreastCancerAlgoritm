# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 08:00:55 2019

@author: OMOTESHO
"""

#importing libraries
import pandas as pd
import numpy as np

#importing dataset
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:,2:32].values
y = dataset.iloc[:,1].values

#missing values
dataset.isnull().sum()
dataset.isna().sum()

#Encoding catergorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting dataset to training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#trainng our model with randomforest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#making predictions
y_pred = classifier.predict(X_test)


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#checking accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)