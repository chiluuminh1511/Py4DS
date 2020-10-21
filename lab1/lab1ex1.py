# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns
#pandas options
pd.set_option('display.max_colwidth',1000,'display.max_rows',None,\
              'display.max_columns',None)
    
#%matplotlib inline
# Plotting options
mpl.style.use('ggplot')
sns.set(style='whitegrid')
#

path = 'spam.csv'

# Read in the data into a pandas dataframe 
dataset_pd = pd.read_csv(path)

# Read in the data into a numpy array
dataset_np = np.genfromtxt(path, delimiter=',')
print(dataset_pd.shape)
print(dataset_np.shape)
print(dataset_pd.head(5))
print(dataset_np[0:5,:])
X = dataset_np[:, :-1]
y = dataset_np[:, -1]
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

# Split dataset #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, "\n\n", X_test.shape, "\n\n", y_train.shape, "\n\n", y_test.shape, "\n\n")


# Import ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Import metrics to evaluate the perfomance of each model
from sklearn import metrics

# Import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier()

##############################

# Fit Decision Tree Classifier
clf = clf.fit(X_train, y_train)
# Predict testset
y_pred = clf.predict(X_test)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")
############################### Fit Logistic Regression Classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)
# Predict testset
y_pred = lr.predict(X_test)
# Evaluate performance of the model
print("LR:  ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross-validation
scores = cross_val_score(lr, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")

clf = SVC()

# Fit SVM Classifier
clf.fit(X_train, y_train)
# Predict testset
y_pred = clf.predict(X_test)
# Evaluate performance of the model
print("SVM Accuracy:  ", metrics.accuracy_score(y_test, y_pred))
print("\n")
# Evaluate a score by cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")
##############################
# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train, y_train)
# Predict testset
y_pred=rdf.predict(X_test)
# Evaluate performance of the model
print("RDF:  ", metrics.accuracy_score(y_test, y_pred))
print("\n")
# Evaluate a score by cross-validation
scores = cross_val_score(rdf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")
##############################
# Fit Logistic Regression Classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)
# Predict testset
y_pred = lr.predict(X_test)
# Evaluate performance of the model
print("LR:  ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross-validation
scores = cross_val_score(lr, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")
##############################
