import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os
df = pd.read_csv("xAPI-Edu-Data.csv")
df.head(10)
df.info()
df.describe()
df.columns
df.rename(index=str, columns={'gender':'Gender', 
                              'NationalITy':'Nationality',
                              'raisedhands':'RaisedHands',
                              'VisITedResources':'VisitedResources'},
                               inplace=True)
df.columns
print("Class Unique Values : ", df["Class"].unique())
print("Topic Unique Values : ", df["Topic"].unique())
print("StudentAbsenceDays Unique Values : ", df["StudentAbsenceDays"].unique())
print("ParentschoolSatisfaction Unique Values : ", df["ParentschoolSatisfaction"].unique())
print("Relation Unique Values : ", df["Relation"].unique())
print("SectionID Unique Values : ", df["SectionID"].unique())
print("Gender Unique Values : ", df["Gender"].unique())
X = df.drop('Class',axis=1)
y = df['Class']
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Cat_Colums = X.dtypes.pipe(lambda X: X[X=='object']).index
for col in Cat_Colums:
    X[col] = label.fit_transform(X[col])
X.head(5)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=52)
###########################################
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Naive Bayes
nb =  GaussianNB()
nb.fit(x_train, y_train)
y_pred=nb.predict(x_test)

print("Accuracy of naive bayees algorithm: ",nb.score(x_test,y_test))
##############################################
from sklearn.svm import SVC

svm=SVC(random_state=1)
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
#accuracy
print("accuracy of svm algorithm: ",svm.score(x_test,y_test))
##########################################
#Logistic Regression
from sklearn.linear_model import LogisticRegression
#fit
lr=LogisticRegression(solver='liblinear')
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
#accuracy
print("accuracy of lr algorithm: ",lr.score(x_test,y_test))

