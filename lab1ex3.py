# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
df=pd.read_csv('mushrooms.csv')
df.head()
y=df['class']
X=df.drop('class',axis=1)
y.head()
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.dtypes
df.head()
df.describe()
############################
df=df.drop(["veil-type"],axis=1)
X=df.drop(['class'], axis=1)
Y=df['class']
x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.1)
############################
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Naive Bayes
nb =  GaussianNB()
nb.fit(x_train, y_train)
y_pred=nb.predict(x_test)

print("Accuracy of naive bayees algorithm: ",nb.score(x_test,y_test))
###############################
from sklearn.svm import SVC

svm=SVC(random_state=1)
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
#accuracy
print("accuracy of svm algorithm: ",svm.score(x_test,y_test))
########################
#Logistic Regression
from sklearn.linear_model import LogisticRegression
#fit
lr=LogisticRegression(solver='liblinear')
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
#accuracy
print("accuracy of lr algorithm: ",lr.score(x_test,y_test))
