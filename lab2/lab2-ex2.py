import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

import seaborn as sns
import os
df = pd.read_csv("xAPI-Edu-Data.csv")
df.head(10)
df.info()
df.describe()
df.columns
df.rename(index=str, columns={'gender':'Gender', 'NationalITy':'Nationality', 'raisedhands':'RaisedHands', 'VisITedResources':'VisitedResources'}, inplace=True)
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
print("accuracy of svm algorithm: ",lr.score(x_test,y_test))

for i in range(1,17):
    print(df.iloc[:,i].value_counts())
    print("*"*20)

sns.pairplot(df,hue='Class')
plt.show()
plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=1,cmap="YlGnBu",annot=True)
plt.yticks(rotation=0)
plt.show()
# plot label 
# kiem tra phan bo cua labels co deu hay khong?
# neu can bang=>co the su dung truc tiep duoc
P_satis = sns.countplot(x = "Class", data = df)

"""nhận xét feature A và B có tương quan cao (0.69),kiểm tra lại mối tương quan của hai fetures"""
df.Class.value_counts(normalize=True).plot(kind='bar')

df.Class.value_counts()

df.Class.value_counts(normalize=True)
plt.show()
plt.subplots(figsize=(20,8))
df["Discussion"].value_counts().sort_index().plot.bar()
plt.title("No. of times vs no. of  student Discussion  on partucular time " )
plt.xlabel("No. of times, student Discussion", fontsize = 14 )
plt.ylabel("No. of student, on partucular time ",fontsize = 14 )
plt.show()

df.Discussion.plot(kind="hist",bins =100 ,figsize= (20,10),grid="True")
plt.xlabel=("Discussion")
plt.legend(loc="upper right")
plt.title("Discussion Histogram")
plt.show()

df.Discussion.plot(kind="hist",bins =10 ,figsize= (20,10),grid="True")
plt.xlabel=("Discussion")
plt.legend(loc="upper right")
plt.title("Discussion Histogram")
plt.show()

Raised_hand=sns.boxplot(x="Class",y="Discussion",data=df)
plt.show()

Facetgrid= sns.FacetGrid(df,hue ="Class" ,height = 6)
Facetgrid.map(sns.kdeplot,"Discussion",shade = True )
Facetgrid.set(xlim=(0,df['Discussion'].max()))
Facetgrid.add_legend()

df.groupby(['ParentschoolSatisfaction'])['Class'].value_counts()

pd.crosstab(df['Class'], df['ParentschoolSatisfaction'])

sns.countplot(x = "ParentschoolSatisfaction", data = df, hue = "Class",palette="bright")
plt.show()

labels=df.ParentschoolSatisfaction.value_counts()
colors=["blue" , "green"]
explode=[0,0]
sizes=df.ParentschoolSatisfaction.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title("Parent school Satisfaction in Data",fontsize=14)
plt.show()
