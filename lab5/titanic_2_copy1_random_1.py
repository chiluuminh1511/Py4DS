'''IDE: Google Colab'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import missingno as msno 
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

path_1 = (r'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv')
path_2 = (r'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/test.csv')

train = pd.read_csv(path_1)
test  = pd.read_csv(path_2)

train.head()

def display_data_train(train) : 

    print("Display the first few columns of the training set")
    print(train.head())
    print("-*-*-*"*20)
    print("\n")

    print("Simple training understanding")
    print(train.shape)
    print("-*-*-*"*20)
    print("\n")

    print("Training information")
    print(train.info())
    print("-*-*-*"*20)
    print("\n")

    print("Training Description")
    print(train.describe())
    print("-*-*-*"*20)
    print("\n")
            
    print("Checking missing values")
    print(train.isnull().sum())
   
print(display_data_train(train))

def display_data_test(test) : 
 
   print("Display the first few columns of the test set")
   print(test.head())
   print("-*-*-*"*20)
   print("\n")
   
   print("Simple training understanding")
   print(test.shape)
   print("-*-*-*"*20)
   print("\n")

   print("Testing information")
   print(test.info())
   print("-*-*-*"*20)
   print("\n")

   print("Testing Description")
   print(test.describe())
   print("-*-*-*"*20)
   print("\n")

   print("Checking missing values")
   print(test.isnull().sum())

print(display_data_test(test))

print("Checking whether the dataset is imbalance: ")
def imbalanced_data(train) : 
    print("There are %s people who came across in the disaster represent by class 1"  % ((train.iloc[:,1]) == 1).sum()) 
    print("There are %s people who passed away in the disaster represent by class 0" %(train.iloc[:,1] == 0).sum())
    print ("The dead rate from class 0: ",((train.iloc[:,1] == 0).sum()/len(train["Survived"]))*100.0)
    print("The survial class from class 1: ",((train.iloc[:,1] == 1).sum()/len(train["Survived"]))*100.0)
    print("61.6% people survived from the class 0. There are this feature definitely impactful")

print(imbalanced_data(train))
print("\nI filter on excel and see that Sex is a proportional feature to survival rate. Female and child were saved first during this tragedy")

family_size = train['SibSp']+train['Parch'] + 1
train["family_size"]  = family_size 
draw  = train.groupby(["Survived","family_size"])["family_size"].sum()
print(draw)
draw.plot(kind = 'bar', color = ['r','b'], label = 'Survived')
plt.legend(['0','1'])
print("The family size has a considerable impact on our outcome whether family")

train.Embarked.replace("","NAN",inplace = True)
train.Embarked.fillna('S',inplace = True)
train.Embarked.isnull().sum() # check

train['Age'].hist(bins = 10)

train.Age.replace("","NAN",inplace = True)
train.Age.fillna(np.random.randint(20,31),inplace = True)
train.Age.isnull().sum()
msno.matrix(train)

train['Title'] = train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=train);
plt.xticks(rotation=45);

train['Title'] = train['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
train['Title'] = train['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=train);
plt.xticks(rotation=45);

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
train['Embarked'] = le.fit_transform(train['Embarked'])
train['Title'] = le.fit_transform(train['Title'])

train

X_train = train[['Pclass','Sex','Age','Fare','Embarked','family_size','Title']].values
X_train

y_train = train['Survived'].values
y_train

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train,y_train)

from sklearn.ensemble import GradientBoostingClassifier

Gc = GradientBoostingClassifier(random_state = 0)
Gc.fit(X_train,y_train)

"""# Work with test set """

test

family_size = test['SibSp']+test['Parch'] + 1
test["family_size"]  = family_size 
draw  = train.groupby(["Survived","family_size"])["family_size"].sum()
print(draw)
draw.plot(kind = 'bar', color = ['r','b'], label = 'Survived')
plt.legend(['0','1'])
print("The family size has a considerable impact on our outcome whether family")

test

test['Title'] = test.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=test);
plt.xticks(rotation=45);

test['Title'] = test['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
test['Title'] = test['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=test);
plt.xticks(rotation=45);

test

test.isnull().sum()

test.Age.replace("","NAN",inplace = True)
test.Age.fillna(np.random.randint(20,31),inplace = True)
test.Age.isnull().sum()
msno.matrix(test)

test.isnull().sum()

test["Fare"].hist()

test["Fare"].replace("","NAN",inplace = True)
test["Fare"].fillna(np.random.randint(0,100),inplace = True)

test.isnull().sum()

test



X_train

le = LabelEncoder()
test['Sex'] = le.fit_transform(test['Sex'])
test['Embarked'] = le.fit_transform(test['Embarked'])
test['Title'] = le.fit_transform(test['Title'])

test

X_test = test[['Pclass', 'Sex', 'Age', 'Fare', 'family_size','Title','Embarked']]

X_test



pred = model.predict(X_test)

y_final = ((pred > 0.5).astype(int).reshape(X_test.shape[0]))

print(y_final[:418])

pred_2 = Gc.predict(X_test)

y_final_2 = ((pred_2 > 0.5).astype(int).reshape(X_test.shape[0]))

print(y_final_2[:417])

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_final})
output.to_csv('gender_submission_2.csv', index=False)

