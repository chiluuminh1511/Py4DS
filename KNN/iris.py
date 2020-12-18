# them thu vien
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

# load data co san trong thu vien sklearn
iris = datasets.load_iris()

# gan nhan cac class
iris_X = iris.data
iris_y = iris.target
print('Number of classes:', len(np.unique(iris_y)))
print('Number of data points:', len(iris_y))

X0 = iris_X[iris_y == 0,:]
print('\nSamples from class 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2,:]
print('\nSamples from class 2:\n', X2[:5,:])

# tao tap train va test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

print('Training size:', len(y_train))
print('Test size    :', len(y_test))

# KNN
# K = 1
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Print results for 20 test data points:')
print('Predicted labels:', y_pred[20:40])
print('Ground truth    :', y_test[20:40])

from sklearn.metrics import accuracy_score
print('Accuracy of 1NN:', accuracy_score(y_test, y_pred))

# K = 10
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy of 10NN with major voting:', accuracy_score(y_test, y_pred))

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy of 10NN (1/distance weights):', accuracy_score(y_test, y_pred))

def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy of 10NN (customized weights):', accuracy_score(y_test, y_pred))