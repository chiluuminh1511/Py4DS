# Them thu vien

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import keras

np.random.seed(2)
'exec(%matplotlib inline)'

# Nhap data

data = pd.read_csv('D:/Bai-tap/Py4DS_Lab2/Py4DS_Lab2/Py4DS_Lab2_Dataset/creditcard.csv')

data.head()

# Tong quan ve data

data.info()
data.corrwith(data.Class).plot.bar(figsize = (20, 10), title = "Correlation with class", fontsize = 15, rot = 45, grid = True)

# Ma tran tuong quan

sn.set(style="white")

# Tinh ma tran tuong quan

corr = data.corr()
corr.head()

# Tao hinh tam giac tren

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Thiet lap hinh dang matplotlib

f, ax = plt.subplots(figsize=(18, 15))

# Tao so do mau phan ra tuy chinh

cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Ve so do

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Kiem tra du lieu bi thieu

data.isna().any()

# Chia dac trung theo trinh do

from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)
data = data.drop(['Time'],axis=1)
data.head()

X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

X.info()
y.head()

# Train mau

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0, criterion = 'gini',  splitter='best', min_samples_leaf=1, min_samples_split=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Decision tree', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index = True)
results

# Mo hinh mang neural

# Them thu vien keras va cac goi

import keras
from keras.models import Sequential
from keras.layers import Dense

# Khoi tao mang luoi

classifier = Sequential()
classifier.add(Dense(units =15 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Phu hop mang luoi vao tap train

classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Du doan ket qua tap thu

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

score = classifier.evaluate(X_test, y_test)
score

# Tao ma tran hon loan

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Xem mau duoc train

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Ma tran hon loan

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))