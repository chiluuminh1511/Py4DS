import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

data= pd.read_csv("creditcard.csv")

count_Class=pd.value_counts(data["Class"], sort= True)
count_Class.plot(kind= 'bar')

No_of_frauds= len(data[data["Class"]==1])
No_of_normals = len(data[data["Class"]==0])
print("The number of fraudulent transactions( Class 1) are: ", No_of_frauds)
print("The number of normal transactions( Class 0) are: ", No_of_normals)
total= No_of_frauds + No_of_normals
Fraud_percent= (No_of_frauds / total)*100
Normal_percent= (No_of_normals / total)*100
print("Class 0 percentage = ", Normal_percent)
print("Class 1 percentage = ", Fraud_percent)

# Danh sach chi so gian lan
fraud_index= np.array(data[data["Class"]==1].index)

# Danh sach chi so binh thuong
normal_index= data[data["Class"]==0].index

# Chon cac chi so binh thuong ngau nhien bang voi chi so gian lan
random_normal_indices= np.random.choice(normal_index, No_of_frauds, replace= False)
random_normal_indices= np.array(random_normal_indices)

# Noi chi so gian lan va chi so binh thuong de tao danh sach cac chi so
undersampled_indices= np.concatenate([fraud_index, random_normal_indices])

# Su dung cac mau khong chi so de lap dataframe
undersampled_data= data.iloc[undersampled_indices, :]

print(undersampled_data.head())

No_of_frauds_sampled= len(undersampled_data[undersampled_data["Class"]== 1])

No_of_normals_sampled = len(undersampled_data[undersampled_data["Class"]== 0])

print("The number of fraudulent transactions( Class 1) are: ", No_of_frauds_sampled)
print("The number of normal transactions( Class 0) are: ", No_of_normals_sampled)
total_sampled= No_of_frauds_sampled + No_of_normals_sampled
print("The total number of rows of both classes are: ", total_sampled)

Fraud_percent_sampled= (No_of_frauds_sampled / total_sampled)*100
Normal_percent_sampled= (No_of_normals_sampled / total_sampled)*100
print("Class 0 percentage = ", Normal_percent_sampled)
print("Class 1 percentage = ", Fraud_percent_sampled)

count_sampled=pd.value_counts(undersampled_data["Class"], sort= True)
count_sampled.plot(kind= 'bar')

# Scale amount

sc= StandardScaler()
undersampled_data["scaled_Amount"]=  sc.fit_transform(undersampled_data.iloc[:,29].values.reshape(-1,1))

# Bo cot thoi gian va old amount
undersampled_data= undersampled_data.drop(["Time","Amount"], axis= 1)

print(undersampled_data.head())

X= undersampled_data.iloc[:, undersampled_data.columns != "Class"].values

y= undersampled_data.iloc[:, undersampled_data.columns == "Class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)
print("The split of the under_sampled data is as follows")
print("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))

# SVM
classifier= SVC(C= 1, kernel= 'rbf', random_state= 0)
classifier.fit(X_train, y_train.ravel())

# Du doan lop su dung X test
y_pred = classifier.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred)

def confusion_matrix_1(CM):
    fig, ax = plot_confusion_matrix(conf_mat=CM)
    plt.title("The Confusion Matrix 1 of Undersampled dataset")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    print("The accuracy is "+str((CM[1,1]+CM[0,0])/(CM[0,0] + CM[0,1]+CM[1,0] + CM[1,1])*100) + " %")
    print("The recall from the confusion matrix is "+ str(CM[1,1]/(CM[1,0] + CM[1,1])*100) +" %")
confusion_matrix_1(cm1)

accuracies = cross_val_score(estimator = classifier, X=X_train, y = y_train.ravel(), cv = 10)
mean_accuracy= accuracies.mean()*100
std_accuracy= accuracies.std()*100
print("The mean accuracy in %: ", accuracies.mean()*100)
print("The standard deviation in % ", accuracies.std()*100)
print("The accuracy of our model in % is betweeen {} and {}".format(mean_accuracy-std_accuracy, mean_accuracy+std_accuracy))

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train.ravel())
best_accuracy = grid_search.best_score_
print("The best accuracy using gridSearch is", best_accuracy)

best_parameters = grid_search.best_params_
print("The best parameters for using this model is", best_parameters)

classifier_with_best_parameters =  SVC(C= best_parameters["C"], kernel= best_parameters["kernel"], random_state= 0)
classifier_with_best_parameters.fit(X_train, y_train.ravel())

# Du doan Class
y_pred_best_parameters = classifier_with_best_parameters.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred_best_parameters)

def confusion_matrix_2(CM):
    fig, ax = plot_confusion_matrix(conf_mat= CM)
    plt.title("The Confusion Matrix 2 of Undersampled dataset using best_parameters")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    print("The accuracy is "+str((CM[1,1]+CM[0,0])/(CM[0,0] + CM[0,1]+CM[1,0] + CM[1,1])*100) + " %")
    print("The recall from the confusion matrix is "+ str(CM[1,1]/(CM[1,0] + CM[1,1])*100) + " %")
confusion_matrix_2(cm2)
# So sanh 2 ma tran
confusion_matrix_1(cm1)

# Tao data moi de test model
datanew= data.copy()

# test model voi ca data
datanew["scaled_Amount"]=  sc.fit_transform(datanew["Amount"].values.reshape(-1,1))

# Bo cot time va amount
datanew= datanew.drop(["Time","Amount"], axis= 1)

# Chia x va y de fit model
X_full= datanew.iloc[:, undersampled_data.columns != "Class"].values

y_full= datanew.iloc[:, undersampled_data.columns == "Class"].values

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size= 0.25, random_state= 0)

print("The split of the full dataset is as follows")
print("X_train_full: ", len(X_train_full))
print("X_test_full: ", len(X_test_full))
print("y_train_full: ", len(y_train_full))
print("y_test_full: ", len(y_test_full))

# y pred toan bo data va ma tran confusion toan bo data

y_pred_full_dataset= classifier_with_best_parameters.predict(X_test_full)

cm3 = confusion_matrix(y_test_full, y_pred_full_dataset)

def confusion_matrix_3(CM):
    fig, ax = plot_confusion_matrix(conf_mat= CM)
    plt.title("The Confusion Matrix 3 of full dataset using best_parameters")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    print("The accuracy is "+str((CM[1,1]+CM[0,0])/(CM[0,0] + CM[0,1]+CM[1,0] + CM[1,1])*100) + " %")
    print("The recall from the confusion matrix is "+ str(CM[1,1]/(CM[1,0] + CM[1,1])*100) +" %")
confusion_matrix_3(cm3)

# So sanh 2 ma tran
confusion_matrix_2(cm2)
