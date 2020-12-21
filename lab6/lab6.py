### It equals one for unsatisfied customers and 0 for satisfied customers 
import seaborn as sns
import csv 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
path_train = ('/home/khoacao/Desktop/dataset/lab06/train.csv')
path_test = ('/home/khoacao/Desktop/dataset/lab06/test.csv') 
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test) 
print(df_train.head())
print(df_test.head())
# preprocessing for train and test 
# check columns and simple analysis

def basic_data_analysis(df_train,df_test) :
    # print length of dataset 
    print("The number of rows in the training set is ",len(df_train))
    print("The number of rows in the test set is ",len(df_test))
    print("The number of columns in the training set is ",len(df_train.columns))
    print("The number of columns in the test set is ",len(df_test.columns))
    # check number of columns
    cols_train = df_train.columns
    cols_test = df_test.columns  
    # statistical table for both train and test
    sta_train = df_train.describe(include  = np.number)
    sta_test  = df_test.describe(include = np.number) 
    # checking mssing values on both set with isnull method 
    mis_train = df_train.isnull().sum()
    mis_test = df_test.isnull().sum() 
    
    return cols_train,cols_test,sta_train,sta_test,mis_train,mis_test

cols_train,cols_test,sta_train,sta_test, mis_train,mis_test = basic_data_analysis(df_train,df_test)

print(cols_train)
print(cols_test)
print(sta_train)
print(sta_test)
print(mis_train)
print(mis_test)
print(pd.set_option('display.max_columns',None))

def target_variable(df_train) : 
     #target
    figure = plt.figure(figsize = (10,5)) 
    target = df_train.TARGET
    val_count = target.value_counts(ascending = True)
    sns.set_theme(style="darkgrid") 
    sns.countplot(x = 'TARGET',data = df_train)
    plt.xlabel("Target")
    plt.ylabel("values")
    plt.title("Visualize target")
    plt.legend()
    plt.show()     
    plt.savefig('countplot.png')
    return val_count 
print(target_variable(df_train)) 
print("The total amount of unstatisfied customers nearly 30010")
print("The total amount of statisfied customer are 73012") 

def format_target(df_train) : 
    # label
    X = df_train.drop('TARGET',axis = 1)
    y = df_train.TARGET.values   
    return X,y 
X,y = format_target(df_train)
print(X)
print(y)
### Applying oversampling and undersampling ############ 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
#######################################################
def format_target(RandomUnderSampler,SMOTE,X,y):
    under = RandomUnderSampler(sampling_strategy = 0.2)
    X_under,y_under = under.fit_resample(X,y)
    X_train,X_test,y_train,y_test = train_test_split(X_under,y_under,test_size = 0.2, random_state = 0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test) 

    df_test_scaled = scaler.transform(df_test)
    
    oversample = SMOTE(random_state = 2020)
    X_sm_train,y_sm_train = oversample.fit_resample(X_train_scaled,y_train) 
    
    return X_sm_train,y_sm_train,X_test_scaled, df_test_scaled,y_test  
X_sm_train, y_sm_train,X_test_scaled, df_test_scaled, y_test = format_target(RandomUnderSampler,SMOTE,X,y)
print(X_sm_train)
print(y_sm_train)
print(df_test_scaled) 
print(X_test_scaled) 
print(y_test) 
### Applying dimension reduction #### 

from sklearn.decomposition import PCA

def PCA_dimentional(PCA,X_sm_train,X_test_scaled,df_test_scaled):  
    pca = PCA(n_components= 10)
    X_train_pca  = pca.fit_transform(X_sm_train) 
    df_test_pca  = pca.transform(df_test_scaled) 
    X_test_pca = pca.fit_transform(X_test_scaled)  

    return X_train_pca, X_test_pca, df_test_pca   

X_train_pca,X_test_pca,df_test_pca = PCA_dimentional(PCA,X_sm_train,X_test_scaled,df_test_scaled) 


PCA_list = [X_train_pca,X_test_pca,df_test_pca] 

for i in PCA_list : 
    if True : 
        print(i) 
# model with sklearn
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC  
from sklearn.neural_network import MLPClassifier
# metrics 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix
def evaluate(X_train_pca,y_sm_train,X_test_pca,df_test_pca) :
    
    list_models = ["LogisticRegression","KNN","DecisionTree",'RandomForestClassifier',"MLPClassifier"]
    
    # declare model
    model1 = LogisticRegression(solver='lbfgs',class_weight='balanced',max_iter=10000)
    model3 = KNeighborsClassifier()
    model4 = DecisionTreeClassifier()
    model5 = MLPClassifier(solver='adam', 
                    activation="relu",
                    # alpha=1e-5,
                    learning_rate="adaptive",
                    learning_rate_init=0.001,
                    hidden_layer_sizes=(100), 
                    max_iter=100,
                    tol=0.001,
                    random_state=1,
                    verbose=True) 

    ## result model
    #results = pd.DataFrame(cols = ['cross_val'], index = list_models) 

    ## fit ## 
    for i,model in enumerate([model1,model3,model4,model5]): 
        model.fit(X_train_pca,y_sm_train)
        predictions = model.predict(X_test_pca)      
    ### metrics ##
        print("Metrics Accuracy",metrics.accuracy_score(y_test,predictions)*100.0)
        print(classification_report(y_test,predictions))
        print(confusion_matrix(y_test,predictions))
        #print(predictions)
        # work with test set and choose the best model to work with 
        if model == model5 : 
          result = model.predict(df_test_pca)
          print(result[:75818])
          data  = pd.DataFrame({"ID":df_test["ID"],"TARGET":result[:75818]})
          data.to_csv("Submission.csv",index = False)
#X_train_pca,y_sm_train,X_test_pca,df_test_pca = evaluate(X_train_pca,y_sm_train,X_test_pca,df_test_pca) 
print( evaluate(X_train_pca,y_sm_train,X_test_pca,df_test_pca))



