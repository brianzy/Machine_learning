from simforest import SimilarityForest
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.datasets import make_blobs

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

dataset = pd.read_csv("Heart_data.csv")

#convert all string values to nan
dataset = dataset.convert_objects(convert_numeric=True)

X1 = dataset.iloc[:, :-1].values
y1= dataset.iloc[:, 13]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy ="mean", axis = 0)
imputer = imputer.fit(X1[:, 0:13])   
X1[:, 0:13] = imputer.transform(X1[:, 0:13])
y1=np.array(y1)


if __name__ == '__main__':
    #test random data
    X, y = make_blobs(n_samples=1000, centers=[(0, 0), (1, 1)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    sf = SimilarityForest(n_estimators=20, n_axes=1)
    sf.fit(X_train, y_train)
    sf_pred = sf.predict(X_test)

    sf_prob = sf.predict_proba(X_test)

    print('Similarity Forest for random sample')
    #print(sf_prob[:, 1])
    #print(y_test)
    print(accuracy_score(y_test, sf_pred))

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)


    rf_prob = rf.predict_proba(X_test)

    print('Random Forest for random sample')
    #print(rf_prob[:, 1])
   # print(y_test)
    print(accuracy_score(y_test, rf_pred))

    svm = SVC(gamma='auto', probability=True)
    svm.fit(X_train, y_train)
    svm_pred=svm.predict(X_test)


    svm_prob = svm.predict_proba(X_test)

    print('SVM for random sample')
    print(accuracy_score(y_test, svm_pred))
    
    #logistic Regression
    lr = LogisticRegression(random_state=0)
    lr.fit(X_train, y_train)
    lr_pred=lr.predict(X_test)
    print('Logistic regression for random sample')
    print(accuracy_score(y_test, lr_pred))
   
    dict_acc_1={'Simlarity forest ':accuracy_score(y_test, sf_pred), \
    'Random forest ':accuracy_score(y_test, rf_pred), 'Svm ':accuracy_score(y_test, svm_pred),\
    'Logistic regression ':accuracy_score(y_test, lr_pred)}
    df_acc_1=pd.DataFrame(list(dict_acc_1.items()))
    df_acc_1.columns=['Model','Accuracy']
    df_acc_1.Accuracy=df_acc_1.Accuracy*100
    ax=sns.barplot(data=df_acc_1,x='Model',y='Accuracy')
    plt.title("Model comparision for randomly generate data")
    plt.ylabel('Accuracy(%)')
    plt.xlabel("Model")
    plt.legend()
    plt.show()
    #print(df_acc_1)

#test breast cancer

    cancer = load_breast_cancer() 
    X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=0)
    sf1 = SimilarityForest(n_estimators=10, n_axes=1)
    sf1.fit(X_train_dia, y_train_dia)

    sf1_pred = sf1.predict(X_test_dia)
    sf1_prob = sf1.predict_proba(X_test_dia)

    print('Similarity Forest for breast cancer')
    #print(sf1_prob[:, 1])
    #print(y_test_dia)
    print(accuracy_score(y_test_dia, sf1_pred))

    rf1 = RandomForestClassifier()
    rf1.fit(X_train_dia, y_train_dia)

    rf1_pred = rf1.predict(X_test_dia)
    rf1_prob = rf1.predict_proba(X_test_dia)

    print('Random Forest for breast cancer')
    #print(rf1_prob[:, 1])
    #print(y_test_dia)
    print(accuracy_score(y_test_dia, rf1_pred))

    svm1 = SVC(gamma='auto', probability=True)
    svm1.fit(X_train_dia, y_train_dia)
    svm1_pred=svm1.predict(X_test_dia)
    svm1_prob = svm1.predict_proba(X_test_dia)

    print('SVM for breast cancer')
   #print(svm1_prob[:, 1])
    #print(y_test_dia)
    print(accuracy_score(y_test_dia, svm1_pred))
    #logistic Regression
    lr1 = LogisticRegression(random_state=0)
    lr1.fit(X_train_dia, y_train_dia)
    lr1_pred=lr1.predict(X_test_dia)
    print('Logistic regression for breast cancer')
    print(accuracy_score(y_test_dia, lr1_pred))

    dict_acc_2={'Simlarity forest ':accuracy_score(y_test_dia, sf1_pred), \
    'Random forest ':accuracy_score(y_test_dia, rf1_pred), \
    'Svm ':accuracy_score(y_test_dia, svm1_pred),'Logistic regression ':accuracy_score(y_test_dia, lr1_pred)}
    df_acc_2=pd.DataFrame(list(dict_acc_2.items()))
    df_acc_2.columns=['Model','Accuracy']
    df_acc_2.Accuracy=df_acc_2.Accuracy*100
    ax=sns.barplot(data=df_acc_2,x='Model',y='Accuracy')
    plt.title("Model comparision for breast cancer data")
    plt.ylabel('Accuracy(%)')
    plt.xlabel("Model")
    plt.legend()
    plt.show()

#test heart dataset
    X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(X1, y1, test_size=0.2, random_state=0)
    sf2 = SimilarityForest(n_estimators=10, n_axes=1)
    sf2.fit(X_train_hr, y_train_hr)

    sf2_pred = sf2.predict(X_test_hr)
    sf2_prob = sf2.predict_proba(X_test_hr)

    print('Similarity Forest for heart disease')
    #print(sf2_prob[:, 1])
    #print(y_test_hr)
    print(accuracy_score(y_test_hr, sf2_pred))

    rf2 = RandomForestClassifier()
    rf2.fit(X_train_hr, y_train_hr)

    rf2_pred = rf2.predict(X_test_hr)
    rf2_prob = rf2.predict_proba(X_test_hr)

    print('Random Forest for heart disease')
   # print(rf2_prob[:, 1])
    #print(y_test_hr)
    print(accuracy_score(y_test_hr, rf2_pred))

    svm2 = SVC(gamma='auto', probability=True)
    svm2.fit(X_train_hr, y_train_hr)
    svm2_pred=svm2.predict(X_test_hr)
    svm2_prob = svm2.predict_proba(X_test_hr)

    print('SVM for heart disease')
    #print(svm2_prob[:, 1])
    #print(y_test_hr)
    print(accuracy_score(y_test_hr, svm2_pred))
    #logistic Regression
    lr2 = LogisticRegression(random_state=0)
    lr2.fit(X_train_hr, y_train_hr)
    lr2_pred=lr2.predict(X_test_hr)
    print('Logistic regression for heart desease')
    print(accuracy_score(y_test_hr, lr2_pred))

    dict_acc_3={'Simlarity forest ':accuracy_score(y_test_hr, sf2_pred), \
    'Random forest ':accuracy_score(y_test_hr, rf2_pred), \
    'Svm ':accuracy_score(y_test_hr, svm2_pred),\
    'Logistic Regression':accuracy_score(y_test_hr, lr2_pred)}
    df_acc_3=pd.DataFrame(list(dict_acc_3.items()))
    df_acc_3.columns=['Model','Accuracy']
    df_acc_3.Accuracy=df_acc_3.Accuracy*100
    ax=sns.barplot(data=df_acc_3,x='Model',y='Accuracy')
    plt.title("Model comparision for heart diease")
    plt.ylabel('Accuracy(%)')
    plt.xlabel("Model")
    plt.legend()
    plt.show()
