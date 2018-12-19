from simforest import SimilarityForest
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
dataset = pd.read_csv("Heart_data.csv")

#convert all string values to nan
dataset = dataset.convert_objects(convert_numeric=True)

X = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 13]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy ="mean", axis = 0)
imputer = imputer.fit(X[:, 0:13])   
X[:, 0:13] = imputer.transform(X[:, 0:13])
y=np.array(y)
if __name__ == '__main__':
#test diabetes
    X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(X, y, test_size=0.2, random_state=0)
    sf1 = SimilarityForest(n_estimators=10, n_axes=1)
    sf1.fit(X_train_dia, y_train_dia)

    sf1_pred = sf1.predict(X_test_dia)
    sf1_prob = sf1.predict_proba(X_test_dia)

    print('Similarity Forest for diabetes')
    print(sf1_prob[:, 1])
    print(y_test_dia)
    print(accuracy_score(y_test_dia, sf1_pred))

    rf1 = RandomForestClassifier()
    rf1.fit(X_train_dia, y_train_dia)

    rf1_pred = rf1.predict(X_test_dia)
    rf1_prob = rf1.predict_proba(X_test_dia)

    print('Random Forest for diabetes')
    print(rf1_prob[:, 1])
    print(y_test_dia)
    print(accuracy_score(y_test_dia, rf1_pred))

    svm1 = SVC(gamma='auto', probability=True)
    svm1.fit(X_train_dia, y_train_dia)
    svm1_pred=svm1.predict(X_test_dia)
    svm1_prob = svm1.predict_proba(X_test_dia)

    print('SVM')
    print(svm1_prob[:, 1])
    print(y_test_dia)
    print(accuracy_score(y_test_dia, svm1_pred))