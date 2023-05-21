# -*- coding: utf-8 -*-
# @Author  : xuhc
# @Time    : 2022/11/19 22:42
# @Function:
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import auc, roc_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, \
    roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")


def evaluation_class(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    # print("confusion matrix:")
    # print(confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print("accuracy:", round(accuracy, 4),
          "precision:", round(precision, 4),
          "recall:", round(recall, 4),
          "F1 score:", round(f1, 4))
    # print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    data_train = pd.read_csv("mixed_train_mol2vec.csv")
    data_test = pd.read_csv("mixed_test_mol2vec.csv")
    X_train = data_train.iloc[:, 3:]
    X_test = data_test.iloc[:, 3:]

    y_train = data_train.PUBCHEM_ACTIVITY_OUTCOME.values
    y_test = data_test.PUBCHEM_ACTIVITY_OUTCOME.values

    print("lr:")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    evaluation_class(lr, X_test, y_test)

    print("knn:")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    evaluation_class(knn, X_test, y_test)

    print("random forest:")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    evaluation_class(rf, X_test, y_test)

    print("decision tree:")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    evaluation_class(dt, X_test, y_test)

    print("SVC:")
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    evaluation_class(svm, X_test, y_test)

