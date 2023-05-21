# -*- coding: utf-8 -*-
# @Author  : xuhc
# @Time    : 2022/10/8 16:35
# @Function:
# 导入依赖库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
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


def getFP(data):
    fps = [Chem.RDKFingerprint(m) for m in data['mol']]
    # fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in data['mol']]

    # convert the RDKit explicit vectors into numpy arrays
    flag = False
    np_fps = np.zeros((data.shape[0], len(fps[0])))
    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        if flag == False:
            np_fps = arr
            flag = True
        else:
            np_fps = np.vstack((np_fps, arr))

    print("完成分子指纹提取\n")
    return np_fps


if __name__ == "__main__":
    data_train = pd.read_csv("../dataset/GPR151_datatable_mixed_train.csv")
    data_test = pd.read_csv("../dataset/GPR151_datatable_mixed_test.csv")
    data_train['mol'] = data_train['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    data_test['mol'] = data_test['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

    print("开始计算训练集分子指纹")
    X_train = getFP(data_train)
    print("开始计算测试集分子指纹")
    X_test = getFP(data_test)

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



