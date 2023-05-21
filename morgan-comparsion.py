# -*- coding: utf-8 -*-
# @Author  : xuhc
# @Time    : 2023/5/12 16:00
# @Function:
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier


def evaluation_class(y_test, y_pred):
    # print("confusion matrix:")
    # print(confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("accuracy:", round(accuracy, 4),
          "precision:", round(precision, 4),
          "recall:", round(recall, 4),
          "F1 score:", round(f1, 4))
    
    
def feature_evaluation(X_train, y_train, fea, y):
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    svm = SVC(probability=True)
    svm.fit(X_train, y_train)

    print("lr:")
    y_pred = lr.predict(fea)
    evaluation_class(y, y_pred)
    print("knn:")
    y_pred = knn.predict(fea)
    evaluation_class(y, y_pred)
    print("random forest:")
    y_pred = rf.predict(fea)
    evaluation_class(y, y_pred)
    print("decision tree:")
    y_pred = dt.predict(fea)
    evaluation_class(y, y_pred)
    print("SVC:")
    y_pred = svm.predict(fea)
    evaluation_class(y, y_pred)
    
    
if __name__ == "__main__":
    X_train = np.load("dataset/mixed_X_train_morgan.npy")
    X_test = np.load("dataset/mixed_X_test_morgan.npy")

    y_train = np.load("mixed_y_train.npy")
    y_test = np.load("mixed_y_test.npy")

    print("主成分分析")
    pca = PCA(n_components=.90)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    feature_evaluation(X_train_pca, y_train, X_test_pca, y_test)

    print("线性判别分析")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)
    feature_evaluation(X_train_lda, y_train, X_test_lda, y_test)

    # 嵌入思想
    print("决策树重要性筛选")
    sfm = SelectFromModel(estimator=DecisionTreeClassifier(), threshold=0.005)
    # threshold=0.005 表示重要性低于0.005时就删除特征
    sfm.fit(X_train, y_train)
    X_train_sfm = sfm.transform(X_train)
    X_test_sfm = sfm.transform(X_test)
    feature_evaluation(X_train_sfm, y_train, X_test_sfm, y_test)