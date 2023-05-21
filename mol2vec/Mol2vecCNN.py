# -*- coding: utf-8 -*-
# @Author  : xuhc
# @Time    : 2022/11/19 22:49
# @Function:

import random

import keras.optimizers
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter

from keras import Sequential, Model, Input
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, \
    confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

myseed = 42
np.random.seed(myseed)
random.seed(myseed)
tf.random.set_seed(myseed)


def evaluation_class(y_test, y_pred):
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("accuracy:", round(accuracy, 4),
          "precision:", round(precision, 4),
          "recall:", round(recall, 4),
          "F1 score:", round(f1, 4))
    # print(classification_report(y_test, y_pred))


def feature_evaluation(dense_output, lr, knn, rf, svm, dt, y):
    print("lr:")
    y_pred = lr.predict(dense_output)
    evaluation_class(y, y_pred)
    print("knn:")
    y_pred = knn.predict(dense_output)
    evaluation_class(y, y_pred)
    print("random forest:")
    y_pred = rf.predict(dense_output)
    evaluation_class(y, y_pred)
    print("decision tree:")
    y_pred = dt.predict(dense_output)
    evaluation_class(y, y_pred)
    print("SVC:")
    y_pred = svm.predict(dense_output)
    evaluation_class(y, y_pred)


if __name__ == "__main__":
    X_train = pd.read_csv("mixed_train_mol2vec.csv").iloc[:, 3:]
    X_test = pd.read_csv("mixed_test_mol2vec.csv").iloc[:, 3:]
    y_train = np.load("../dataset/mixed_y_train.npy")
    y_test = np.load("../dataset/mixed_y_test.npy")

    X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train_ = to_categorical(y_train, num_classes=2)
    y_test_ = to_categorical(y_test, num_classes=2)

    # 1.搭建适合mol2vec100维特征的卷积神经网络提取特征，并使用softmax分类器进行活性分类
    model = Sequential()
    model.add(Conv1D(32, 4, activation='relu'))
    model.add(Conv1D(16, 4, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', name="dense_2"))
    model.add(Dense(2, activation='softmax', name="dense_proba"))

    opt = keras.optimizers.Adam(learning_rate=0.00025)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')

    history = model.fit(X_train, y_train_, epochs=100, batch_size=50)
    model.summary()

    # 获得训练集和验证集的acc和loss曲线
    acc = history.history['accuracy']
    loss = history.history['loss']

    # 绘制acc曲线
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # 绘制loss曲线
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

    result = model.predict(X_test)
    y_predict = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    print("mol2vec+cnn 预测结果:")
    evaluation_class(y_test, y_predict)

    # 2.提取cnn在dense_2层产生的256维特征，投入多种基本分类器
    dense_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    # 以model的中间特征作为输出
    dense_test_output = dense_layer_model.predict(X_test)
    print(dense_test_output.shape)
    dense_train_output = dense_layer_model.predict(X_train)
    print(dense_train_output.shape)

    lr = LogisticRegression(random_state=42)
    lr.fit(dense_train_output, y_train)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(dense_train_output, y_train)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(dense_train_output, y_train)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(dense_train_output, y_train)

    svm = SVC(probability=True)
    svm.fit(dense_train_output, y_train)

    print("======mol2vec + cnn + basic classifier======")
    feature_evaluation(dense_test_output, lr, knn, rf, svm, dt, y_test)

    # 3.对爬取的zinc20数据集分子活性进行预测并写入csv文件中
    print("===========zinc20 dataset===============")
    zinc_similiar = pd.read_csv("ZINC_SIMILIAR_GPR151_v2_Mol2Vec.csv")
    zinc_mol2vec = zinc_similiar.iloc[:, 8:]
    zinc_mol2vec = np.array(zinc_mol2vec).reshape(zinc_mol2vec.shape[0], zinc_mol2vec.shape[1], 1)

    dense_zinc_mol2vec = dense_layer_model.predict(zinc_mol2vec)
    print(dense_zinc_mol2vec.shape)
    y_pred = svm.predict(dense_zinc_mol2vec)
    zinc_similiar_label = zinc_similiar.iloc[:, 0:8]
    zinc_similiar_label['gpr_active_pred'] = y_pred
    zinc_similiar_label.to_csv("ZINC_SIMILIAR_GPR151_v2_pred.csv", index=False)


