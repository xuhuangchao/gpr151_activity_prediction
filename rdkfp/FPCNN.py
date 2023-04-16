# -*- coding: utf-8 -*-
# @Author  : xuhc
# @Time    : 2022/10/8 15:59
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
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, concatenate
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, \
    confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    print(classification_report(y_test, y_pred))


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


def one_hot_encoder(array):
    encoder_result = []
    max_len = 0
    for i in range(array.shape[0]):
        index_vec = []
        for j in range(array.shape[1]):
            if array[i, j] == 1:
                index_vec.append(j+1)
        encoder_result.append(index_vec)
        length = len(index_vec)
        max_len = max(max_len, length)
    return encoder_result, max_len


def getRDKFP(data):
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


def getMorganFP(data):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in data['mol']]

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
    # X_train = np.load("RDKFP/mixed_X_train_rdk.npy")
    # X_test = np.load("RDKFP/mixed_X_test_rdk.npy")
    X_train = np.load("MorganFP/mixed_X_train_morgan.npy")
    X_test = np.load("MorganFP/mixed_X_test_morgan.npy")

    y_train = np.load("mixed_y_train.npy")
    y_test = np.load("mixed_y_test.npy")

    y_train_ = to_categorical(y_train, num_classes=2)
    y_test_ = to_categorical(y_test, num_classes=2)

    encode_X_train, max_len_train = one_hot_encoder(X_train)
    encode_X_test, max_len_test = one_hot_encoder(X_test)
    # print(encode_X_train)
    # print(encode_X_test)

    max_length = max(max_len_train, max_len_test)
    padded_X_train = pad_sequences(encode_X_train, maxlen=max_length, padding='post')
    padded_X_test = pad_sequences(encode_X_test, maxlen=max_length, padding='post')

    model = Sequential()
    model.add(Embedding(2049, 64, input_length=max_length))
    model.add(Conv1D(32, 8, activation='relu'))
    model.add(Conv1D(16, 8, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', name="dense_1"))
    model.add(Dense(256, activation='relu', name="dense_2"))
    model.add(Dense(2, activation='softmax', name="dense_proba"))

    opt = keras.optimizers.Adam(learning_rate=0.00025)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')
    model.summary()

    history = model.fit(padded_X_train, y_train_, epochs=15, batch_size=100)

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

    result = model.predict(padded_X_test)
    y_predict = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    print("CNN 混合数据集预测结果：")
    evaluation_class(y_test, y_predict)

    dense_layer_model = Model(inputs=model.input,
                              outputs=model.get_layer('dense_2').output)
    # 以这个model的预测值作为输出
    dense_test_output = dense_layer_model.predict(padded_X_test)
    print(dense_test_output.shape)
    dense_train_output = dense_layer_model.predict(padded_X_train)
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

    print("=======================CNN Features Test set===============================")
    feature_evaluation(dense_test_output, lr, knn, rf, svm, dt, y_test)

    print("=======================dense plot TSNE and PCA===============================")

    dense_test_tsne = TSNE(n_components=2, random_state=33).fit_transform(dense_test_output)
    X_test_tsne = TSNE(n_components=2, random_state=33).fit_transform(X_test)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_test, label="MorganFP t-SNE")
    plt.legend()
    plt.subplot(122)
    plt.scatter(dense_test_tsne[:, 0], dense_test_tsne[:, 1], c=y_test, label="MorganFP+CNN t-SNE")
    plt.legend()
    plt.savefig('./images/dense_test_tsne.png', dpi=120)
    plt.show()

    print("=======================ZINC20 Features Test==================================")
    zinc_similiar = pd.read_csv("../zinc-result/ZINC_SIMILIAR_GPR151_v2.csv")
    zinc_similiar['mol'] = zinc_similiar['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    # X_zinc = getRDKFP(zinc_similiar)
    X_zinc = getMorganFP(zinc_similiar)
    encode_X_zinc, max_len_zinc = one_hot_encoder(X_zinc)
    padded_X_zinc = pad_sequences(encode_X_zinc, maxlen=max_length, padding='post')
    y_zinc_result = model.predict(padded_X_zinc)
    zinc_similiar['gpr_active_pred'] = np.argmax(y_zinc_result, axis=1)
    # zinc_similiar.to_csv("ZINC_SIMILIAR_GPR151_v2_MorganFPCNN_pred.csv", index=False)


    # proba_layer_model = Model(inputs= model.input,
    #                           outputs=model.get_layer("dense_proba").output)
    # proba_test_output = proba_layer_model.predict(padded_X_test)
    # proba_result = np.column_stack((proba_test_output, y_predict, y_test))

    # print("集成分类器投票结果：")
    # lr_proba = lr.predict_proba(dense_test_output)
    # knn_proba = knn.predict_proba(dense_test_output)
    # rf_proba = rf.predict_proba(dense_test_output)
    # svm_proba = svm.predict_proba(dense_test_output)
    # dt_proba = dt.predict_proba(dense_test_output)
    # embedding_proba = (lr_proba + knn_proba + rf_proba + svm_proba + dt_proba)/5
    # y_voting = np.argmax(embedding_proba, axis=1)
    # evaluation_class(y_test, y_voting)

    # y_voting_02 = y_voting[index_02]
    # print("AID_1508602数据集的预测结果：")
    # evaluation_class(y_test_02, y_voting_02)
    #
    # y_voting_10 = y_voting[index_10]
    # print("AID_1508610数据集的预测结果：")
    # evaluation_class(y_test_10, y_voting_10)









