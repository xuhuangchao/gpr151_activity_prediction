# -*- coding: utf-8 -*-
# @Author  : xuhc
# @Time    : 2022/10/31 14:09
# @Function:
import os

import pandas as pd
import numpy as np
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem

def generate_word_embedding(data_file, out_file, model_file='model_300dim.pkl'):
    """
    Uses a pre-trained model to generate word embeddings for a list of molecules.
    Prepends the embeddings as columns to the input data file
    and writes an output csv file.

    :param data_file: path to a csv data file containing a 'SMILES' column of SMILES strings
    :param out_file: path to the output csv file
    :param model_file: path to a pre-trained model
    :return: data from the new csv file written to disk, as a pandas DataFrame
    """
    data = pd.read_csv(data_file).dropna(axis=0)
    smiles = data['smiles']
    mol = [Chem.MolFromSmiles(i) for i in smiles]
    sentence = [MolSentence(mol2alt_sentence(i, radius=1)) for i in mol]
    w2v_model = word2vec.Word2Vec.load(model_file)
    embedding = [DfVec(x) for x in sentences2vec(sentence, w2v_model)]
    data_mol2vec = np.array([x.vec for x in embedding])
    data_mol2vec = pd.DataFrame(data_mol2vec)
    new_data = pd.concat([data, data_mol2vec], axis=1)
    new_data.to_csv(out_file, index=False)
    return data_mol2vec, new_data


test_mol2vec, new_test = generate_word_embedding("../dataset/GPR151_datatable_mixed_test.csv", "mixed_test_mol2vec.csv")
train_mol2vec, new_train = generate_word_embedding("../dataset/GPR151_datatable_mixed_train.csv", "mixed_train_mol2vec.csv")


