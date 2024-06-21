#!/usr/bin/env python
# coding: utf-8

import seaborn as sn
import pandas as pd
import json, os

# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from torch.optim import Adam
# from transformers import get_linear_schedule_with_warmup
# from torch.nn.utils import clip_grad_norm_
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from transformers import AutoTokenizer, AutoModel

from transformers import AutoTokenizer, TFAutoModel, GPT2LMHeadModel, TFGPT2LMHeadModel, RobertaTokenizer
from transformers import set_seed

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Embedding, MaxPool1D
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Bidirectional, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import glorot_uniform, RandomUniform, lecun_uniform, Constant, TruncatedNormal
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPool1D, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool1D
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Layer

import numpy as np
import csv

import matplotlib.pyplot as plt
import random
from collections import OrderedDict
import time
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import random

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from collections import defaultdict
from imblearn.under_sampling import RandomUnderSampler

from sklearn.utils import shuffle


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Define method name and root path of the repository

method = "embeddingsExtraction"

root_path = os.path.join('..', '..')
dataset = pd.read_csv(os.path.join(root_path, 'data', 'dataset.csv'))

# Define specific seeder for all experiments and processes
seeders = [123456, 789012, 345678, 901234, 567890, 123, 456, 789, 135, 680]
seed = seeders[0]
print(seed)
np.random.seed(seed)
random.seed(seed)
#torch.manual_seed(seed)
tf.random.set_seed(seed)
set_seed(seed)


embedding_algorithm = "bert" # "bert" # "gpt"

# define functions
def padSequences(sequences, max_len):
    lines_pad = []
    for sequence in sequences:
        seq = sequence['input_ids'].numpy()[0]
        if len(seq) < max_len:
            for i in range(len(seq), max_len):
                seq = np.append(seq, 0)
        lines_pad.append(seq)
    return lines_pad

def get_max_len(sequences):
    max_len = 0

    for seq in sequences:
        if len(seq['input_ids'].numpy()[0]) > max_len:
            max_len = len(seq['input_ids'].numpy()[0])

    return max_len

def getMaxLen(X):

    # Code for identifying max length of the data samples after tokenization using transformer tokenizer
    
    max_length = 0
    # Iterate over each sample in your dataset
    for i, input_ids in enumerate(X['input_ids']):
        # Calculate the length of the tokenized sequence for the current sample
        length = tf.math.reduce_sum(tf.cast(input_ids != 1, tf.int32)).numpy()
        # Update max_length and max_row if the current length is greater
        if length > max_length:
            max_length = length
            max_row = i

    print("Max length of tokenized data:", max_length)
    print("Row with max length:", max_row)

    #X['input_ids'] = np.delete(X['input_ids'], max_row, axis=0)
    
    return max_length

# Evaluation functions
def recall_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
        return recall

def precision_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
        return precision

def f1_metric(y_true, y_pred):

    prec = precision_metric(y_true, y_pred)
    rec = recall_metric(y_true, y_pred)
    f1 = 2*((prec*rec)/(prec+rec+K.epsilon()))
    return f1

def f2_metric(y_true, y_pred):

    prec = precision_metric(y_true, y_pred)
    rec = recall_metric(y_true, y_pred)
    f2 = 5*((prec*rec)/(4*prec+rec+K.epsilon()))
    return f2

# Deep Learning Models - Classifiers
def buildLstm(max_len, top_words, dim, seed, embedding_matrix, optimizer, n_categories):
    model=Sequential()
    kernel_initializer = glorot_uniform() # glorot_uniform, RandomUniform, lecun_uniform, Constant, TruncatedNormal
    model.add(Embedding(input_dim=top_words, output_dim=dim, input_length=None, weights=[embedding_matrix], mask_zero=True, trainable=False))
    model.add(LSTM(500, activation='tanh', dropout=0.2, return_sequences=True, stateful=False, kernel_constraint=max_norm(3), bias_constraint=max_norm(3), kernel_initializer=kernel_initializer)) # , recurrent_constraint=max_norm(3)
    model.add(LSTM(100, activation='tanh', dropout=0.1, return_sequences=True, stateful=False, kernel_initializer=kernel_initializer))
    model.add(LSTM(200, activation='tanh', dropout=0.1, stateful=False, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization()) # default momentum=0.99
    #model.add(Dropout(0.2))
    
    #optimizer = optimizers.SGD(lr=learning_rate, decay=0.1, momentum=0.2, nesterov=True)
    #optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8, decay=0.0)
    #optimizer = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.004)
    #optimizer = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    if n_categories > 2:
        model.add(Dense(units = n_categories, activation = 'softmax', kernel_initializer=kernel_initializer))
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    else:
        model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer=kernel_initializer))
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[f1_metric])
    return model

def buildGru(max_len, top_words, dim, seed, embedding_matrix, optimizer, n_categories):
    model=Sequential()
    kernel_initializer = glorot_uniform() # glorot_uniform, RandomUniform, lecun_uniform, Constant, TruncatedNormal
    model.add(Embedding(input_dim=top_words, output_dim=dim, input_length=None, weights=[embedding_matrix], mask_zero=True, trainable=False))
    model.add(GRU(500, activation='tanh', dropout=0.2, return_sequences=True, stateful=False, kernel_constraint=max_norm(3), bias_constraint=max_norm(3), kernel_initializer=kernel_initializer)) # , recurrent_constraint=max_norm(3)
    model.add(GRU(100, activation='tanh', dropout=0.1, return_sequences=True, stateful=False, kernel_initializer=kernel_initializer))
    model.add(GRU(200, activation='tanh', dropout=0.1, stateful=False, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization()) # default momentum=0.99
    #model.add(Dropout(0.2))
    
    #optimizer = optimizers.SGD(lr=learning_rate, decay=0.1, momentum=0.2, nesterov=True)
    #optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8, decay=0.0)
    #optimizer = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.004)
    #optimizer = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    if n_categories > 2:
        model.add(Dense(units = n_categories, activation = 'softmax', kernel_initializer=kernel_initializer))
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    else:
        model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer=kernel_initializer))
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[f1_metric]) 
    return model

def buildBiLstm(max_len, top_words, dim, seed, embedding_matrix, optimizer, n_categories):
    model=Sequential()
    kernel_initializer = glorot_uniform() # glorot_uniform, RandomUniform, lecun_uniform, Constant, TruncatedNormal
    model.add(Embedding(input_dim=top_words, output_dim=dim, input_length=None, weights=[embedding_matrix], mask_zero=True, trainable=False))
    model.add(Bidirectional(LSTM(500, activation='tanh', dropout=0.2, return_sequences=True, stateful=False, kernel_constraint=max_norm(3), bias_constraint=max_norm(3), kernel_initializer=kernel_initializer))) # , recurrent_constraint=max_norm(3)
    model.add(Bidirectional(LSTM(100, activation='tanh', dropout=0.1, return_sequences=True, stateful=False, kernel_initializer=kernel_initializer)))
    model.add(Bidirectional(LSTM(200, activation='tanh', dropout=0.1, stateful=False, kernel_initializer=kernel_initializer)))
    model.add(BatchNormalization()) # default momentum=0.99
    #model.add(Dropout(0.2))
    
    #optimizer = optimizers.SGD(lr=learning_rate, decay=0.1, momentum=0.2, nesterov=True)
    #optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8, decay=0.0)
    #optimizer = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.004)
    #optimizer = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    if n_categories > 2:
        model.add(Dense(units = n_categories, activation = 'softmax', kernel_initializer=kernel_initializer))
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    else:
        model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer=kernel_initializer))
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[f1_metric]) 
    return model

def buildBiGru(max_len, top_words, dim, seed, embedding_matrix, optimizer, n_categories):
    model=Sequential()
    kernel_initializer = glorot_uniform() # glorot_uniform, RandomUniform, lecun_uniform, Constant, TruncatedNormal
    model.add(Embedding(input_dim=top_words, output_dim=dim, input_length=None, weights=[embedding_matrix], mask_zero=True, trainable=False))
    model.add(Bidirectional(GRU(500, activation='tanh', dropout=0.2, return_sequences=True, stateful=False, kernel_constraint=max_norm(3), bias_constraint=max_norm(3), kernel_initializer=kernel_initializer))) # , recurrent_constraint=max_norm(3)
    model.add(Bidirectional(GRU(100, activation='tanh', dropout=0.1, return_sequences=True, stateful=False, kernel_initializer=kernel_initializer)))
    model.add(Bidirectional(GRU(200, activation='tanh', dropout=0.1, stateful=False, kernel_initializer=kernel_initializer)))
    model.add(BatchNormalization()) # default momentum=0.99
    #model.add(Dropout(0.2))
    
    #optimizer = optimizers.SGD(lr=learning_rate, decay=0.1, momentum=0.2, nesterov=True)
    #optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8, decay=0.0)
    #optimizer = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.004)
    #optimizer = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    if n_categories > 2:
        model.add(Dense(units = n_categories, activation = 'softmax', kernel_initializer=kernel_initializer))
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    else:
        model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer=kernel_initializer))
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[f1_metric])  
    return model

def buildCnn(max_len, top_words, dim, seed, embedding_matrix, optimizer, n_categories):
    cnn_model = Sequential()
    cnn_model.add(Embedding(top_words, dim, input_length=None, weights=[embedding_matrix], mask_zero=True, trainable=False))
    cnn_model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu'))
    '''cnn_model.add(MaxPooling1D(pool_size = 5))
    cnn_model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu'))
    cnn_model.add(MaxPooling1D(pool_size = 5))
    cnn_model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu'))'''
    cnn_model.add(GlobalMaxPool1D())
    #cnn_model.add(Dense(units = 128, activation = 'relu'))
    
    if n_categories > 2:
        cnn_model.add(Dense(units = n_categories, activation = 'softmax'))
        cnn_model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    else:
        cnn_model.add(Dense(units = 1, activation = 'sigmoid'))
        cnn_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[f1_metric])
    return cnn_model


# data split
val_ratio = 0.1
num_of_ratio = int(val_ratio * len(dataset))
data = dataset.iloc[0:-num_of_ratio, :]
test_data = dataset.iloc[-num_of_ratio:, :]
train_data = data.iloc[0:-num_of_ratio, :]
val_data = data.iloc[-num_of_ratio:, :]

# Shuffle dataset
train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
print(train_data.head())
print(len(train_data))

train_data = train_data[train_data["project"] != "Chrome"]
print(len(train_data))

train_data = train_data[["processed_func", "target"]]
train_data.head()

train_data = train_data.dropna(subset=["processed_func"])

word_counts = train_data["processed_func"].apply(lambda x: len(x.split()))
max_length = word_counts.max()
print("Maximum number of words:", max_length)

vc = train_data["target"].value_counts()

print(vc)

print("Percentage: ", (vc[1] / vc[0])*100, '%')

n_categories = len(vc)
print(n_categories)

train_data = pd.DataFrame(({'text': train_data['processed_func'], 'label': train_data['target']}))
#train_data = train_data[0:100]
train_data.head()


val_data = val_data[val_data["project"] != "Chrome"]

val_data = pd.DataFrame(({'text': val_data['processed_func'], 'label': val_data['target']}))
val_data.head()

test_data = test_data[test_data["project"] != "Chrome"]

test_data = pd.DataFrame(({'text': test_data['processed_func'], 'label': test_data['target']}))


# Pre-processing step: Under-sampling

sampling = False
if n_categories == 2 and sampling == True:
    # Apply under-sampling with the specified strategy
    class_counts = pd.Series(train_data["label"]).value_counts()
    print("Class distribution ", class_counts)

    majority_class = class_counts.idxmax()
    print("Majority class ", majority_class)

    minority_class = class_counts.idxmin()
    print("Minority class ", minority_class)

    target_count = 2 * class_counts[class_counts.idxmin()] # class_counts[class_counts.idxmin()] # int(class_counts.iloc[0] / 2)
    print("Targeted number of majority class", target_count)

    # under
    sampling_strategy = {majority_class: target_count}
    rus = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strategy)

    x_train_resampled, y_train_resampled = rus.fit_resample(np.array(train_data["text"]).reshape(-1, 1), train_data["label"])
    print("Class distribution after augmentation", pd.Series(y_train_resampled).value_counts())


    # Shuffle the resampled data while preserving the correspondence between features and labels
    x_train_resampled, y_train_resampled = shuffle(x_train_resampled, y_train_resampled, random_state=seed)

    # rename
    X_train = x_train_resampled
    Y_train = y_train_resampled

    X_train = pd.Series(X_train.reshape(-1))

else:
    X_train = train_data["text"]
    Y_train = train_data["label"]


# Choose transformer model

# microsoft/codebert-base-mlm # microsoft/codebert-base

# # PYTORCH
# if embedding_algorithm == "bert":
#     model_variation = "microsoft/codebert-base-mlm"
#     tokenizer = AutoTokenizer.from_pretrained(model_variation, do_lower_case=True) #Tokenizer
#     #bert-base-uncased #bert-base #albert-base-v2 # roberta-base # distilbert-base-uncased #distilbert-base
#     # Define New tokens for string and numerical i.e., strId$ and numId$
#     new_tokens = ["strId$", "numId$"]
#     for new_token in new_tokens:
#         if new_token not in tokenizer.get_vocab().keys():
#             tokenizer.add_tokens(new_token)

#     bert = AutoModel.from_pretrained(model_variation, num_labels=n_categories)

#     bert.resize_token_embeddings(len(tokenizer))

#     embedding_matrix = bert.embeddings.word_embeddings.weight.detach().cpu().numpy()

#     num_words = len(embedding_matrix)
#     print(num_words)
#     dim = len(embedding_matrix[0])
#     print(dim)

#     sentences = X_train.tolist()
#     sequences = [tokenizer.encode(sente, truncation=True, add_special_tokens=False, return_tensors="pt").numpy() for sente in sentences] # Tokenize the complete sentences

#     lines_pad_x_train = []
#     for seq in sequences:
#         lines_pad_x_train.append(torch.tensor(seq[0]))

#     lines_pad_x_train = pad_sequence(lines_pad_x_train, batch_first=True, padding_value=0)
#     max_len = lines_pad_x_train.size()[1]


#     sentences = val_data["Input"]
#     sequences = [tokenizer.encode(sente, truncation=True, add_special_tokens=False, return_tensors="pt").numpy() for sente in sentences]
#     lines_pad_x_val = []
#     for seq in sequences:
#         lines_pad_x_val.append(torch.tensor(seq[0]))
#     lines_pad_x_val = pad_sequence(lines_pad_x_val, batch_first=True, padding_value=0)

#     sentences = test_data["Input"]
#     sequences = [tokenizer.encode(sente, truncation=True, add_special_tokens=False, return_tensors="pt").numpy() for sente in sentences]
#     lines_pad_x_test = []
#     for seq in sequences:
#         lines_pad_x_test.append(torch.tensor(seq[0]))
#     lines_pad_x_test = pad_sequence(lines_pad_x_test, batch_first=True, padding_value=0)

# TENSORFLOW
if embedding_algorithm == "bert":
    model_variation = "microsoft/codebert-base-mlm"
    #bert-base-uncased #bert-base #albert-base-v2 # roberta-base # distilbert-base-uncased #distilbert-base # "microsoft/codebert-base"

    tokenizer = AutoTokenizer.from_pretrained(model_variation, do_lower_case=True) #Tokenizer

    # tokenizer = RobertaTokenizer(vocab_file="../tokenizer_training/cpp_tokenizer/cpp_tokenizer-vocab.json",
    #                          merges_file="../tokenizer_training/cpp_tokenizer/cpp_tokenizer-merges.txt")

    # Define New tokens for string and numerical i.e., strId$ and numId$
#     new_tokens = ["strId$", "numId$"]
#     for new_token in new_tokens:
#         if new_token not in tokenizer.get_vocab().keys():
#             tokenizer.add_tokens(new_token)

    bert = TFAutoModel.from_pretrained(model_variation)

    #bert.resize_token_embeddings(len(tokenizer))

    bert_embeddings = bert.get_input_embeddings()
    embedding_matrix = bert_embeddings.weights[0].numpy()

    num_words = embedding_matrix.shape[0]
    print(num_words)
    dim = embedding_matrix.shape[1]
    print(dim)

    sentences = X_train.tolist()
    sequences = [tokenizer(sente, truncation=True, max_length=510, add_special_tokens=False, return_tensors="tf") for sente in sentences] # Tokenize the complete sentences

    max_len = get_max_len(sequences)
    print(max_len)

    lines_pad_x_train = padSequences(sequences, max_len)
    lines_pad_x_train = [arr.tolist() for arr in lines_pad_x_train]
    lines_pad_x_train = np.array(lines_pad_x_train)

    val_sentences = val_data["text"].tolist()
    val_sequences = [tokenizer(sente, truncation=True, max_length=510, add_special_tokens=False, return_tensors="tf") for sente in val_sentences]

    lines_pad_x_val = padSequences(val_sequences, max_len)
    lines_pad_x_val = [arr.tolist() for arr in lines_pad_x_val]
    lines_pad_x_val = np.array(lines_pad_x_val)

    test_sentences = test_data["text"].tolist()
    test_sequences = [tokenizer(sente, truncation=True, max_length=510, add_special_tokens=False, return_tensors="tf") for sente in test_sentences]

    lines_pad_x_test = padSequences(test_sequences, max_len)
    lines_pad_x_test = [arr.tolist() for arr in lines_pad_x_test]
    lines_pad_x_test = np.array(lines_pad_x_test)


# # PYTORCH
# if embedding_algorithm == "gpt":
#     model_variation = "gpt2" # "microsoft/CodeGPT-small-py-adaptedGPT2" # "gpt2" # "microsoft/CodeGPT-small-py"
#     tokenizer = AutoTokenizer.from_pretrained(model_variation, do_lower_case=True) #Tokenizer
#     # Define New tokens for string and numerical i.e., strId$ and numId$
#     new_tokens = ["strId$", "numId$"]
#     for new_token in new_tokens:
#         if new_token not in tokenizer.get_vocab().keys():
#             tokenizer.add_tokens(new_token)

#     gpt = AutoModel.from_pretrained(model_variation, num_labels=n_categories)

#     gpt.resize_token_embeddings(len(tokenizer))

#     embedding_matrix = gpt.wte.weight.detach().cpu().numpy()

#     num_words = len(embedding_matrix)
#     print(num_words)
#     dim = len(embedding_matrix[0])
#     print(dim)

#     sentences = X_train.tolist()
#     sequences = [tokenizer.encode(sente, truncation=True, add_special_tokens=False, return_tensors="pt").numpy() for sente in sentences] # Tokenize the complete sentences

#     lines_pad_x_train = []
#     for seq in sequences:
#         lines_pad_x_train.append(torch.tensor(seq[0]))

#     lines_pad_x_train = pad_sequence(lines_pad_x_train, batch_first=True, padding_value=0)
#     max_len = lines_pad_x_train.size()[1]


#     sentences = val_data["Input"]
#     sequences = [tokenizer.encode(sente, truncation=True, add_special_tokens=False, return_tensors="pt").numpy() for sente in sentences]
#     lines_pad_x_val = []
#     for seq in sequences:
#         lines_pad_x_val.append(torch.tensor(seq[0]))
#     lines_pad_x_val = pad_sequence(lines_pad_x_val, batch_first=True, padding_value=0)

#     sentences = test_data["Input"]
#     sequences = [tokenizer.encode(sente, truncation=True, add_special_tokens=False, return_tensors="pt").numpy() for sente in sentences]
#     lines_pad_x_test = []
#     for seq in sequences:
#         lines_pad_x_test.append(torch.tensor(seq[0]))
#     lines_pad_x_test = pad_sequence(lines_pad_x_test, batch_first=True, padding_value=0)


# TENSORFLOW
if embedding_algorithm == "gpt":
    model_variation = "microsoft/CodeGPT-small-py" # "microsoft/CodeGPT-small-py-adaptedGPT2" # "gpt2" # "microsoft/CodeGPT-small-py"
    tokenizer = AutoTokenizer.from_pretrained(model_variation, do_lower_case=True) #Tokenizer
    #bert-base-uncased #bert-base #albert-base-v2 # roberta-base # distilbert-base-uncased #distilbert-base
    # Define New tokens for string and numerical i.e., strId$ and numId$
#     new_tokens = ["strId$", "numId$"]
#     for new_token in new_tokens:
#         if new_token not in tokenizer.get_vocab().keys():
#             tokenizer.add_tokens(new_token)

    gpt = TFGPT2LMHeadModel.from_pretrained(model_variation, num_labels=n_categories)

    #gpt.resize_token_embeddings(len(tokenizer))

    embedding_matrix = gpt.transformer.wte.weights[0]

    num_words = embedding_matrix.shape[0]
    print(num_words)
    dim = embedding_matrix.shape[1]
    print(dim)

#     X = tokenizer(
#         text=X_train.tolist(),
#         add_special_tokens=False,
#         max_length=512,
#         truncation=True,
#         padding=True,
#         return_tensors='tf',
#         return_token_type_ids=False,
#         return_attention_mask=True,
#         verbose=True
#     )

#     max_len = getMaxLen(X)
    max_len = 512

    sentences = X_train.tolist()
    sequences = [tokenizer.encode(sente, truncation=True, add_special_tokens=False, return_tensors="tf").numpy() for sente in sentences] # Tokenize the complete sentences

    lines_pad_x_train = []
    for seq in sequences:
        lines_pad_x_train.append(seq[0])

    lines_pad_x_train = pad_sequences(lines_pad_x_train, padding = 'post', maxlen = max_len)

    sentences = val_data["text"].tolist()
    sequences = [tokenizer.encode(sente, truncation=True, add_special_tokens=False, return_tensors="tf").numpy() for sente in sentences]
    lines_pad_x_val = []
    for seq in sequences:
        lines_pad_x_val.append(seq[0])
    lines_pad_x_val = pad_sequences(lines_pad_x_val, padding = 'post', maxlen = max_len)

    sentences = test_data["text"].tolist()
    sequences = [tokenizer.encode(sente, truncation=True, add_special_tokens=False, return_tensors="tf").numpy() for sente in sentences]
    lines_pad_x_test = []
    for seq in sequences:
        lines_pad_x_test.append(seq[0])
    lines_pad_x_test = pad_sequences(lines_pad_x_test, padding = 'post', maxlen = max_len)

    embedding_matrix = embedding_matrix.numpy()

# release memory space
del dataset
del data
del train_data
del X_train

Y_train = np.array(Y_train)
Y_val = np.array(val_data["label"])
Y_test = np.array(test_data["label"])
len(Y_train), len(Y_val), len(Y_test)

# Select Hyper-parameters

n_epochs = 100
patience = 5
batch_size = 64
lr = 0.001
optimizer = optimizers.Adam(learning_rate=lr)


print("Training...")
milli_sec1 = int(round(time.time() * 1000))

userModel = "bigru"

if userModel == "cnn":
    myModel = buildCnn(max_len, num_words, dim, seed, embedding_matrix, optimizer, n_categories)
elif userModel == "lstm":
    myModel = buildLstm(max_len, num_words, dim, seed, embedding_matrix, optimizer, n_categories)
elif userModel == "bilstm":
    myModel = buildBiLstm(max_len, num_words, dim, seed, embedding_matrix, optimizer, n_categories)
elif userModel == "gru":
    myModel = buildGru(max_len, num_words, dim, seed, embedding_matrix, optimizer, n_categories)
elif userModel == "bigru":
    myModel = buildBiGru(max_len, num_words, dim, seed, embedding_matrix, optimizer, n_categories)

print("model summary\m", myModel.summary())

csv_logger = CSVLogger('log.csv', append=True, separator=',')
es = EarlyStopping(monitor='val_f1_metric', mode='max', verbose=1, patience=patience)
mc = ModelCheckpoint('best_model.h5', monitor='val_f1_metric', mode='max', verbose=1, save_best_only=True)

history = myModel.fit(lines_pad_x_train, Y_train, validation_data=(lines_pad_x_val, Y_val), epochs = n_epochs, batch_size = batch_size, shuffle=False, verbose=1, callbacks=[csv_logger,es,mc]) #, class_weight=class_weights

milli_sec2 = int(round(time.time() * 1000))
print("Training is completed after", milli_sec2-milli_sec1)


# Load best model

#model = load_model('best_model.h5')
myModel.load_weights("best_model.h5")


# Classification report on validation set

print(classification_report(Y_val, (myModel.predict(lines_pad_x_val) > 0.5).astype("int32")))


# Prediction and Evaluation on testing set

#scores = myModel.evaluate(lines_pad_x_test, Y_test, verbose=0)
#predictions = myModel.predict_classes(X_test, verbose=0)
predScores = myModel.predict(lines_pad_x_test)
predictions = (predScores > 0.5).astype("int32")

accuracy=accuracy_score(Y_test, predictions)
if n_categories > 2:
    precision=precision_score(Y_test, predictions, average='macro')
    recall=recall_score(Y_test, predictions, average='macro')
    f1=f1_score(Y_test, predictions, average='macro')
else:
    precision=precision_score(Y_test, predictions)
    recall=recall_score(Y_test, predictions)
    f1=f1_score(Y_test, predictions)
    roc_auc=roc_auc_score(Y_test, predictions)
f2=5*precision*recall / (4*precision+recall)

cm = confusion_matrix(Y_test, predictions)
#print(cm)
sn.heatmap(cm, annot=True)
tn, fp, fn, tp = cm.ravel()

print("TP=",tp)
print("TN=",tn)
print("FP=",fp)
print("FN=",fn)

acc = ((tp+tn)/(tp+tn+fp+fn))

print("Accuracy:%.2f%%"%(acc*100))
print("Precision:%.2f%%"%(precision*100))
print("Recall:%.2f%%"%(recall*100))
print("F1 score:%.2f%%"%(f1*100))
print("Roc_Auc score:%.2f%%"%(roc_auc*100))
print("F2 score:%.2f%%"%(f2*100))
print(classification_report(Y_test, predictions))


# Export classification report

# Create the path
path = os.path.join(root_path, 'results', model_variation.split("/")[-1], method, userModel, str(seed))

# Create directory if it doesn't exist
os.makedirs(path, exist_ok=True)

# Define the CSV file path
csv_file_path = os.path.join(path, f"{seed}.csv")

# Write data to CSV
data = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "f2": f2,
    "roc_auc": roc_auc
}

# Write to CSV
with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data.keys())
    writer.writeheader()
    writer.writerow(data)


# Compute the average values of the classication metrics considering the results for all different seeders

# Define a dictionary to store cumulative sum of metrics
cumulative_metrics = defaultdict(float)
count = 0  # Counter to keep track of number of CSV files

# Iterate over all CSV files in the results folder
results_folder = os.path.join(root_path, "results", model_variation.split("/")[-1], method, userModel)

for root, dirs, files in os.walk(results_folder):
    for filename in files:
        if filename.endswith(".csv") and filename != "avg.csv":
            csv_file_path = os.path.join(root, filename)

            with open(csv_file_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    for metric, value in row.items():
                        cumulative_metrics[metric] += float(value)
            count += 1

# Compute average values
average_metrics = {metric: total / count for metric, total in cumulative_metrics.items()}

# Print average values
print(average_metrics)

# Define the path for the average CSV file
avg_csv_file_path = os.path.join(root_path, "results", model_variation.split("/")[-1], method, userModel, "avg.csv")

# Write average metrics to CSV
with open(avg_csv_file_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=average_metrics.keys())
    writer.writeheader()
    writer.writerow(average_metrics)


# Clean up
del myModel
#tf.keras.backend.clear_session()


