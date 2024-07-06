#!/usr/bin/env python
# coding: utf-8

import seaborn as sn
import pandas as pd
import json, os, io
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
import time
import random

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec
from gensim.models import FastText

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
from tensorflow.keras.initializers import glorot_uniform, RandomUniform, lecun_uniform, Constant
from collections import OrderedDict
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPool1D
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool1D
from keras_preprocessing.text import tokenizer_from_json

from imblearn.under_sampling import RandomUnderSampler
from collections import defaultdict
from sklearn.utils import shuffle


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

root_path = os.path.join('..', '..', '..')
dataset = pd.read_csv(os.path.join(root_path, 'data', 'dataset.csv'))

# Define specific seeder for all experiments and processes
seeders = [123456, 789012, 345678, 901234, 567890, 123, 456, 789, 135, 680]
seed = seeders[0]
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

method = "w2v"

# data split
val_ratio = 0.1
num_of_ratio = int(val_ratio * len(dataset))
data = dataset.iloc[0:-num_of_ratio, :]
test_data = dataset.iloc[-num_of_ratio:, :]
train_data = data.iloc[0:-num_of_ratio, :]
val_data = data.iloc[-num_of_ratio:, :]


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

train_data = pd.DataFrame(({'Text': train_data['processed_func'], 'Labels': train_data['target']}))
train_data.head()

val_data = val_data[val_data["project"] != "Chrome"]

val_data = pd.DataFrame(({'Text': val_data['processed_func'], 'Labels': val_data['target']}))
val_data.head()

test_data = test_data[test_data["project"] != "Chrome"]

test_data = pd.DataFrame(({'Text': test_data['processed_func'], 'Labels': test_data['target']}))

sampling = False
if n_categories == 2 and sampling == True:
    # Apply under-sampling with the specified strategy
    class_counts = pd.Series(train_data["Labels"]).value_counts()
    print("Class distribution ", class_counts)

    majority_class = class_counts.idxmax()
    print("Majority class ", majority_class)

    minority_class = class_counts.idxmin()
    print("Minority class ", minority_class)

    target_count = 4 * class_counts[class_counts.idxmin()] # int(class_counts[class_counts.idxmax()] / 2) # 2 * class_counts[class_counts.idxmin()] # class_counts[class_counts.idxmin()] # int(class_counts.iloc[0] / 2)  
    print("Targeted number of majority class", target_count)

    # under
    sampling_strategy = {majority_class: target_count}        
    rus = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strategy)

    x_train_resampled, y_train_resampled = rus.fit_resample(np.array(train_data["Text"]).reshape(-1, 1), train_data["Labels"]) 
    print("Class distribution after augmentation", pd.Series(y_train_resampled).value_counts())


    # Shuffle the resampled data while preserving the correspondence between features and labels
    x_train_resampled, y_train_resampled = shuffle(x_train_resampled, y_train_resampled, random_state=seed)

    # rename
    X_train = x_train_resampled
    Y_train = y_train_resampled

    X_train = pd.Series(X_train.reshape(-1))

else:
    X_train = train_data["Text"]
    Y_train = train_data["Labels"]


def stringToList(string):
    codeLinesList = []
    for line in string.split():
        codeLinesList.append(line)
    return codeLinesList


allTokens = []
for seq in X_train:
    listSeq = stringToList(seq)
    allTokens.append(listSeq)

X_train = allTokens

X_train = pd.Series(X_train)

allTokens = []
for seq in val_data["Text"]:
    listSeq = stringToList(seq)
    allTokens.append(listSeq)

val_data["Tokens"] = allTokens

allTokens = []
for seq in test_data["Text"]:
    listSeq = stringToList(seq)
    allTokens.append(listSeq)

test_data["Tokens"] = allTokens

# word embedding 
embeddings_index = {}
f = open('w2v_embeddings.txt', encoding="utf-8")
for line in f:    
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs   
f.close() 

dim = 100

concatenated_data = pd.concat([X_train, val_data["Tokens"], test_data["Tokens"]])

tokenizer_obj = Tokenizer()   
tokenizer_obj.fit_on_texts(concatenated_data)

tokenizer_json = tokenizer_obj.to_json()
tokenizerFile = 'w2v_tokenizer.json'

with io.open(tokenizerFile, 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

with open(tokenizerFile) as f:
    dataTokenizer = json.load(f)
    tokenizer_obj = tokenizer_from_json(dataTokenizer)

sequences = tokenizer_obj.texts_to_sequences(concatenated_data)
word_index = tokenizer_obj.word_index

lines_pad = pad_sequences(sequences, padding = 'post', maxlen = max_length)

num_words = len(word_index) + 1 # +1 for the unknown-zeros

embedding_matrix = np.zeros((num_words, dim))
for word, i in word_index.items():
    if i > num_words:
        continue
    #embedding_vector = embeddings_index.get(word)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# truncate sequences
max_len = 512
lines_pad = lines_pad[:, 0:512]

train_x = lines_pad[0:len(X_train)]
val_x = lines_pad[len(X_train):len(X_train)+len(val_data["Tokens"])]
test_x = lines_pad[len(X_train)+len(val_data["Tokens"]):len(X_train)+len(val_data["Tokens"])+len(test_data["Tokens"])]


# rename
x_train = train_x
x_val = val_x
x_test = test_x


y_train = np.array(Y_train)
y_val = np.array(val_data["Labels"])
y_test = np.array(test_data["Labels"])

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

n_epochs = 100
patience = 10
batch_size = 64
lr = 0.001
optimizer = optimizers.Adam(learning_rate=lr)

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

history = myModel.fit(x_train, y_train, validation_data=(x_val, y_val), epochs = n_epochs, batch_size = batch_size, shuffle=False, verbose=1, callbacks=[csv_logger,es,mc]) #, class_weight=class_weights

milli_sec2 = int(round(time.time() * 1000))
print("Training is completed after", milli_sec2-milli_sec1)

#model = load_model('best_model.h5')
myModel.load_weights("best_model.h5")

print(classification_report(y_val, (myModel.predict(x_val) > 0.5).astype("int32")))

#scores = myModel.evaluate(lines_pad_x_test, Y_test, verbose=0)
#predictions = myModel.predict_classes(X_test, verbose=0)
predScores = myModel.predict(x_test)
predictions = (predScores > 0.5).astype("int32")

accuracy=accuracy_score(y_test, predictions)
if n_categories > 2:
    precision=precision_score(y_test, predictions, average='macro')
    recall=recall_score(y_test, predictions, average='macro')
    f1=f1_score(y_test, predictions, average='macro')
else:
    precision=precision_score(y_test, predictions)
    recall=recall_score(y_test, predictions)
    f1=f1_score(y_test, predictions)
    roc_auc=roc_auc_score(y_test, predictions)
f2=5*precision*recall / (4*precision+recall)

cm = confusion_matrix(y_test, predictions)
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
print(classification_report(y_test, predictions))

# Create the path
path = os.path.join(root_path, 'results', userModel, method, str(seed))

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

# Define a dictionary to store cumulative sum of metrics
cumulative_metrics = defaultdict(float)
count = 0  # Counter to keep track of number of CSV files

# Iterate over all CSV files in the results folder
results_folder = os.path.join(root_path, "results", userModel, method)

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
avg_csv_file_path = os.path.join(root_path, "results", userModel, method, "avg.csv")

# Write average metrics to CSV
with open(avg_csv_file_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=average_metrics.keys())
    writer.writeheader()
    writer.writerow(average_metrics)



