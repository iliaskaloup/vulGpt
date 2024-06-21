#!/usr/bin/env python
# coding: utf-8

import seaborn as sn
import pandas as pd
import json, os
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

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle


seeders = [123456, 789012, 345678, 901234, 567890, 123, 456, 789, 135, 680]

seed = seeders[0]

np.random.seed(seed)
random.seed(seed)


root_path = os.path.join('..', '..')


dataset = pd.read_csv(os.path.join(root_path, 'data', 'dataset.csv'))

data = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
print(data.head())
print(len(data))


data = data[["func", "vul"]]
data.head()


data = data.dropna(subset=["func"])


word_counts = data["func"].apply(lambda x: len(x.split()))
max_length = word_counts.max()
print("Maximum number of words:", max_length)


vc = data["vul"].value_counts()

print(vc)

print("Percentage: ", (vc[1] / vc[0])*100, '%')

n_categories = len(vc)
print(n_categories)


data = pd.DataFrame(({'text': data['func'], 'label': data['vul']}))
#train_data = train_data[0:100]
data.head()

val_ratio = 0.10

train_val_data, test_data = train_test_split(data, test_size=val_ratio, random_state=seed, stratify=data['label'])
train_data, val_data = train_test_split(train_val_data, test_size=val_ratio, random_state=seed, stratify=train_val_data['label'])

sampling = True
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


# textual code data
X_val = val_data["text"]

X_test = test_data["text"]


# labels
y_train = Y_train
y_val = val_data["label"]
y_test = test_data["label"]


# apply BoW
vectorizer = TfidfVectorizer(norm='l2', max_features=1000)
vectorizer = vectorizer.fit(X_train)


X_train = np.asarray(vectorizer.transform(X_train).todense())
X_val = np.asarray(vectorizer.transform(X_val).todense())
X_test = np.asarray(vectorizer.transform(X_test).todense())


# define model
myModel = RandomForestClassifier(n_estimators=1000,
                            n_jobs=-1,
                            verbose=1)

# myModel = SVC(kernel='rbf', gamma=100)
# myModel = tree.DecisionTreeClassifier(max_depth=120)
# myModel = GaussianNB()
# myModel = KNeighborsClassifier(n_neighbors=1)


# train model
myModel.fit(X_train, y_train)


# make predictions
val_preds = myModel.predict(X_val)
preds = myModel.predict(X_test)


# evaluate on validation data
f1 = f1_score(y_true=y_val, y_pred=val_preds)
precision = precision_score(y_true=y_val, y_pred=val_preds)
recall = recall_score(y_true=y_val, y_pred=val_preds)
f2 = 5*precision*recall / (4*precision+recall)
print("Evaluation Results on validation data")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F2 Score: {f2}")

print("\n")

# evaluate on test data
f1 = f1_score(y_true=y_test, y_pred=preds)
precision = precision_score(y_true=y_test, y_pred=preds)
recall = recall_score(y_true=y_test, y_pred=preds)
f2 = 5*precision*recall / (4*precision+recall)
print("Evaluation Results on test data")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F2 Score: {f2}")

cm = confusion_matrix(y_test, preds)
#print(cm)
sn.heatmap(cm, annot=True)
print(classification_report(y_test, preds))



