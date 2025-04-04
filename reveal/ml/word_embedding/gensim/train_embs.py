#!/usr/bin/env python
# coding: utf-8

import csv, os
from gensim.models import Word2Vec
from gensim.models import FastText
import pandas as pd

from sklearn.model_selection import train_test_split

import nltk

seed = 123456



# hyperparams
dim = 100
method = "w2v" # ft
min_count = 10
epochs = 10


# read embeddings corpus corpus_embeddings
root_path = os.path.join('..', '..', '..')

dataset = pd.read_csv(os.path.join(root_path, 'data', 'dataset.csv'))

data = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
print(data.head())
print(len(data))

data = data[["func", "vul"]]
data = data.dropna(subset=["func"])
data = pd.DataFrame(({'text': data['func'], 'label': data['vul']}))
val_ratio = 0.10
train_data, test_data = train_test_split(data, test_size=val_ratio, random_state=seed, stratify=data['label'])
train_data, val_data = train_test_split(train_data, test_size=val_ratio, random_state=seed, stratify=train_data['label'])

train_data = train_data[["text"]]
train_data.head()

word_counts = train_data["text"].apply(lambda x: len(x.split()))
max_length = word_counts.max()
print("Maximum number of words:", max_length)


train_data = pd.DataFrame(({'Text': train_data['text']}))
#train_data = train_data[0:100]
train_data.head()

text = train_data["Text"].values.tolist()

# Write each function to the file
file_path = os.path.join(root_path, 'data', 'tokenizer_train_data.txt')

with open(file_path, "w", encoding="utf-8") as file:
    for function in text:
        file.write(function + "\n")


with open(os.path.join(root_path, 'data', 'tokenizer_train_data.txt'), 'r', encoding='utf-8') as file:
    corpus = file.read() #.lower().replace('\n', ' ')


corpusList = corpus.split("}\n")


def dropBlank(tokens0):
    tokens = []
    for i in range(0, len(tokens0)):
        temp = tokens0[i]
        if temp != '':
            tokens.append(temp)
    return tokens

def dropEmpty(tokens0):
    tokens = []
    for i in range(0, len(tokens0)):
        temp = tokens0[i]
        if temp != []:
            tokens.append(temp)
    return tokens


def stringToList(string):
    codeLinesList = []
    for line in string.splitlines():
        codeLinesList.append(line)
    return codeLinesList

def tokenizeLines(codeLinesList):
    codeTokens = []
    
    for line in codeLinesList:
        templineTokens = nltk.word_tokenize(line)
        codeTokens.extend(templineTokens)
    
    return codeTokens

def dataTokenization(corpus):
    
    allTokens = []
    for i in range(0, len(corpus)):
        stringLines = corpus[i]
        
        #convert source code from string to list of lines
        lines = stringToList(stringLines)
        
        #tokenize lines to list of words
        tokens0 = tokenizeLines(lines)
        
        #remove blank lines
        tokens = dropBlank(tokens0)
        
        #lower case
        for w in range(0, len(tokens)):
            tokens[w] = tokens[w].lower()
         
        allTokens.append(tokens)
        
    return allTokens

def embVectors(dim, epochs, min_count, method, corpusList): 

    data = dataTokenization(corpusList)
    
    data = dropEmpty(data)
    
    if method == "w2v": 
        model = Word2Vec(data, vector_size=dim, workers=4, epochs=epochs, min_count=min_count) #, window=20
        fileEmb = method + '_embeddings.txt'
        model.wv.save_word2vec_format(fileEmb, binary=False)
    elif method == "ft":
        model_ted = FastText(vector_size=dim, min_count=min_count)
        model_ted.build_vocab(corpus_iterable=data)
        model_ted.train(corpus_iterable=data, total_examples=len(data), epochs=epochs)
        fileEmb = method + '_embeddings.txt'
        model_ted.wv.save_word2vec_format(fileEmb, binary=False)
    
    return fileEmb

fileEmb = embVectors(dim, epochs, min_count, method, corpusList)

