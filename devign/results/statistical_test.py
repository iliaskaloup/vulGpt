#!/usr/bin/env python
# coding: utf-8

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import wilcoxon
import statistics

relative_path1 = os.path.join("CodeGPT-small-py", "forSequence")

#relative_path1 = os.path.join("CodeGPT-small-py", "forSequence")

relative_path2 = os.path.join("CodeGPT-small-py", "embeddingsExtraction", "bigru")

var_of_interest = "f1"

all_data1 = pd.DataFrame()

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(relative_path1):
    if root != relative_path1:
        for file in files:
            # Check if the file is a CSV file
            if file.endswith(".csv"):
                # Construct the full path of the CSV file
                file_path = os.path.join(root, file)
                # Read the CSV file into a dataframe
                df = pd.read_csv(file_path)
                # Append the dataframe to the master dataframe
                all_data1 = pd.concat([all_data1, df], ignore_index=True)


print(all_data1[var_of_interest])
print(all_data1.describe())
#print("avg   ", round(sum(all_data1[var_of_interest].values.tolist())/len(all_data1[var_of_interest].values.tolist()), 6))


all_data2 = pd.DataFrame()

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(relative_path2):
    if root != relative_path2:
        for file in files:
            # Check if the file is a CSV file
            if file.endswith(".csv"):
                # Construct the full path of the CSV file
                file_path = os.path.join(root, file)
                # Read the CSV file into a dataframe
                df = pd.read_csv(file_path)
                # Append the dataframe to the master dataframe
                all_data2 = pd.concat([all_data2, df], ignore_index=True)


print(all_data2[var_of_interest])
print(all_data2.describe())

# conduct the Wilcoxon-Signed Rank Test: pvalue < 0.05 --> statistically significant differentiation of the results
res = wilcoxon(all_data1[var_of_interest].values.tolist(), all_data2[var_of_interest].values.tolist())
print(res)
coef = res[0]
print("statistic coefficient: ", coef)
pvalue = res[1]
print("p value: ", pvalue)

if pvalue < 0.05:
    print("statistically significant differentiation")
else:
    print("no statistical significance")
