#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import os


# In[2]:


root_path = os.path.join('..')
root_path


# In[3]:


def readData(filename):
    with open(filename,'r+') as file:
        dataset  = json.load(file)
    return dataset


# In[4]:


data = readData(os.path.join(root_path, 'data', 'function.json'))


# In[5]:


df = pd.DataFrame(data)


# In[6]:


df.index


# In[7]:


df.head()


# In[8]:


df['target'].value_counts()


# In[9]:


print(df["project"].value_counts())


# In[13]:


print(df["commit_id"].value_counts())


# In[14]:


df_data = pd.DataFrame(({'func': df['func'], 'vul': df['target']}))
df_data.head()


# In[15]:


df_data.to_csv(os.path.join('..', "data", 'dataset.csv'), index=False)


# In[ ]:




