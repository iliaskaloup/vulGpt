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


data_vul = readData(os.path.join(root_path, 'data', 'vulnerables.json'))


# In[7]:


df_vul = pd.DataFrame(data_vul)


# In[8]:


df_vul.index


# In[13]:


df_vul["target"] = 1


# In[14]:


df_vul.head()


# In[5]:


data_clean = readData(os.path.join(root_path, 'data', 'non-vulnerables.json'))


# In[10]:


df_clean = pd.DataFrame(data_clean)


# In[11]:


df_clean.index


# In[15]:


df_clean["target"] = 0


# In[16]:


df_clean.head()


# In[18]:


df = pd.concat([df_vul, df_clean])


# In[21]:


df.head()


# In[22]:


df['target'].value_counts()


# In[23]:


print(df["project"].value_counts())


# In[24]:


print(df["hash"].value_counts())


# In[25]:


print(df["size"].value_counts())


# In[27]:


df_data = pd.DataFrame(({'func': df['code'], 'vul': df['target']}))
df_data.head()


# In[28]:


df_data.to_csv(os.path.join('..', "data", 'dataset.csv'), index=False)


# In[ ]:




