#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
header=['url','state']
df=pd.read_csv('datathon_train.csv', names=header)
print(df)


# In[136]:


df.describe()


# In[137]:


df.info()


# 

# In[119]:


df['url']=df['url'].apply(lambda x:x.replace("/"," "))
df['url']=df['url'].apply(lambda x:x.replace("."," "))
df['url']=df['url'].apply(lambda x:x.replace("%20"," "))
df['url']=df['url'].apply(lambda x:x.replace("-"," "))
print(df['url'])


# In[120]:


df.shape


# In[127]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vector=vectorizer.fit_transform(df['url'])


# In[128]:



vector.shape


# In[132]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer1 = TfidfVectorizer()
vector1=vectorizer1.fit_transform(df['url'])


# In[133]:


vector1.shape


# In[ ]:




