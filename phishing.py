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


# In[143]:


q=df['state']
q


# In[148]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


# In[145]:


x_train,x_test,y_train,y_test=train_test_split(vector,q,test_size=0.2,random_state=42)
reg=linear_model.LogisticRegression()


# In[147]:


reg.fit(x_train,y_train)


# In[150]:


y_pred=reg.predict(x_test)
print(y_pred)


# In[151]:


print(vector)


# In[152]:


from sklearn.metrics import accuracy_score
print("accuracy",accuracy_score(y_test,y_pred))


# In[ ]:




