#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing packages and reading the file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
header=['url','state']
df=pd.read_csv('datathon_train.csv', names=header)
print(df)


# In[3]:


#describe the data frame
df.describe()


# In[4]:


#data info
df.info()


# 

# In[5]:


#converting url into sentence by using lambda function
df['url']=df['url'].apply(lambda x:x.replace("/"," "))
df['url']=df['url'].apply(lambda x:x.replace("."," "))
df['url']=df['url'].apply(lambda x:x.replace("%20"," "))
df['url']=df['url'].apply(lambda x:x.replace("-"," "))
print(df['url'])


# In[120]:


#shape of the df
df.shape


# In[6]:


#usage of countervectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vector=vectorizer.fit_transform(df['url'])


# In[8]:


#shape of the vector
vector.shape


# In[10]:


#using of Tfidvectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer1 = TfidfVectorizer()
vector1=vectorizer1.fit_transform(df['url'])


# In[11]:


#shape of vector1
vector1.shape


# In[12]:


#q
q=df['state']
q


# In[13]:


#logistic regression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(vector,q,test_size=0.2,random_state=42)
reg=linear_model.LogisticRegression()


# In[16]:


reg.fit(x_train,y_train)


# In[17]:


y_pred=reg.predict(x_test)
print(y_pred)


# In[151]:


print(vector)


# In[18]:


#finding accuracy
from sklearn.metrics import accuracy_score
print("accuracy",accuracy_score(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




