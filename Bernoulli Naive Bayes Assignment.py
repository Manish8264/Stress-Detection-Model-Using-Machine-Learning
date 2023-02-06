#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np    #for linear algebra
import pandas as pd   # for data analysis 
import seaborn as sns  # for visualization
from matplotlib import pyplot as plt   # for data visualization
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer   # for sparse matrix 


# In[2]:


df = pd.read_csv("spam.csv",encoding= "latin -1 ")    


# In[4]:


#visualizing dataset
df.head(n=20)


# In[5]:


df.shape


# In[6]:


#to check whether target attribute is binary or not
np.unique(df['class'])


# In[7]:


np.unique(df['message'])


# In[10]:


#creating sparse matrix

x=df["message"].values
y=df["class"].values


# In[11]:


#create count vectorizer object
cv=CountVectorizer()

x=cv.fit_transform(x)
v=x.toarray()

print(v)


# In[12]:


first_col=df.pop('message')     # popping out the message and making it first column 
df.insert(0,'message',first_col)
df         #position of two columns is shifted to make it easier to calcualte


# In[13]:


#splitting train + test  3:1        #taking 70 and 30 combination 

train_x=x[:4180]
train_y=y[:4180]

test_x=x[4180:]
test_y=y[4180:]


# In[14]:


#model building by sklearn  librarry which has bernoulli
bnb=BernoulliNB(binarize=0.0)   #0.0 is binarization factor 
model=bnb.fit(train_x,train_y)    #fitting everything traing data  in our bnb model 

#prediction based on train and test data 

y_pred_train=bnb.predict(train_x)   
y_pred_test=bnb.predict(test_x)


# In[15]:


#training score
print(bnb.score(train_x,train_y)*100)

#testing score
print(bnb.score(test_x,test_y)*100)


# In[16]:


#report of our traning data 
from sklearn.metrics import classification_report
print(classification_report(train_y,y_pred_train))


# In[17]:


#report of testing data 
from sklearn.metrics import classification_report
print(classification_report(test_y,y_pred_test))


# In[ ]:




