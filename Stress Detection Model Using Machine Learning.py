# -*- coding: utf-8 -*-
"""ML Stress Detection Project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yjB0nFy40lH-xg4Goo4r93POPmXHuCCd
"""

import numpy as np
import pandas as pd

df=pd.read_csv('stress.csv')
df.head()

df.describe()

df.isnull().sum()

import nltk
import re
from nltk. corpus import stopwords
import string
nltk. download( 'stopwords' )
stemmer = nltk. SnowballStemmer("english")
stopword=set (stopwords . words ( 'english' ))

def clean(text):
    text = str(text) . lower()  #returns a string where all characters are lower case. Symbols and Numbers are ignored.
    text = re. sub('\[.*?\]',' ',text)  #substring and returns a string with replaced values.
    text = re. sub('https?://\S+/www\. \S+', ' ', text)#whitespace char with pattern
    text = re. sub('<. *?>+', ' ', text)#special char enclosed in square brackets
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)#eliminate punctuation from string
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)#word character ASCII punctuation
    text = [word for word in text. split(' ') if word not in stopword]  #removing stopwords
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]#remove morphological affixes from words
    text = " ". join(text)
    return text
df [ "text"] = df["text"]. apply(clean)

import matplotlib. pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ". join(i for i in df. text)
stopwords = set (STOPWORDS)
wordcloud = WordCloud( stopwords=stopwords,background_color="white") . generate(text)
plt. figure(figsize=(10, 10) )
plt. imshow(wordcloud )
plt. axis("off")
plt. show( )

from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split

#targetting text and label column from our dataset
x = np.array (df["text"])      #x has count of text values
y = np.array (df["label"])     #y has count of label values that is in form of 0 and 1 which is" 0 for stress and 1 for not stress"

cv = CountVectorizer ()     
X = cv. fit_transform(x)    # fits data(mean,sd) to scale the output in x as input
print(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33,random_state = 40)    #data is splitted into70,30 format   
  #random_state function is used for initialization

from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)

user=input("Enter the text")
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)