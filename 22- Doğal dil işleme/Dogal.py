# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 19:50:44 2020

@author: icabi
"""
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
yorumlar = pd.read_csv('Restaurant_Reviews.csv')

import re

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('stopwords')

# Preprocessing [Ön işleme]
derlem = []
for i in range(1000):
    yorum = re.sub("[^a-zA-Z]",' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
    
    
    
# Feautre Extraction [ Öznitelik Çıkarımı ]    
# Bag of Words [ BOW ]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 2000)
X = cv.fit_transform(derlem).toarray() 
Y = yorumlar.iloc[:,1].values

# Makine Öğrenimi ...
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=0)

#---------------------------------------
# Naive_Bayes denemesi
#---------------------------------------
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('Naive_Bayes')
print(cm) # ♦72.5 accuracy

#---------------------------------------
# KN-N denemesi
#---------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)

#---------------------------------------
# Random Forest denemesi
#---------------------------------------

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5, criterion= 'entropy')

rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)
