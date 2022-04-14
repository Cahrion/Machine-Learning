# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 20:05:48 2021

@author: icabi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('Wine.csv')


X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

from sklearn.linear_model import LogisticRegression
# PCA ile yapılmayan yapı
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
# PCA ile yapılan yapı
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)
y_pred2 = classifier2.predict(X_test2)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('PCA ile yapılmayan yapımız')
print(cm)
cm = confusion_matrix(y_test, y_pred2)
print('PCA ile yapılan yapımız')
print(cm)
cm = confusion_matrix(y_pred, y_pred2)
print('PCA vs PCA"sız')
print(cm)
