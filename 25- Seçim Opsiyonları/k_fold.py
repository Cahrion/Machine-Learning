# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:27:42 2020

@author: icabi
"""
#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('Social_Network_Ads.csv')

x = veriler.iloc[:,2:4].values # Bağımsız degişkenler
y = veriler.iloc[:,4:].values


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.svm import SVC
classifiers = SVC(kernel='rbf', random_state=0)
classifiers.fit(X_train,y_train)

y_pred = classifiers.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

# Çapraz Doğrulama
from sklearn.model_selection import cross_val_score
basari = cross_val_score(estimator=classifiers,X=X_train,y=y_train,cv = 4)
print(basari.mean())
print(basari.max())


























