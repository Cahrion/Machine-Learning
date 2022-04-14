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

veriler = pd.read_excel('iris.xls')

x = veriler.iloc[:,1:4].values # Bağımsız degişkenler
y = veriler.iloc[:,4:].values

# Test bölümü
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

# Azaltma uygulanır.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#-------------------------------------------------------------
# Buradan itibaren sınıflandırma algoritmaları başlar.
#-------------------------------------------------------------
# 1. Logistik Regression
#-------------------------------------------------------------

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)


cm = confusion_matrix(y_test,y_pred)
print('linear')
print(cm)

#-------------------------------------------------------------
# 2. KNN
#-------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)

#-------------------------------------------------------------
# 3. SVC 
#-------------------------------------------------------------

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

#-------------------------------------------------------------
# 4.Naive_Bayes
#-------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

#-------------------------------------------------------------
# 5. Decision Tree
#-------------------------------------------------------------

from sklearn import tree

dtc = tree.DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
        
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

#-------------------------------------------------------------
# 6. Random Forest
#-------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=6, criterion= 'entropy')

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

print('RFC')
print(cm)

#-------------------------------------------------------------
# 7. ROC , TPR, FPR değerleri
#-------------------------------------------------------------

"""
y_proba = rfc.predict_proba(X_test)
print(y_proba)

from sklearn import metrics

fpr, tpr,thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)

"""
















