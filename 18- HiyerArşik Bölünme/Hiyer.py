# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:57:35 2020

@author: icabi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('musteriler.csv')

#---------------------------------------------------------
# Başlangıç
#---------------------------------------------------------

X = veriler.iloc[:,3:].values
Y = veriler.iloc[:,2:3]

#---------------------------------------------------------
# Kmeans kütüphanesi
#---------------------------------------------------------

from sklearn.cluster import KMeans
km = KMeans(n_clusters=4,init= 'k-means++',)

Y_tahminKmeans = km.fit_predict(X)

"""
# Deney Platformu...

sonuclar = []
for i in range(1,11):
    km = KMeans(n_clusters= i,init= 'k-means++', random_state=123)
    km.fit(X)
    sonuclar.append(km.inertia_)
    
    
plt.plot(range(1,11), sonuclar)
plt.show()
"""
plt.title('KMeans')
plt.scatter(X[Y_tahminKmeans==0,0],X[Y_tahminKmeans==0,1],s=100, c='red')
plt.scatter(X[Y_tahminKmeans==1,0],X[Y_tahminKmeans==1,1],s=100, c='blue')
plt.scatter(X[Y_tahminKmeans==2,0],X[Y_tahminKmeans==2,1],s=100, c='green')
plt.scatter(X[Y_tahminKmeans==3,0],X[Y_tahminKmeans==3,1],s=100, c='black')
plt.show()

#---------------------------------------------------------
# Agglomerative Clustering / Hiyerarşiv Cluster
#---------------------------------------------------------

from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=4,affinity='euclidean', linkage='ward')
Y_tahmin = agc.fit_predict(X)

plt.title('HiyerArşiv')
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='black')

plt.show()

#---------------------------------------------------------
# HiyerArşivin Dendrogram modu
#---------------------------------------------------------

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.show()



















