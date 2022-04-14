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


X = veriler.iloc[:,3:].values
Y = veriler.iloc[:,2:3]

from sklearn.cluster import KMeans
km = KMeans(n_clusters=4,init= 'k-means++',)


"""
# Deney Platformu...

sonuclar = []
for i in range(1,11):
    km = KMeans(n_clusters= i,init= 'k-means++', random_state=123)
    km.fit(X)
    sonuclar.append(km.inertia_)
    
    
plt.plot(range(1,11), sonuclar)
"""


