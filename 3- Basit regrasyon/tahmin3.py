# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 22:07:24 2020

@author: icabi
"""
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)


aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]


satislar2 = veriler.iloc[:,:1].values


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#verilerin olceklenmesi
"""
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.transform(y_test)
"""
# Model insası(linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


tahmin = lr.predict(x_test)












