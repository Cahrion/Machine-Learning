# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 02:01:04 2020

@author: icabi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri yükleme
veriler = pd.read_csv('veriler.csv')

# data frame dilimleme (slice)
x = veriler.iloc[:,1:3]
y = veriler.iloc[:,3:4]

# Numpy (dizi)array dönüşümü
X = x.values
Y = y.values



from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0,n_estimators = 6)
rf_reg.fit(X,Y.ravel())


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.40, random_state=0)


rf_reg.fit(x_train,y_train)

y_pred = rf_reg.predict(x_test)

















