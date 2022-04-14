# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:05:21 2020

@author: icabi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri yükleme
veriler = pd.read_csv('maaslar.csv')

# data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# Numpy (dizi)array dönüşümü
X = x.values
Y = y.values

#Linear regression
# Doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



#polynomial regression
#Doğrusal olmayan (nonlinear model) oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10) #
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# Doğrusal olan mdoelin remi
plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg.predict(X), color ='blue')
plt.show()

# Doğrusal olmayan modelin resmi
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()
#--------------

# Tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))


print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))





from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y)
plt.plot(X,r_dt.predict(X))

plt.plot(x,r_dt.predict(Z), color='green')
plt.plot(x,r_dt.predict(K), color='yellow')

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
