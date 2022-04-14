# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:00:08 2020

@author: icabi
"""
#1.kutuphaneler
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
