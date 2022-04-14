# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 02:01:04 2020

@author: icabi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Veri yükleme
veriler = pd.read_csv('maaslar.csv')

# data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# Numpy (dizi)array dönüşümü
X = x.values
Y = y.values
Z = X + 0.5
K = X - 0.4
#Linear regression
# Doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

a = r2_score(Y,lin_reg.predict(X))
b = r2_score(Y,lin_reg.predict(Z))
c = r2_score(Y,lin_reg.predict(K))

print('Linear R2 değeri')
print(a)
print(b)
print(c)



#polynomial regression
#Doğrusal olmayan (nonlinear model) oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10) #
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
"""
# Doğrusal olan mdoelin remi
plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg.predict(X), color ='blue')
plt.show()

# Doğrusal olmayan modelin resmi
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()
"""
#--------------

a = r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X)))
b = r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(Z)))
c = r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(K)))

print('Poly (Polinom) R2 değeri')
print(a)
print(b)
print(c)



# Tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))


print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

# ---SVR---

from sklearn.preprocessing import StandardScaler

sc1= StandardScaler()

x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)
"""
plt.scatter(x_olcekli,y_olcekli, color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli), color='blue')
"""
s= r2_score(y_olcekli,svr_reg.predict(x_olcekli))


print('SVR R2 değeri')
print(s)



# Agaç yapısı tahmini

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)
"""
plt.scatter(X,Y)
plt.plot(X,r_dt.predict(X))

plt.plot(x,r_dt.predict(Z), color='green')
plt.plot(x,r_dt.predict(K), color='yellow')
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

"""
a = r2_score(Y,r_dt.predict(X))
b = r2_score(Y,r_dt.predict(K))
c = r2_score(Y,r_dt.predict(Z))

print('Decision Tree R2 değeri')
print(a)
print(b)
print(c)


# Random Forest tahmini
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0,n_estimators = 10)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.4]]))
"""
plt.scatter(X, Y, color='red')
plt.plot(X,rf_reg.predict(X), color='black')


plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(x,rf_reg.predict(K), color='yellow')

"""

a = r2_score(Y,rf_reg.predict(X))
b = r2_score(Y,rf_reg.predict(K))
c = r2_score(Y,rf_reg.predict(Z))


print('Random Forest R2 değeri')
print(a)
print(b)
print(c)












