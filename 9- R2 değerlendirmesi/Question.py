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
veriler = pd.read_csv('maaslar_yeni.csv')
artis = veriler.sort_values(by=['maas'])
# data frame dilimleme (slice)
x = artis.iloc[:,2:5]
y = artis.iloc[:,5:]

# Numpy (dizi)array dönüşümü"
X = x.values
Y = y.values


print(veriler.corr())
#--------------------------------------------------------------------------
#Linear regression
#--------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

a = r2_score(Y,lin_reg.predict(X))


print('Linear R2 değeri')
print(a)





#--------------------------------------------------------------------------
# Polly 
#--------------------------------------------------------------------------

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10) #
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

a = r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X)))
print('Poly (Polinom) R2 değeri')
print(a)

#--------------------------------------------------------------------------
# SVR öncesi ölceklendirme yapılması gerekir.
#--------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

sc1= StandardScaler()

x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

s= r2_score(y_olcekli,svr_reg.predict(x_olcekli))

print('SVR R2 değeri')
print(s)

#--------------------------------------------------------------------------
# Agaç yapısı
#--------------------------------------------------------------------------

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)

a = r2_score(Y,r_dt.predict(X))

print('Decision Tree R2 değeri')
print(a)

#--------------------------------------------------------------------------
# Agaç yapısı
#--------------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0,n_estimators = 30)
rf_reg.fit(X,Y.ravel())

a = r2_score(Y,rf_reg.predict(X))


print('Random Forest R2 değeri')
print(a)


print(lin_reg.predict([[10,10,100]]))
print(lin_reg2.predict(poly_reg.fit_transform([[10,10,100]])))
print(svr_reg.predict([[10,10,100]]))
print(r_dt.predict([[10,10,100]]))
print(rf_reg.predict([[10,10,100]]))



plt.plot(X,rf_reg.predict(X), color='black')


