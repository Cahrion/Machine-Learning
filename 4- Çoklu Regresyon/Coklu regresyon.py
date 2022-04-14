# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Kodlar


# Eksik veriler
veriler = pd.read_csv('veriler.csv')



# sci-  kit learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


Yas = veriler.iloc[:,1:4].values

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])

#encoder Nominal Ordinal -> Numeric

ulke = veriler.iloc[:,0:1].values
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

 # Etiket atama --> OneHotEncoder
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()



gender = veriler.iloc[:,-1:].values
from sklearn import preprocessing

les = preprocessing.LabelEncoder()

gender[:,-1] = les.fit_transform(veriler.iloc[:,-1])

 # Etiket atama --> OneHotEncoder
ohes = preprocessing.OneHotEncoder()
gender = ohes.fit_transform(gender).toarray()






#numpy dizileri dataframe dönüşümü

sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])


sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns=['boy','kilo','yas'])


cinsiyet = veriler.iloc[:,-1].values


sonuc3 = pd.DataFrame(data = gender[:,:1], index = range(22), columns = ['cinsiyet'])

#dataframe birleştirme işlemi


s = pd.concat([sonuc,sonuc2], axis =1)


s2 = pd.concat([s,sonuc3],axis=1)
print(s2)

# Verilerin bölünmesi test egitim için bölünmesi

"""
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
"""
boy = sonuc2.iloc[:,:1].values

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag], axis=1)


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(sag,boy,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


"""
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())



"""


