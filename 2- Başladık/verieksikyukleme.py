# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 00:46:40 2020

@author: icabi
"""

# :=_=:
# Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Kodlar


# Eksik veriler
veriler = pd.read_csv('eksikveriler.csv')
print(veriler[['kilo']])


# sci-  kit learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)