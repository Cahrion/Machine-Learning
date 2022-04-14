# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 01:10:13 2020

@author: icabi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
#-------------------------------------
# Random Selection
#-------------------------------------
N = 10000
d = 10
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()


