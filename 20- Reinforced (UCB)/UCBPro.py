# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 01:10:13 2020

@author: icabi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

#-------------------------------------
# Random Selection
#-------------------------------------
"""
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

"""
#-------------------------------------
# UCB
#-------------------------------------
N = 10000 # 10.000 tıklama
d = 10 # toplam 10 ilan
oduller = [0] * d # İlk basta bütün ilanların odulu 0
tiklamalar = [0] * d  # Tıklamalar o ana kadar ki
toplam = 0 # Toplam odul
secilenler = []

for n in range(0,N):
    ad = 0
    max_ucb = 0
    for i in range(0,d):
        if(tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n,ad]
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul

print('Toplam Ödül: ')
print(toplam)

plt.hist(secilenler)
plt.show()