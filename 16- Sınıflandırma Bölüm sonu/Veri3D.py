# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:06:13 2020

@author: icabi
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5


plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150,azim=110)
ax.scatter(iris.data[:,0], iris.data[:,1], iris.data[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title('IRIS Verisi')
ax.set_xlabel('birinci özellik')
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel('İkinci özellik')
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel('üçüncü özellik')
ax.w_zaxis.set_ticklabels([])

plt.show()







