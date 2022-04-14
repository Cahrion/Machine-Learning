import pandas as pd

url = 'https://bilkav.com/satislar.csv'

veriler = pd.read_csv(url)

X = veriler.iloc[:,0:1].values
Y = veriler.iloc[:,1].values

bolme = 0.33

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size = bolme)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
print(lr.predict(X_test))

import pickle

dosya = 'model.kayit'
pickle.dump(lr,open(dosya,'wb'))

yuklenen = pickle.load(open(dosya,'rb'))
# Sonradan yüklenen dosya
print('-------------------------')
print(yuklenen.predict(X_test))