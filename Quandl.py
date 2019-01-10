# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 03:01:05 2019

@author: hp
"""
import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import pickle

from matplotlib import style

style.use('ggplot')


quandl.ApiConfig.api_key = 'C7_Lrb9EY6hMduTn-GAb'

df = quandl.get("WIKI/GOOGL")
#Defining what we gonna get
df = df[[ 'Open'    ,'High'       ,'Adj. Close'  ,'Adj. Volume' ]]
#Adding the relationship using rows
df['HL_PCT'] = ( df['High'] - df['Adj. Close'] ) / ( df['Adj. Close'] * 100.0)
df['PCT_change'] = ( df['Adj. Close'] - df['Open'] ) / ( df['Open'] * 100.0)
#Taking what we care about the most
df = df[[ 'HL_PCT'    ,'PCT_change'       ,'Adj. Close'  ,'Adj. Volume' ]]
#print(df.head())

forecast_col = 'Adj. Close'


#Replacing the nan data !important
df.fillna(-99999, inplace=True)

forecast_out = math.ceil( 0.1 * len(  df   )    )

df['label'] = df[ forecast_col ].shift( - forecast_out )


X = np.array( df.drop(['label','Adj. Close'], 1) )


X = preprocessing.scale( X )
X_lately = X[ - forecast_out:]
X = X[: - forecast_out:]
df.dropna(inplace=True)

#print(df.tail())
y = np.array( df['label'] )

y = np.array( df['label'] )

#print( len(X), len(y) )

X_train , X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2 )


#ONCE A MONTH

clf = LinearRegression(n_jobs=-1)
#another algo for machine learning
#clf = svm.SVR()
'''
clf.fit( X_train , y_train )

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
'''
#END ONCE A MONTH
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load( pickle_in )

accuracy = clf.score(X_test, y_test)

#print( accuracy )

forecast_set = clf.predict( X_lately )


#print( forecast_set, accuracy, forecast_out )

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()

one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len( df.columns )-1)] + [i]
    
#print( df.head() )
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend( loc = 4)
plt.xlabel( 'Date')
plt.ylabel( 'Price')
plt.show()