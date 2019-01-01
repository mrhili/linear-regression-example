# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 03:01:05 2019

@author: hp
"""
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


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

forecast_out = math.ceil( 0.01 * len(  df   )    )

df['label'] = df[ forecast_col ].shift( - forecast_out )
df.dropna(inplace=True)

#print(df.tail())

X = np.array( df.drop(['label'], 1) )
y = np.array( df['label'] )

X = preprocessing.scale( X )


y = np.array( df['label'] )

#print( len(X), len(y) )

X_train , X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2 )

clf = LinearRegression(n_jobs=-1)
#another algo for machine learning
#clf = svm.SVR()
clf.fit( X_train , y_train )

accuracy = clf.score(X_test, y_test)

print( accuracy )