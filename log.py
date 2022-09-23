# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:48:11 2022

@author: CHAITRA
"""

import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/CHAITRA/Desktop/IRIS.csv")
print(data.isnull().sum())
print(data.head())
print(data.info())
x=data.iloc[:,2:4].values
y=data.iloc[:,-1].values

print(x.shape)
print(y.shape)

y=y.reshape(y.shape[0],1)
print(y.shape)


from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
data['species']=Le.fit_transform(data['species'])
print(data.head(5))
print(data.corr())

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.5,random_state=23) 

from sklearn.linear_model import LogisticRegression

Log_reg=LogisticRegression()
Log_reg.fit(X_train,Y_train)
Y_pred=Log_reg.predict(X_test)

print(Log_reg.predict([[4.4,1.4]]))

import matplotlib.pyplot as plt
pl=data.iloc[:,2];
pw=data.iloc[:,3];

plt.scatter(pw,pl,'species','rgb','osd');
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()
    







