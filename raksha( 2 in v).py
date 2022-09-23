# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 07:46:05 2022

@author: CHAITRA
"""


import pandas as pd
data=pd.read_csv("C:/Users/CHAITRA/Desktop/50_Startups (1).csv")
print(data.isna().sum())
from sklearn.impute import SimpleImputer
SI=SimpleImputer()

SI_R__D=SI.fit(data[['R&D Spend']])
data['R&D Spend']=SI_R__D.transform(data[['R&D Spend']])


SI_Marketing_Spend=SI.fit(data[['Marketing Spend']])
data['Marketing Spend']=SI_Marketing_Spend.transform(data[['Marketing Spend']])


SI_State=SimpleImputer(strategy='most_frequent')
SI_State=SI_State.fit(data[['State']])
data['State']=SI_State.transform(data[['State']])
print(data.isna().sum())

print(data.corr()['Profit'])
data=data.drop('State',axis=1)

import matplotlib.pyplot as plt
plt.plot(data['R&D Spend'],data['Profit'])
plt.xlabel("R&D spend")
plt.ylabel("Profit spend")
plt.show()

plt.plot(data['Profit'],data['Marketing Spend'])
plt.xlabel('Profit spend')
plt.ylabel("Marketing Spend")
plt.show()

X=data.iloc[:,0:3:2].values
Y=data.iloc[:,-1].values
print(X)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split  
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
reg=KNeighborsRegressor(n_neighbors=2)
#reg=SVR(kernel='poly',degree=2)
#reg= DecisionTreeRegressor(random_state=(0),max_depth=(4))
reg.fit(X,Y)

y_pred_Decision=reg.predict(X_test)
y_pred_SVR=reg.predict(X_test)
y_pred_KNN=reg.predict(X_test)
from sklearn.metrics import r2_score
score=r2_score(Y_test,y_pred_SVR)
score=r2_score(Y_test,y_pred_KNN)
score=r2_score(Y_test,y_pred_Decision)
print(score)