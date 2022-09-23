# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:07:09 2022

@author: CHAITRA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:/Users/CHAITRA/Desktop/heart.csv")
print(data.isnull().sum())
print(data.head())
print(data.iloc[:,0:13:2])
y=data['target'].values
y=y.reshape(y.shape[0],1)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=23)

from sklearn.neighbors import KNeighborsClassifier
list=[]
for i in range(1,10):
  model=KNeighborsClassifier(n_neighbors=i)
  model.fit(x_train,y_train)
  y_pred=model.predict(x_test)
  from sklearn.metrics import confusion_matrix
  cm=confusion_matrix(y_test, y_pred)
  print(cm)
 
  from sklearn.metrics import accuracy_score
  score=accuracy_score(y_test,y_pred)
  print(score)
  list.append(score)

x=[1,2,3,4,5,6,7,8,9]

plt.plot(x,list,color='b',label='accuracy score')
plt.legend()
plt.show()





