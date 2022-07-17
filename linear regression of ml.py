import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/CHAITRA/Desktop/Salary_Data.csv")
print(data.isnull().sum())
x=data[['YearsExperience']].values
y=data[['Salary']].values
print(x.shape)
print(y.shape)
       
#x=x.reshape(x.shape[0],1)
#y=y.reshape(y.shape[0],1)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=2)

from sklearn.linear_model import LinearRegression
LG=LinearRegression()
LG.fit(x_train,y_train)
u=LG.predict(x_test)
t=LG.predict([[2.7]])

