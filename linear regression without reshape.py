import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/CHAITRA/Desktop/Salary_Data.csv")
print(data.isnull().sum())
x=data[['YearsExperience']].values
y=data[['Salary']].values
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=2)

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(x_train,y_train)
u=LR.predict(x_test)
t=LR.predict([[2.5]])
