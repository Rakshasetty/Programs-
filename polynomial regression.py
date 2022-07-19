import pandas as pd
data=pd.read_csv("C:/Users/CHAITRA/Desktop/Position_Salaries.csv")
x=data.iloc[:,1].values
y=data.iloc[:,-1].values
print(x)
print(y)
x=x.reshape(x.shape[0],1)
y=y.reshape(y.shape[0],1)

from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x,y)
y_predict=Lr.predict(x)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_trans=poly.fit_transform(x)
Lr.fit(x_trans,y)
y_pred_poly=Lr.predict(x_trans)

import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.scatter(x,y,color='r')
x2=plt.plot(x,y_predict,color='b')
plt.subplot(2,1,2)
plt.scatter(x,y,color='r')
plt.plot(x,y_pred_poly,color='b')
plt.legend()
plt.show()

