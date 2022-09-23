

import pandas as pd
data=pd.read_csv("C:/Users/CHAITRA/Desktop/Social_Network_Ads.csv")
print(data.isnull().sum())
x=data.iloc[:,2:5]
x_corr=x.corr()
print(x_corr)

x_ind=data.iloc[:,2:4].values
y=data.iloc[:,-1].values

y=y.reshape(y.shape[0],1)
print(y.shape)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_ind=sc.fit_transform(x_ind) #here values are not in same range so we perform scalar index some is in thousands ans aome is in lakhs

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_ind,y,test_size=0.2,random_state=23)

from sklearn.linear_model import LogisticRegression
leg=LogisticRegression()
leg.fit(x_train,y_train)
y_pred=leg.predict(x_test)

print(leg.predict([[25,35000]]))
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_pred,y_test)
print(acc)
