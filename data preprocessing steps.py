import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/CHAITRA/Desktop/Data preprocessing1 ML")
print(data.head(3))
print(data.isnull().sum())
X=data.iloc[:,0:3].values
Y=data.iloc[:,-1].values


#print missing values

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy='mean')
X[:,1:3]=si.fit_transform(X[:,1:3])
print(X)

#labeling parts

from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
X[:,0]=Le.fit_transform(X[:,0])
print(X)

#onehot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
Ct=ColumnTransformer([('Tf1',OneHotEncoder(drop='first'),[0])],remainder='passthrough')
X=Ct.fit_transform(X)
print(X)

#saclling

from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
X=Sc.fit_transform(X)
print(X)

#splitting

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3,random_state=2)
print(X_test)
print(Y_test)

