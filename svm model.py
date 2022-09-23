

import pandas as pd
data=pd.read_csv("C:/Users/CHAITRA/Desktop/heart.csv")
print(data.isna().sum())
print(data.head())


X=data.iloc[:,0:13].values
Y=data.iloc[:,-1].values
Y=Y.reshape(Y.shape[0],1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=23)

from sklearn.svm import SVC
model=SVC(kernel='rbf',random_state=0)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_pred,Y_test)
print(cm)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_pred,Y_test)
print(score)



