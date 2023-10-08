import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("suv_data.csv")
data.drop(['User ID','Gender'],inplace=True, axis=1)
X=data.drop("Purchased",axis=1)
y=data["Purchased"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#model
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)

predicition=log_reg.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
c_matrix=confusion_matrix(y_test,predicition)
print(c_matrix)
accuracy_score=accuracy_score(y_test,predicition)
print(accuracy_score)
#graph
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, edgecolor='k')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()