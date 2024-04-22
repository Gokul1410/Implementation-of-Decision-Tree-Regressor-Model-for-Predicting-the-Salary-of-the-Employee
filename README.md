# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Gokul C
RegisterNumber:  212223240040
*/

import pandas as pd

data=pd.read_csv("/content/Salary_EX7.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])

from sklearn.tree import DecisionTreeRegressor,plot_tree

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plot_tree(dt,feature_names=x.columns,class_names=['Salary'], filled=True)

plt.show()


```

## Output:

![Screenshot 2024-04-02 182649](https://github.com/Gokul1410/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/153058321/ac01e851-7e7d-4437-aeb5-b9784f30102e)
![Screenshot 2024-04-02 182702](https://github.com/Gokul1410/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/153058321/752c9cff-657a-40ea-b775-37894c047b5b)
![Screenshot 2024-04-02 182726](https://github.com/Gokul1410/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/153058321/3380f5ba-c3f1-43a6-8b90-dd7f065e1775)
![Screenshot 2024-04-02 182800](https://github.com/Gokul1410/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/153058321/77078bdb-8b4a-4b97-aa84-64c4bbe2fc52)






## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
