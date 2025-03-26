# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy, confusion matrices

5.Display the result. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SAI SANJAY R 
RegisterNumber: 212223040178 
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])

x=data1.iloc[:,:-1]
x
print(x)
y=data1["status"]
y
print(y)
print()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
HEAD:
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
![Screenshot 2025-03-25 175443](https://github.com/user-attachments/assets/becfbfe8-8463-4817-850b-60795a3d344a)

COPY:
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
![Screenshot 2025-03-25 175513](https://github.com/user-attachments/assets/3d1ededb-a385-40ab-b34e-e7ca0a3d4daa)

FIT TRANSFORM:
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
```
![Screenshot 2025-03-25 175552](https://github.com/user-attachments/assets/75fe1751-6d6c-4577-96a3-96596ed5a00a)

LOGISTIC REGRESSION:
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
```
![Screenshot 2025-03-25 175627](https://github.com/user-attachments/assets/e27a58e3-f183-4f51-8762-750416708328)

ACCURACY SCORE:
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
```
![Screenshot 2025-03-25 175648](https://github.com/user-attachments/assets/1f200c5e-079a-4a7a-84d7-111d9a38f2ee)

CONFUSSION MATRIX:
```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
```
![Screenshot 2025-03-25 175658](https://github.com/user-attachments/assets/9707c0df-a39e-4862-a955-f6b3faf9b3da)

CLASSIFICATION REFORT:
```
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```
![Screenshot 2025-03-25 175722](https://github.com/user-attachments/assets/ade13549-ea69-4730-bbac-f26f0b3362bf)

PREDICTION:
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![Screenshot 2025-03-25 175747](https://github.com/user-attachments/assets/8fd3a68d-8fb3-4dcf-a13e-e155bdf78364)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
