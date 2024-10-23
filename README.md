# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
 ```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df= pd.read_csv("/content/bmi.csv")
df
```

![image](https://github.com/user-attachments/assets/9ea8edd1-4b81-4bc2-940e-5d8d18b8e4f0)


```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/e601a2e7-af6d-42ff-80e5-0fbd7f834cd1)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```

![image](https://github.com/user-attachments/assets/f52d309f-ba68-4b7d-8a9e-1415637e0b44)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/58d950e3-5e14-40f2-bcb7-870c48f5bf1b)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/ee98e9e9-54e4-4950-b913-94aa084584ac)

```
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/694cd1bf-73ee-4530-b4e6-c34419f94cd6)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head()
```

![image](https://github.com/user-attachments/assets/0c5bd13d-25a8-47ec-9673-8653fa3123c0)

```
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```
```
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```

![image](https://github.com/user-attachments/assets/21daffc8-b911-48cd-850f-ce38be85bf96)

```
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/e961e449-b813-4ffb-b99a-433feb4c622a)

```
missing=data[data.isnull().any(axis=1)]
missing
```

![image](https://github.com/user-attachments/assets/fbbac5b9-e0a4-4761-9d74-d34c395cb12e)

```
data2=data.dropna(axis=0)
data2
```

![image](https://github.com/user-attachments/assets/b4c4a3bc-29ea-414a-87f3-b55bdc247b91)

```
sal=data['SalStat']
data2['SalStat']=data['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```


![image](https://github.com/user-attachments/assets/dd6ae99d-38d6-43a9-b2e1-5a65cd23b2c7)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```


![image](https://github.com/user-attachments/assets/9b2cfebc-fcfa-4e2d-905c-15008f45a0a0)


```
data2
```


![image](https://github.com/user-attachments/assets/44494a39-c872-4c7a-8107-45ce2c819c25)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```


![image](https://github.com/user-attachments/assets/5c839ffb-9026-43be-acd3-80cc0bf5d2af)

```
columns_list=list(new_data.columns)
print (columns_list)
```

![image](https://github.com/user-attachments/assets/71ac1b70-7f9d-4f07-a83e-81fc312c4a22)

```
features=list(set(columns_list))
print(features)
```


![image](https://github.com/user-attachments/assets/5efa78d8-f82f-40a9-9744-011e342d48cd)

```
y=new_data['SalStat'].values
print(y
```


![image](https://github.com/user-attachments/assets/c6b1eccf-c27e-472e-9e93-cd1f8978a829)

```
x=new_data[features].values
print(x)
```

![image](https://github.com/user-attachments/assets/4c3069a9-0fb1-4cb4-a2fd-731f01a764a9)

```
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```


![image](https://github.com/user-attachments/assets/cf746ca8-2502-437b-a4ee-c7eb600a41f3)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```


![image](https://github.com/user-attachments/assets/2ce335c8-01f3-4123-9ab7-4a854646dfe9)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```


![image](https://github.com/user-attachments/assets/8ed42482-9265-4700-8622-72e21f3aa981)

```
print('Misclassified samples: %d' % (test_y !=prediction).sum())
```


![image](https://github.com/user-attachments/assets/9b71dfdf-f169-473b-b411-a6fd98d583db)

```
data.shape
```

![image](https://github.com/user-attachments/assets/55fa3451-ad3d-425a-bc45-e3ce186e20b6)




# RESULT:
        Thus perform Feature Scaling and Feature Selection process and save the data to a file successfully.
