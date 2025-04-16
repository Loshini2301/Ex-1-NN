## NAME: LOSHINI.G
## REGISTER NUMBER: 212223220051
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
### IMPORT LIBRARIES:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```
### LOAD DATASET:
```
df=pd.read_csv('Churn_Modelling.csv',index_col="RowNumber")
df.head()
```
### DATA PREPROCESSING:
```
df.isnull().sum()
df.duplicated().sum()
df = df.drop(['Surname', 'Geography', 'Gender'], axis=1)
df.head()  
scaler=StandardScaler()
df=pd.DataFrame(scaler.fit_transform(df))
df.head()
```
### ASSINGNING X AND Y VALUES:
```
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
```
### SPLITTING DATA FOR TRAINING AND TESTING:
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x.shape
y.shape
print("X training data shape=",x_train.shape)
print("X testing data shape=",x_test.shape)
print("Y training data shape=",y_train.shape)
print("Y testing data shape=",y_test.shape)
```


## OUTPUT:
## DATASET:
![image](https://github.com/user-attachments/assets/58a79152-6e11-4de0-a8f1-b4bfa64b4dbe)
## SUMMATION OF NULL VALUES:
![image](https://github.com/user-attachments/assets/b3d71b71-b2d3-42c1-a6a8-435bea798d0a)
## DATASET AFTER SCALING:
![image](https://github.com/user-attachments/assets/772e0070-6c5a-412c-b2b1-c2c4dbdde1b4)
## SPLITTING TRAINING AND TESTING DATA:
![image](https://github.com/user-attachments/assets/79a3b601-6914-4e0c-9b4a-6fdd36635586)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


