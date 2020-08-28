#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the datasets
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

#Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=0)

#Feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
"""
