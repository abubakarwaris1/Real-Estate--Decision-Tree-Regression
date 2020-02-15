# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\\Udemy - Machine Learning\\practice\\Real Estate-Decision Tree\\Real estate.csv')
X = dataset.iloc[:, 1:7].values
y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_test=sc_y.fit_transform(y_test.reshape(-1,1))

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred=sc_y.inverse_transform(y_pred)
y_test=sc_y.inverse_transform(y_test)

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((len(y),1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6]]
regressor_OLS= sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS= sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
