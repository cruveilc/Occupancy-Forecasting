from sklearn.neural_network import MLPClassifier
import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import tree

data = pd.read_excel('project.xlsx'
)
print(data.isnull().values.any())
print(data.columns)

y = data[' label']
x=data[[' Tin', ' Tout', ' humidity', ' detected_motions', ' power',
       ' office_CO2_concentration', ' door', ' CO2_corridor',
       ' acoustic_pressure']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

reg = LinearRegression().fit(x_train, y_train)
y_predict=reg.predict(x_test)

reg1 = RandomForestRegressor(n_estimators=50, random_state=0)
reg1.fit(x_train, y_train)
y_predict1 = reg1.predict(x_test)

reg2 = tree.DecisionTreeRegressor()
reg2.fit(x_train,y_train)
y_predict2 = reg2.predict(x_test)

print(r2_score(y_test,y_predict))
print(mean_squared_error(y_test,y_predict))
print(r2_score(y_test,y_predict1))
print(mean_squared_error(y_test,y_predict1))
print(r2_score(y_test,y_predict2))
print(mean_squared_error(y_test,y_predict2))
