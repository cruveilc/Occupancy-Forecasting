import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from sklearn.model_selection import train_test_split

data = pd.read_excel('project.xlsx',
sheetname=0,
header=0,
index_col=[0],
keep_default_na=True
)
print(data.isnull().values.any())
print(data.columns)
#data.plot(figsize=(18,5))
#plt.show()
#sns.pairplot(data)
#plt.show()
lab = data[' label']
x=data[[' Tin', ' Tout', ' humidity', ' detected_motions', ' power',
       ' office_CO2_concentration', ' door', ' CO2_corridor',
       ' acoustic_pressure']]

x_train, x_test, y_train, y_test = train_test_split(x, lab, test_size = 0.3)

#LinearRegression
# model = sklearn.linear_model.LinearRegression()
# model.fit(x_train, y_train)
#
# print(model.coef_)
#
# predictions = model.predict(x_test)
# plt.scatter(y_test, predictions)
# plt.show()
# np.sqrt(sklearn.metrics.mean_squared_error(y_test, predictions))

#RandomForest

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

plt.scatter(y_test, y_pred)
plt.show()

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Tin=data[' Tin']
#Code Explanation: model = LinearRegression() creates a linear regression model and the for loop divides the dataset into three folds (by shuffling its indices). Inside the loop, we fit the data and then assess its performance by appending its score to a list (scikit-learn returns
# the R² score which is simply the coefficient of determination).
# X = pd.DataFrame(co2)
# y = pd.DataFrame(lab)
# model = sklearn.linear_model.LinearRegression()
# scores = []
# kfold = KFold(n_splits=3, shuffle=True, random_state=42)
# for i, (train, test) in enumerate(kfold.split(X, y)):
#  model.fit(X.iloc[train,:], y.iloc[train,:])
#  score = model.score(X.iloc[test,:], y.iloc[test,:])
#  scores.append(score)
# print(scores)







