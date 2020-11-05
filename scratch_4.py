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
from pandas.plotting import register_matplotlib_converters

from sklearn.model_selection import train_test_split

register_matplotlib_converters()

data = pd.read_excel('values_office.xlsx',
sheetname=0,
header=0,
index_col=[0],
keep_default_na=True
)
print(data.isnull().values.any())
print(data.columns)

#data.plot(figsize=(18,5))
#plt.show()



data_training = data[34544:39432]

lab = data_training['label']
x=data_training[['Toffice_reference', 'humidity', 'detected_motions', 'power',
       'office_CO2_concentration', 'door']]

x_train, x_test, y_train, y_test = train_test_split(x, lab, test_size = 0.3)


#LinearRegression
model = sklearn.linear_model.LinearRegression()
model.fit(x_train, y_train)

print(model.coef_)

predictions = model.predict(x_test)
plt.scatter(y_test, predictions)
plt.show()
np.sqrt(sklearn.metrics.mean_squared_error(y_test, predictions))



#RandomForest

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

#plt.scatter(y_test, y_pred)
#plt.show()
#plt.figure(figsize=(11, 9))
#plt.plot(y_test, label='Orginal')
#plt.plot(y_pred, label='Predicted', color='Orange')
#plt.show()

#data_estimated = data[34544:39432]

data_estimated = pd.concat([data[0:34544],data[39432:]])

x_estimated=data_estimated[['Toffice_reference', 'humidity', 'detected_motions', 'power',
       'office_CO2_concentration', 'door']]
data_estimated['label']=regressor.predict(x_estimated)

plt.plot(data_estimated['label'])
data['label']= pd.concat([data_estimated['label'][0:34544],data['label'][34544:39432],data_estimated['label'][34544:]])

data.label.plot(figsize=(18,5))
plt.show()


#SVM Regression
# svm_reg = LinearSVR(epsilon=0.5)
# svm_reg.fit(x_train,y_train)
# y_pred2 = regressor.predict(x_test)


#plt.scatter(y_test, y_pred2)
#plt.show()




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
