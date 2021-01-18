import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.svm import LinearSVR
from pandas.plotting import register_matplotlib_converters
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import datetime

#register_matplotlib_converters()



tStart=time.time()


data = pd.read_excel('values_office.xlsx',header=0,index_col=None)
print(data.isnull().values.any())
print(data.columns)

temps=data['time']
Toffice_reference=data['Toffice_reference']
humidity=data['humidity']
detected_motions=data['detected_motions']
power=data['power']
office_CO2_concentration=data['office_CO2_concentration']
door=data['door']
occupancy=data['label']

ploting_datas=[Toffice_reference,humidity,detected_motions,power,office_CO2_concentration,door]

# Toffice_reference: select=0
# humidity: select=1
# detected_motions: select=2
# power: select=3
# office_CO2_concentration: select=4
# door: select=5
want_plot=False

if want_plot==True:
    legende = ['Office Co2 concentration (ppm)', 'label (occupancy) * 100']
    select = 4
    label_select = True
    if select != None:
        data = ploting_datas[select]
        plt.plot(temps[34544:39432], data[34544:39432])
        if label_select == True:
            plt.plot(temps[34544:39432], occupancy[34544:39432] * 100)
        plt.legend(legende)
        plt.xlabel('time')
        plt.gcf().autofmt_xdate()
        plt.show()
        plt.close()


data_training = data[34544:39432]

lab = data_training['label']
x=data_training[['Toffice_reference', 'humidity', 'detected_motions', 'power',
       'office_CO2_concentration', 'door']]

x_train, x_test, y_train, y_test = train_test_split(x, lab, test_size = 0.3)


#LinearRegression
# model = sklearn.linear_model.LinearRegression()
# model.fit(x_train, y_train)
#
# print(model.coef_)
#
# y_predict = model.predict(x_test)
# plt.scatter(y_test, y_predict)
# plt.xlabel('Real Occupancy')
# plt.ylabel('Estimated Occupancy')
# plt.show()
#
#
# MSE=mean_squared_error(y_test,y_predict)
# EVS=explained_variance_score(y_test,y_predict)
# R2S=r2_score(y_test,y_predict)
#
# print('LinearRegression')
#
# print('mean_squared_error : ',MSE)

# print('r2_score : ',R2S)

# GradientBoostingRegressor


# params = {'n_estimators': 500, 'max_depth': 6,
#         'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
# clf = GradientBoostingRegressor(**params).fit(x_train, y_train)

# clf = GradientBoostingRegressor()
# clf= clf.fit(x_train, y_train)
#
# y_predict = clf.predict(x_test)
# plt.scatter(y_test, y_predict)
# plt.xlabel('Real Occupancy')
# plt.ylabel('Estimated Occupancy')
# plt.show()
#
# MSE=mean_squared_error(y_test,y_predict)
# EVS=explained_variance_score(y_test,y_predict)
# R2S=r2_score(y_test,y_predict)
# #
# print('GradientBoostingRegressor')
#
# print('mean_squared_error : ',MSE)
#
# print('r2_score : ',R2S)

# #RandomForest
MSE_list=[]
N_estimators=[]
for i in range (50,150):

    regressor = RandomForestRegressor(n_estimators=i)
    print(regressor.get_params(True))
    regressor = regressor.fit(x_train, y_train)
    y_predict = regressor.predict(x_test)

    MSE=mean_squared_error(y_test,y_predict)
    MSE_list.append(MSE*100)
    N_estimators.append(i)

    # R2S=r2_score(y_test,y_predict)
plt.plot(N_estimators,MSE_list)
plt.xlabel('Number of trees (n_estimators)')
plt.ylabel('Mean Squared Error (%)')
plt.show()

# print('RandomForest')
#
# print('mean_squared_error : ',MSE)
# print('explained_variance_score : ',EVS)
# print('r2_score : ',R2S)
#
# plt.scatter(y_test, y_predict)
#
# plt.xlabel('Real Occupancy')
# plt.ylabel('Estimated Occupancy')
# plt.show()



#data.label.plot(figsize=(18,5))
# plt.scatter(y_test, y_pred)
# plt.show()

# data.to_excel("output.xlsx")

#
# #SVM Regression
# svm_reg = LinearSVR(epsilon=0.5)
# svm_reg.fit(x_train,y_train)
# y_pred2 = regressor.predict(x_test)
#
#
# mse = mean_squared_error(y_test, clf.predict(x_test))
# r2 = r2_score(y_test, clf.predict(x_test))
#
# print("MSE: %.4f" % mse)
# print("R2: %.4f" % r2)
#
# #plt.scatter(y_test, y_pred2)
# #plt.show()
#
#
#
#
# print('Mean Squared Error:', (metrics.mean_squared_error(y_test, y_pred)))

#data_estimated = data[34544:39432]

# data_estimated = pd.concat([data[0:34544],data[39432:]])
#
#
# x_estimated=data_estimated[['Toffice_reference', 'humidity', 'detected_motions', 'power',
#        'office_CO2_concentration', 'door']]
# data_estimated['label'] = regressor.predict(x_estimated)
# data['label']= pd.concat([data_estimated['label'][0:34544],data['label'][34544:39432],data_estimated['label'][34544:]])

print ("time=",time.time()-tStart )

