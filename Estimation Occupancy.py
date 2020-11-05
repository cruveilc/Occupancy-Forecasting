import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# data management and plot

dataFrame = pd.read_excel('project.xlsx',index_col=None,header=0)


print(dataFrame.columns)


time=dataFrame['time']
Tin=dataFrame[' Tin']
Tout=dataFrame[' Tout']
humidity=dataFrame[' humidity']
detected_motions=dataFrame[' detected_motions']
power=dataFrame[' power']
office_CO2_concentration=dataFrame[' office_CO2_concentration']
door=dataFrame[' door']
CO2_corridor=dataFrame[' CO2_corridor']
acoustic_pressure=dataFrame[' acoustic_pressure']
label=dataFrame[' label']

ploting_datas=[Tin,Tout,humidity,detected_motions,power,office_CO2_concentration,door,CO2_corridor,acoustic_pressure]


# Tin: select=0
# Tout: select=1
# humidity: select=2
# detected_motions: select=3
# power: select=4
# office_CO2_concentration: select=5
# door: select=6
# CO2_corridor: select=7
# acoustic_pressure: select=8


select=None
label_select=True
if select != None:
    data = ploting_datas[select]
    plt.plot(time, data)
    if label_select==True:
        plt.plot(time, label)
    plt.show()
    plt.close()


#Datas selection for training and testing

X=dataFrame[[' Tin', ' Tout', ' humidity', ' detected_motions', ' power',' office_CO2_concentration', ' door',
             ' CO2_corridor',' acoustic_pressure']]

X_train, X_test, y_train, y_test = train_test_split(X, label,test_size=0.33)



#SVM method

# SVM_method = svm.SVR(kernel='rbf',epsilon=0.1,gamma='scale',C=100)
# SVM_method.fit(X_train,y_train)
# y_predict=SVM_method.predict(X_test)
#
# print('explained_variance_score : ', explained_variance_score(y_test,y_predict))
# print('mean_squared_error : ',mean_squared_error(y_test,y_predict))
# print('r2_score : ',r2_score(y_test,y_predict))
#
# plt.scatter(y_predict,y_test)
#
# plt.show()

#RandomForest method

# RDF_method= RandomForestRegressor(n_estimators=90,criterion='mse',random_state=0)
# RDF_method.fit(X_train,y_train)
# y_predict=RDF_method.predict(X_test)
# MSE=mean_squared_error(y_test,y_predict)
# EVS=explained_variance_score(y_test,y_predict)
# R2S=r2_score(y_test,y_predict)
#
# print('mean_squared_error : ',MSE)
# print('explained_variance_score : ',EVS)
# print('r2_score : ',R2S)
#
# plt.scatter(y_test,y_predict)
# plt.show()

#Neural network method

# MLP=MLPClassifier(hidden_layer_sizes=(500,),activation='logistic',solver='lbfgs',alpha=0.0001,random_state=1,max_iter=1000)
# y_train=y_train.astype("int")
#
# MLP.fit(X_train, y_train)
# y_predict=MLP.predict(X_test)
# MSE=mean_squared_error(y_test,y_predict)
# EVS=explained_variance_score(y_test,y_predict)
# R2S=r2_score(y_test,y_predict)
#
# print('mean_squared_error : ',MSE)
# print('explained_variance_score : ',EVS)
# print('r2_score : ',R2S)

#LSTM method

LSTM_method=Sequential()
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)

print(X_test,y_test)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


LSTM_method.add(LSTM(units=128,input_shape=(X_train.shape[1],X_train.shape[2])))
LSTM_method.add(Dropout(rate=0.1))
LSTM_method.add(Dense(units=1))

LSTM_method.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')

LSTM_method.fit(X_train,y_train, epochs = 1000, batch_size = 30)
y_predict=LSTM_method.predict(X_test)

MSE=mean_squared_error(y_test,y_predict)

print('mean_squared_error : ',MSE)



plt.scatter(y_test,y_predict)
plt.show()