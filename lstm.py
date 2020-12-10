

import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense,LSTM,GRU

data = pd.read_excel('output.xlsx',
sheet_name=0,
header=0,
index_col=[0],
keep_default_na=True
)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('label%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('label%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('label%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

data2=data[['label']]

#print(data.iloc[:,0])
labelEncoder = LabelEncoder()
#data.iloc[:,0] = labelEncoder.fit_transform(data.iloc[:,0])
values = data2.values
print(values.shape)
values = values.astype('float32')


scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled,4*4*24,4*24)

values1 = reframed.values
n_train = 50000
train = values1[:n_train]
test = values1[n_train:]
trainX,trainY = train[:,:-24*4],train[:,4*4*24:]
testX,testY = test[:,:-24*4],test[:,4*4*24:]

trainX = trainX.reshape(trainX.shape[0],1,trainX.shape[1])
testX = testX.reshape(testX.shape[0],1,testX.shape[1])

stop_noimprovement = EarlyStopping(patience=10)
model = Sequential()
model.add(LSTM(50,input_shape=(trainX.shape[1],trainX.shape[2]),dropout=0.2))
model.add(Dense(4*24))
model.compile(loss="mae",optimizer="adam")

history= model.fit(trainX,trainY,validation_data=(testX,testY),epochs=30,verbose=2,callbacks=[stop_noimprovement],shuffle=False)

from sklearn.metrics import r2_score
ypred = model.predict(testX)
#ypred=ypred.reshape((testY.shape[0]))
plt.figure(figsize=(11,9))
plt.plot(testY,label='Orginal')
plt.plot(ypred,label='Predicted',color='Orange')
#plt.legend(loc='best')
plt.title('Test results for 24 hours prediction, R2_score:%f'%r2_score(testY,ypred))
print(r2_score(testY,ypred))