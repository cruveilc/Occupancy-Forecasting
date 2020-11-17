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

data = pd.read_excel('output.xlsx',
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
#lab = data['label']
colist=['Tin', 'Tout', 'humidity', 'detected_motions', 'power',
      'office_CO2_concentration', 'door', 'CO2_corridor',
       'acoustic_pressure']

shiftted=data-data.shift()
shiftted.dropna(inplace=True)


def test_stationary(timeseries):
    from statsmodels.tsa.stattools import adfuller as adf
    # determining rolling statistic
    movingaverage = timeseries.rolling(window=24).mean()
    movingstd = timeseries.rolling(window=24).std()
    movingaverage.dropna(inplace=True)
    # plt
    plt.figure(figsize=(15, 11))
    orig = plt.plot(timeseries, color='blue', label='Orginal')
    movav = plt.plot(movingaverage, color='red', label='Moving Average')
    movstd = plt.plot(movingstd, color='black', label='Moving Std')
    plt.title('Rolling statistic plot')
    plt.show()

    print('Results of Dickey-Fuller:')
    stat = adf(timeseries['label'], autolag='AIC')
    serie = pd.Series(stat[0:4], index=['Test statistic', 'p_value', 'Lags used', 'Number of observations'])
    for key, value in stat[4].items():
        serie['Critical value( %s)' % key] = value
    print(serie)
    return timeseries - movingaverage

test_stationary(shiftted)


# Data decompostion
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(shiftted['label'],period=30)
trend = decomposition.trend
seasonal = decomposition.seasonal
resid = decomposition.resid
resid.dropna(inplace=True)
plt.figure(figsize=(13, 11))
plt.subplot(411)
plt.plot(data[0:2500], label='Orginal')
plt.legend(loc=('best'))
plt.subplot(412)
plt.plot(trend, color='red', label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, color='black', label
='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(resid, color='green', label='Residual')
plt.legend('Residual')
plt.legend(loc='best')
#plt.show()


#x_train, x_test, y_train, y_test = train_test_split(x, lab, test_size = 0.3)

from numpy import array


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence

n_steps = 12*12
# split into samples
X, y = split_sequence(data.label, n_steps)

# summarize the data
#for i in range(len(X)):
     #print(X[i], y[i])
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))


# In[46]:


from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, TimeDistributed, RepeatVector
from keras.models import Sequential
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from sklearn.metrics import r2_score

#
#

model = Sequential()
model.add(LSTM(15, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()



X.shape

trainx = X[0:40060]
trainy = y[0:40060]
testx = X[40060:]
testy = y[40060:]

network = model.fit(trainx, trainy, epochs=(10))

yhat = model.predict(testx, verbose=0)
plt.figure(figsize=(11, 9))
plt.plot(testy, label='Orginal')
plt.plot(yhat, label='Predicted', color='Orange')
plt.legend(loc='best')
plt.title('Test results with LSTM, R2_score:%f' % r2_score(testy, yhat))
print(r2_score(testy, yhat))
plt.show()
