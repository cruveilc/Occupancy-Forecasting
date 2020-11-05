#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
import csv
import math
import os
import pyramid
import random
import seaborn as sns
import statsmodels.tsa.stattools as ts
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose


# In[38]:


data = pd.read_excel('french_temp.xlsx',sheet_name=0,header=0,index_col=[0],keep_default_na=True)
print(data)

shiftted=data-data.shift()
shiftted.dropna(inplace=True)


# In[40]:


def test_stationary(timeseries):
    from statsmodels.tsa.stattools import adfuller as adf
    #determining rolling statistic
    movingaverage=timeseries.rolling(window=24).mean()
    movingstd=timeseries.rolling(window=24).std()
    movingaverage.dropna(inplace=True)
    #plt
    plt.figure(figsize=(15,11))
    orig=plt.plot(timeseries,color='blue',label='Orginal')
    movav=plt.plot(movingaverage,color='red',label='Moving Average')
    movstd=plt.plot(movingstd,color='black',label='Moving Std')
    plt.title('Rolling statistic plot')
    plt.show(block=False)
    
    print('Results of Dickey-Fuller:')
    stat=adf(timeseries['load'],autolag='AIC')
    serie=pd.Series(stat[0:4],index=['Test statistic','p_value','Lags used','Number of observations'])
    for key,value in stat[4].items():
        serie['Critical value( %s)'%key]=value
    print(serie)
    return timeseries-movingaverage


# In[41]:


test_stationary(shiftted)
plt.show()


# In[42]:


# Data decompostion
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(shiftted)
trend=decomposition.trend
seasonal=decomposition.seasonal
resid=decomposition.resid
resid.dropna(inplace=True)
plt.figure(figsize=(13,11))
plt.subplot(411)
plt.plot(data[0:2500],label='Orginal')
plt.legend(loc=('best'))
plt.subplot(412)
plt.plot(trend,color='red',label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,color='black',label
         ='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(resid,color='green',label='Residual')
plt.legend('Residual')
plt.legend(loc='best')
plt.show()


# In[43]:


shiftted


# In[44]:


Arima_model=auto_arima(data, start_p=1, start_q=1, max_p=4, max_q=4, start_P=0, start_Q=0, max_P=3, max_Q=3, m=24, seasonal=True, trace=True, error_action='warn', suppress_warnings=True, random_state = 20, n_fits=20)


# In[45]:


from numpy import array
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
 
# define input sequence

n_steps = 24
# split into samples
X, y = split_sequence(data.load, n_steps)
# summarize the data
for i in range(len(X)):
    print(X[i], y[i])
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))


# In[46]:


from keras.layers import LSTM,Dense,Conv1D,MaxPooling1D,Flatten,TimeDistributed,RepeatVector
from keras.models import Sequential
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from sklearn.metrics import r2_score


# In[47]:


model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
model.summary()


# In[48]:


X.shape


# In[49]:


trainx=X[0:5760]
trainy=y[0:5760]
testx=X[5760:]
testy=y[5760:]
network=model.fit(trainx,trainy,epochs=150)


# In[50]:


yhat = model.predict(testx, verbose=0)
plt.figure(figsize=(11,9))
plt.plot(testy,label='Orginal')
plt.plot(yhat,label='Predicted',color='Orange')
plt.legend(loc='best')
plt.title('Test results with LSTM, R2_score:%f'%r2_score(testy,yhat))
print(r2_score(testy,yhat))


# In[51]:


model=Sequential()
model.add(Conv1D(filters=64,kernel_size=3,activation='relu',input_shape=(24,1)))
model.add(Conv1D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(1))
model.add(LSTM(200,activation='relu',return_sequences=True))
model.add(TimeDistributed(Dense(100,activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer=Adam(),loss=mean_squared_error)
model.summary()


# In[52]:


trainx=X[0:5760]
trainy=y[0:5760]
testx=X[5760:]
testy=y[5760:]
trainy=trainy.reshape(trainy.shape[0],1,1)
model.fit(trainx,trainy,epochs=50)


# In[53]:


ypred = model.predict(testx, verbose=0)
ypred=ypred.reshape((testy.shape[0]))
plt.figure(figsize=(11,9))
plt.plot(testy,label='Orginal')
plt.plot(ypred,label='Predicted',color='Orange')
plt.legend(loc='best')
plt.title('Test results with CNN-LSTM, R2_score:%f'%r2_score(testy,ypred))
print(r2_score(testy,ypred))


# In[54]:


from statsmodels.tsa.stattools import acf,pacf 
ac=acf(data['load'],nlags=100,unbiased=True)
pac=pacf(data['load'],nlags=100)
plt.figure(figsize=(11,9))
plt.title('ACF plot')
plt.grid()
plt.plot(ac)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.figure(figsize=(11,9))
plt.title('PACF plot')
plt.grid()
plt.plot(pac)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.show()


# In[55]:


train=data[:5760]
test=data[5760:]
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(train,order=(1,1,10),seasonal_order=(2,0,2,24),enforce_stationarity=True,enforce_invertibility=False)
fitted=model.fit()
print(fitted.summary())


# In[56]:


forecast=round(fitted.forecast(steps=3024),2)
forecast=pd.DataFrame(forecast.values,index=test.index)
print('r2_score is %4f'%r2_score(test,forecast))
plt.figure(figsize=(11,9))
#plt.title('MSE is %4f'%mean_squared_error(test,forecast))
plt.plot(test,color='blue',label='Orginal')
plt.plot(forecast,color='orange',label='Forecast')
plt.legend(loc='best')
plt.grid('Bold')
plt.show()


# In[21]:


autocorrs = pd.Series(ac)
list_of_regressors = autocorrs.loc[autocorrs > 0.6].index
list_of_regressors = list_of_regressors[1:11]
list_of_regressors


# In[22]:


def create_regressor_attributes(df, attribute, list_of_prev_t_instants) :
    
    """
    Ensure that the index is of datetime type
    Creates features with previous time instant values
    """
        
    list_of_prev_t_instants.sort_values
    start = list_of_prev_t_instants[-1] 
    end = len(df)
    df['datetime'] = df.index
    df.reset_index(drop=True)

    df_copy = df[start:end]
    df_copy.reset_index(inplace=True, drop=True)

    for attribute in attribute :
            foobar = pd.DataFrame()

            for prev_t in list_of_prev_t_instants :
                new_col = pd.DataFrame(df[attribute].iloc[(start - prev_t) : (end - prev_t)])
                new_col.reset_index(drop=True, inplace=True)
                new_col.rename(columns={attribute : '{}_(t-{})'.format(attribute, prev_t)}, inplace=True)
                foobar = pd.concat([foobar, new_col], sort=False, axis=1)

            df_copy = pd.concat([df_copy, foobar], sort=False, axis=1)
            
    df_copy.set_index(['datetime'], drop=True, inplace=True)
    
    return df_copy


# In[23]:


cons = data.loc[:, [ 'load']]
df_consum = create_regressor_attributes(cons, ['load'], list_of_regressors)
df_consum['temp']=data.temp[24:]


# In[24]:


df_consum


# In[25]:


def train_test_valid_split_plus_scaling(df, valid_set_size, test_set_size):
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    
    df_copy = df.reset_index(drop=True)
    
    df_test = df_copy.iloc[ int(np.floor(len(df_copy)*(1-test_set_size))) : ]
    df_train_plus_valid = df_copy.iloc[ : int(np.floor(len(df_copy)*(1-test_set_size))) ]

    df_train = df_train_plus_valid.iloc[ : int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) ]
    df_valid = df_train_plus_valid.iloc[ int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) : ]


    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]
    X_valid, y_valid = df_valid.iloc[:, 1:], df_valid.iloc[:, 0]
    X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]
    
    global Target_scaler
    
    Target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    Feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    
    X_train_scaled = Feature_scaler.fit_transform(np.array(X_train))
    X_valid_scaled = Feature_scaler.fit_transform(np.array(X_valid))
    X_test_scaled = Feature_scaler.fit_transform(np.array(X_test))
    
    y_train_scaled = Target_scaler.fit_transform(np.array(y_train).reshape(-1,1))
    y_valid_scaled = Target_scaler.fit_transform(np.array(y_valid).reshape(-1,1))
    y_test_scaled = Target_scaler.fit_transform(np.array(y_test).reshape(-1,1))
    
    print('Shape of training inputs, training target:', X_train_scaled.shape, y_train_scaled.shape)
    print('Shape of validation inputs, validation target:', X_valid_scaled.shape, y_valid_scaled.shape)
    print('Shape of test inputs, test targets:', X_test_scaled.shape, y_test_scaled.shape)

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled


# In[26]:


valid_set_size = 0.3
test_set_size = 0.3
X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_valid_split_plus_scaling(df_consum, 
                                                                                         valid_set_size, 
                                                                                        test_set_size)


# In[27]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
X_train=scaler.fit_transform(X_train)
X_valid=scaler.fit_transform(X_valid)
X_test=scaler.fit_transform(X_test)
y_train=scaler.fit_transform(y_train)
y_valid=scaler.fit_transform(y_valid)
y_test=scaler.fit_transform(y_test)


# In[ ]:


# Regression with temprature
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
yhat=reg.predict(X_test)
plt.figure(figsize=(9,7))
plt.scatter(y_test,yhat)
plt.xlabel('y_test')
plt.ylabel('y_predict')
plt.title('Predict')
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print('MSE is %4f'%mean_squared_error(y_test,yhat))
print('MAE is %4f'%mean_absolute_error(y_test,yhat))
print('R_score is %4f'%r2_score(y_test,yhat))


# In[28]:


print('r2_score for Linear regression is %4f'%r2_score(y_test,yhat))
print('MSE for Linear Regression is %4f'%mean_squared_error(y_test,yhat))
plt.figure(figsize=(11,9))
plt.plot(y_test,label='Actual')
plt.plot(yhat,color='orange',label='Predicted')
plt.grid('Bold')
plt.legend(loc='best')
plt.title('results from Linear Regression')
plt.show()


# In[29]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
yhat=reg.predict(X_test)
plt.figure(figsize=(9,7))
plt.scatter(y_test,yhat)
plt.xlabel('y_test')
plt.ylabel('y_predict')
plt.title('Predict')
plt.show()


# In[30]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print('MSE is %4f'%mean_squared_error(y_test,yhat))
print('MAE is %4f'%mean_absolute_error(y_test,yhat))
print('R_score is %4f'%r2_score(y_test,yhat))


# In[31]:


print('r2_score for Linear regression is %4f'%r2_score(y_test,yhat))
print('MSE for Linear Regression is %4f'%mean_squared_error(y_test,yhat))
plt.figure(figsize=(11,9))
plt.plot(y_test,label='Actual')
plt.plot(yhat,color='orange',label='Predicted')
plt.grid('Bold')
plt.legend(loc='best')
plt.title('results from Linear Regression')
plt.show()


# In[32]:


from keras.models import Model,Sequential
from keras.layers import Flatten,Dense, Conv1D, LSTM, MaxPool1D,Input,concatenate,Average,Dropout
from keras.optimizers import RMSprop,Adam
from keras.losses import mean_squared_error


# In[33]:


input_=Input(shape=(4,),dtype='float32')
dense1=Dense(27,activation='linear')(input_)
dense2=Dense(18,activation='linear')(dense1)
dense3=Dense(18,activation='linear')(dense2)
dropout=Dropout(0.2)(dense2)
final=Dense(1,activation='linear')(dense2)
mymodel=Model(input_,final)
mymodel.summary()


# In[34]:


mymodel.compile(loss=mean_squared_error,optimizer=Adam(),metrics=['accuracy'])


# In[35]:


network=mymodel.fit(X_train,y_train,epochs=50,batch_size=1,validation_data=(X_valid,y_valid))


# In[36]:


ypredict=mymodel.predict(X_test)
print('r2_score for ANN is %4f'%r2_score(y_test,ypredict))
plt.figure(figsize=(11,9))
plt.plot(y_test,label='Actual')
plt.plot(ypredict,color='orange',label='Predicted')
plt.grid('Bold')
plt.legend(loc='best')
plt.title('r2_score from Linear regression is %4f'%r2_score(y_test,ypredict))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




