# Occupancy-Forecasting
In this university project for energy systems students who wanted a first insight into the "Smart Building" notion we focused on a small office with different sensors that enabled to get a good estimation of occupancy (see Manar Amayri, Stéphane Ploix, Sanghamitra Bandyopadhyay. Estimating Occupancy in an Office Setting. )
In this small project the intent was to recreate a simple estimation model based on recently made literature and to focus on diferent techniques of forecasting to be able to forecast 24 hours ahead in the building.
It is an initiative project to machine learning techniques so most of the models are simple except for the LightGBM model that was inspired by miniaxixi https://github.com/minaxixi/Kaggle-M5-Forecasting-Accuracy/blob/master/model_recursive.ipynb
Tested models were MLP, CNN-LSTM, LSTM with different features (date, calendar, feature engineering...etc...) for different forecasting horizons but also different time sample.
The models are basics and is a good way for a beginner to have an insight into today's great tools to simple make forecasting and any comment or advice is welcome.

This forecasting project is accompanied by an example of energy savings that could be done with good forecasting results. It is a simplified predictive controller coded in python with an example of Kivy Application that could be implmented to make an interactive and low cost dashboard for inhabitants by using a Raspberry Pi (compatible with Kivy).

The Github is organized as following:
1. Dataset
2. Estimation model
3. Forecasting model
4. Energy savings
