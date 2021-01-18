import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.svm import LinearSVR
from pandas.plotting import register_matplotlib_converters
import time
from sklearn.model_selection import train_test_split
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import datetime
import plotly.graph_objects as go

# register_matplotlib_converters()


tStart = time.time()

dataFrame = pd.read_excel('project.xlsx', header=0, index_col=None)

print(dataFrame.columns)

# date_actuelle=dataFrame[' time'][4556]
# print(date_actuelle)

# data.plot(figsize=(18,5))
# plt.show()

X = dataFrame[
    [' Tin', ' Tout', ' humidity', ' detected_motions', ' power', ' office_CO2_concentration', ' door', ' CO2_corridor',
     ' acoustic_pressure']]
label = dataFrame[' occupancy']

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.33)

date_actuelle = dataFrame['time'][2740]
idate = 2740
print(date_actuelle)

# LinearRegression
# model = sklearn.linear_model.LinearRegression()
# model.fit(x_train, y_train)
#
# print(model.coef_)
#
# predictions = model.predict(x_test)
# plt.scatter(y_test, predictions)
# plt.show()
# np.sqrt(sklearn.metrics.mean_squared_error(y_test, predictions))

# GradientBoostingRegressor

# from sklearn.metrics import mean_squared_error,r2_score
# from sklearn.ensemble import GradientBoostingRegressor
#
# params = {'n_estimators': 500, 'max_depth': 6,
#         'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
# clf = GradientBoostingRegressor(**params).fit(x_train, y_train)
#
# mse = mean_squared_error(y_test, clf.predict(x_test))
# r2 = r2_score(y_test, clf.predict(x_test))
#
# print("MSE: %.4f" % mse)
# print("R2: %.4f" % r2)

# RandomForest

RDF_method = RandomForestRegressor(n_estimators=90, criterion='mse', random_state=0)
RDF_method.fit(X_train, y_train)
y_predict = RDF_method.predict(X_test)
MSE = round(mean_squared_error(y_test, y_predict),4)
EVS = round(explained_variance_score(y_test, y_predict),4)
R2S = round(r2_score(y_test, y_predict),4)

# plt.scatter(y_test, y_pred)
# plt.show()

# data_estimated = data[34544:39432]

print("time=", time.time() - tStart)

# data.label.plot(figsize=(18,5))
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


# Tin=data[' Tin']
# Code Explanation: model = LinearRegression() creates a linear regression model and the for loop divides the dataset into three folds (by shuffling its indices). Inside the loop, we fit the data and then assess its performance by appending its score to a list (scikit-learn returns
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



fig=go.Figure()
fig.add_trace(go.Table(header=dict(values=["MSE", "R2S", "EVS"],line_color='burlywood',fill_color='lightskyblue' ,font_size=20,height=30,align='center'),
             cells=dict(values=[MSE, R2S, EVS],font_size=20,height=30,align='center')
             ))

fig.update_layout(autosize=True)

app = dash.Dash(__name__)

# App layout
app.layout = html.Div([

    html.H1("Energy management dashboard", style={'text-align': 'center', 'color': '#1891cd','font-siz':'30'}),

    #first row
    html.Div(children=[
        #first column of first row
        html.Div(children=[
            dcc.Dropdown(id='select_features',
                 options=[{'label': ' Tin', 'value': ' Tin'},
                          {'label': ' Tout', 'value': ' Tout'},
                          {'label': ' humidity', 'value': ' humidity'},
                          {'label': ' detected_motions', 'value': ' detected_motions'},
                          {'label': ' power', 'value': ' power'},
                          {'label': ' office_CO2_concentration', 'value': ' office_CO2_concentration'},
                          {'label': ' door', 'value': ' door'},
                          {'label': ' CO2_corridor', 'value': ' CO2_corridor'},
                          {'label': ' acoustic_pressure', 'value': ' acoustic_pressure'},
                          {'label': ' occupancy', 'value': ' occupancy'}],
                 value=' Tin',
                 style={'font-size':'23px','height':'5px','width':'212%','z-index':'1','background-color':'#1891cd','font-weight':'bold'},

                 )

        ], style={ 'display': 'inline-block','vertical-align': 'top', 'margin-left': '4vw'}
        ),

        #second column of first row
        html.H2(children=[
            html.Div(id='output_container', children=[],
                     style={'z-index':'2'})

        ], style={'display': 'inline-block','vertical-align': 'top', 'margin-left': '14vw', 'margin-top': '0.3vw','font-size':'25px'})
    ], className='row',
    ),

    #second row
    html.Div(children=[
        #first column of second row
        html.Div(children=[
            dcc.Graph(figure=fig,id='tableau erreurs')

        ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '0vw','margin-top': '-1vw'}),


        #second column of second row
        html.Div(children=[
            dcc.Checklist(
                options=[
                    {'label': 'Cooling', 'value': 'Cool'},
                    {'label': 'Open window', 'value': 'OW'}
                ],
                value=['Cool'],
                inputStyle={'width':'20px','height':'20px','cursor':'pointer','background-color':'#F0FFFF'},
                style={'background-color':'#1E90FF','border-radius': '10px','color':'white'}

            )

        ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '0vw', 'margin-top': '4.7vw','font-size':'30px'})


    ], className='row'),

    #third row
    html.Div(children=[
        #first column of third row
        html.Div(children=[
            dcc.Graph(id='label_plot',
                        figure={})
        ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '0vw', 'margin-top': '-13vw'}),

        #second column of third row
        html.Div(children=[
            dcc.Graph(id='features_plot',
                figure={})

            ], style={ 'display': 'inline-block','vertical-align': 'top', 'margin-left': '0vw', 'margin-top': '-13vw'})
    ], className='row'),
])


# @app.callback(
#     [Output(component_id='slider_container', component_property='style'),
#      Output(component_id='slider_months', component_property='value')],
#     [Input(component_id='select_period',component_property='value')]
# )
#
# def set_sliders(select_period):
#     if select_period=='months':
#         return {'display':'block'},1
#     else:
#         return {'display':'none'},0

@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='features_plot', component_property='figure'),
     Output(component_id='label_plot', component_property='figure')],
    [Input(component_id='select_features', component_property='value')]
)
def update_graph(select_features):

    container = "date : " + str(date_actuelle)

    data_est = dataFrame.copy()

    temps = data_est['time']
    feature = data_est[select_features]

    abscissa = temps[2586:idate]
    ordinate = feature[2586:idate]



    fig_feat = px.line(
        data_frame=data_est,
        x=abscissa,
        y=ordinate,
        labels = {'x': 'time', 'y':select_features}

    )

    fig_lab = px.line(
        data_frame=data_est,
        x=abscissa,
        y=data_est[' occupancy'][2586:idate],
        labels={'x':'time','y':'label'}

    )
    return  container,fig_feat, fig_lab



if __name__ == '__main__':
    app.run_server(debug=True)
