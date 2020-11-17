from sklearn.datasets import load_breast_cancer
X, y = data = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor


data = pd.read_excel('values_office.xlsx',
sheetname=0,
header=0,
index_col=[0],
keep_default_na=True
)
data_training = data[34544:39432]

lab = data_training['label']
x=data_training[['Toffice_reference', 'humidity', 'detected_motions', 'power',
       'office_CO2_concentration', 'door']]

x_train, x_test, y_train, y_test = train_test_split(x, lab, test_size = 0.3)


names=[]
train_scores =[]
test_scores =[]

models={'OLS': LinearRegression(),
       'Ridge': Ridge(),
       'Lasso': Lasso(),
       'ElasticN': ElasticNet(),
       'GBReg': GradientBoostingRegressor()}

for name, model in models.items():
    name_model = model
    name_fit = name_model.fit(x_train, y_train)
    name_pred = name_model.predict(x_test)
    name_train_score = name_model.score(x_train, y_train).round(4)
    name_test_score = name_model.score(x_test, y_test).round(4)
    names.append(name)
    train_scores.append(name_train_score)
    test_scores.append(name_test_score)

print(names)
print(name_train_score)
print(name_test_score)


# score_df = pd.DataFrame(names, train_scores, test_scores)
# score_df






