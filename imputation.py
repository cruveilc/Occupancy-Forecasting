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

data = pd.read_excel('values_office.xlsx',sheet_name=0,header=0,index_col=[0],keep_default_na=True)

print(data)

data.replace(-1.0, np.nan, inplace=True)


from missingpy import MissForest
imputer = MissForest()
X_imputed = imputer.fit_transform(data)

data = pd.DataFrame(data=X_imputed, index=data.index, columns=data.columns)
data.label.plot(figsize=(18,5))
plt.show()
