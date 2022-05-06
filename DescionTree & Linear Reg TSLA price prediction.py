import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas_datareader.data as web
import datetime as dt
from datetime import datetime
import plotly.express as px

df = pd.read_csv('TSLA.csv')

df1 = df.copy()
df1 = df1.loc[2394:]
df1.reset_index(inplace = True)
df1.drop('index',axis=1,inplace= True)

plt.figure(figsize=(16,6))
plt.title = 'Tesla'
plt.xlabel('Days')
plt.ylabel('Close Price USD($)')
plt.plot(df1['Adj Close'])

df1 = df1[['Adj Close']]

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

future_days = 30
df1['Prediction'] = df1[['Adj Close']].shift(-future_days)

X = np.array(df1.drop('Prediction',axis =1))[:-future_days]
y = np.array(df1['Prediction'])[:-future_days]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Creating Models
# Descison Tree Regressor Model
tree = DecisionTreeRegressor().fit(X_train.round(),y_train.round())
# Linear regression Model
lr = LinearRegression().fit(X_train.round(),y_train.round())

X_future = df1.drop('Prediction',axis = 1)[:-future_days]
X_future = X_future.tail(future_days)
X_future = np.array(X_future)

# Show Tree Model
tree_pred = tree.predict(X_future)

# Show Linear Model
lr_pred = lr.predict(X_future)

# visualising descision model
valid = df1[X.shape[0]:]
valid['Predictions'] = tree_pred

plt.figure(figsize=(16,6))
# plt.title('Descision Tree Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD($)')
plt.plot(df1['Adj Close'])
plt.plot(valid[['Adj Close','Predictions']])
plt.legend(['Orig','Val','Pred'])

# visualising linear regression model
valid = df1[X.shape[0]:]
valid['Predictions'] = lr_pred

plt.figure(figsize=(16,6))
# plt.title('Linear Regression Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD($)')
plt.plot(df1['Adj Close'])
plt.plot(valid[['Adj Close','Predictions']])
plt.legend(['Orig','Val','Pred'])

# Conclusion: Descison Tree Regressor Model works better than Linear regression Model