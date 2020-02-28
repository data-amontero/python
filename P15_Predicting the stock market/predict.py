import pandas as pd
from datetime import datetime
from datetime import timedelta  
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

stocks = pd.read_csv('sphist.csv')
#print(stocks.head())

stocks['Date'] = pd.to_datetime(stocks['Date'])
#print(stocks.dtypes)

##sorting data in ascending order according to date. 
stocks = stocks.sort_values('Date', ascending=True)
#print(stocks.head())

#start date = 1951-01-03
#end date = 2015-12-07

stocks_test = stocks[stocks['Date']>datetime(year=1951, month=1, day=2)]
#print(stocks_test.head())

stocks['day_5'] = 0
stocks['day_30'] = 0
stocks['day_365'] = 0

stocks['day_5'] = stocks['Close'].rolling(window=5).mean()
stocks['day_5'] = stocks['Close'].rolling(5).mean()
stocks['day_5'] = stocks['day_5'].shift(1)
stocks['day_30'] = stocks['Close'].rolling(30).mean()
stocks['day_30'] = stocks['day_30'].shift(1)
stocks['day_365'] = stocks['Close'].rolling(365).mean()
stocks['day_365'] = stocks['day_365'].shift(1)
stocks = stocks.dropna(axis = 0)

stocks = stocks[stocks['Date']>datetime(year=1951, month=1, day=2)]

#train and test
train = stocks[stocks['Date']<datetime(year=2013, month=1, day=1)]
test = stocks[stocks['Date']>=datetime(year=2013, month=1, day=1)]

lm = LinearRegression()
features = ['day_5', 'day_30', 'day_365']
lm.fit(train[features], train['Close'])
predictions = lm.predict(test[features])
rmse = mean_squared_error(predictions, test['Close'])**0.5

print(rmse)
###worst project eve