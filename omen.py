import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from alpha_vantage.foreignexchange import ForeignExchange

#set constants
ALPHA = "PLXD9IM2VJWSJAPE"
FIELD = "4. close"

#fetch data
api = ForeignExchange(key=ALPHA)
#use list of all currencies
#intraday is also available
#use full output size
#9pm utc is 8am aedt
data, _ = api.get_currency_exchange_intraday(from_symbol="AUD", to_symbol="USD", outputsize="full", interval="60min")

#extract required data
time_series = []
for date in data:
	time_series.append((date, float(data[date][FIELD])))
#remove data points from when market is closed
time_series.sort(key=lambda tup : tup[0])
print(len(time_series))
last = time_series.pop()
dates = []
prices = []
for date, price in time_series:
	dates.append(date)
	prices.append(price)
	print(str(date) + " " + str(price))
print("goal:")
print(last[1])

#build data structures
df = pd.DataFrame(data = prices, index = dates, columns = ["price"])
df["prediction"] = df[["price"]].shift(-1)
x = np.array(df.drop(["prediction"], 1))
x = preprocessing.scale(x)
x_forecast = x[-1:]
x = x[:-1]
y = np.array(df["prediction"])
y = y[:-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#run linear regression
reg = LinearRegression()
reg.fit(x_train, y_train)
future = reg.predict(x_forecast)
print("prediction:")
print(future[0])
