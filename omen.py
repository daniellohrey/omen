import json
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from alpha_vantage.foreignexchange import ForeignExchange

#set constants
ALPHA = "PLXD9IM2VJWSJAPE"
DATE = "%Y-%m-%d %H:%M:%S"
FIELD = "4. close"
SYMBOL = "AX.A2M"

#get the data
api = ForeignExchange(key=ALPHA)
#currencies are AUD/JPY, AUD/NZD, AUD/USD, etc. come back to it
#intraday is also available
#use full in production
#9pm utc is 8am aedt (daylight savings)
data, _ = api.get_currency_exchange_intraday(from_symbol="AUD", to_symbol="USD", outputsize="compact", interval="60min")
#print(datetime.datetime.now())
#print(data[FIELD])
#print(data.to_string())
#sys.exit()
#print(data)
#formatted_data = {}
#for k in data:
	#print(k)
#sys.exit()
#dates = sorted([datetime.strptime(k,DATE) for k in data])
#for date in dates:
	#print(date + " " + str(data[date.strftime(DATE)][FIELD]) + " " + str(data[date.strftime(DATE)]["1. open"]))
	#price = float(data[date.strftime(DATE)][FIELD])
	#formatted_data[date.timestamp()] = price
#data = formatted_data
#print(data)

for date in data:
	print(date + " " + str(data[date]["1. open"]) + " " + str(data[date]["4. close"]))
sys.exit()



#get required data
prices = []
dates = []
for date in data:
	dates.append(date)
	prices.append(data[date])

#build data structures
df = pd.DataFrame(data = prices, index = dates, columns = ["price"])
df["prediction"] = df[["price"]].shift(-30)
x = np.array(df.drop(["prediction"], 1))
x = preprocessing.scale(x)
x_forecast = x[-30:]
x = x[:-30]
y = np.array(df["prediction"])
y = y[:-30]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#run linear regression
clf = LinearRegression()
clf.fit(x_train, y_train)
future = clf.predict(x_forecast)

#set up data for formatting
prediction = []
for price in future:
	prediction.append(round(price, 3))
now = datetime.date.today()
day = datetime.timedelta(days = 1)
days3 = datetime.timedelta(data = 3)

#build final dictionary
prophesy = {}
for price in prediction:
	if now.weekday() == 4:
		now = now + days3
	else:
		now = now + day
	epoch = datetime.datetime.strptime(str(now), DATE)
	prophesy[epoch.timestamp()] = float(price)

print(prophesy)
