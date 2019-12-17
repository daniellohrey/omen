import time
import datetime
import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from alpha_vantage.foreignexchange import ForeignExchange

def omen(key, from_symbol, to_symbol, interval=False):
	"""Takes Alpha Vantage API key, currency pair and optional interval, and returns the predicted next value."""

	#set constants
	FIELD = "4. close"

	#fetch data
	api = ForeignExchange(key=key)
	if interval:
		data, _ = api.get_currency_exchange_intraday(from_symbol=from_symbol, to_symbol=to_symbol, outputsize="full", interval=interval)
	else:
		data, _ = api.get_currency_exchange_daily(from_symbol=from_symbol, to_symbol=to_symbol, outputsize="full")

	#extract required data
	time_series = []
	for date in data:
		time_series.append((date, float(data[date][FIELD])))
	time_series.sort(key=lambda tup : tup[0])
	dates = []
	prices = []
	for date, price in time_series:
		dates.append(date)
		prices.append(price)

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
	return prices[-1], future[0]

#main
parser = argparse.ArgumentParser(description = "Predect future values of forex")
parser.add_argument("-k", "--key", default="PLXD9IM2VJWSJAPE", help="API key")
parser.add_argument("-i", "--intraday", action="store_true", help="Set for hourly")
args = parser.parse_args()

assets = [("AUD","JPY"), ("AUD","NZD"), ("AUD","USD"), ("CAD","JPY"), ("CHF","JPY"), ("EUR","AUD"), ("EUR","GBP"), ("EUR","JPY"), ("EUR","USD"), ("GBP","AUD"), ("GBP","JPY"), ("GBP","USD"), ("NZD","JPY"), ("NZD","USD"), ("USD","CAD"), ("USD","CHF"), ("USD","JPY")] 

if args.intraday:
	interval = "60min"
else:
	interval = False

i = 1
for asset in assets:
	#set file name
	file_name = asset[0] + "-" + asset[1] + "_"
	if interval:
		file_name += interval + ".txt"
	else:
		file_name += "daily.txt"

	#do prediction
	last, future = omen(args.key, asset[0], asset[1], interval)

	#set write data
	now = datetime.datetime.now()
	now = now - datetime.timedelta(minutes=now.minute, seconds=now.second, microseconds=now.microsecond)
	to_write = "last : " + str(last) + " next: " + str(future) + " - "
	if last < future:
		to_write += "CALL" + "\n"
	else:
		to_write += "PUT" + "\n"

	#write to file
	with open(file_name, "a") as f:
		f.write(str(now) + " - " + to_write)
			
	#avoid rate limiting
	if i % 5 == 0:
		time.sleep(300) #5 mins, api rate limit
	i += 1
