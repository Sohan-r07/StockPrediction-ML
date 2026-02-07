import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Market data
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

#ML Model selection, i decided to go with Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

#Alpaca API setup
APIK = "APIKEY"
APIS = "APISECRET"

Stock = "NVDA"
Start = "2020-01-01"
End = "2025-12-31"

client = StockHistoricalDataClient(APIK,APIS,)
request_params = StockBarsRequest(
    symbol_or_symbols=Stock,
    timeframe=TimeFrame.Day,
    start=Start,
    end=End,

)

bars = client.get_stock_bars(request_params)

#Clean up the data into a Pandas DataFrame

df = bars.df.reset_index()

df = df[df["symbol"]==Stock]

#Adding technical indicators to help the Model with the Predictions

df["return"] = df["close"].pct_change()
df["Result"] = (df["return"] > 0).astype(int)
df["volatility"] = df["return"].rolling(5).std()

#Moving Averages
df["ma5"] = df["close"].rolling(5).mean()
df["ma10"] = df["close"].rolling(10).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma50"] = df["close"].rolling(50).mean()

df["target"] = df["close"].shift(-1)
df = df.dropna()

features=[
    "close",
    "Result",
    "volume",
    "volatility",
    "ma5",
    "ma10",
    "ma20",
    "ma50",
]

#Splitting Data into Test and train data sets
x = df[features]
y=df["target"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.17,shuffle=False)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    random_state=2
)

model.fit(xtrain,ytrain)

predictions = model.predict(xtest)
mae = mean_absolute_error(ytest, predictions)
rmse = root_mean_squared_error(ytest,predictions)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

#Plotting
plt.figure(figsize=(12,6))
plt.plot(ytest.values,label="Actual Prices")
plt.plot(predictions,label="Model Predictions")
plt.title("Stock Price Prediction using Random Forest Regressor")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


#latest_data = x.iloc[-1:].values
#pred = model.predict(latest_data)[0]
#print(f"Next day Prediction for NVDA: {pred:.2f}")

#This part of the code is was used to verify if the Model was at the least able to prexict the directional movement of the stock in a day i.e +ve or -ve

#correct=0
#incorrect = 0
#print(len(predictions))
#for i in range(0,248,1):
#    if((predictions[i]>0).astype(int)==ytest.iloc[i]):
#        correct+=1
#    else:
#        incorrect+=1


#print("Correct Predictions: ",correct)
#print("Incorrect Predictions: ", incorrect)
#print("Win Percentage: ",((correct/len(ytest))*100.0))
