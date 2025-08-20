# src/run_prediction.py

#%%
# STEP 1: IMPORTS & SETUP
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Define the ticker and forecast period
ticker_symbol = 'AAPL'
future_days = 30

#%%
# STEP 2: DATA RETRIEVAL
print(f"Attempting to pull data for {ticker_symbol}...")
ticker_data = yf.Ticker(ticker_symbol)
df = ticker_data.history(period='5y')

if not df.empty:
    print("✅ Success! Data retrieved.")
    print(df.tail())
else:
    print("❌ Failed to retrieve data.")

#%%
# STEP 3: FEATURE ENGINEERING
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['Volume_Change'] = df['Volume'].pct_change()
df['Prediction'] = df['Close'].shift(-future_days)

print("\n--- Data with Features and Target ---")
print(df.tail(5))

#%%
# STEP 4: DATA PREPARATION FOR MODEL
X = df[['Close', 'SMA_20', 'SMA_50', 'Volume_Change']]
X = X.iloc[:-future_days]
y = df['Prediction'].iloc[:-future_days]

X = X.dropna()
y = y[X.index] 

print("\n--- Final Cleaned Feature Set (X) ---")
print(X.tail())

#%%
# STEP 5: MODEL TRAINING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LinearRegression()
model = RandomForestRegressor(n_estimators=100, random_state=42) # <-- Add this

model.fit(X_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
print(f"\nModel Root Mean Squared Error: ${rmse:.2f}")

#%%
# STEP 6: MAKE A FUTURE PREDICTION
X_to_forecast = df[['Close', 'SMA_20', 'SMA_50', 'Volume_Change']].iloc[-future_days:]
forecasted_prices = model.predict(X_to_forecast)

print("\n--- Forecasted Prices for the next 30 days ---")
# Print the first 5 forecasted prices
print(forecasted_prices[:5])

#%%
# STEP 7: VISUALIZE THE FORECAST
plt.style.use('dark_background')
plt.figure(figsize=(14, 7))
plt.title(f'{ticker_symbol} Price Prediction')

# Plot the last year of historical data
# The .plot() from pandas can handle the legend argument, but let's be consistent
df['Close'].iloc[-252:].plot(label='Historical Close Price')

# Create a future date range for the forecast
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

# Plot the forecasted data
# The error was here: plt.plot() does not take a 'legend' argument
plt.plot(future_dates, forecasted_prices, label='Forecasted Price', color='orange', linestyle='--')

# This single command draws the legend for all labeled plots above
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend() # This is the correct way to show the legend
plt.grid(True, alpha=0.3)
plt.show()