import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fetch_data(ticker_symbol, period="5y"):
    """Fetches historical stock data from Yahoo Finance."""
    print(f"Attempting to pull data for {ticker_symbol}...")
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(period=period)
    if not df.empty:
        print("✅ Success! Data retrieved.")
    else:
        print(f"❌ Failed to retrieve data for {ticker_symbol}.")
        return None
    return df

def engineer_features(df, future_days=30):
    """Creates features and the target variable for the model."""
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # The target variable we want to predict
    df['Prediction'] = df['Close'].shift(-future_days)
    
    print("--- Features and Target engineered ---")
    return df

def prepare_data(df, future_days=30):
    """Prepares the data for training by creating X and y sets."""
    features = ['Close', 'SMA_20', 'SMA_50', 'Volume_Change']
    X = df[features]
    
    # Remove the last 'future_days' rows for which we don't have a target
    X = X.iloc[:-future_days]
    y = df['Prediction'].iloc[:-future_days]
    
    # Drop any rows with NaN values (from rolling windows)
    X = X.dropna()
    y = y[X.index] # Ensure y aligns with the cleaned X
    
    print("--- Final Cleaned Feature Set (X) and Target (y) prepared ---")
    return X, y

def train_model(X, y):
    """Splits data and trains a RandomForestRegressor model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using RandomForestRegressor as it performed better
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model Root Mean Squared Error: ${rmse:.2f}")
    
    return model

def make_prediction(model, df, future_days=30):
    """Uses the trained model to forecast future prices."""
    features = ['Close', 'SMA_20', 'SMA_50', 'Volume_Change']
    
    # Get the most recent data to forecast from
    X_to_forecast = df[features].iloc[-future_days:]
    
    forecasted_prices = model.predict(X_to_forecast)
    print("--- Forecasted Prices for the next 30 days ---")
    print(forecasted_prices[:5]) # Print the first 5 forecasts
    
    return forecasted_prices

def plot_results(df, forecast, ticker_symbol, future_days=30):
    """Visualizes the historical data and the forecast."""
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 7))
    plt.title(f'{ticker_symbol} Price Prediction')

    # Plot the last year of historical data
    df['Close'].iloc[-252:].plot(label='Historical Close Price')

    # Create a future date range for the forecast
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    # Plot the forecasted data
    plt.plot(future_dates, forecast, label='Forecasted Price', color='orange', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main function to run the stock prediction pipeline."""
    # --- Configuration ---
    ticker_symbol = 'AAPL'
    future_days_to_predict = 30
    
    # --- Pipeline ---
    stock_df = fetch_data(ticker_symbol)
    if stock_df is not None:
        df_featured = engineer_features(stock_df, future_days_to_predict)
        X, y = prepare_data(df_featured, future_days_to_predict)
        trained_model = train_model(X, y)
        forecast = make_prediction(trained_model, df_featured, future_days_to_predict)
        plot_results(df_featured, forecast, ticker_symbol, future_days_to_predict)

# This ensures the main() function is called only when the script is executed directly
if __name__ == "__main__":
    main()