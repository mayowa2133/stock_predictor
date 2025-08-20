# app.py

import streamlit as st
import pandas as pd

# Import the functions from your original script
from src.run_prediction import (
    fetch_data, 
    engineer_features, 
    prepare_data, 
    train_model, 
    make_prediction, 
    plot_results
)

# --- Streamlit App Interface ---

# Set up the title and a brief description
st.title("Stock Price Predictor")
st.write("""
This app predicts the 30-day future price of a stock using a machine learning model.
Enter a stock ticker symbol to get started.
""")

# Create a text input widget for the ticker symbol
ticker_symbol = st.text_input("Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL").upper()

# Create a button that will trigger the prediction
if st.button("Get Forecast"):
    if not ticker_symbol:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching data and making predictions for {ticker_symbol}..."):
            # --- Run the Prediction Pipeline ---
            # We call the functions we imported, one by one
            
            stock_df = fetch_data(ticker_symbol)
            
            if stock_df is not None:
                df_featured = engineer_features(stock_df)
                X, y = prepare_data(df_featured)
                trained_model = train_model(X, y)
                forecast = make_prediction(trained_model, df_featured)
                
                # --- Display the Results ---
                st.success(f"Forecast for {ticker_symbol} generated successfully!")
                
                # Display the last few rows of the raw data
                st.subheader("Latest Historical Data")
                st.dataframe(stock_df.tail())
                
                # Display the forecast plot
                st.subheader("Prediction Plot")
                # We call our modified plot_results function which returns a figure
                fig = plot_results(df_featured, forecast, ticker_symbol)
                st.pyplot(fig) # Streamlit's function to display a matplotlib plot
            else:
                st.error(f"Could not retrieve data for {ticker_symbol}. Please check the symbol.")