import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import datetime

# 1. APP TITLE & SIDEBAR 
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 End-to-End Stock Price Prediction App")

# Sidebar for user input
st.sidebar.header("User Input")
symbol = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, MSFT, GOOGL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

if st.sidebar.button("Train Model & Predict"):
    
    #2. DATA LOADING 
    st.write(f"### Fetching Data for {symbol}...")
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            st.error("No data found. Please check the ticker symbol.")
        else:
            # Show raw data preview
            with st.expander("View Raw Data"):
                st.dataframe(data.tail())

            # --- 3. FEATURE ENGINEERING (Your Code) ---
            # We copy your exact logic here
            data['Prev_Close'] = data['Close'].shift(1)
            data['MA_5'] = data['Close'].rolling(5).mean()
            data['MA_10'] = data['Close'].rolling(10).mean()
            data.dropna(inplace=True)

            X = data[['Prev_Close','MA_5','MA_10']]
            y = data['Close']

            # --- 4. MODEL TRAINING (Your Code) ---
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))

            # --- 5. RESULTS DISPLAY ---
            col1, col2 = st.columns(2)
            col1.metric("Model RMSE (Error)", f"{rmse:.2f}")
            col1.success(f"Model Trained Successfully on {len(X_train)} data points!")

            # --- 6. VISUALIZATION (Your Matplotlib Code adapted for Streamlit) ---
            st.subheader("Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_test.values, label="Actual Price", color='blue')
            ax.plot(predictions, label="Predicted Price", color='red', linestyle='--')
            ax.legend()
            ax.set_title(f"{symbol} Price Prediction")
            st.pyplot(fig)
            
            # --- 7. BONUS: PREDICT TOMORROW (The "End-to-End" magic) ---
            # We take the very last data point to predict the next unknown day
            last_row = data.iloc[[-1]][['Prev_Close', 'MA_5', 'MA_10']]
            next_pred = model.predict(last_row)
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔮 Next Day Prediction")
            st.sidebar.write(f"Predicted Price: **${next_pred[0]:.2f}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("👈 Enter a stock ticker in the sidebar and click 'Train Model' to start.")