import yfinance as yf
import pandas as pd
import streamlit as st
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Indian Stock Recommender", layout="centered")

st.title("📈 Indian Stock Buy/Sell/Hold Recommender (Free & AI-based)")
st.markdown("Enter an Indian stock symbol like `TCS.NS`, `RELIANCE.NS`, `INFY.NS`")

# Input
symbol = st.text_input("Stock Symbol", value="TCS.NS")

if st.button("Get Recommendation"):
    with st.spinner("Fetching data and predicting..."):

        # Step 1: Get historical data
        df = yf.download(symbol, period="6mo", interval="1d")

        if df.empty:
            st.error("⚠️ Could not fetch data. Try another symbol like `INFY.NS`.")
        else:
            # Step 2: Add indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['macd'] = ta.trend.MACD(df['Close']).macd()
            df['ema_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
            df = df.dropna()

            # Step 3: Label data for training
            df['pct_change'] = df['Close'].pct_change().shift(-1)
            df['label'] = 'Hold'
            df.loc[df['pct_change'] > 0.02, 'label'] = 'Buy'
            df.loc[df['pct_change'] < -0.02, 'label'] = 'Sell'

            # Step 4: Prepare for model
            features = ['rsi', 'macd', 'ema_20']
            X = df[features]
            y = df['label']

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

            # Step 5: Train the model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Step 6: Predict for the latest data
            latest = X.tail(1)
            prediction_encoded = model.predict(latest)[0]
            prediction = label_encoder.inverse_transform([prediction_encoded])[0]

            # Output
            st.success(f"🟢 AI Recommendation for **{symbol}**: **{prediction}**")
            st.caption("Model based on past 6 months of price data and technical indicators.")
