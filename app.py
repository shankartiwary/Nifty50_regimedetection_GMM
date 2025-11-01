import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler

# --- App Title and Description ---
st.title('Nifty 50 Market Regime Detection')
st.write("""
This app uses a Hidden Markov Model (HMM) to identify different market regimes
for the Nifty 50 index. The data is fetched dynamically from Yahoo Finance.
""")

# --- Data Fetching and Model Training (with Caching) ---
@st.cache_data
def get_data_and_model():
    """
    Fetches Nifty 50 data from Yahoo Finance, processes it,
    and fits an HMM model to identify market regimes.
    The data is fetched from 2015 to the current date.
    """
    start_date = "2015-01-01"
    end_date = datetime.date.today()
    symbol = "^NSEI"  # Nifty 50 symbol

    # Download data
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("Could not download data. Please check the ticker symbol or date range.")
        return None, None

    # Feature Engineering
    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Returns"] = df["Close"].pct_change()
    df["Range"] = (df["High"] / df["Low"]) - 1
    df['Volatility'] = df['Returns'].rolling(window=30).std()
    df.dropna(inplace=True)

    # Prepare data for HMM
    X_train = df[["Returns", "Range", "Volatility"]]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Fit HMM model
    model = GaussianHMM(n_components=4, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_scaled)

    # Predict hidden states (regimes)
    hidden_states = model.predict(X_scaled)
    df['Regime'] = hidden_states

    return df, model

# --- Main App Logic ---
df, model = get_data_and_model()

if df is not None:
    # --- Display Market Regime Chart ---
    st.subheader('Nifty 50 Price with Market Regimes')

    fig, ax = plt.subplots(figsize=(18, 10))

    prices = df['Close']

    # Create arrays for each regime's plot
    for i in range(model.n_components):
        state_prices = np.full(prices.shape, np.nan)
        state_prices[df['Regime'] == i] = prices[df['Regime'] == i]
        ax.plot(df.index, state_prices, label=f'Regime {i}')

    ax.set_title('Nifty 50 Market Regimes')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # --- Display Raw Data ---
    st.subheader('Raw Data (2015-Present)')
    st.dataframe(df.tail(100)) # Display the last 100 rows
else:
    st.write("Please try again later.")
