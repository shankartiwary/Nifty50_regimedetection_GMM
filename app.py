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
This app uses a Hidden Markov Model (HMM) to identify and describe different market
regimes for the Nifty 50 index. The data is fetched dynamically from Yahoo Finance.
""")

# --- Data Fetching and Model Training (with Caching) ---
@st.cache_data
def get_data_and_model():
    """
    Fetches Nifty 50 data, processes it, fits an HMM model, analyzes the regimes,
    and generates descriptive labels for each regime.
    """
    start_date = "2015-01-01"
    end_date = datetime.date.today()
    symbol = "^NSEI"

    # Download data
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("Could not download data.")
        return None, None, None, None

    # Feature Engineering
    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Returns"] = df["Close"].pct_change()
    df["Range"] = (df["High"] / df["Low"]) - 1
    df['Volatility'] = df['Returns'].rolling(window=15).std() # Shortened window
    df.dropna(inplace=True)

    # Prepare and scale data
    X_train = df[["Returns", "Range", "Volatility"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Fit HMM model
    model = GaussianHMM(n_components=5, covariance_type="full", n_iter=100, random_state=42) # Increased components
    model.fit(X_scaled)

    # Predict hidden states
    hidden_states = model.predict(X_scaled)

    # Analyze, Sort, and Describe Regimes
    means = model.means_
    sorted_vol_indices = np.argsort(means[:, 2])
    regime_map = {original: new for new, original in enumerate(sorted_vol_indices)}
    remapped_states = np.array([regime_map[s] for s in hidden_states])
    df['Regime'] = remapped_states

    sorted_means = means[sorted_vol_indices]

    # Generate dynamic descriptions
    regime_labels = []
    for i in range(model.n_components):
        mean_return = sorted_means[i, 0]

        # Volatility description (based on sorted order)
        if i < 2:
            vol_desc = "Low Volatility"
        elif i < 4:
            vol_desc = "Mid Volatility"
        else:
            vol_desc = "High Volatility"

        # Trend description
        if mean_return > 0.05:
            trend_desc = "Bullish"
        elif mean_return < -0.05:
            trend_desc = "Bearish"
        else:
            trend_desc = "Neutral"

        regime_labels.append(f"Regime {i}: {vol_desc}, {trend_desc}")

    regime_characteristics = pd.DataFrame(
        sorted_means,
        columns=['Mean Return', 'Mean Range', 'Mean Volatility'],
        index=[f'Regime {i}' for i in range(model.n_components)] # Use simple index for the table
    )
    # Add the descriptive label as a column
    regime_characteristics['Description'] = [label.split(': ')[1] for label in regime_labels]


    return df, model, regime_characteristics, regime_labels

# --- Main App Logic ---
df, model, regime_characteristics, regime_labels = get_data_and_model()

if df is not None:
    # --- Display Market Regime Chart ---
    st.subheader('Nifty 50 Price with Market Regimes')

    fig, ax = plt.subplots(figsize=(18, 10))

    prices = df['Close']

    for i in range(model.n_components):
        state_prices = np.full(prices.shape, np.nan)
        state_prices[df['Regime'] == i] = prices[df['Regime'] == i]
        ax.plot(df.index, state_prices, label=regime_labels[i])

    ax.set_title('Nifty 50 Market Regimes')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # --- Display Regime Characteristics Table ---
    st.subheader('Regime Characteristics')
    st.write("The regimes are sorted by their mean volatility (from lowest to highest).")
    st.dataframe(regime_characteristics)

    # --- Display Raw Data ---
    st.subheader('Recent Data with Predicted Regimes')
    st.dataframe(df.tail(100))
else:
    st.write("Please try again later.")
