import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# Data Extraction
start_date = "2017-01-01"
end_date = "2022-06-01"
symbol = "^NSEI"  # Nifty 50 symbol
data = yf.download(symbol, start=start_date, end=end_date)
data = data[["Open", "High", "Low", "Close", "Volume"]]

# Create a copy
df = data.copy()
df["Returns"] = df["Close"].pct_change()
df["Range"] = (df["High"] / df["Low"]) - 1
df.dropna(inplace=True)

# Structure Data
X_train = df[["Returns", "Range"]]

# Fit Model
hmm_model = GaussianHMM(n_components=4, covariance_type="full", n_iter=100).fit(X_train)
print("Model Score:", hmm_model.score(X_train))

# Check results
hidden_states = hmm_model.predict(X_train)
print("Hidden States:", hidden_states[:5])

prices = df['Close'].values

# Create empty arrays for plotting
plot_0 = np.full(prices.shape, np.nan)
plot_1 = np.full(prices.shape, np.nan)
plot_2 = np.full(prices.shape, np.nan)
plot_3 = np.full(prices.shape, np.nan)

# Fill arrays based on hidden states
plot_0[hidden_states == 0] = prices[hidden_states == 0]
plot_1[hidden_states == 1] = prices[hidden_states == 1]
plot_2[hidden_states == 2] = prices[hidden_states == 2]
plot_3[hidden_states == 3] = prices[hidden_states == 3]


# Plot chart
fig = plt.figure(figsize = (18,10))
plt.plot(df.index, plot_0, color="red", label='State 0')
plt.plot(df.index, plot_1, color="green", label='State 1')
plt.plot(df.index, plot_2, color="black", label='State 2')
plt.plot(df.index, plot_3, color="orange", label='State 3')
plt.title('Nifty 50 Market Regimes')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig("nifty_50_regimes.png")
print("Chart saved to nifty_50_regimes.png")
