import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Daten holen (z. B. Apple)
ticker = 'AAPL'
data = yf.download(ticker, start='2022-01-01', end='2023-12-31')
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# 2. Strategie: Kaufen, wenn SMA50 > SMA200, sonst verkaufen
data['Signal'] = 0
data['Signal'][50:] = \
    (data['SMA50'][50:] > data['SMA200'][50:]).astype(int)
data['Position'] = data['Signal'].diff()

# 3. Plotten
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Kurs', alpha=0.5)
plt.plot(data['SMA50'], label='SMA50')
plt.plot(data['SMA200'], label='SMA200')
plt.plot(data[data['Position'] == 1].index,
         data['SMA50'][data['Position'] == 1],
         '^', label='Kaufen', color='g', markersize=10)
plt.plot(data[data['Position'] == -1].index,
         data['SMA50'][data['Position'] == -1],
         'v', label='Verkaufen', color='r', markersize=10)
plt.legend()
plt.title(f'Simulation SMA-Crossover: {ticker}')
plt.grid()
plt.show()