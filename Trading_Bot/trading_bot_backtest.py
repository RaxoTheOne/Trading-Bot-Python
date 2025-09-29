# trading_bot_backtest.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# -------------------------------------
# Parameter
# -------------------------------------
ticker = "BTC-USD"                    # Kryptowährung Bitcoin in USD
end = datetime.today().strftime("%Y-%m-%d")
start = (datetime.today() - timedelta(days=700)).strftime("%Y-%m-%d")  # ~2 Jahre
interval = "1h"                        # 1-Stunden-Kerzen

print(f"Lade Daten für {ticker} von {start} bis {end} mit Intervall {interval} …")

# -------------------------------------
# Daten abrufen
# -------------------------------------
df = yf.download(
    ticker,
    start=start,
    end=end,
    interval=interval,
    progress=False
)

# -------------------------------------
# Prüfen und anzeigen
# -------------------------------------
if df.empty:
    print("FEHLER: Keine Daten empfangen. Prüfe Ticker oder Zeitbereich.")
else:
    print(f"{len(df)} Kerzen geladen.")
    print("Erste 5 Zeilen:")
    print(df.head())
    print("\nLetzte 5 Zeilen:")
    print(df.tail())

    # Beispiel: einfachen gleitenden Durchschnitt berechnen
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    print("\nMit SMA_50 (Beispiel für Indikator):")
    print(df[['Close', 'SMA_50']].tail())

    # Optional: Als CSV speichern
    df.to_csv("btc_usd_hourly.csv")
    print("\nDaten in 'btc_usd_hourly.csv' gespeichert.")
