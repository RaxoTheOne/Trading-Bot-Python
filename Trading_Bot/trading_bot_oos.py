"""
trading_bot_oos.py
Ein erweiterter Backtest-Starter:
- SMA-Crossover (Short/Long)
- Positionsgrößen basierend auf ATR & fixed risk fraction
- Fees & Slippage berücksichtigt
- In-Sample / Out-Of-Sample (OOS) Split
- Trades -> trades.csv
- Report -> report.csv (Metriken für IS und OOS)
Hinweis: KEINE Live-Orders. Paper/backtest only.
"""

from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import csv, os

# ----------------- KONFIG -----------------
YF_TICKER = "BTC-USD"          # yfinance Ticker
START = (datetime.today() - timedelta(days=700)).strftime("%Y-%m-%d")
END = datetime.today().strftime("%Y-%m-%d")                   # None = bis heute
TIMEFRAME = "1h"               # '1h' mapping in fetch
SMA_SHORT = 50
SMA_LONG = 200
ATR_PERIOD = 14

INITIAL_CAPITAL = 10000.0      # Startkapital für Simulation
RISK_PER_TRADE = 0.01          # 1% des Kapitals pro Trade (Risk fraction)
FEE_PCT = 0.0015               # Gesamtgebühr pro Trade (z.B. 0.15% -> 0.0015)
SLIPPAGE_PCT = 0.0008          # Slippage pro Trade (z.B. 0.08% -> 0.0008)

OOS_SPLIT = 0.2                # letzter Anteil der Daten = Out-Of-Sample (z.B. 0.2 = 20%)
LOG_TRADES = "trades.csv"
REPORT = "report.csv"
# -----------------------------------------

# --- Hilfsfunktionen ---
def fetch_history_yf(ticker, timeframe='1h', start=None, end=None):
    interval_map = {'1m':'1m','5m':'5m','15m':'15m','1h':'60m','4h':'4h','1d':'1d'}
    interval = interval_map.get(timeframe, '60m')
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def sma(series, window):
    return series.rolling(window=window).mean()

def atr(df, n=14):
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def max_drawdown(equity_series):
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    return dd.min()

def sharpe_ratio(returns, periods_per_year=252):
    # returns: periodic returns (e.g. daily returns). If hourly, adjust periods_per_year accordingly.
    mean = np.nanmean(returns)
    std = np.nanstd(returns, ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return (mean / std) * (periods_per_year ** 0.5)

# --- Strategy Backtest mit Fees & Slippage ---
def backtest_with_fees(df, initial_capital=10000.0, risk_per_trade=0.01,
                       fee_pct=0.0015, slippage_pct=0.0008,
                       sma_short=50, sma_long=200, atr_period=14):
    df = df.copy()
    df['sma_short'] = sma(df['close'], sma_short)
    df['sma_long'] = sma(df['close'], sma_long)
    df['atr'] = atr(df, atr_period)
    capital = initial_capital
    position = 0.0  # units
    entry_price = 0.0
    equity_curve = []
    trades = []

    # we compute returns per step for equity series (use close-to-close)
    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        price = row['close']

        # append current equity (cash + position*market_price)
        net_worth = capital + position * price
        equity_curve.append({'timestamp': date, 'equity': net_worth})

        # need past value to generate cross -> skip until we have previous
        if i == 0: continue
        prev = df.iloc[i-1]

        buy_signal = (prev['sma_short'] <= prev['sma_long']) and (row['sma_short'] > row['sma_long'])
        sell_signal = (prev['sma_short'] >= prev['sma_long']) and (row['sma_short'] < row['sma_long'])

        # ENTRY
        if position == 0 and buy_signal and not np.isnan(row['atr']):
            # stop based on ATR
            stop = price - row['atr'] * 1.5
            risk_per_unit = price - stop
            if risk_per_unit <= 0:
                continue
            max_risk = capital * risk_per_trade
            units = max_risk / risk_per_unit
            # cost and apply slippage and fee
            effective_buy_price = price * (1.0 + slippage_pct)
            cost = units * effective_buy_price
            fee = cost * fee_pct
            total_cost = cost + fee
            if total_cost > capital:
                # reduce units to fit capital
                units = capital / (effective_buy_price * (1.0 + fee_pct))
                cost = units * effective_buy_price
                fee = cost * fee_pct
                total_cost = cost + fee
            if units <= 0: continue
            # execute buy
            position = units
            entry_price = effective_buy_price
            capital -= total_cost
            trades.append({
                'timestamp': date.isoformat(), 'side':'BUY', 'price': float(effective_buy_price),
                'units': float(units), 'fee': float(fee), 'capital_after': float(capital), 'reason':'SMA_CROSS', 'stop': float(stop)
            })

        # EXIT
        elif position > 0:
            # stoploss example (3*ATR from entry) OR SMA cross
            stop_price = entry_price - row['atr'] * 3 if not np.isnan(row['atr']) else None
            exit_by_stop = (stop_price is not None) and (price <= stop_price)
            if sell_signal or exit_by_stop:
                effective_sell_price = price * (1.0 - slippage_pct)
                proceeds = position * effective_sell_price
                fee = proceeds * fee_pct
                net = proceeds - fee
                capital += net
                trades.append({
                    'timestamp': date.isoformat(), 'side':'SELL', 'price': float(effective_sell_price),
                    'units': float(position), 'fee': float(fee), 'capital_after': float(capital), 'reason': 'STOP' if exit_by_stop else 'SMA_CROSS', 'stop': float(stop_price) if stop_price is not None else ''
                })
                position = 0.0
                entry_price = 0.0

    # close any open position at end
    final_price = df['close'].iloc[-1]
    if position > 0:
        effective_sell_price = final_price * (1.0 - slippage_pct)
        proceeds = position * effective_sell_price
        fee = proceeds * fee_pct
        net = proceeds - fee
        capital += net
        trades.append({
            'timestamp': df.index[-1].isoformat(), 'side':'SELL', 'price': float(effective_sell_price),
            'units': float(position), 'fee': float(fee), 'capital_after': float(capital), 'reason': 'CLOSE_END', 'stop': ''
        })
        position = 0.0

    equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
    equity_df.index = pd.to_datetime(equity_df.index)
    return trades, equity_df

# --- Utilities Schreiben/Auswertung ---
def append_trades_csv(trades, filename=LOG_TRADES):
    write_header = not os.path.exists(filename)
    with open(filename, 'a', newline='') as f:
        cols = ['timestamp','side','price','units','fee','capital_after','reason','stop']
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header: writer.writeheader()
        for t in trades:
            writer.writerow(t)

def evaluate_equity(equity_df, initial_capital):
    equity_df = equity_df.sort_index()
    equity = equity_df['equity']
    # periodic returns: pct change
    returns = equity.pct_change().fillna(0).values
    total_return = (equity.iloc[-1] / initial_capital) - 1.0
    # compute CAGR approximate: use time delta in years between first and last
    days = (equity.index[-1] - equity.index[0]).total_seconds() / (3600*24)
    years = days / 365.25 if days > 0 else 1/365.25
    cagr = (equity.iloc[-1] / initial_capital) ** (1.0/years) - 1.0 if years > 0 else np.nan
    mdd = max_drawdown(equity)
    # Sharpe: estimate returns per business-day equivalent.
    # If timeframe hourly, estimate periods_per_year = 24*365 = 8760
    # We infer period from median diff
    if len(equity.index) > 1:
        median_diff_seconds = np.median(np.diff(equity.index.astype('int64')))/1e9
        periods_per_year = int(round(365.25 * 24 * 3600 / median_diff_seconds))
    else:
        periods_per_year = 252
    sr = sharpe_ratio(returns, periods_per_year=periods_per_year)
    return {
        'total_return': float(total_return),
        'cagr': float(cagr) if not np.isnan(cagr) else None,
        'max_drawdown': float(mdd),
        'sharpe': float(sr) if not np.isnan(sr) else None,
        'final_equity': float(equity.iloc[-1])
    }

# --- Main: IS/OOS Split, Run, Report ---
def run():
    print("Lade historische Daten...")
    df = fetch_history_yf(YF_TICKER, timeframe=TIMEFRAME, start=START, end=END)
    if df is None or len(df) < SMA_LONG + 50:
        raise SystemExit("Nicht genug Datenpunkte für Backtest. Ticker/Zeitraum anpassen.")
    # OOS split by fraction
    split_idx = int(len(df) * (1.0 - OOS_SPLIT))
    df_is = df.iloc[:split_idx].copy()
    df_oos = df.iloc[split_idx:].copy()

    print(f"Gesamt-Datapoints: {len(df)}. IS: {len(df_is)}. OOS: {len(df_oos)}")

    # Run on IS
    trades_is, equity_is = backtest_with_fees(df_is, initial_capital=INITIAL_CAPITAL,
                                              risk_per_trade=RISK_PER_TRADE, fee_pct=FEE_PCT,
                                              slippage_pct=SLIPPAGE_PCT,
                                              sma_short=SMA_SHORT, sma_long=SMA_LONG, atr_period=ATR_PERIOD)
    # start OOS with same starting capital as initial (or you could start from IS final equity)
    trades_oos, equity_oos = backtest_with_fees(df_oos, initial_capital=INITIAL_CAPITAL,
                                                risk_per_trade=RISK_PER_TRADE, fee_pct=FEE_PCT,
                                                slippage_pct=SLIPPAGE_PCT,
                                                sma_short=SMA_SHORT, sma_long=SMA_LONG, atr_period=ATR_PERIOD)

    # Save trades
    if trades_is:
        append_trades_csv(trades_is, LOG_TRADES)
    if trades_oos:
        append_trades_csv(trades_oos, LOG_TRADES)

    # Evaluate
    eval_is = evaluate_equity(equity_is, INITIAL_CAPITAL)
    eval_oos = evaluate_equity(equity_oos, INITIAL_CAPITAL)

    report_row = {
        'timestamp': datetime.utcnow().isoformat(),
        'ticker': YF_TICKER,
        'sma_short': SMA_SHORT,
        'sma_long': SMA_LONG,
        'atr_period': ATR_PERIOD,
        'initial_capital': INITIAL_CAPITAL,
        'risk_per_trade': RISK_PER_TRADE,
        'fee_pct': FEE_PCT,
        'slippage_pct': SLIPPAGE_PCT,
        'oos_split': OOS_SPLIT,
        # IS metrics
        'is_total_return': eval_is['total_return'],
        'is_cagr': eval_is['cagr'],
        'is_max_drawdown': eval_is['max_drawdown'],
        'is_sharpe': eval_is['sharpe'],
        'is_final_equity': eval_is['final_equity'],
        # OOS metrics
        'oos_total_return': eval_oos['total_return'],
        'oos_cagr': eval_oos['cagr'],
        'oos_max_drawdown': eval_oos['max_drawdown'],
        'oos_sharpe': eval_oos['sharpe'],
        'oos_final_equity': eval_oos['final_equity'],
        'notes': 'Backtest mit Fees & Slippage; OOS initial capital = same as IS (configurable)'
    }

    # append report CSV
    write_header = not os.path.exists(REPORT)
    with open(REPORT, 'a', newline='') as f:
        cols = list(report_row.keys())
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header: writer.writeheader()
        writer.writerow(report_row)

    # Ausgabe kurz
    print("=== Zusammenfassung ===")
    print("IS Final Equity:", eval_is['final_equity'], "Total Return:", eval_is['total_return'])
    print("OOS Final Equity:", eval_oos['final_equity'], "Total Return:", eval_oos['total_return'])
    print(f"Trades (IS): {len(trades_is)} | Trades (OOS): {len(trades_oos)}")
    print("Trades wurden nach", LOG_TRADES, "geschrieben. Report in", REPORT)

if __name__ == "__main__":
    run()
