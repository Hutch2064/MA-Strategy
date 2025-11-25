import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib

# ============================================
# CONFIG — MATCH STREAMLIT EXACTLY
# ============================================

DEFAULT_START_DATE = "2011-11-24"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "GLD": 1.0,   # 3/3
    "TQQQ": 1/3,
    "BTC-USD": 1/3,
}

RISK_OFF_WEIGHTS = {
    "SHY": 1.0,
}

# ============================================
# DATA LOADING (no cache)
# ============================================

def load_price_data(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
    else:
        px = data["Close"].copy()

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    return px.dropna(how="all")

# ============================================
# TECHNICALS — TESTFOL ACCURATE VERSION
# ============================================

def compute_ma_matrix(price, lengths, ma_type):
    ma_dict = {}

    if ma_type == "ema":
        for L in lengths:
            ma = price.ewm(span=L, adjust=False).mean()
            ma_dict[L] = ma.shift(1)
    else:
        for L in lengths:
            ma = price.rolling(window=L, min_periods=L).mean()
            ma_dict[L] = ma.shift(1)

    return ma_dict


def generate_testfol_signal_vectorized(price, ma, tol):
    px = price.shift(1).values
    ma_vals = ma.values
    n = len(px)

    upper = ma_vals * (1 + tol)
    lower = ma_vals * (1 - tol)

    sig = np.zeros(n, dtype=bool)

    first_valid = np.nanargmin(np.isnan(ma_vals))
    if first_valid == 0:
        first_valid = 1
    start_index = first_valid + 1

    for t in range(start_index, n):
        if not sig[t-1]:
            sig[t] = px[t] > upper[t]
        else:
            sig[t] = not (px[t] < lower[t])

    return pd.Series(sig, index=ma.index).fillna(False)

# ============================================
# BACKTEST — LOG RETURNS + DAILY REBALANCE
# ============================================

def build_weight_df(prices, signal, risk_on_weights, risk_off_weights):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for asset, w in risk_on_weights.items():
        if asset in prices.columns:
            weights.loc[signal, asset] = w

    for asset, w in risk_off_weights.items():
        if asset in prices.columns:
            weights.loc[~signal, asset] = w

    return weights


def compute_performance(log_returns, equity_curve, rf=0.0):
    cagr = np.exp(log_returns.mean() * 252) - 1
    vol = log_returns.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol > 0 else np.nan
    dd = equity_curve / equity_curve.cummax() - 1
    max_dd = dd.min()
    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": total_ret,
    }


def backtest(prices, signal, risk_on_weights, risk_off_weights):
    log_px = np.log(prices)
    log_rets = log_px.diff().fillna(0)

    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)
    strat_log_rets = (weights.shift(1).fillna(0) * log_rets).sum(axis=1)
    eq = np.exp(strat_log_rets.cumsum())

    return {
        "returns": strat_log_rets,
        "equity_curve": eq,
        "weights": weights,
        "signal": signal,
        "performance": compute_performance(strat_log_rets, eq),
    }

# ============================================
# GRID SEARCH — SAME AS STREAMLIT
# ============================================

def run_grid_search(prices, risk_on_weights, risk_off_weights):
    btc = prices["BTC-USD"]

    best_sharpe = -1e9
    best_trades = np.inf
    best_cfg = None
    best_result = None

    lengths = list(range(21, 253))
    types = ["sma", "ema"]
    tolerances = np.arange(0.0, 0.1001, 0.002)

    ma_cache = {t: compute_ma_matrix(btc, lengths, t) for t in types}

    for ma_type in types:
        for L in lengths:
            ma_series = ma_cache[ma_type][L]

            for tol in tolerances:
                signal = generate_testfol_signal_vectorized(btc, ma_series, tol)

                result = backtest(prices, signal, risk_on_weights, risk_off_weights)
                sharpe = result["performance"]["Sharpe"]

                sig_arr = result["signal"].astype(int)
                switches = sig_arr.diff().abs().sum()
                trades_per_year = switches / (len(sig_arr) / 252)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_trades = trades_per_year
                    best_cfg = (L, ma_type, tol)
                    best_result = result

                elif sharpe == best_sharpe and trades_per_year < best_trades:
                    best_trades = trades_per_year
                    best_cfg = (L, ma_type, tol)
                    best_result = result

    return best_cfg, best_result

# ============================================
# EMAIL
# ============================================

def attach_file(msg, filepath):
    with open(filepath, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{filepath}"')
    msg.attach(part)


def send_email(regime, cfg, perf):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    SEND_TO = os.getenv("SEND_TO")

    L, ma_type, tol = cfg

    html = f"""
    <html>
      <body>
        <h2>BTC MA Optimized Portfolio — Daily Signal</h2>
        <p><b>Regime:</b> {regime}</p>

        <h3>Performance</h3>
        <ul>
          <li><b>CAGR:</b> {perf['CAGR']:.2%}</li>
          <li><b>Volatility:</b> {perf['Volatility']:.2%}</li>
          <li><b>Sharpe:</b> {perf['Sharpe']:.3f}</li>
          <li><b>Max Drawdown:</b> {perf['MaxDrawdown']:.2%}</li>
          <li><b>Total Return:</b> {perf['TotalReturn']:.2%}</li>
        </ul>

        <h3>Best Parameters</h3>
        <ul>
          <li>Length: {L}</li>
          <li>Type: {ma_type.upper()}</li>
          <li>Tolerance: {tol:.3f}</li>
        </ul>
      </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["Subject"] = f"BTC MA Optimized Portfolio — {regime}"
    msg["From"] = EMAIL_USER
    msg["To"] = SEND_TO
    msg.attach(MIMEText(html, "html"))

    attach_file(msg, "equity_curve.png")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, SEND_TO, msg.as_string())

# ============================================
# MAIN — HEADLESS DAILY ENGINE
# ============================================

if __name__ == "__main__":
    tickers = ["BTC-USD", "GLD", "TQQQ", "SHY"]
    prices = load_price_data(tickers, DEFAULT_START_DATE).dropna(how="any")

    best_cfg, best_result = run_grid_search(prices, RISK_ON_WEIGHTS, RISK_OFF_WEIGHTS)

    perf = best_result["performance"]
    sig = best_result["signal"]
    eq = best_result["equity_curve"]

    latest_signal = bool(sig.iloc[-1])
    regime = "RISK-ON" if latest_signal else "RISK-OFF"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq, label="Optimized Strategy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.close()

    send_email(regime, best_cfg, perf)



