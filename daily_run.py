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
# CONFIG
# ============================================

DEFAULT_START_DATE = "2011-11-24"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "GLD": 1/3,
    "TQQQ": 1/3,
    "BTC-USD": 1/3,
}

RISK_OFF_WEIGHTS = {
    "UUP": 1.0,
}

# ============================================
# DATA (IDENTICAL TO STREAMLIT VERSION, minus cache)
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
# TECHNICALS
# ============================================

def compute_ma(series, length, ma_type):
    if ma_type == "ema":
        return series.ewm(span=length, adjust=False).mean()
    return series.rolling(window=length, min_periods=length).mean()

def generate_signal(price, length, ma_type, tol):
    ma = compute_ma(price, length, ma_type)
    sig = price > ma * (1 + tol)
    return sig.fillna(False)

# ============================================
# BACKTEST
# ============================================

def build_weight_df(prices, signal, risk_on_weights, risk_off_weights):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for a, w in risk_on_weights.items():
        if a in prices.columns:
            weights.loc[signal, a] = w

    for a, w in risk_off_weights.items():
        if a in prices.columns:
            weights.loc[~signal, a] = w

    return weights

def compute_performance(returns, equity_curve, rf=0.0):
    n = len(returns)
    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (1 + total_ret)**(252/n) - 1

    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol > 0 else np.nan

    dd = equity_curve / equity_curve.cummax() - 1
    max_dd = dd.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": total_ret,
    }

def backtest(prices, signal, risk_on_weights, risk_off_weights):
    rets = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strat_rets = (weights.shift(1).fillna(0) * rets).sum(axis=1)
    eq = (1 + strat_rets).cumprod()

    return {
        "returns": strat_rets,
        "equity_curve": eq,
        "weights": weights,
        "signal": signal,
        "performance": compute_performance(strat_rets, eq),
    }

# ============================================
# GRID SEARCH (IDENTICAL)
# ============================================

def run_grid_search(prices, risk_on_weights, risk_off_weights):
    btc = prices["BTC-USD"]

    best_sharpe = -1e9
    best = None
    best_cfg = None

    lengths = range(21, 253)
    types = ["sma", "ema"]
    tolerances = np.arange(0.0, 0.1001, 0.001)

    for length in lengths:
        for ma_type in types:
            for tol in tolerances:
                signal = generate_signal(btc, length, ma_type, tol)
                result = backtest(prices, signal, risk_on_weights, risk_off_weights)
                sharpe = result["performance"]["Sharpe"]

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_cfg = (length, ma_type, tol)
                    best = result

    return best_cfg, best

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
        <h2>BTC MA Model — Daily Signal</h2>
        <p><b>Regime:</b> {regime}</p>

        <h3>Performance</h3>
        <ul>
          <li><b>CAGR:</b> {perf['CAGR']:.2%}</li>
          <li><b>Volatility:</b> {perf['Volatility']:.2%}</li>
          <li><b>Sharpe:</b> {perf['Sharpe']:.3f}</li>
          <li><b>Max Drawdown:</b> {perf['MaxDrawdown']:.2%}</li>
          <li><b>Total Return:</b> {perf['TotalReturn']:.2%}</li>
        </ul>

        <h3>Best MA Params</h3>
        <ul>
          <li>Length: {L}</li>
          <li>Type: {ma_type.upper()}</li>
          <li>Tolerance: {tol:.3f}</li>
        </ul>
      </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["Subject"] = f"BTC Trend Model — {regime}"
    msg["From"] = EMAIL_USER
    msg["To"] = SEND_TO

    msg.attach(MIMEText(html, "html"))

    attach_file(msg, "equity_curve.png")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, SEND_TO, msg.as_string())

# ============================================
# MAIN — HEADLESS STREAMLIT
# ============================================

if __name__ == "__main__":
    tickers = ["BTC-USD", "GLD", "TQQQ", "UUP"]
    prices = load_price_data(tickers, DEFAULT_START_DATE)
    prices = prices.dropna(how="any")

    best_cfg, best_result = run_grid_search(prices, RISK_ON_WEIGHTS, RISK_OFF_WEIGHTS)

    perf = best_result["performance"]
    signal = best_result["signal"]
    eq = best_result["equity_curve"]

    is_on = bool(signal.iloc[-1])
    regime = "RISK-ON (33/33/33)" if is_on else "RISK-OFF (100% UUP)"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq, label="Optimized Strategy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.close()

    send_email(regime, best_cfg, perf)


