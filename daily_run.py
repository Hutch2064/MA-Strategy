import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dataclasses import dataclass

# =========================
# CONFIG
# =========================

DEFAULT_START_DATE = "2011-11-24"
DEFAULT_END_DATE = None

TICKERS = ["BTC-USD", "GLD", "TQQQ", "UUP"]

RISK_ON_WEIGHTS = {
    "GLD": 3.0 / 3.0,
    "TQQQ": 1.0 / 3.0,
    "BTC-USD": 1.0 / 3.0,
}

RISK_OFF_WEIGHTS = {
    "UUP": 1.0,
}

RISK_FREE_RATE = 0.0


# =========================
# DATA LOADING
# =========================

def load_price_data_raw(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    px = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px.dropna(how="all")


# =========================
# MOVING AVERAGES
# =========================

def compute_ma(series, length, ma_type):
    if ma_type == "ema":
        return series.ewm(span=length, adjust=False).mean()
    return series.rolling(window=length, min_periods=length).mean()


# =========================
# SIGNAL
# =========================

def generate_signal(price, length, ma_type, tolerance):
    ma = compute_ma(price, length, ma_type)
    return (price > ma * (1 + tolerance)).fillna(False)


# =========================
# BACKTEST ENGINE
# =========================

def build_weight_df(prices, signal, ron_w, roff_w):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for a, w in ron_w.items():
        if a in weights.columns:
            weights.loc[signal, a] = w
    for a, w in roff_w.items():
        if a in weights.columns:
            weights.loc[~signal, a] = w
    return weights


def compute_performance(returns, equity_curve):
    n = len(returns)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (1 + total_return) ** (252 / n) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else -np.inf
    dd = (equity_curve / equity_curve.cummax()) - 1
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": dd.min(),
        "TotalReturn": total_return,
    }


def backtest(prices, signal, ron_w, roff_w):
    rets = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, ron_w, roff_w)
    strat_rets = (weights.shift(1).fillna(0) * rets).sum(axis=1)
    eq = (1 + strat_rets).cumprod()
    return strat_rets, eq


# =========================
# DETERMINISTIC GRID SEARCH
# =========================

def run_grid_search(prices, ron_w, roff_w):
    btc = prices["BTC-USD"]

    lengths = range(21, 253)
    types = ["sma", "ema"]
    tolerances = np.round(np.arange(0.0, 0.1001, 0.001), 3)

    best_params = None
    best_result = None
    best_sharpe = -np.inf

    for ma_type in types:
        for L in lengths:
            for tol in tolerances:

                sig = generate_signal(btc, L, ma_type, tol)
                strat_rets, eq = backtest(prices, sig, ron_w, roff_w)
                perf = compute_performance(strat_rets, eq)

                if perf["Sharpe"] > best_sharpe:
                    best_sharpe = perf["Sharpe"]
                    best_params = (L, ma_type, tol)
                    best_result = {
                        "returns": strat_rets,
                        "equity_curve": eq,
                        "signal": sig,
                        "performance": perf,
                    }

    return best_params, best_result


# =========================
# PLOT
# =========================

def plot_equity_curve(eq, prices, filename="equity_curve.png"):
    rets = prices.pct_change().fillna(0)
    cols = ["GLD", "TQQQ", "BTC-USD"]

    base = (rets[cols] * np.array([1/3, 1/3, 1/3])).sum(axis=1)
    base_eq = (1 + base).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq.values, label="Optimized Strategy", linewidth=2)
    plt.plot(base_eq.index, base_eq.values,
             label="Pure Risk-On (33/33/33)", linestyle="--", linewidth=2)
    plt.title("Equity Curve – Optimized Strategy")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =========================
# EMAIL SENDER
# =========================

def attach_file(msg, filepath, mime_subtype="octet-stream"):
    if not os.path.exists(filepath):
        return
    with open(filepath, "rb") as f:
        part = MIMEBase("application", mime_subtype)
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition",
                    f'attachment; filename="{os.path.basename(filepath)}"')
    msg.attach(part)


def send_email(regime, params, perf):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    SEND_TO = os.getenv("SEND_TO")

    L, ma_type, tol = params

    html = f"""
    <html>
      <body>
        <h2>Optimized MA Signal</h2>
        <p><b>Today's Regime:</b> {regime}</p>

        <h3>Performance</h3>
        <ul>
          <li><b>CAGR:</b> {perf['CAGR']:.2%}</li>
          <li><b>Volatility:</b> {perf['Volatility']:.2%}</li>
          <li><b>Sharpe:</b> {perf['Sharpe']:.3f}</li>
          <li><b>Max Drawdown:</b> {perf['MaxDrawdown']:.2%}</li>
          <li><b>Total Return:</b> {perf['TotalReturn']:.2%}</li>
        </ul>

        <h3>Optimized Parameters</h3>
        <ul>
          <li>MA Length: {L}</li>
          <li>Type: {ma_type.upper()}</li>
          <li>Tolerance: {tol:.3f}</li>
        </ul>

        <p>Equity curve attached.</p>
      </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["Subject"] = f"MA Strategy Report – {regime}"
    msg["From"] = EMAIL_USER
    msg["To"] = SEND_TO

    msg_alt = MIMEMultipart("alternative")
    msg.attach(msg_alt)
    msg_alt.attach(MIMEText(html, "html"))

    attach_file(msg, "equity_curve.png")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, SEND_TO, msg.as_string())


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    prices = load_price_data_raw(TICKERS, DEFAULT_START_DATE, DEFAULT_END_DATE)
    prices = prices.dropna()

    best_params, best_result = run_grid_search(
        prices,
        RISK_ON_WEIGHTS,
        RISK_OFF_WEIGHTS,
    )

    perf = best_result["performance"]
    signal = best_result["signal"]
    eq = best_result["equity_curve"]

    plot_equity_curve(eq, prices, "equity_curve.png")

    is_on = bool(signal.iloc[-1])
    regime = "RISK-ON (33/33/33)" if is_on else "RISK-OFF (100% UUP)"

    send_email(regime, best_params, perf)

