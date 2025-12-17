# ============================================================
# DAILY HYBRID SIG ALERT — REGIME ONLY
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import optuna
import smtplib
import ssl
import os
from email.message import EmailMessage

# ============================================================
# CONFIG (LOCKED)
# ============================================================

DEFAULT_START_DATE = "2000-01-01"

RISK_ON_WEIGHTS = {
    "BTC-USD": 0.5,
    "FNGO": 0.5,
}

RISK_OFF_WEIGHTS = {
    "GLD": 0.55,
    "STRC": 0.45,
}

FLIP_COST = 0.00
START_RISKY = 0.70
START_SAFE  = 0.30

# ============================================================
# DATA LOADING
# ============================================================

def load_price_data(tickers, start_date):
    data = yf.download(tickers, start=start_date, progress=False)
    px = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px.dropna(how="any")

# ============================================================
# PORTFOLIO INDEX
# ============================================================

def build_portfolio_index(prices, weights):
    rets = prices.pct_change().fillna(0)
    idx = pd.Series(0.0, index=rets.index)
    for a, w in weights.items():
        if a in rets.columns:
            idx += rets[a] * w
    return (1 + idx).cumprod()

# ============================================================
# MA + SIGNAL
# ============================================================

def compute_ma(px, L):
    return px.rolling(L, min_periods=1).mean().shift(1)

def generate_signal(px, ma, tol):
    sig = np.zeros(len(px), dtype=bool)
    upper = ma * (1 + tol)
    lower = ma * (1 - tol)
    for i in range(1, len(px)):
        if not sig[i - 1]:
            sig[i] = px.iloc[i] > upper.iloc[i]
        else:
            sig[i] = not (px.iloc[i] < lower.iloc[i])
    return pd.Series(sig, index=px.index)

# ============================================================
# OPTUNA — MA PARAMS (OOS)
# ============================================================

def optimize_ma(prices, n_trials=200):
    idx = build_portfolio_index(prices, RISK_ON_WEIGHTS)
    train, test = idx[:-756], idx[-756:]

    def objective(trial):
        L = trial.suggest_int("L", 100, 252)
        tol = trial.suggest_float("tol", 0.01, 0.03)
        ma = compute_ma(train, L)
        sig = generate_signal(train, ma, tol)
        rets = train.pct_change().fillna(0)[sig]
        return -(rets.mean() / rets.std()) if rets.std() > 0 else 1e9

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params["L"], study.best_params["tol"]

# ============================================================
# EMAIL SENDER
# ============================================================

def send_email(body):

    msg = EmailMessage()
    msg["Subject"] = "Daily Hybrid SIG Regime"
    msg["From"] = os.environ["EMAIL_USER"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(os.environ["EMAIL_USER"], os.environ["EMAIL_PASS"])
        smtp.send_message(msg)

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    tickers = sorted(set(
        list(RISK_ON_WEIGHTS.keys()) +
        list(RISK_OFF_WEIGHTS.keys())
    ))

    prices = load_price_data(tickers, DEFAULT_START_DATE)

    # Optimize MA
    L, tol = optimize_ma(prices)

    # Build signal
    idx = build_portfolio_index(prices, RISK_ON_WEIGHTS)
    ma = compute_ma(idx, L)
    sig = generate_signal(idx, ma, tol)

    latest_signal = bool(sig.iloc[-1])

    # EMAIL BODY — REGIME ONLY
    if latest_signal:
        body = "RISK-ON"
    else:
        body = (
            "RISK-OFF\n"
            "When in risk off allocate 55% to GLD -!: 45% STRC."
        )

    send_email(body)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()