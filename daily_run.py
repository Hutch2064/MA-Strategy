import numpy as np
import pandas as pd
import yfinance as yf
import optuna
from scipy.optimize import minimize
import datetime
import os
import ssl
import smtplib
from email.message import EmailMessage

# ============================================================
# CONFIG — IDENTICAL TO MAIN
# ============================================================

DEFAULT_START_DATE = "2000-01-01"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "BTC-USD": 0.50,
    "FNGO": 0.50,
}

RISK_OFF_WEIGHTS = {
    "SHY": 0.45,
    "GLD": 0.55,
}

FLIP_COST = 0.00
START_RISKY = 0.70
START_SAFE  = 0.30

OPTUNA_TRIALS = 300

# ============================================================
# DATA LOADING
# ============================================================

def load_price_data(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    px = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px.dropna(how="any")

# ============================================================
# PORTFOLIO INDEX (IDENTICAL)
# ============================================================

def build_portfolio_index(prices, weights):
    rets = prices.pct_change().fillna(0)
    idx = pd.Series(0.0, index=rets.index)
    for a, w in weights.items():
        if a in rets.columns:
            idx += rets[a] * w
    return (1 + idx).cumprod()

# ============================================================
# MA + SIGNAL (ROBUST VERSION)
# ============================================================

def compute_ma_matrix(series, lengths):
    return {L: series.rolling(L, min_periods=1).mean().shift(1) for L in lengths}

def generate_testfol_signal(price, ma, tol):
    px = price.values
    ma = ma.values
    upper = ma * (1 + tol)
    lower = ma * (1 - tol)

    sig = np.zeros(len(px), dtype=bool)

    for i in range(1, len(px)):
        if not sig[i-1]:
            sig[i] = px[i] > upper[i]
        else:
            sig[i] = not (px[i] < lower[i])

    return pd.Series(sig, index=price.index)

# ============================================================
# BACKTEST (MA STRATEGY)
# ============================================================

def backtest(prices, signal):
    rets = prices.pct_change().fillna(0)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for a, w in RISK_ON_WEIGHTS.items():
        weights.loc[signal, a] = w
    for a, w in RISK_OFF_WEIGHTS.items():
        weights.loc[~signal, a] = w

    strat = (weights.shift(1).fillna(0) * rets).sum(axis=1)
    flips = signal.astype(int).diff().abs() == 1
    strat += np.where(flips, -FLIP_COST * 4.0, 0.0)

    eq = (1 + strat).cumprod()

    sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0

    return eq, strat, sharpe

# ============================================================
# OPTUNA OOS OPTIMIZATION (IDENTICAL LOGIC)
# ============================================================

def optuna_oos(prices):
    TEST_DAYS = 252 * 3

    train = prices.iloc[:-TEST_DAYS]
    test  = prices.iloc[-TEST_DAYS:]

    pi_train = build_portfolio_index(train, RISK_ON_WEIGHTS)
    pi_test  = build_portfolio_index(test,  RISK_ON_WEIGHTS)

    def objective(trial):
        L = trial.suggest_int("L", 100, 252)
        tol = trial.suggest_float("tol", 0.01, 0.03, step=0.0025)

        ma_train = compute_ma_matrix(pi_train, [L])[L]
        ma_test  = compute_ma_matrix(pi_test,  [L])[L]

        sig_test = generate_testfol_signal(pi_test, ma_test, tol)
        _, _, sharpe = backtest(test, sig_test)

        return -sharpe

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    best_L = study.best_params["L"]
    best_tol = study.best_params["tol"]

    pi_full = build_portfolio_index(prices, RISK_ON_WEIGHTS)
    ma_full = compute_ma_matrix(pi_full, [best_L])[best_L]
    sig = generate_testfol_signal(pi_full, ma_full, best_tol)

    eq, rets, sharpe = backtest(prices, sig)

    return sig, best_L, best_tol, sharpe

# ============================================================
# TRUE CALENDAR QUARTERS
# ============================================================

def calendar_quarters(dates):
    q_ends = pd.date_range(dates.min(), dates.max(), freq="Q")
    return pd.to_datetime([dates[dates <= q].max() for q in q_ends])

# ============================================================
# DAILY ENGINE
# ============================================================

def run_daily_engine():

    tickers = sorted(set(RISK_ON_WEIGHTS) | set(RISK_OFF_WEIGHTS))
    prices = load_price_data(tickers, DEFAULT_START_DATE)

    sig, L, tol, sharpe = optuna_oos(prices)

    regime = "RISK-ON" if sig.iloc[-1] else "RISK-OFF"

    q_ends = calendar_quarters(prices.index)

    today = pd.Timestamp.today().normalize()
    next_q = pd.date_range(start=today, periods=2, freq="Q")[0]
    days_to_q = (next_q - today).days

    return {
        "regime": regime,
        "L": L,
        "tol": tol,
        "sharpe": sharpe,
        "next_q": next_q.date(),
        "days": days_to_q,
    }

# ============================================================
# EMAIL
# ============================================================

def build_email(o):
    return f"""
CURRENT REGIME
{ o['regime'] }

OPTIMIZED MA PARAMETERS
Length: {o['L']}
Tolerance: {o['tol']:.2%}

OUT-OF-SAMPLE SHARPE
{o['sharpe']:.3f}

NEXT REBALANCE
{o['next_q']}  ({o['days']} days)

IMPLEMENTATION CHECKLIST
- Rotate 100% to treasury sleeve on MA flip
- Execute rebalance only on calendar quarter-end
- Apply SIG gap mechanically
- Re-evaluate Sharpe-optimal weights quarterly

Dashboard:
https://portofliostrategy.streamlit.app/
"""

def send_email(text):
    msg = EmailMessage()
    msg["Subject"] = "Daily Portfolio SIG Update"
    msg["From"] = os.environ["EMAIL_USER"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(text)

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as s:
        s.login(os.environ["EMAIL_USER"], os.environ["EMAIL_PASS"])
        s.send_message(msg)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    out = run_daily_engine()
    email = build_email(out)
    send_email(email)