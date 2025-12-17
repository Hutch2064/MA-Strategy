# ============================================================
# DAILY HYBRID SIG RUN — CANONICAL VERSION
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import optuna
import datetime
import smtplib
from email.mime.text import MIMEText

# ============================================================
# CONFIG (IDENTICAL)
# ============================================================

DEFAULT_START_DATE = "2000-01-01"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "BTC-USD": 0.5,
    "FNGO": 0.5,
}

RISK_OFF_WEIGHTS = {
    "SHY": 0.45,
    "GLD": 0.55,
}

FLIP_COST = 0.00
START_RISKY = 0.70
START_SAFE  = 0.30

# ============================================================
# DATA LOADING (IDENTICAL)
# ============================================================

def load_price_data(tickers, start_date):
    data = yf.download(tickers, start=start_date, progress=False)
    px = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px.dropna(how="any")

# ============================================================
# BUILD PORTFOLIO INDEX (IDENTICAL)
# ============================================================

def build_portfolio_index(prices, weights):
    rets = prices.pct_change().fillna(0)
    out = pd.Series(0.0, index=rets.index)
    for a, w in weights.items():
        if a in rets.columns:
            out += rets[a] * w
    return (1 + out).cumprod()

# ============================================================
# MA + SIGNAL (IDENTICAL)
# ============================================================

def compute_ma_matrix(px, lengths):
    return {L: px.rolling(L, min_periods=1).mean().shift(1) for L in lengths}

def generate_testfol_signal_vectorized(px, ma, tol):
    upper = ma * (1 + tol)
    lower = ma * (1 - tol)
    sig = np.zeros(len(px), dtype=bool)
    for i in range(1, len(px)):
        if not sig[i-1]:
            sig[i] = px.iloc[i] > upper.iloc[i]
        else:
            sig[i] = not (px.iloc[i] < lower.iloc[i])
    return pd.Series(sig, index=px.index)

# ============================================================
# PERFORMANCE (IDENTICAL)
# ============================================================

def compute_performance(returns, eq):
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1
    vol  = returns.std() * np.sqrt(252)
    sharpe = returns.mean() * 252 / vol if vol > 0 else 0
    dd = eq / eq.cummax() - 1
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": dd.min(),
        "TotalReturn": eq.iloc[-1] / eq.iloc[0] - 1,
        "DD_Series": dd,
    }

# ============================================================
# SIG ENGINE (IDENTICAL)
# ============================================================

def run_sig_engine(
    risk_on_returns,
    risk_off_returns,
    quarterly_target,
    ma_signal,
    pure_sig_rw=None,
    pure_sig_sw=None,
    quarter_end_dates=None,
    quarterly_multiplier=2.0,
    ma_flip_multiplier=4.0,
):

    dates = risk_on_returns.index
    eq = 10000.0
    risky_val = eq * START_RISKY
    safe_val  = eq * START_SAFE

    eq_curve = []
    risky_w = []
    safe_w  = []

    sig = ma_signal.astype(int)
    flip_mask = sig.diff().abs() == 1
    q_set = set(quarter_end_dates)

    frozen_r = frozen_s = None

    for i, d in enumerate(dates):
        r_on  = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])

        if i > 0 and flip_mask.iloc[i]:
            eq *= (1 - FLIP_COST * ma_flip_multiplier)

        if ma_on:
            if frozen_r is not None:
                risky_val = eq * pure_sig_rw.iloc[i]
                safe_val  = eq * pure_sig_sw.iloc[i]
                frozen_r = frozen_s = None

            risky_val *= (1 + r_on)
            safe_val  *= (1 + r_off)

            if d in q_set:
                eq *= (1 - FLIP_COST * quarterly_multiplier)

            eq = risky_val + safe_val
            rw = risky_val / eq
            sw = safe_val / eq
        else:
            if frozen_r is None:
                frozen_r, frozen_s = risky_val, safe_val
            eq *= (1 + r_off)
            rw, sw = 0.0, 1.0

        eq_curve.append(eq)
        risky_w.append(rw)
        safe_w.append(sw)

    return (
        pd.Series(eq_curve, index=dates),
        pd.Series(risky_w, index=dates),
        pd.Series(safe_w, index=dates),
    )

# ============================================================
# OPTUNA — MA PARAMS ONLY (IDENTICAL OBJECTIVE)
# ============================================================

def optuna_ma_params(prices, n_trials=300):
    idx = build_portfolio_index(prices, RISK_ON_WEIGHTS)
    train, test = idx[:-756], idx[-756:]

    def obj(trial):
        L = trial.suggest_int("L", 100, 252)
        tol = trial.suggest_float("tol", 0.01, 0.03)
        ma = compute_ma_matrix(train, [L])[L]
        sig = generate_testfol_signal_vectorized(train, ma, tol)
        rets = train.pct_change().fillna(0)[sig]
        return -rets.mean() / rets.std() if rets.std() > 0 else 1e9

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=n_trials)
    return study.best_params["L"], study.best_params["tol"]

# ============================================================
# DAILY RUN
# ============================================================

def main():

    tickers = list(RISK_ON_WEIGHTS) + list(RISK_OFF_WEIGHTS)
    prices = load_price_data(tickers, DEFAULT_START_DATE)

    # MA params
    L, tol = optuna_ma_params(prices)

    # MA signal
    idx = build_portfolio_index(prices, RISK_ON_WEIGHTS)
    ma = compute_ma_matrix(idx, [L])[L]
    sig = generate_testfol_signal_vectorized(idx, ma, tol)

    # Quarterly logic
    q_ends = pd.date_range(prices.index.min(), prices.index.max(), freq="Q")
    q_ends = [prices.index[prices.index <= q].max() for q in q_ends]

    # Quarterly target
    ron = sum(prices[a].pct_change().fillna(0) * w for a, w in RISK_ON_WEIGHTS.items())
    roff = sum(prices[a].pct_change().fillna(0) * w for a, w in RISK_OFF_WEIGHTS.items())
    bh_cagr = (1 + ron).prod() ** (252 / len(ron)) - 1
    quarterly_target = (1 + bh_cagr) ** 0.25 - 1

    # PURE SIG
    pure_eq, pure_rw, pure_sw = run_sig_engine(
        ron, roff, quarterly_target, pd.Series(True, index=sig.index),
        quarter_end_dates=q_ends, ma_flip_multiplier=0.0
    )

    # HYBRID SIG (THIS IS THE PORTFOLIO)
    hybrid_eq, hybrid_rw, hybrid_sw = run_sig_engine(
        ron, roff, quarterly_target, sig,
        pure_rw, pure_sw, q_ends
    )

    hybrid_returns = hybrid_eq.pct_change().fillna(0)
    perf = compute_performance(hybrid_returns, hybrid_eq)

    print("=== DAILY HYBRID SIG ===")
    for k, v in perf.items():
        if k != "DD_Series":
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()