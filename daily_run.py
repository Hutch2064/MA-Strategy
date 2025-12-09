# ============================================================
# daily_run.py — Hybrid SIG Daily Update (Standalone Version)
# Fully replicates the logic of the main Streamlit app,
# but outputs ONLY: Current MA Regime + Implementation Checklist
# + Hybrid SIG equity curve PNG for email attachment.
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import datetime
import smtplib
from email.message import EmailMessage
import io
import os


# ============================================================
# CONFIG (identical to main)
# ============================================================

DEFAULT_START_DATE = "1999-01-01"

RISK_ON_WEIGHTS = {
    "UGL": 0.25,
    "TQQQ": 0.30,
    "BTC-USD": 0.45,
}

RISK_OFF_WEIGHTS = {
    "SHY": 1.0,
}

FLIP_COST = 0.045

START_RISKY = 0.70
START_SAFE = 0.30


# ============================================================
# DATA LOADING
# ============================================================

def load_price_data(tickers, start_date):
    data = yf.download(tickers, start=start_date, progress=False)
    if "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
    else:
        px = data["Close"].copy()

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px.dropna(how="all")


# ============================================================
# SIMPLE PORTFOLIO INDEX
# ============================================================

def build_portfolio_index(prices, weights):
    rets = prices.pct_change().fillna(0)
    idx = pd.Series(0.0, index=rets.index)
    for a, w in weights.items():
        if a in rets.columns:
            idx += rets[a] * w
    return (1 + idx).cumprod()


# ============================================================
# MA MATRIX
# ============================================================

def compute_ma_matrix(price_series, lengths, ma_type):
    ma_dict = {}
    if ma_type == "ema":
        for L in lengths:
            ma = price_series.ewm(span=L, adjust=False).mean().shift(1)
            ma_dict[L] = ma
    else:
        for L in lengths:
            ma = price_series.rolling(L, min_periods=L).mean().shift(1)
            ma_dict[L] = ma
    return ma_dict


# ============================================================
# TESTFOL SIGNAL (MA with tolerance)
# ============================================================

def generate_testfol_signal(price, ma, tol):
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


# ============================================================
# SIG ENGINE — IDENTICAL TO MAIN
# ============================================================

def run_sig_engine(risk_on, risk_off, target_q, ma_signal, pure_rw=None, pure_sw=None, quarter_ends=None):
    dates = risk_on.index
    n = len(dates)

    sig_arr = ma_signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    eq = 10000.0
    risky_val = eq * START_RISKY
    safe_val = eq * START_SAFE

    frozen_r = None
    frozen_s = None

    eq_curve = []
    risky_w = []
    safe_w = []
    risky_val_series = []
    safe_val_series = []
    rebals = []

    qset = set(quarter_ends)

    for i in range(n):
        date = dates[i]
        r_on = risk_on.iloc[i]
        r_off = risk_off.iloc[i]
        ma_on = bool(ma_signal.iloc[i])

        if ma_on:

            if frozen_r is not None:
                risky_val = eq * pure_rw.iloc[i]
                safe_val = eq * pure_sw.iloc[i]
                frozen_r = None
                frozen_s = None

            risky_val *= (1 + r_on)
            safe_val *= (1 + r_off)

            if date in qset:
                prev_qs = [qd for qd in quarter_ends if qd < date]
                if prev_qs:
                    prev_q = prev_qs[-1]
                    idx_prev = dates.get_loc(prev_q)
                    risky_qstart = risky_val_series[idx_prev]
                    goal = risky_qstart * (1 + target_q)

                    if risky_val > goal:
                        excess = risky_val - goal
                        risky_val -= excess
                        safe_val += excess
                        rebals.append(date)
                    elif risky_val < goal:
                        need = goal - risky_val
                        move = min(need, safe_val)
                        safe_val -= move
                        risky_val += move
                        rebals.append(date)

                    eq *= (1 - FLIP_COST * target_q)

            eq = risky_val + safe_val
            rw = risky_val / eq
            sw = safe_val / eq

            if flip_mask.iloc[i]:
                eq *= (1 - FLIP_COST)

        else:
            if frozen_r is None:
                frozen_r = risky_val
                frozen_s = safe_val
            eq *= (1 + r_off)
            rw = 0.0
            sw = 1.0

        eq_curve.append(eq)
        risky_w.append(rw)
        safe_w.append(sw)
        risky_val_series.append(risky_val)
        safe_val_series.append(safe_val)

    return (
        pd.Series(eq_curve, index=dates),
        pd.Series(risky_w, index=dates),
        pd.Series(safe_w, index=dates),
        rebals
    )


# ============================================================
# DAILY RUN PIPELINE
# ============================================================

def run_daily():
    tickers = list(RISK_ON_WEIGHTS.keys()) + list(RISK_OFF_WEIGHTS.keys())
    prices = load_price_data(tickers, DEFAULT_START_DATE)

    # GRID SEARCH (identical to main)
    portfolio_index = build_portfolio_index(prices, RISK_ON_WEIGHTS)

    lengths = range(21, 253)
    types = ["sma", "ema"]
    tolerances = np.arange(0.0, 0.0501, 0.002)

    best_sharpe = -1e9
    best_cfg = None
    best_signal = None
    best_result_eq = None

    simple_rets = prices.pct_change().fillna(0)
    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in RISK_ON_WEIGHTS.items():
        risk_on_simple += simple_rets[a] * w

    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in RISK_OFF_WEIGHTS.items():
        risk_off_daily += simple_rets[a] * w

    for t in types:
        ma_cache = compute_ma_matrix(portfolio_index, lengths, t)
        for L in lengths:
            ma = ma_cache[L]
            for tol in tolerances:
                sig = generate_testfol_signal(portfolio_index, ma, tol)

                strat_returns = (sig.shift(1).fillna(False) * risk_on_simple + 
                                (~sig.shift(1).fillna(True)) * risk_off_daily)
                eq = (1 + strat_returns).cumprod()

                cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1
                vol = strat_returns.std() * np.sqrt(252)
                sharpe = cagr / vol if vol > 0 else -1e9

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_cfg = (L, t, tol)
                    best_signal = sig
                    best_result_eq = eq

    best_len, best_type, best_tol = best_cfg

    # MA Regime
    current_regime = "RISK-ON" if best_signal.iloc[-1] else "RISK-OFF"

    # Build MA for hybrid SIG engine
    ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]

    # Quarter ends
    dates = prices.index
    q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq="Q")
    q_ends = pd.to_datetime([dates[dates <= q].max() for q in q_ends])

    # Compute quarterly target
    risk_on_eq = (1 + risk_on_simple).cumprod()
    bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
    quarterly_target = (1 + bh_cagr) ** (1/4) - 1

    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)
    pure_eq, pure_rw, pure_sw, _ = run_sig_engine(
        risk_on_simple, risk_off_daily, quarterly_target,
        pure_sig_signal, quarter_ends=q_ends
    )

    hybrid_eq, rw, sw, rebals = run_sig_engine(
        risk_on_simple, risk_off_daily, quarterly_target,
        best_signal, pure_rw=pure_rw, pure_sw=pure_sw,
        quarter_ends=q_ends
    )

    # Plot equity curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hybrid_eq / hybrid_eq.iloc[0] * 10000, label="Hybrid SIG")
    ax.set_title("Hybrid SIG Equity Curve")
    ax.grid(alpha=0.3)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Email body
    text_output = f"""
Current MA Regime: {current_regime}

Implementation Checklist:
- Rebalance whenever the MA regime flips.
- At each calendar quarter-end, input your portfolio value at last rebalance & today's portfolio value.
- Execute the exact dollar adjustment recommended by the model on the rebalance date.
- Do NOT rebalance intra-quarter unless the MA signal flips.
- Let weights drift naturally day by day — this is part of the model.
"""

    return text_output, buf


# ============================================================
# EMAIL SENDER (GitHub Action fills env vars)
# ============================================================

def send_email(body, png_buf):

    msg = EmailMessage()
    msg["Subject"] = "Daily Hybrid SIG Update"
    msg["From"] = os.environ["EMAIL_FROM"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(body)

    msg.add_attachment(
        png_buf.read(),
        maintype="image",
        subtype="png",
        filename="hybrid_sig_equity_curve.png"
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.environ["EMAIL_FROM"], os.environ["EMAIL_PASS"])
        smtp.send_message(msg)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    body, png_buf = run_daily()
    send_email(body, png_buf)

