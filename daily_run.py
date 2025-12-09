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
# DAILY RUN PIPELINE (Option A — hard-coded default values)
# ============================================================

def run_daily():
    # ---------------------------------------------
    # 1. Load data identical to main app
    # ---------------------------------------------
    tickers = list(RISK_ON_WEIGHTS.keys()) + list(RISK_OFF_WEIGHTS.keys())
    prices = load_price_data(tickers, DEFAULT_START_DATE)

    # Simple returns
    simple_rets = prices.pct_change().fillna(0)

    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in RISK_ON_WEIGHTS.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in RISK_OFF_WEIGHTS.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    # ---------------------------------------------
    # 2. Run MA grid search EXACTLY like main
    # ---------------------------------------------
    best_cfg, best_result = run_grid_search(
        prices, RISK_ON_WEIGHTS, RISK_OFF_WEIGHTS, FLIP_COST
    )
    best_len, best_type, best_tol = best_cfg
    sig = best_result["signal"]

    # Current regime
    current_regime = "RISK-ON" if sig.iloc[-1] else "RISK-OFF"

    # ---------------------------------------------
    # 3. Build portfolio index and optimal MA
    # ---------------------------------------------
    portfolio_index = build_portfolio_index(prices, RISK_ON_WEIGHTS)
    opt_ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]

    # ---------------------------------------------
    # 4. TRUE CALENDAR QUARTER LOGIC (identical)
    # ---------------------------------------------
    dates = prices.index

    true_q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq='Q')

    mapped_q_ends = []
    for qd in true_q_ends:
        mapped_q_ends.append(dates[dates <= qd].max())
    mapped_q_ends = pd.to_datetime(mapped_q_ends)

    today_date = pd.Timestamp.today().normalize()

    # next calendar quarter end
    next_q_end = pd.date_range(start=today_date, periods=2, freq="Q")[0]
    # last completed quarter end
    last_q_end = pd.date_range(end=today_date, periods=2, freq="Q")[0]

    days_to_next_q = (next_q_end - today_date).days

    # ---------------------------------------------
    # 5. Quarterly target identical to main code
    # ---------------------------------------------
    risk_on_eq = (1 + risk_on_simple).cumprod()
    bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
    quarterly_target = (1 + bh_cagr) ** (1/4) - 1

    # ---------------------------------------------
    # 6. PURE SIG + HYBRID SIG (identical logic)
    # ---------------------------------------------
    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)

    pure_eq, pure_rw, pure_sw, pure_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        pure_sig_signal,
        quarter_end_dates=mapped_q_ends
    )

    hybrid_eq, hybrid_rw, hybrid_sw, hybrid_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        sig,
        pure_sig_rw=pure_rw,
        pure_sig_sw=pure_sw,
        quarter_end_dates=mapped_q_ends
    )

    # Last hybrid SIG rebalance date
    if len(hybrid_rebals) > 0:
        last_reb_date = hybrid_rebals[-1].strftime("%Y-%m-%d")
        quarter_start_date = hybrid_rebals[-1]
    else:
        last_reb_date = "None yet"
        quarter_start_date = dates[0]

    # ---------------------------------------------
    # 7. Hard-coded default account values (Option A)
    # ---------------------------------------------
    qs_cap_1 = 75815.26
    qs_cap_2 = 10074.83
    qs_cap_3 = 4189.76

    real_cap_1 = 73165.78
    real_cap_2 = 9264.46
    real_cap_3 = 4191.56

    # ---------------------------------------------
    # 8. Compute quarter progress EXACTLY like the app
    # ---------------------------------------------
    def get_sig_progress(qs_cap, today_cap):
        risky_start = qs_cap * float(hybrid_rw.loc[quarter_start_date])
        risky_today = today_cap * float(hybrid_rw.iloc[-1])
        return compute_quarter_progress(risky_start, risky_today, quarterly_target)

    prog1 = get_sig_progress(qs_cap_1, real_cap_1)
    prog2 = get_sig_progress(qs_cap_2, real_cap_2)
    prog3 = get_sig_progress(qs_cap_3, real_cap_3)

    # ---------------------------------------------
    # 9. Build equity curve PNG for email
    # ---------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hybrid_eq / hybrid_eq.iloc[0] * 10000, label="Hybrid SIG", linewidth=2)
    ax.set_title("Hybrid SIG Equity Curve")
    ax.grid(alpha=0.3)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # ---------------------------------------------
    # 10. Email text — now includes last + next rebalance
    # ---------------------------------------------
    text_output = f"""
Current MA Regime: {current_regime}

Last Rebalance: {last_reb_date}
Next Rebalance (Calendar Quarter-End): {next_q_end.date()}  ({days_to_next_q} days)

Quarterly Target Growth Rate: {quarterly_target:.2%}

--- ACCOUNT PROGRESS ---

Taxable:
    Gap: ${prog1["Gap ($)"]:.2f}
    Target Next Rebalance: ${prog1["Deployed Capital Target Next Rebalance ($)"]:.2f}

Tax-Sheltered:
    Gap: ${prog2["Gap ($)"]:.2f}
    Target Next Rebalance: ${prog2["Deployed Capital Target Next Rebalance ($)"]:.2f}

Joint:
    Gap: ${prog3["Gap ($)"]:.2f}
    Target Next Rebalance: ${prog3["Deployed Capital Target Next Rebalance ($)"]:.2f}

--- IMPLEMENTATION CHECKLIST ---
- Rebalance whenever the MA regime flips.
- At each calendar quarter-end, use LAST REBALANCE deployed capital × today's risky weight.
- Execute the exact dollar increase/decrease on the rebalance date.
- No intra-quarter rebalancing unless MA flips.
- Let weights drift day by day.
"""

    return text_output, buf

# ============================================================
# EMAIL SENDER (GitHub Action fills env vars)
# ============================================================

def send_email(body, png_buf):

    msg = EmailMessage()
    msg["Subject"] = "Daily Hybrid SIG Update"
    msg["From"] = os.environ["EMAIL_USER"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(body)

    msg.add_attachment(
        png_buf.read(),
        maintype="image",
        subtype="png",
        filename="hybrid_sig_equity_curve.png"
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.environ["EMAIL_USER"], os.environ["EMAIL_PASS"])
        smtp.send_message(msg)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    body, png_buf = run_daily()
    send_email(body, png_buf)

