import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from scipy.optimize import minimize
import datetime
import os

# ============================================================
# CONFIG (IDENTICAL)
# ============================================================

DEFAULT_START_DATE = "1999-01-01"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "UGL": .25,
    "TQQQ": .30,
    "BTC-USD": .45,
}

RISK_OFF_WEIGHTS = {
    "SHY": 1.0,
}

FLIP_COST = 0.045
START_RISKY = 0.70
START_SAFE  = 0.30


# ============================================================
# DATA LOADING (Streamlit removed)
# ============================================================

def load_price_data(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
    else:
        px = data["Close"].copy()

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    return px.dropna(how="all")


# ============================================================
# BUILD PORTFOLIO INDEX — SIMPLE RETURNS
# ============================================================

def build_portfolio_index(prices, weights_dict):
    simple_rets = prices.pct_change().fillna(0)
    idx_rets = pd.Series(0.0, index=simple_rets.index)

    for a, w in weights_dict.items():
        if a in simple_rets.columns:
            idx_rets += simple_rets[a] * w

    return (1 + idx_rets).cumprod()


# ============================================================
# MA MATRIX
# ============================================================

def compute_ma_matrix(price_series, lengths, ma_type):
    ma_dict = {}
    if ma_type == "ema":
        for L in lengths:
            ma = price_series.ewm(span=L, adjust=False).mean()
            ma_dict[L] = ma.shift(1)
    else:
        for L in lengths:
            ma = price_series.rolling(window=L, min_periods=L).mean()
            ma_dict[L] = ma.shift(1)
    return ma_dict


# ============================================================
# TESTFOL SIGNAL LOGIC
# ============================================================

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
        if not sig[t - 1]:
            sig[t] = px[t] > upper[t]
        else:
            sig[t] = not (px[t] < lower[t])

    return pd.Series(sig, index=ma.index).fillna(False)


# ============================================================
# SIG ENGINE — IDENTICAL
# ============================================================

def run_sig_engine(
    risk_on_returns,
    risk_off_returns,
    target_quarter,
    ma_signal,
    pure_sig_rw=None,
    pure_sig_sw=None,
    flip_cost=FLIP_COST,
    quarter_end_dates=None
):

    dates = risk_on_returns.index
    n = len(dates)

    if quarter_end_dates is None:
        raise ValueError("quarter_end_dates must be supplied")

    quarter_end_set = set(quarter_end_dates)
    sig_arr = ma_signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    eq = 10000.0
    risky_val = eq * START_RISKY
    safe_val  = eq * START_SAFE

    frozen_risky = None
    frozen_safe  = None

    equity_curve = []
    risky_w_series = []
    safe_w_series = []
    risky_val_series = []
    safe_val_series = []
    rebalance_dates = []

    for i in range(n):
        date = dates[i]
        r_on = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])

        if ma_on:

            if frozen_risky is not None:
                w_r = pure_sig_rw.iloc[i]
                w_s = pure_sig_sw.iloc[i]
                risky_val = eq * w_r
                safe_val  = eq * w_s
                frozen_risky = None

            risky_val *= (1 + r_on)
            safe_val  *= (1 + r_off)

            if date in quarter_end_set:
                prev_qs = [qd for qd in quarter_end_dates if qd < date]
                if prev_qs:
                    prev_q = prev_qs[-1]
                    idx_prev = dates.get_loc(prev_q)
                    risky_at_qstart = risky_val_series[idx_prev]
                    goal_risky = risky_at_qstart * (1 + target_quarter)

                    if risky_val > goal_risky:
                        excess = risky_val - goal_risky
                        risky_val -= excess
                        safe_val  += excess
                        rebalance_dates.append(date)

                    elif risky_val < goal_risky:
                        needed = goal_risky - risky_val
                        move = min(needed, safe_val)
                        safe_val -= move
                        risky_val += move
                        rebalance_dates.append(date)

                    eq *= (1 - flip_cost * target_quarter)

            eq = risky_val + safe_val
            risky_w = risky_val / eq
            safe_w  = safe_val  / eq

            if flip_mask.iloc[i]:
                eq *= (1 - flip_cost)

        else:
            if frozen_risky is None:
                frozen_risky = risky_val
                frozen_safe  = safe_val

            eq *= (1 + r_off)
            risky_w = 0.0
            safe_w  = 1.0

        equity_curve.append(eq)
        risky_w_series.append(risky_w)
        safe_w_series.append(safe_w)
        risky_val_series.append(risky_val)
        safe_val_series.append(safe_val)

    return (
        pd.Series(equity_curve, index=dates),
        pd.Series(risky_w_series, index=dates),
        pd.Series(safe_w_series, index=dates),
        rebalance_dates
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

def build_weight_df(prices, signal, risk_on_weights, risk_off_weights):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for a, w in risk_on_weights.items():
        if a in prices.columns:
            weights.loc[signal, a] = w

    for a, w in risk_off_weights.items():
        if a in prices.columns:
            weights.loc[~signal, a] = w

    return weights


def compute_performance(simple_returns, eq_curve, rf=0.0):
    cagr = (eq_curve.iloc[-1] / eq_curve.iloc[0]) ** (252 / len(eq_curve)) - 1
    vol = simple_returns.std() * np.sqrt(252)
    sharpe = (simple_returns.mean() * 252 - rf) / vol if vol > 0 else np.nan
    dd = eq_curve / eq_curve.cummax() - 1

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": dd.min(),
        "TotalReturn": eq_curve.iloc[-1] / eq_curve.iloc[0] - 1,
        "DD_Series": dd
    }


def backtest(prices, signal, risk_on_weights, risk_off_weights, flip_cost):
    simple = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    flip_costs = np.where(flip_mask, -flip_cost, 0.0)
    strat_adj = strategy_simple + flip_costs

    eq = (1 + strat_adj).cumprod()

    return {
        "returns": strat_adj,
        "equity_curve": eq,
        "signal": signal,
        "weights": weights,
        "performance": compute_performance(strat_adj, eq),
        "flip_mask": flip_mask,
    }


# ============================================================
# GRID SEARCH (Streamlit removed, logic identical)
# ============================================================

def run_grid_search(prices, risk_on_weights, risk_off_weights, flip_cost):
    best_sharpe = -1e9
    best_cfg = None
    best_result = None
    best_trades = np.inf

    portfolio_index = build_portfolio_index(prices, risk_on_weights)

    lengths = list(range(21, 253))
    types = ["sma", "ema"]
    tolerances = np.arange(0.0, .0501, .002)

    # removed progress bar

    ma_cache = {t: compute_ma_matrix(portfolio_index, lengths, t) for t in types}

    for ma_type in types:
        for length in lengths:
            ma = ma_cache[ma_type][length]
            for tol in tolerances:
                signal = generate_testfol_signal_vectorized(portfolio_index, ma, tol)
                result = backtest(prices, signal, risk_on_weights, risk_off_weights, flip_cost)

                sig_arr = signal.astype(int)
                switches = sig_arr.diff().abs().sum()
                trades_per_year = switches / (len(sig_arr) / 252)
                sharpe = result["performance"]["Sharpe"]

                if sharpe > best_sharpe or (
                    sharpe == best_sharpe and trades_per_year < best_trades
                ):
                    best_sharpe = sharpe
                    best_trades = trades_per_year
                    best_cfg = (length, ma_type, tol)
                    best_result = result

    return best_cfg, best_result
# ============================================================
# QUARTERLY PROGRESS HELPER
# ============================================================

def compute_quarter_progress(risky_start, risky_today, quarterly_target):
    target_risky = risky_start * (1 + quarterly_target)
    gap = target_risky - risky_today
    pct_gap = gap / risky_start if risky_start > 0 else 0

    return {
        "Deployed Capital at Last Rebalance ($)": risky_start,
        "Deployed Capital Today ($)": risky_today,
        "Deployed Capital Target Next Rebalance ($)": target_risky,
        "Gap ($)": gap,
        "Gap (%)": pct_gap,
    }


def normalize(eq):
    return eq / eq.iloc[0] * 10000


# ============================================================
# MAIN DAILY ENGINE (Streamlit removed)
# ============================================================

def run_daily_engine():

    # ------------- identical defaults -------------
    start = DEFAULT_START_DATE
    end_val = None

    risk_on_tickers = list(RISK_ON_WEIGHTS.keys())
    risk_on_weights = RISK_ON_WEIGHTS.copy()

    risk_off_tickers = list(RISK_OFF_WEIGHTS.keys())
    risk_off_weights = RISK_OFF_WEIGHTS.copy()

    qs_cap_1 = 75815.26
    qs_cap_2 = 10074.83
    qs_cap_3 = 4189.76

    real_cap_1 = 73165.78
    real_cap_2 = 9264.46
    real_cap_3 = 4191.56

    # --------------------------------------------------------

    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers))
    prices = load_price_data(all_tickers, start, end_val).dropna(how="any")

    # ============================================================
    # MA GRID SEARCH (IDENTICAL)
    # ============================================================

    best_cfg, best_result = run_grid_search(
        prices, risk_on_weights, risk_off_weights, FLIP_COST
    )
    best_len, best_type, best_tol = best_cfg
    sig = best_result["signal"]
    perf = best_result["performance"]

    latest_signal = sig.iloc[-1]
    current_regime = "RISK-ON" if latest_signal else "RISK-OFF"

    # MA line
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    opt_ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]

    # ============================================================
    # RISK-ON PERFORMANCE (unchanged)
    # ============================================================

    simple_rets = prices.pct_change().fillna(0)
    risk_on_simple = pd.Series(0.0, index=simple_rets.index)

    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_on_eq = (1 + risk_on_simple).cumprod()

    # risk-on μ and cov
    risk_on_px = prices[risk_on_tickers].dropna()
    risk_on_rets = risk_on_px.pct_change().dropna()

    mu = risk_on_rets.mean().values
    cov = risk_on_rets.cov().values + np.eye(len(mu)) * 1e-10

    def neg_sharpe(w):
        r = np.dot(mu, w)
        v = np.sqrt(np.dot(w.T, cov @ w))
        return -(r / v) if v > 0 else 1e9

    n = len(mu)
    bounds = [(0, 1)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    res = minimize(neg_sharpe, np.ones(n) / n, bounds=bounds, constraints=cons)
    w_opt = res.x

    sharp_returns = (risk_on_rets * w_opt).sum(axis=1)
    sharp_eq = (1 + sharp_returns).cumprod()

    # ============================================================
    # TRUE CALENDAR QUARTER-ENDS (IDENTICAL)
    # ============================================================

    dates = prices.index

    true_q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq='Q')
    mapped_q_ends = []

    for qd in true_q_ends:
        mapped_q_ends.append(dates[dates <= qd].max())

    mapped_q_ends = pd.to_datetime(mapped_q_ends)

    today_date = pd.Timestamp.today().normalize()

    true_next_q = pd.date_range(start=today_date, periods=2, freq="Q")[0]
    next_q_end = true_next_q

    true_prev_q = pd.date_range(end=today_date, periods=2, freq="Q")[0]
    past_q_end = true_prev_q

    days_to_next_q = (next_q_end - today_date).days

    # ============================================================
    # HYBRID SIG ENGINE (IDENTICAL)
    # ============================================================

    bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
    quarterly_target = (1 + bh_cagr) ** (1/4) - 1

    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)

    pure_sig_eq, pure_sig_rw, pure_sig_sw, pure_sig_rebals = run_sig_engine(
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
        pure_sig_rw=pure_sig_rw,
        pure_sig_sw=pure_sig_sw,
        quarter_end_dates=mapped_q_ends
    )

    # ============================================================
    # QUARTER PROGRESS (IDENTICAL)
    # ============================================================

    if len(hybrid_rebals) > 0:
        quarter_start_date = hybrid_rebals[-1]
    else:
        quarter_start_date = dates[0]

    def get_sig_progress(qs_cap, today_cap):
        risky_start = qs_cap * float(hybrid_rw.loc[quarter_start_date])
        risky_today = today_cap * float(hybrid_rw.iloc[-1])
        return compute_quarter_progress(risky_start, risky_today, quarterly_target)

    prog_1 = get_sig_progress(qs_cap_1, real_cap_1)
    prog_2 = get_sig_progress(qs_cap_2, real_cap_2)
    prog_3 = get_sig_progress(qs_cap_3, real_cap_3)

    # ============================================================
    # BUILD OUTPUT DICTIONARY FOR EMAIL
    # ============================================================

    outputs = {
        "regime": current_regime,
        "next_rebalance": next_q_end.date(),
        "days_to_rebalance": days_to_next_q,
        "quarterly_progress": {
            "Taxable": prog_1,
            "Tax-Sheltered": prog_2,
            "Joint": prog_3,
        }   
    }

    return outputs
# ============================================================
# EMAIL BUILDER
# ============================================================

def build_email_text(outputs):
    regime = outputs["regime"]
    next_reb = outputs["next_rebalance"]
    days = outputs["days_to_rebalance"]
    prog = outputs["quarterly_progress"]

    t1 = prog["Taxable"]
    t2 = prog["Tax-Sheltered"]
    t3 = prog["Joint"]

    body = f"""
Current Regime: {regime}

Next Rebalance (Calendar Quarter-End): {next_reb}  ({days} days from today)

IMPLEMENTATION CHECKLIST
- Rotate 100% of portfolio to treasury sleeve whenever the MA regime flips.
- At each calendar quarter-end, input your portfolio value at last rebalance & today's portfolio value.
- Execute the exact dollar adjustment recommended by the model (increase/decrease deployed sleeve) on the rebalance date.
- At each rebalance, re-evaluate the Sharpe-optimal portfolio weighting.


Dashboard Link:
https://portofliostrategy.streamlit.app/
"""
    return body


# ============================================================
# EMAIL SENDER (using GitHub Action secrets)
# ============================================================

import smtplib
from email.message import EmailMessage
import ssl
import os

def send_email(body_text):

    msg = EmailMessage()
    msg["Subject"] = "Daily Portfolio Update"
    msg["From"] = os.environ["EMAIL_USER"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(body_text)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(os.environ["EMAIL_USER"], os.environ["EMAIL_PASS"])
        smtp.send_message(msg)


# ============================================================
# MAIN RUNNER
# ============================================================

if __name__ == "__main__":
    outputs = run_daily_engine()

    body_text = build_email_text(outputs)

    send_email(body_text)
