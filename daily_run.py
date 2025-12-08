import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import datetime

# ============================================================
# CONFIG — MATCH MAIN STREAMLIT APP EXACTLY
# ============================================================

DEFAULT_START_DATE = "1999-01-01"
RISK_FREE_RATE = 0.0
FLIP_COST = 0.045
QUARTER_DAYS = 63

RISK_ON_WEIGHTS = {
    "UGL": .25,
    "TQQQ": .30,
    "BTC-USD": .45,
}

RISK_OFF_WEIGHTS = {
    "SHY": 1.0,
}

START_RISKY = 0.60
START_SAFE  = 0.40


# ============================================================
# DATA LOADING
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
# PORTFOLIO INDEX — SIMPLE RETURNS
# ============================================================

def build_portfolio_index(prices, weights_dict):
    simple_rets = prices.pct_change().fillna(0)
    idx_rets = pd.Series(0.0, index=simple_rets.index)

    for a, w in weights_dict.items():
        if a in simple_rets.columns:
            idx_rets += simple_rets[a] * w

    return (1 + idx_rets).cumprod()


# ============================================================
# MOVING AVERAGE MATRIX
# ============================================================

def compute_ma_matrix(price_series, lengths, ma_type):
    out = {}
    if ma_type == "ema":
        for L in lengths:
            ma = price_series.ewm(span=L, adjust=False).mean()
            out[L] = ma.shift(1)
    else:
        for L in lengths:
            ma = price_series.rolling(window=L, min_periods=L).mean()
            out[L] = ma.shift(1)
    return out


# ============================================================
# TESTFOL SIGNAL ENGINE (HYSTERESIS)
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
        if not sig[t-1]:
            sig[t] = px[t] > upper[t]
        else:
            sig[t] = not (px[t] < lower[t])

    return pd.Series(sig, index=ma.index).fillna(False)


# ============================================================
# BACKTEST ENGINE FOR MA STRATEGY
# ============================================================

def build_weight_df(prices, signal, ron, roff):
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for a, pct in ron.items():
        if a in w.columns:
            w.loc[signal, a] = pct

    for a, pct in roff.items():
        if a in w.columns:
            w.loc[~signal, a] = pct

    return w


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
        "DD_Series": dd,
    }


def backtest(prices, signal, ron, roff, flip_cost):
    simple = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, ron, roff)

    strat_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    flips = signal.astype(int).diff().abs() == 1
    flip_hits = np.where(flips, -flip_cost, 0.0)

    adj = strat_simple + flip_hits
    eq = (1 + adj).cumprod()

    return {
        "returns": adj,
        "equity_curve": eq,
        "signal": signal,
        "flip_mask": flips,
        "performance": compute_performance(adj, eq),
    }


# ============================================================
# GRID SEARCH — IDENTICAL TO MAIN APP
# ============================================================

def run_grid_search(prices, ron, roff, flip_cost):
    best_sharpe = -1e12
    best_cfg = None
    best_res = None
    best_trades = np.inf

    portfolio_index = build_portfolio_index(prices, ron)
    lengths = list(range(21, 253))
    types = ["sma", "ema"]
    tolerances = np.arange(0.0, 0.0501, 0.002)

    ma_cache = {t: compute_ma_matrix(portfolio_index, lengths, t) for t in types}

    for ma_type in types:
        for L in lengths:
            ma = ma_cache[ma_type][L]

            for tol in tolerances:
                sig = generate_testfol_signal_vectorized(portfolio_index, ma, tol)
                res = backtest(prices, sig, ron, roff, flip_cost)

                sig_arr = sig.astype(int)
                trades = sig_arr.diff().abs().sum() / (len(sig_arr) / 252)

                sharpe = res["performance"]["Sharpe"]

                if (sharpe > best_sharpe) or (sharpe == best_sharpe and trades < best_trades):
                    best_sharpe = sharpe
                    best_trades = trades
                    best_cfg = (L, ma_type, tol)
                    best_res = res

    return best_cfg, best_res
# ============================================================
# SIG ENGINE — PURE SIG + HYBRID SIG (IDENTICAL TO MAIN APP)
# ============================================================

def run_sig_engine(
    risk_on_returns,
    risk_off_returns,
    target_quarter,
    ma_signal,
    pure_sig_rw=None,
    pure_sig_sw=None,
    flip_cost=FLIP_COST
):
    dates = risk_on_returns.index
    n = len(dates)

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
    rebalance_events = 0

    for i in range(n):
        r_on  = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])

        if ma_on:

            if frozen_risky is not None:
                w_r = pure_sig_rw.iloc[i]
                w_s = pure_sig_sw.iloc[i]
                risky_val = eq * w_r
                safe_val  = eq * w_s
                frozen_risky = None
                frozen_safe  = None

            risky_val *= (1 + r_on)
            safe_val  *= (1 + r_off)

            if i >= QUARTER_DAYS and (i % QUARTER_DAYS == 0):

                past_risky_val = risky_val_series[i - QUARTER_DAYS]
                goal_risky = past_risky_val * (1 + target_quarter)

                if risky_val > goal_risky:
                    excess = risky_val - goal_risky
                    risky_val -= excess
                    safe_val  += excess
                    rebalance_events += 1

                elif risky_val < goal_risky:
                    needed = goal_risky - risky_val
                    move = min(needed, safe_val)
                    safe_val  -= move
                    risky_val += move
                    rebalance_events += 1

                quarter_fee = flip_cost * target_quarter
                eq *= (1 - quarter_fee)

            eq = risky_val + safe_val

            risky_w = risky_val / eq
            safe_w  = safe_val / eq

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
        rebalance_events
    )


# ============================================================
# QUARTERLY PROGRESS HELPER
# ============================================================

def compute_quarter_progress(risky_start, risky_today, quarterly_target):
    target_risky = risky_start * (1 + quarterly_target)
    gap = target_risky - risky_today
    pct_gap = gap / risky_start if risky_start > 0 else 0

    return {
        "Implied Deployed Capital at Qtr Start ($)": risky_start,
        "Implied Deployed Capital Today ($)": risky_today,
        "Deployed Capital Target Qtr End ($)": target_risky,
        "Gap ($)": gap,
        "Gap (%)": pct_gap,
    }


# ============================================================
# ADVANCED METRICS — MATCH MAIN APP
# ============================================================

def time_in_drawdown(dd): 
    return (dd < 0).mean()

def ulcer(dd): 
    return np.sqrt((dd**2).mean()) if (dd**2).mean() != 0 else np.nan

def pain_gain(cagr, dd): 
    u = ulcer(dd)
    return cagr / u if u != 0 else np.nan

def mar_ratio(cagr, maxdd): 
    return cagr / abs(maxdd) if maxdd != 0 else np.nan


def compute_stats(perf, returns, dd, flips, tpy):
    return {
        "CAGR": perf["CAGR"],
        "Volatility": perf["Volatility"],
        "Sharpe": perf["Sharpe"],
        "MaxDD": perf["MaxDrawdown"],
        "Total": perf["TotalReturn"],
        "MAR": mar_ratio(perf["CAGR"], perf["MaxDrawdown"]),
        "TID": time_in_drawdown(dd),
        "PainGain": pain_gain(perf["CAGR"], dd),
        "Skew": returns.skew(),
        "Kurtosis": returns.kurt(),
        "Trades/year": tpy,
    }


# ============================================================
# SHARPE OPTIMAL PORTFOLIO (IDENTICAL TO MAIN APP)
# ============================================================

def compute_sharpe_optimal(prices, ron_weights):
    tickers = [t for t in ron_weights.keys() if t in prices.columns]
    px = prices[tickers].dropna()
    rets = px.pct_change().dropna()

    mu = rets.mean().values
    cov = rets.cov().values + np.eye(len(mu)) * 1e-10

    def neg_sharpe(w):
        r = np.dot(mu, w)
        v = np.sqrt(np.dot(w.T, cov @ w))
        if v == 0: 
            return 1e9
        return -(r / v)

    n = len(mu)
    bounds = [(0, 1)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    res = minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=cons)
    w_opt = res.x

    sharp_returns = (rets * w_opt).sum(axis=1)
    sharp_eq = (1 + sharp_returns).cumprod()
    sharp_perf = compute_performance(sharp_returns, sharp_eq)

    return tickers, w_opt, sharp_returns, sharp_eq, sharp_perf


# ============================================================
# REGIME STATS (IDENTICAL TO MAIN APP)
# ============================================================

def compute_regime_stats(signal):
    sig_int = signal.astype(int)
    flips = sig_int.diff().fillna(0).ne(0)

    segments = []
    current = sig_int.iloc[0]
    seg_start = sig_int.index[0]

    for date, sw in flips.iloc[1:].items():
        if sw:
            segments.append((current, seg_start, date))
            current = sig_int.loc[date]
            seg_start = date

    segments.append((current, seg_start, sig_int.index[-1]))

    out = []
    for r, s, e in segments:
        out.append([
            "RISK-ON" if r == 1 else "RISK-OFF",
            s.date(),
            e.date(),
            (e - s).days
        ])

    df = pd.DataFrame(out, columns=["Regime", "Start", "End", "Duration (days)"])

    avg_on  = df[df["Regime"] == "RISK-ON"]["Duration (days)"].mean()
    avg_off = df[df["Regime"] == "RISK-OFF"]["Duration (days)"].mean()

    return df, avg_on, avg_off
# ============================================================
# EMAIL HELPERS
# ============================================================

def fmt_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "—"

def fmt_dec(x):
    return f"{x:.3f}" if pd.notna(x) else "—"

def fmt_num(x):
    return f"{x:,.2f}" if pd.notna(x) else "—"


def attach_file(msg, filepath):
    with open(filepath, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(filepath)}"')
    msg.attach(part)


def send_email(html):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    SEND_TO = os.getenv("SEND_TO")

    msg = MIMEMultipart()
    msg["Subject"] = "Daily Portfolio Strategy Update"
    msg["From"] = EMAIL_USER
    msg["To"] = SEND_TO

    msg.attach(MIMEText(html, "html"))
    attach_file(msg, "equity_curve.png")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, SEND_TO, msg.as_string())


# ============================================================
# MAIN — HEADLESS DAILY ENGINE
# ============================================================

if __name__ == "__main__":

    # Load the same universe as the main app
    tickers = sorted(set(list(RISK_ON_WEIGHTS.keys()) + list(RISK_OFF_WEIGHTS.keys())))
    prices = load_price_data(tickers, DEFAULT_START_DATE).dropna(how="any")

    # 1. GRID SEARCH for optimal MA regime
    best_cfg, best_result = run_grid_search(prices, RISK_ON_WEIGHTS, RISK_OFF_WEIGHTS, FLIP_COST)
    best_len, best_type, best_tol = best_cfg

    sig = best_result["signal"]
    perf = best_result["performance"]

    # MA series
    portfolio_index = build_portfolio_index(prices, RISK_ON_WEIGHTS)
    opt_ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]

    # Current regime
    latest_signal = bool(sig.iloc[-1])
    regime = "RISK-ON" if latest_signal else "RISK-OFF"

    # Trades per year
    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252)

    # 2. Always-ON Risk-On Benchmark
    simple_rets = prices.pct_change().fillna(0)
    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in RISK_ON_WEIGHTS.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_on_eq = (1 + risk_on_simple).cumprod()
    risk_on_perf = compute_performance(risk_on_simple, risk_on_eq)

    # 3. SHARPE-OPTIMAL PORTFOLIO
    so_tickers, so_weights, so_returns, so_eq, so_perf = compute_sharpe_optimal(
        prices, RISK_ON_WEIGHTS
    )

    # 4. BUILD HYBRID + PURE SIG ENGINES
    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in RISK_OFF_WEIGHTS.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
    quarterly_target = (1 + bh_cagr) ** (1/4) - 1

    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)

    pure_eq, pure_rw, pure_sw, _ = run_sig_engine(
        risk_on_simple, risk_off_daily, quarterly_target, pure_sig_signal
    )

    hyb_eq, hyb_rw, hyb_sw, _ = run_sig_engine(
        risk_on_simple, risk_off_daily, quarterly_target, sig,
        pure_sig_rw=pure_rw, pure_sig_sw=pure_sw
    )

    # 5. FIND QUARTER START (IDENTICAL TO MAIN APP)
    last_index = len(hyb_rw) - 1
    quarter_indices = [i for i in range(len(hyb_rw)) if i % QUARTER_DAYS == 0]
    q_start_idx = max(idx for idx in quarter_indices if idx <= last_index)

    quarter_start_date = prices.index[q_start_idx]
    today_date = prices.index[-1]

    # 6. NEXT QUARTER DATE
    next_q = quarter_start_date + pd.Timedelta(days=QUARTER_DAYS)
    while next_q <= today_date:
        next_q += pd.Timedelta(days=QUARTER_DAYS)

    days_to_next_q = (next_q - today_date).days

    # 7. QUARTERLY PROGRESS TABLE (IDENTICAL CALC)
    def sig_progress(real_cap):
        risky_start = float(hyb_rw.iloc[q_start_idx]) * real_cap
        risky_today = float(hyb_rw.iloc[-1]) * real_cap
        return compute_quarter_progress(risky_start, risky_today, quarterly_target)

    cap1, cap2, cap3 = 25000, 25000, 25000
    prog1 = sig_progress(cap1)
    prog2 = sig_progress(cap2)
    prog3 = sig_progress(cap3)

    # 8. DISTANCE TO NEXT MA SIGNAL
    latest_date = opt_ma.dropna().index[-1]
    P = float(portfolio_index.loc[latest_date])
    MA = float(opt_ma.loc[latest_date])

    upper = MA * (1 + best_tol)
    lower = MA * (1 - best_tol)

    if latest_signal:
        delta = (P - lower) / P
        flip_direction = "Drop Required for RISK-OFF"
    else:
        delta = (upper - P) / P
        flip_direction = "Gain Required for RISK-ON"

    # 9. REGIME STATS
    regime_df, avg_on, avg_off = compute_regime_stats(sig)

    # 10. STRATEGY STATS TABLE (ALL FIVE PORTFOLIOS)
    hyb_simple = hyb_eq.pct_change().fillna(0)
    pure_simple = pure_eq.pct_change().fillna(0)

    hyb_perf = compute_performance(hyb_simple, hyb_eq)
    pure_perf = compute_performance(pure_simple, pure_eq)

    strat_stats = compute_stats(
        perf, best_result["returns"], perf["DD_Series"],
        best_result["flip_mask"], trades_per_year
    )

    risk_stats = compute_stats(
        risk_on_perf, risk_on_simple, risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_simple), dtype=bool), 0
    )

    hyb_stats = compute_stats(
        hyb_perf, hyb_simple, hyb_perf["DD_Series"],
        np.zeros(len(hyb_simple)), 0
    )

    pure_stats = compute_stats(
        pure_perf, pure_simple, pure_perf["DD_Series"],
        np.zeros(len(pure_simple)), 0
    )

    # 11. BUILD EQUITY CURVE PLOT
    plt.figure(figsize=(12, 6))
    plt.plot((best_result["equity_curve"]/best_result["equity_curve"].iloc[0])*10000,
             label="MA Strategy", linewidth=2)
    plt.plot((so_eq/so_eq.iloc[0])*10000, label="Sharpe-Optimal", linewidth=2, color="magenta")
    plt.plot((risk_on_eq/risk_on_eq.iloc[0])*10000, label="100% Risk-On", alpha=0.6)
    plt.plot((hyb_eq/hyb_eq.iloc[0])*10000, label="Hybrid SIG", linewidth=2, color="blue")
    plt.plot((pure_eq/pure_eq.iloc[0])*10000, label="Pure SIG", linewidth=2, color="orange")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.close()

    # 12. BUILD HTML EMAIL
    html = f"""
    <h2>Daily Portfolio Strategy Update</h2>

    <h3>Current MA Regime: <b>{regime}</b></h3>

    <h3>Optimal Signal Parameters</h3>
    <ul>
      <li><b>Type:</b> {best_type.upper()}</li>
      <li><b>Length:</b> {best_len}</li>
      <li><b>Tolerance:</b> {best_tol:.2%}</li>
    </ul>

    <h3>Distance to Next Signal</h3>
    <p><b>{flip_direction}:</b> {delta:.2%}</p>

    <h3>Quarterly SIG Progress</h3>
    <p>Quarter started: {quarter_start_date.date()}<br>
       Next rebalance: {next_q.date()} ({days_to_next_q} days)</p>

    <table border="1" cellspacing="0" cellpadding="4">
      <tr><th></th><th>Taxable</th><th>Tax-Sheltered</th><th>Joint</th></tr>
      <tr><td>Start Risky ($)</td><td>{fmt_num(prog1['Implied Deployed Capital at Qtr Start ($)'])}</td><td>{fmt_num(prog2['Implied Deployed Capital at Qtr Start ($)'])}</td><td>{fmt_num(prog3['Implied Deployed Capital at Qtr Start ($)'])}</td></tr>
      <tr><td>Risky Today ($)</td><td>{fmt_num(prog1['Implied Deployed Capital Today ($)'])}</td><td>{fmt_num(prog2['Implied Deployed Capital Today ($)'])}</td><td>{fmt_num(prog3['Implied Deployed Capital Today ($)'])}</td></tr>
      <tr><td>Qtr Target ($)</td><td>{fmt_num(prog1['Deployed Capital Target Qtr End ($)'])}</td><td>{fmt_num(prog2['Deployed Capital Target Qtr End ($)'])}</td><td>{fmt_num(prog3['Deployed Capital Target Qtr End ($)'])}</td></tr>
      <tr><td>Gap ($)</td><td>{fmt_num(prog1['Gap ($)'])}</td><td>{fmt_num(prog2['Gap ($)'])}</td><td>{fmt_num(prog3['Gap ($)'])}</td></tr>
      <tr><td>Gap (%)</td><td>{fmt_pct(prog1['Gap (%)'])}</td><td>{fmt_pct(prog2['Gap (%)'])}</td><td>{fmt_pct(prog3['Gap (%)'])}</td></tr>
    </table>

    <h3>Regime Statistics</h3>
    <p>Avg RISK-ON: {avg_on:.1f} days<br>
       Avg RISK-OFF: {avg_off:.1f} days</p>

    <h3>Attached Chart</h3>
    <p>The file <b>equity_curve.png</b> contains the equity curves for all strategies.</p>
    """

    # 13. SEND EMAIL
    send_email(html)
