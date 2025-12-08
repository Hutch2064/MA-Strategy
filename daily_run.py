# ===============================================
# DAILY EMAIL SCRIPT — BLOCK 1
# All defaults preserved exactly from Streamlit app
# ===============================================
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import ssl
import io
import base64

# ============================================================
# CONFIG  (identical to Streamlit)
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
QUARTER_DAYS = 63

# SIG engine starting point (unchanged)
START_RISKY = 0.60
START_SAFE  = 0.40

# ============================================================
# REAL ACCOUNT VALUES (A1 — hardcoded)
# ============================================================

REAL_CAP_1 = 72558.41     # Taxable
REAL_CAP_2 = 9177.97      # Tax-Sheltered
REAL_CAP_3 = 4151.11      # Joint

# ============================================================
# DATA LOADING (no Streamlit caching here)
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
# MA MATRIX (same as Streamlit)
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
# TESTFOL MA SIGNAL (same)
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
# PERFORMANCE ENGINE HELPERS
# ============================================================

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

# ============================================================
# BACKTEST ENGINE (MA Strategy)
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
# GRID SEARCH (EXACT from Streamlit — no UI)
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

    ma_cache = {t: compute_ma_matrix(portfolio_index, lengths, t) for t in types}

    total_loops = len(lengths) * len(types) * len(tolerances)
    loop = 0

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

                # Selection logic identical to app
                if (
                    sharpe > best_sharpe or
                    (sharpe == best_sharpe and trades_per_year < best_trades)
                ):
                    best_sharpe = sharpe
                    best_trades = trades_per_year
                    best_cfg = (length, ma_type, tol)
                    best_result = result

                loop += 1

    return best_cfg, best_result
# ===============================================
# DAILY EMAIL SCRIPT — BLOCK 2
# SIG Engine, performance stats, quarterly progress
# ===============================================

# ============================================================
# SIG ENGINE
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

            # Restore pure SIG weights upon re-entering
            if frozen_risky is not None:
                w_r = pure_sig_rw.iloc[i]
                w_s = pure_sig_sw.iloc[i]
                risky_val = eq * w_r
                safe_val  = eq * w_s
                frozen_risky = None
                frozen_safe  = None

            risky_val *= (1 + r_on)
            safe_val  *= (1 + r_off)

            # Quarterly rebalance
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

                # Small quarterly fee
                quarter_fee = flip_cost * target_quarter
                eq *= (1 - quarter_fee)

            eq = risky_val + safe_val
            risky_w = risky_val / eq
            safe_w  = safe_val / eq

            if flip_mask.iloc[i]:
                eq *= (1 - flip_cost)

        else:
            # Freeze SIG weights for re-entry
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
        rebalance_events,
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
# NORMALIZER
# ============================================================

def normalize(eq):
    return eq / eq.iloc[0] * 10000

# ============================================================
# MAIN ENGINE EXECUTION (headless version of Streamlit logic)
# ============================================================

def run_full_engine():

    # Load all prices
    tickers = sorted(set(RISK_ON_WEIGHTS.keys()) | set(RISK_OFF_WEIGHTS.keys()))
    prices = load_price_data(tickers, DEFAULT_START_DATE)

    # Run MA optimization
    best_cfg, best_result = run_grid_search(
        prices, RISK_ON_WEIGHTS, RISK_OFF_WEIGHTS, FLIP_COST
    )
    best_len, best_type, best_tol = best_cfg

    sig = best_result["signal"]
    perf = best_result["performance"]

    # Current MA regime
    latest_signal = sig.iloc[-1]
    current_regime = "RISK-ON" if latest_signal else "RISK-OFF"

    # Rebuild MA curve
    portfolio_index = build_portfolio_index(prices, RISK_ON_WEIGHTS)
    opt_ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]

    # Always-on risk-on benchmark
    simple_rets = prices.pct_change().fillna(0)

    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in RISK_ON_WEIGHTS.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_on_eq = (1 + risk_on_simple).cumprod()
    risk_on_perf = compute_performance(risk_on_simple, risk_on_eq)

    # SHARPE-OPTIMAL PORTFOLIO
    ron_px = prices[list(RISK_ON_WEIGHTS.keys())].dropna()
    ron_rets = ron_px.pct_change().dropna()

    mu = ron_rets.mean().values
    cov = ron_rets.cov().values + np.eye(len(mu)) * 1e-10

    def neg_sharpe(w):
        r = np.dot(mu, w)
        v = np.sqrt(np.dot(w.T, cov @ w))
        return -(r / v) if v > 0 else 1e9

    n = len(mu)
    res = minimize(neg_sharpe, np.ones(n)/n,
                   bounds=[(0,1)]*n,
                   constraints={"type":"eq","fun":lambda w: w.sum()-1})
    w_opt = res.x

    sharp_returns = (ron_rets * w_opt).sum(axis=1)
    sharp_eq = (1 + sharp_returns).cumprod()
    sharp_perf = compute_performance(sharp_returns, sharp_eq)

    # HYBRID SIG ENGINE
    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in RISK_OFF_WEIGHTS.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    # Quarterly target
    bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
    quarterly_target = (1 + bh_cagr) ** (1/4) - 1

    # PURE SIG (always risk-on)
    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)
    pure_eq, pure_rw, pure_sw, _ = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        pure_sig_signal
    )

    # HYBRID SIG
    hybrid_eq, hybrid_rw, hybrid_sw, _ = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        sig,
        pure_sig_rw=pure_rw,
        pure_sig_sw=pure_sw
    )

    # QUARTER START DETECTION
    last_index = len(hybrid_rw) - 1
    quarter_indices = [i for i in range(len(hybrid_rw)) if i % QUARTER_DAYS == 0]
    q_start_idx = max([i for i in quarter_indices if i <= last_index])

    quarter_start_date = prices.index[q_start_idx]
    today_date = prices.index[-1]

    # PROGRESS FOR EACH ACCOUNT
    def get_sig_progress(real_cap):
        risky_start = float(hybrid_rw.iloc[q_start_idx]) * real_cap
        risky_today = float(hybrid_rw.iloc[-1]) * real_cap
        return compute_quarter_progress(risky_start, risky_today, quarterly_target)

    prog_1 = get_sig_progress(REAL_CAP_1)
    prog_2 = get_sig_progress(REAL_CAP_2)
    prog_3 = get_sig_progress(REAL_CAP_3)

    # NEXT QUARTER DATE
    next_q = quarter_start_date + pd.Timedelta(days=QUARTER_DAYS)
    while next_q <= today_date:
        next_q += pd.Timedelta(days=QUARTER_DAYS)
    days_to_next = (next_q - today_date).days

    # ADVANCED METRICS
    def time_in_drawdown(dd): return (dd < 0).mean()
    def mar(c, dd): return c / abs(dd) if dd != 0 else np.nan
    def ulcer(dd): return np.sqrt((dd**2).mean()) if (dd**2).mean() != 0 else np.nan
    def pain_gain(c, dd): return c / ulcer(dd) if ulcer(dd) != 0 else np.nan

    # Stats builder
    def compute_stats(perf, returns, dd, flips, tpy):
        return {
            "CAGR": perf["CAGR"],
            "Volatility": perf["Volatility"],
            "Sharpe": perf["Sharpe"],
            "MaxDD": perf["MaxDrawdown"],
            "Total": perf["TotalReturn"],
            "MAR": mar(perf["CAGR"], perf["MaxDrawdown"]),
            "TID": time_in_drawdown(dd),
            "PainGain": pain_gain(perf["CAGR"], dd),
            "Skew": returns.skew(),
            "Kurtosis": returns.kurt(),
            "Trades/year": tpy,
        }

    # MA Strategy
    ma_simple = best_result["returns"]
    ma_perf = perf
    ma_stats = compute_stats(
        perf,
        ma_simple,
        perf["DD_Series"],
        best_result["flip_mask"],
        (sig.astype(int).diff().abs().sum() / (len(sig) / 252)),
    )

    # Buy & Hold (100% Risk-On)
    bh_stats = compute_stats(
        risk_on_perf,
        risk_on_simple,
        risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_simple), dtype=bool),
        0,
    )

    # Sharpe-optimal
    sharp_stats = compute_stats(
        sharp_perf,
        sharp_returns,
        sharp_perf["DD_Series"],
        np.zeros(len(sharp_returns), dtype=bool),
        0,
    )

    # Hybrid SIG
    hybrid_simple = hybrid_eq.pct_change().fillna(0)
    hybrid_perf = compute_performance(hybrid_simple, hybrid_eq)
    hybrid_stats = compute_stats(
        hybrid_perf,
        hybrid_simple,
        hybrid_perf["DD_Series"],
        np.zeros(len(hybrid_simple)),
        0,
    )

    # Pure SIG
    pure_simple = pure_eq.pct_change().fillna(0)
    pure_stats = compute_stats(
        pure_sig_perf := compute_performance(pure_simple, pure_eq),
        pure_simple,
        pure_sig_perf["DD_Series"],
        np.zeros(len(pure_simple)),
        0,
    )

    # REGIME STATISTICS
    sig_int = sig.astype(int)
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

    regime_rows = []
    for r, s, e in segments:
        regime_rows.append({
            "regime": "RISK-ON" if r == 1 else "RISK-OFF",
            "start": s,
            "end": e,
            "duration": (e - s).days
        })

    # MA SIGNAL DISTANCE
    latest_date = opt_ma.dropna().index[-1]
    P = float(portfolio_index.loc[latest_date])
    MA_val = float(opt_ma.loc[latest_date])

    upper = MA_val * (1 + best_tol)
    lower = MA_val * (1 - best_tol)

    if latest_signal:
        ma_distance = ("Drop Required for RISK-OFF", (P - lower) / P)
    else:
        ma_distance = ("Gain Required for RISK-ON", (upper - P) / P)

    # RETURN EVERYTHING FOR BLOCK 3 EMAIL FORMATTING
    return {
        "prices": prices,
        "signal": sig,
        "portfolio_index": portfolio_index,
        "opt_ma": opt_ma,
        "best_cfg": best_cfg,
        "current_regime": current_regime,
        "ma_distance": ma_distance,
        "next_q": next_q,
        "days_to_next": days_to_next,
        "quarter_start": quarter_start_date,
        "progress": (prog_1, prog_2, prog_3),
        "performances": {
            "MA": ma_stats,
            "Sharpe": sharp_stats,
            "BuyHold": bh_stats,
            "Hybrid": hybrid_stats,
            "PureSIG": pure_stats,
        },
        "equity_curves": {
            "MA": best_result["equity_curve"],
            "Sharpe": sharp_eq,
            "BuyHold": risk_on_eq,
            "Hybrid": hybrid_eq,
            "PureSIG": pure_eq,
            "MA_Curve": opt_ma,
        },
        "sharp_weights": (ron_px.columns, w_opt),
        "regimes": regime_rows,
    }
# ============================================================
# BLOCK 3 — EMAIL OUTPUT (HTML + PLOT + SEND)
# ============================================================

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


# ------------------------------------------------------------
# Email credentials (use environment variables)
# ------------------------------------------------------------
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO   = os.getenv("EMAIL_TO")


# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------
def fmt_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "—"

def fmt_dec(x):
    return f"{x:.3f}" if pd.notna(x) else "—"

def fmt_num(x):
    return f"{x:,.2f}" if pd.notna(x) else "—"

def html_row(label, *vals):
    tds = "".join(f"<td style='text-align:right'>{v}</td>" for v in vals)
    return f"<tr><td>{label}</td>{tds}</tr>"


# ------------------------------------------------------------
# Build HTML table for performance stats
# ------------------------------------------------------------
def build_stats_table(stats):
    rows = [
        ("CAGR", "CAGR"),
        ("Volatility", "Volatility"),
        ("Sharpe", "Sharpe"),
        ("Max Drawdown", "MaxDD"),
        ("Total Return", "Total"),
        ("MAR Ratio", "MAR"),
        ("Time in Drawdown (%)", "TID"),
        ("Pain-to-Gain", "PainGain"),
        ("Skew", "Skew"),
        ("Kurtosis", "Kurtosis"),
        ("Trades/year", "Trades/year"),
    ]

    html = """
    <table border="1" cellspacing="0" cellpadding="5">
      <thead>
        <tr>
          <th>Metric</th>
          <th>MA Strategy</th>
          <th>Sharpe-Optimal</th>
          <th>100% Risk-On</th>
          <th>Hybrid SIG</th>
          <th>Pure SIG</th>
        </tr>
      </thead>
      <tbody>
    """

    sMA     = stats["MA"]
    sSharp  = stats["Sharpe"]
    sBH     = stats["BuyHold"]
    sHybrid = stats["Hybrid"]
    sPure   = stats["PureSIG"]

    for label, key in rows:

        def get_value(obj):
            v = obj.get(key, None)
            if key in ["CAGR", "Volatility", "MaxDD", "Total", "TID"]:
                return fmt_pct(v)
            elif key in ["Sharpe", "MAR", "PainGain", "Skew", "Kurtosis"]:
                return fmt_dec(v)
            else:
                return fmt_num(v)

        html += html_row(
            label,
            get_value(sMA),
            get_value(sSharp),
            get_value(sBH),
            get_value(sHybrid),
            get_value(sPure),
        )

    html += "</tbody></table>"
    return html


# ------------------------------------------------------------
# Build regime table HTML
# ------------------------------------------------------------
def build_regime_table(regimes):

    html = """
    <table border="1" cellspacing="0" cellpadding="4">
      <thead>
        <tr>
          <th>Regime</th>
          <th>Start</th>
          <th>End</th>
          <th>Duration (days)</th>
        </tr>
      </thead>
      <tbody>
    """

    for row in regimes:
        html += (
            f"<tr>"
            f"<td>{row['regime']}</td>"
            f"<td>{row['start'].date()}</td>"
            f"<td>{row['end'].date()}</td>"
            f"<td style='text-align:right'>{row['duration']}</td>"
            f"</tr>"
        )

    html += "</tbody></table>"
    return html


# ------------------------------------------------------------
# Build quarterly SIG progress table (Taxable / TS / Joint)
# ------------------------------------------------------------
def build_quarter_progress_table(prog_1, prog_2, prog_3):

    df = pd.concat([
        pd.DataFrame.from_dict(prog_1, orient='index', columns=['Taxable']),
        pd.DataFrame.from_dict(prog_2, orient='index', columns=['Tax-Sheltered']),
        pd.DataFrame.from_dict(prog_3, orient='index', columns=['Joint']),
    ], axis=1)

    df.loc["Gap (%)"] = df.loc["Gap (%)"].apply(lambda x: f"{x:.2%}")

    return df.to_html(border=1, justify="center")


# ------------------------------------------------------------
# Main function that builds and sends the email
# ------------------------------------------------------------
def send_daily_email(results):

    # Unpack everything from results
    best_len, best_type, best_tol = results["best_cfg"]
    current_regime = results["current_regime"]
    ma_distance_str, ma_distance_val = results["ma_distance"]
    next_q = results["next_q"]
    days_to_next = results["days_to_next"]
    quarter_start = results["quarter_start"]
    prog_1, prog_2, prog_3 = results["progress"]
    stats = results["performances"]
    regimes = results["regimes"]
    eq = results["equity_curves"]
    sharp_cols, sharp_w = results["sharp_weights"]

    # ---------------------------------------------------------
    # Build PLOT: combine all equity curves
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(normalize(eq["MA"]),     label="MA Strategy", linewidth=2)
    ax.plot(normalize(eq["Sharpe"]), label="Sharpe-Optimal", linewidth=2, color="magenta")
    ax.plot(normalize(eq["BuyHold"]),label="100% Risk-On", alpha=0.7)
    ax.plot(normalize(eq["Hybrid"]), label="Hybrid SIG", linewidth=2, color="blue")
    ax.plot(normalize(eq["PureSIG"]),label="Pure SIG", linewidth=2, color="orange")

    ax.legend()
    ax.grid(alpha=.3)
    plt.tight_layout()

    plt.savefig("email_chart.png")
    plt.close()


    # ---------------------------------------------------------
    # HTML BUILD
    # ---------------------------------------------------------
    html = f"""
    <html>
    <body>

    <h2>Daily MA + SIG Strategy Report</h2>

    <h3>Current MA Regime: {current_regime}</h3>
    <p><b>MA Type:</b> {best_type.upper()} — <b>Length:</b> {best_len} — <b>Tolerance:</b> {best_tol:.2%}</p>

    <h3>MA Signal Distance</h3>
    <p><b>{ma_distance_str}:</b> {ma_distance_val:.2%}</p>

    <h3>Quarterly SIG Progress</h3>
    <p><b>Quarter Start:</b> {quarter_start.date()}<br>
       <b>Next Rebalance:</b> {next_q.date()} ({days_to_next} days)</p>

    {build_quarter_progress_table(prog_1, prog_2, prog_3)}

    <h3>Performance Comparison Table</h3>
    {build_stats_table(stats)}

    <h3>Sharpe-Optimal Weights</h3>
    <ul>
    """
    for t, w in zip(sharp_cols, sharp_w):
        html += f"<li><b>{t}:</b> {w:.2%}</li>"
    html += "</ul>"

    html += f"""
    <h3>Regime Durations</h3>
    {build_regime_table(regimes)}

    <p>The attached chart shows all strategy equity curves normalized to 10,000.</p>

    </body>
    </html>
    """

    # ---------------------------------------------------------
    # Assemble Email
    # ---------------------------------------------------------
    msg = MIMEMultipart()
    msg["Subject"] = f"Daily Portfolio Signal — {current_regime}"
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(html, "html"))

    # Attach plot
    with open("email_chart.png", "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        'attachment; filename="strategy_chart.png"'
    )
    msg.attach(part)

    # ---------------------------------------------------------
    # SEND EMAIL
    # ---------------------------------------------------------
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())

    print("Email sent successfully.")


# ------------------------------------------------------------
# FINAL EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":

    results = run_full_engine()
    send_daily_email(results)
