import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import io
from scipy.optimize import minimize
import datetime

# ============================================================
# CONFIG
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

# Starting weights inside the SIG engine (unchanged)
START_RISKY = 0.60
START_SAFE  = 0.40


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=True)
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
# SIG ENGINE (no user inputs, internal cadence, deterministic)
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

    # Flip detection identical to MA strategy
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

            # Restore pure-SIG weights after exiting RISK-OFF
            if frozen_risky is not None:
                w_r = pure_sig_rw.iloc[i]
                w_s = pure_sig_sw.iloc[i]
                risky_val = eq * w_r
                safe_val  = eq * w_s
                frozen_risky = None
                frozen_safe  = None

            # Apply daily returns
            risky_val *= (1 + r_on)
            safe_val  *= (1 + r_off)

            # Quarterly rebalance logic — pure Kelly SIG
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

                # Quarterly drag fee (small)
                quarter_fee = flip_cost * target_quarter
                eq *= (1 - quarter_fee)

            eq = risky_val + safe_val
            risky_w = risky_val / eq
            safe_w  = safe_val  / eq

            # Flip cost identical to MA strategy
            if flip_mask.iloc[i]:
                eq *= (1 - flip_cost)

        else:
            # Freeze risky/safe values for re-entry
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
# GRID SEARCH
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

    progress = st.progress(0.0)
    total = len(lengths) * len(types) * len(tolerances)
    idx = 0

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

                idx += 1
                if idx % 200 == 0:
                    progress.progress(idx / total)

                if (
                    sharpe > best_sharpe or
                    (sharpe == best_sharpe and trades_per_year < best_trades)
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
        "Implied Deployed Capital at Qtr Start ($)": risky_start,
        "Implied Deployed Capital Today ($)": risky_today,
        "Deployed Capital Target Qtr End ($)": target_risky,
        "Gap ($)": gap,
        "Gap (%)": pct_gap,
    }


# ============================================================
# NORMALIZATION FOR PLOTTING
# ============================================================

def normalize(eq):
    return eq / eq.iloc[0] * 10000
    
# ============================================================
# PRECOMPUTE QUARTER START DATE (before running backtest)
# ============================================================

# ---- PREVIEW QUARTER START DATE WITHOUT RUNNING ANY STRATEGY ----
# Load only prices for quarter detection (fast, minimal)
preview_prices = load_price_data(
    sorted(set(RISK_ON_WEIGHTS.keys()) | set(RISK_OFF_WEIGHTS.keys())),
    DEFAULT_START_DATE,
    None
)

# Build index just to align with dates
preview_index = build_portfolio_index(preview_prices, RISK_ON_WEIGHTS)

# Determine quarter start index (same method the engine uses)
preview_quarter_indices = [i for i in range(len(preview_index)) if i % QUARTER_DAYS == 0]
preview_q_start_idx = max(preview_quarter_indices)

# Sidebar needs this date BEFORE running the strategy
preview_quarter_start_date = preview_index.index[preview_q_start_idx].date()
# ============================================================
# STREAMLIT APP
# ============================================================

def main():

    st.set_page_config(page_title="Portfolio MA Regime Strategy", layout="wide")
    st.title("Portfolio Strategy")

    # ------------------------------------------------------------
    # SIDEBAR: Backtest Inputs
    # ------------------------------------------------------------
    st.sidebar.header("Backtest Settings")
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")

    # ------------------------------------------------------------
    # SIDEBAR: Asset Sleeves
    # ------------------------------------------------------------
    st.sidebar.header("Deployed Capital Sleeve")
    risk_on_tickers_str = st.sidebar.text_input(
        "Tickers", ",".join(RISK_ON_WEIGHTS.keys())
    )
    risk_on_weights_str = st.sidebar.text_input(
        "Weights", ",".join(str(w) for w in RISK_ON_WEIGHTS.values())
    )

    st.sidebar.header("Treasury Sleeve")
    risk_off_tickers_str = st.sidebar.text_input(
        "Tickers", ",".join(RISK_OFF_WEIGHTS.keys())
    )
    risk_off_weights_str = st.sidebar.text_input(
        "Weights", ",".join(str(w) for w in RISK_OFF_WEIGHTS.values())
    )

    # ------------------------------------------------------------
    # SIDEBAR: Quarter Start + User Inputs for Real-World Balances
    # ------------------------------------------------------------
    st.sidebar.header("Quarterly Tracking Inputs")

    # Show the correct quarter start date before clicking RUN
    st.sidebar.write(f"**Current Quarter Start Date:** {preview_quarter_start_date}")

    st.sidebar.write("Enter your real portfolio values:")

    real_cap_1_start = st.sidebar.number_input(
        "Taxable – Portfolio Value at Quarter Start ($)",
        min_value=0.0, value=0.0, step=100.0
    )
    real_cap_1_today = st.sidebar.number_input(
        "Taxable – Portfolio Value Today ($)",
        min_value=0.0, value=0.0, step=100.0
    )

    real_cap_2_start = st.sidebar.number_input(
        "Tax-Sheltered – Portfolio Value at Quarter Start ($)",
        min_value=0.0, value=0.0, step=100.0
    )
    real_cap_2_today = st.sidebar.number_input(
        "Tax-Sheltered – Portfolio Value Today ($)",
        min_value=0.0, value=0.0, step=100.0
    )

    real_cap_3_start = st.sidebar.number_input(
        "Joint – Portfolio Value at Quarter Start ($)",
        min_value=0.0, value=0.0, step=100.0
    )
    real_cap_3_today = st.sidebar.number_input(
        "Joint – Portfolio Value Today ($)",
        min_value=0.0, value=0.0, step=100.0
    )
    
    # ------------------------------------------------------------
    # Parse sleeve inputs
    # ------------------------------------------------------------
    risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",")]
    risk_on_weights_list = [float(x) for x in risk_on_weights_str.split(",")]
    risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))

    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))

    # ------------------------------------------------------------
    # Load prices
    # ------------------------------------------------------------
    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers))
    end_val = end if end.strip() else None

    prices = load_price_data(all_tickers, start, end_val).dropna(how="any")

    # ------------------------------------------------------------
    # MA Optimization (Grid Search)
    # ------------------------------------------------------------
    best_cfg, best_result = run_grid_search(
        prices, risk_on_weights, risk_off_weights, FLIP_COST
    )
    best_len, best_type, best_tol = best_cfg

    sig = best_result["signal"]
    perf = best_result["performance"]

    # ------------------------------------------------------------
    # CURRENT MA REGIME
    # ------------------------------------------------------------
    latest_signal = sig.iloc[-1]
    current_regime = "RISK-ON" if latest_signal else "RISK-OFF"

    st.subheader(f"Current MA Regime: {current_regime}")
    st.write(f"**MA Type:** {best_type.upper()}  —  **Length:** {best_len}  —  **Tolerance:** {best_tol:.2%}")

    # Compute MA used for final signal
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    opt_ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]

    # Trades per year calculation
    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252)

    # ------------------------------------------------------------
    # Always-ON RISK-ON Benchmark
    # ------------------------------------------------------------
    simple_rets = prices.pct_change().fillna(0)

    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_on_eq = (1 + risk_on_simple).cumprod()
    risk_on_perf = compute_performance(risk_on_simple, risk_on_eq)

    # ------------------------------------------------------------
    # Sharpe-Optimal Portfolio (unchanged)
    # ------------------------------------------------------------
    risk_on_px = prices[[t for t in risk_on_tickers if t in prices.columns]].dropna()
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
    sharp_perf = compute_performance(sharp_returns, sharp_eq)

    # ------------------------------------------------------------
    # HYBRID SIG ENGINE (Backtested, No User Inputs)
    # ------------------------------------------------------------
    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    # Annualized CAGR → quarterly Kelly target
    bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
    quarterly_target = (1 + bh_cagr) ** (1/4) - 1

    # PURE SIG (always RISK-ON)
    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)

    pure_sig_eq, pure_sig_rw, pure_sig_sw, pure_sig_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        pure_sig_signal
    )

    # HYBRID SIG (MA Filter)
    hybrid_eq, hybrid_rw, hybrid_sw, hybrid_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        sig,
        pure_sig_rw=pure_sig_rw,
        pure_sig_sw=pure_sig_sw
    )

    # ------------------------------------------------------------
    # AUTOMATIC QUARTER START (Derived From Backtest)
    # ------------------------------------------------------------
    last_index = len(hybrid_rw) - 1

    # Find most recent quarter boundary
    quarter_indices = [i for i in range(len(hybrid_rw)) if i % QUARTER_DAYS == 0]
    q_start_idx = max(idx for idx in quarter_indices if idx <= last_index)

    quarter_start_date = prices.index[q_start_idx]
    today_date = prices.index[-1]

    # ------------------------------------------------------------
    # DERIVE risky_start / risky_today FOR EACH ACCOUNT (C1)
    # ------------------------------------------------------------
    def get_sig_progress(real_start_val, real_today_val):
        risky_start = real_start_val * float(hybrid_rw.iloc[q_start_idx])
        risky_today = real_today_val * float(hybrid_rw.iloc[-1])
        return compute_quarter_progress(risky_start, risky_today, quarterly_target)

    prog_1 = get_sig_progress(real_start_1, real_today_1)
    prog_2 = get_sig_progress(real_start_2, real_today_2)
    prog_3 = get_sig_progress(real_start_3, real_today_3)

    # ------------------------------------------------------------
    # NEXT QUARTER DATE (automatic)
    # ------------------------------------------------------------
    next_q = quarter_start_date + pd.Timedelta(days=QUARTER_DAYS)
    while next_q <= today_date:
        next_q += pd.Timedelta(days=QUARTER_DAYS)

    days_to_next_q = (next_q - today_date).days
    # ------------------------------------------------------------
    # TEXT FUNCTION FOR SIG REBALANCE GUIDANCE
    # ------------------------------------------------------------
    def rebalance_text(gap, next_q, days_to_next_q):
        date_str = next_q.strftime("%m/%d/%Y")
        days_str = f"{days_to_next_q} days"

        if gap > 0:
            return f"Increase deployed sleeve by **${gap:,.2f}** on **{date_str}** ({days_str})"
        elif gap < 0:
            return f"Decrease deployed sleeve by **${abs(gap):,.2f}** on **{date_str}** ({days_str})"
        else:
            return f"No rebalance needed until **{date_str}** ({days_str})"

    # ------------------------------------------------------------
    # DISPLAY SIG QUARTERLY PROGRESS RESULTS
    # ------------------------------------------------------------
    st.subheader("Strategy Summary")
    st.write(f"**Quarter start:** {quarter_start_date.date()}")
    st.write(f"**Next rebalance date:** {next_q.date()} ({days_to_next_q} days)")

    prog_df = pd.concat([
        pd.DataFrame.from_dict(prog_1, orient='index', columns=['Taxable']),
        pd.DataFrame.from_dict(prog_2, orient='index', columns=['Tax-Sheltered']),
        pd.DataFrame.from_dict(prog_3, orient='index', columns=['Joint']),
    ], axis=1)

    prog_df.loc["Gap (%)"] = prog_df.loc["Gap (%)"].apply(lambda x: f"{x:.2%}")
    st.dataframe(prog_df)

    st.write("### Rebalance Recommendations")
    st.write("**Taxable:** " + rebalance_text(prog_1["Gap ($)"], next_q, days_to_next_q))
    st.write("**Tax-Sheltered:** " + rebalance_text(prog_2["Gap ($)"], next_q, days_to_next_q))
    st.write("**Joint:** " + rebalance_text(prog_3["Gap ($)"], next_q, days_to_next_q))

    # ------------------------------------------------------------
    # ADVANCED METRICS
    # ------------------------------------------------------------
    def time_in_drawdown(dd): return (dd < 0).mean()
    def mar(c, dd): return c / abs(dd) if dd != 0 else np.nan
    def ulcer(dd): return np.sqrt((dd**2).mean()) if (dd**2).mean() != 0 else np.nan
    def pain_gain(c, dd): return c / ulcer(dd) if ulcer(dd) != 0 else np.nan

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

    # Compute strategy stats
    hybrid_simple = hybrid_eq.pct_change().fillna(0)
    hybrid_perf = compute_performance(hybrid_simple, hybrid_eq)

    pure_sig_simple = pure_sig_eq.pct_change().fillna(0)
    pure_sig_perf = compute_performance(pure_sig_simple, pure_sig_eq)

    strat_stats = compute_stats(
        perf,
        best_result["returns"],
        perf["DD_Series"],
        best_result["flip_mask"],
        trades_per_year,
    )

    risk_stats = compute_stats(
        risk_on_perf,
        risk_on_simple,
        risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_simple), dtype=bool),
        0,
    )

    hybrid_stats = compute_stats(
        hybrid_perf,
        hybrid_simple,
        hybrid_perf["DD_Series"],
        np.zeros(len(hybrid_simple), dtype=bool),
        0,
    )

    pure_sig_stats = compute_stats(
        pure_sig_perf,
        pure_sig_simple,
        pure_sig_perf["DD_Series"],
        np.zeros(len(pure_sig_simple), dtype=bool),
        0,
    )

    # ------------------------------------------------------------
    # STAT TABLE DISPLAY
    # ------------------------------------------------------------
    st.subheader("MA vs Sharpe-Optimal vs Buy & Hold vs Hybrid SIG/MA vs Pure SIG")

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
        ("Trades per year", "Trades/year"),
    ]

    def fmt_pct(x): return f"{x:.2%}" if pd.notna(x) else "—"
    def fmt_dec(x): return f"{x:.3f}" if pd.notna(x) else "—"
    def fmt_num(x): return f"{x:,.2f}" if pd.notna(x) else "—"

    table_data = []
    for label, key in rows:
        sv = strat_stats.get(key, np.nan)
        sh = sharp_perf.get(key, np.nan)
        rv = risk_stats.get(key, np.nan)
        hv = hybrid_stats.get(key, np.nan)
        ps = pure_sig_stats.get(key, np.nan)

        if key in ["CAGR", "Volatility", "MaxDD", "Total", "TID"]:
            row = [label, fmt_pct(sv), fmt_pct(sh), fmt_pct(rv), fmt_pct(hv), fmt_pct(ps)]
        elif key in ["Sharpe", "MAR", "PainGain", "Skew", "Kurtosis"]:
            row = [label, fmt_dec(sv), fmt_dec(sh), fmt_dec(rv), fmt_dec(hv), fmt_dec(ps)]
        else:
            row = [label, fmt_num(sv), fmt_num(sh), fmt_num(rv), fmt_num(hv), fmt_num(ps)]

        table_data.append(row)

    stat_table = pd.DataFrame(
        table_data,
        columns=[
            "Metric",
            "MA Strategy",
            "Sharpe-Optimal",
            "Buy & Hold",
            "Hybrid SIG",
            "Pure SIG",
        ],
    )

    st.dataframe(stat_table, use_container_width=True)

    # ------------------------------------------------------------
    # ACCOUNT-LEVEL ALLOCATIONS
    # ------------------------------------------------------------

    def compute_allocations(account_value, risky_w, safe_w, ron_w, roff_w):
        risky_dollars = account_value * risky_w
        safe_dollars  = account_value * safe_w

        alloc = {"Total Risky $": risky_dollars, "Total Safe $": safe_dollars}

        for t, w in ron_w.items():
            alloc[t] = risky_dollars * w

        for t, w in roff_w.items():
            alloc[t] = safe_dollars * w

        return alloc

    def compute_sharpe_alloc(account_value, tickers, weights):
        return {t: account_value * w for t, w in zip(tickers, weights)}

    def add_pct(df_dict):
        out = pd.DataFrame.from_dict(df_dict, orient="index", columns=["$"])

        # If this is a SIG-style table (has total risky/safe rows)
        if "Total Risky $" in out.index and "Total Safe $" in out.index:
            total_portfolio = float(out.loc["Total Risky $","$"]) + float(out.loc["Total Safe $","$"])
            out["% Portfolio"] = (out["$"] / total_portfolio * 100).apply(lambda x: f"{x:.2f}%")
            return out

        # Otherwise this is a simple ticker-only allocation (Sharpe-optimal, MA-only, 100% Risk-On, etc.)
        total = out["$"].sum()
        out["% Portfolio"] = (out["$"] / total * 100).apply(lambda x: f"{x:.2f}%")
        return out

    st.subheader("Account-Level Allocations")

    # CURRENT hybrid weights
    hyb_r = float(hybrid_rw.iloc[-1])
    hyb_s = float(hybrid_sw.iloc[-1])

    pure_r = float(pure_sig_rw.iloc[-1])
    pure_s = float(pure_sig_sw.iloc[-1])

    # Risk-On is 100% risky sleeve
    # MA strategy depends on signal today
    latest_signal = sig.iloc[-1]

    tab1, tab2, tab3 = st.tabs(["Taxable", "Tax-Sheltered", "Joint"])

    accounts = [
        ("Taxable", real_cap_1),
        ("Tax-Sheltered", real_cap_2),
        ("Joint", real_cap_3),
    ]

    for (label, cap), tab in zip(accounts, (tab1, tab2, tab3)):
        with tab:
            st.write(f"### {label} — Hybrid SIG")
            st.dataframe(add_pct(compute_allocations(cap, hyb_r, hyb_s, risk_on_weights, risk_off_weights)))

            st.write(f"### {label} — Pure SIG")
            st.dataframe(add_pct(compute_allocations(cap, pure_r, pure_s, risk_on_weights, risk_off_weights)))

            st.write(f"### {label} — 100% Risk-On Portfolio")
            st.dataframe(add_pct(compute_allocations(cap, 1.0, 0.0, risk_on_weights, {"SHY": 0})))

            st.write(f"### {label} — Sharpe-Optimal")
            st.dataframe(add_pct(compute_sharpe_alloc(cap, risk_on_px.columns, w_opt)))

            st.write(f"### {label} — MA Strategy")
            if latest_signal:
                ma_alloc = compute_allocations(cap, 1.0, 0.0, risk_on_weights, {"SHY": 0})
            else:
                ma_alloc = compute_allocations(cap, 0.0, 1.0, {}, risk_off_weights)
            st.dataframe(add_pct(ma_alloc))

    # ------------------------------------------------------------
    # MA SIGNAL DISTANCE
    # ------------------------------------------------------------
    st.subheader("Next MA Signal Distance")

    latest_date = opt_ma.dropna().index[-1]
    P = float(portfolio_index.loc[latest_date])
    MA = float(opt_ma.loc[latest_date])

    upper = MA * (1 + best_tol)
    lower = MA * (1 - best_tol)

    if latest_signal:
        delta = (P - lower) / P
        st.write(f"**Drop Required for RISK-OFF:** {delta:.2%}")
    else:
        delta = (upper - P) / P
        st.write(f"**Gain Required for RISK-ON:** {delta:.2%}")

    # ------------------------------------------------------------
    # REGIME STATS
    # ------------------------------------------------------------
    st.subheader("Regime Statistics")

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
        regime_rows.append([
            "RISK-ON" if r == 1 else "RISK-OFF",
            s.date(), e.date(),
            (e - s).days
        ])

    regime_df = pd.DataFrame(regime_rows, columns=["Regime", "Start", "End", "Duration (days)"])
    st.dataframe(regime_df)

    st.write(f"**Avg RISK-ON duration:** {regime_df[regime_df['Regime']=='RISK-ON']['Duration (days)'].mean():.1f} days")
    st.write(f"**Avg RISK-OFF duration:** {regime_df[regime_df['Regime']=='RISK-OFF']['Duration (days)'].mean():.1f} days")

    # ------------------------------------------------------------
    # FINAL PLOT
    # ------------------------------------------------------------
    st.subheader("Portfolio Strategy Performance Comparison")

    plot_index = build_portfolio_index(prices, risk_on_weights)
    plot_ma = compute_ma_matrix(plot_index, [best_len], best_type)[best_len]

    plot_index_norm = normalize(plot_index)
    plot_ma_norm = normalize(plot_ma.dropna())

    strat_eq_norm  = normalize(best_result["equity_curve"])
    sharp_eq_norm  = normalize(sharp_eq)
    hybrid_eq_norm = normalize(hybrid_eq)
    pure_sig_norm  = normalize(pure_sig_eq)
    risk_on_norm   = normalize(risk_on_eq)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strat_eq_norm, label="MA Strategy", linewidth=2)
    ax.plot(sharp_eq_norm, label="Sharpe-Optimal", linewidth=2, color="magenta")
    ax.plot(risk_on_norm, label="100% Risk-On", alpha=0.65)
    ax.plot(hybrid_eq_norm, label="Hybrid SIG", linewidth=2, color="blue")
    ax.plot(pure_sig_norm, label="Pure SIG", linewidth=2, color="orange")
    ax.plot(plot_ma_norm, label=f"MA({best_len}) {best_type.upper()}", linestyle="--", color="black", alpha=0.6)

    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)


# ============================================================
# LAUNCH APP
# ============================================================

if __name__ == "__main__":
    main()


