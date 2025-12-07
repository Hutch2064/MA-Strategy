import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import io
from scipy.optimize import minimize

# ============================================
# CONFIG
# ============================================

DEFAULT_START_DATE = "1999-01-01"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "UGL": .22,
    "QLD": .39,
    "BTC-USD": .39,
}

RISK_OFF_WEIGHTS = {
    "SHY": 1.0,
}

FLIP_COST = 0.05

QUARTER_DAYS = 63
START_RISKY = 0.60
START_SAFE = 0.40


# ============================================
# DATA LOADING
# ============================================

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


# ============================================
# BUILD PORTFOLIO INDEX — SIMPLE RETURNS
# ============================================

def build_portfolio_index(prices, weights_dict):
    simple_rets = prices.pct_change().fillna(0)
    idx_rets = pd.Series(0.0, index=simple_rets.index)

    for a, w in weights_dict.items():
        if a in simple_rets.columns:
            idx_rets += simple_rets[a] * w

    return (1 + idx_rets).cumprod()


# ============================================
# MA MATRIX
# ============================================

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


# ============================================
# TESTFOL SIGNAL LOGIC
# ============================================

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
# NEW: SIG ENGINE — TRUE JASON KELLY LOGIC (Enhanced)
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
    
    # === Flip detection identical to MA strategy ===
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
    rebalance_events = 0
    risky_val_series = []
    safe_val_series = []

    for i, date in enumerate(dates):
        r_on  = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])

        if ma_on:

            if frozen_risky is not None:
                if pure_sig_rw is not None and pure_sig_sw is not None:
                    w_r = pure_sig_rw.iloc[i]
                    w_s = pure_sig_sw.iloc[i]
                else:
                    w_r = START_RISKY
                    w_s = START_SAFE

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
                    
                # === Quarterly drag fee (NEW STEP 3) ===
                quarter_fee = flip_cost * target_quarter
                eq *= (1 - quarter_fee)

            eq = risky_val + safe_val
            risky_w = risky_val / eq if eq > 0 else 0
            safe_w  = safe_val  / eq if eq > 0 else 0
            
            # === Hybrid flip cost identical to MA strategy ===
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
# ============================================
# BACKTEST ENGINE (unchanged)
# ============================================

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
    max_dd = dd.min()

    total_ret = eq_curve.iloc[-1] / eq_curve.iloc[0] - 1

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": total_ret,
        "DD_Series": dd,
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


# ============================================
# GRID SEARCH (unchanged)
# ============================================

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

                sharpe_adj = result["performance"]["Sharpe"]

                idx += 1
                if idx % 200 == 0:
                    progress.progress(idx / total)

                if (
                    sharpe_adj > best_sharpe or
                    (sharpe_adj == best_sharpe and trades_per_year < best_trades)
                ):
                    best_sharpe = sharpe_adj
                    best_trades = trades_per_year
                    best_cfg = (length, ma_type, tol)
                    best_result = result

    return best_cfg, best_result


def normalize(eq):
    return eq / eq.iloc[0] * 10000

# ===== NEW: Quarter Progress Helper =====
def compute_quarter_progress(risky_start, risky_today, quarterly_target):
    target = risky_start * (1 + quarterly_target)
    gap = target - risky_today
    pct_gap = gap / risky_start if risky_start > 0 else 0

    return {
        "Risky Start ($)": risky_start,
        "Risky Today ($)": risky_today,
        "Quarterly Target ($)": target,
        "Gap ($)": gap,
        "Gap (%)": pct_gap
    }

# ============================================
# STREAMLIT APP
# ============================================

def main():

    st.set_page_config(page_title="Portfolio MA Regime Strategy", layout="wide")
    st.title("Portfolio Strategy")

    st.sidebar.header("Backtest Settings")
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")

    st.sidebar.header("Risk-ON Portfolio")
    risk_on_tickers_str = st.sidebar.text_input("Tickers", ",".join(RISK_ON_WEIGHTS.keys()))
    risk_on_weights_str = st.sidebar.text_input("Weights", ",".join(str(w) for w in RISK_ON_WEIGHTS.values()))

    st.sidebar.header("Risk-OFF Portfolio")
    risk_off_tickers_str = st.sidebar.text_input("Tickers", ",".join(RISK_OFF_WEIGHTS.keys()))
    risk_off_weights_str = st.sidebar.text_input("Weights", ",".join(str(w) for w in RISK_OFF_WEIGHTS.values()))

    st.sidebar.header("SIG Rebalance Settings")
    strategy_start_date = st.sidebar.date_input(
        "Enter the date you LAST rebalanced",
        value=pd.Timestamp.today().date()
    )
    
    strategy_start_date = pd.Timestamp(strategy_start_date)

    st.sidebar.header("SIG Rebalancing Inputs (RISKY dollars only)")

    # ACCOUNT 1 — Taxable
    risky_start_1 = st.sidebar.number_input(
        "Taxable – Deployed Dollars at Quarter Start",
        min_value=0.0,
        value=6000.0,   # example default: 60% of 10k
        step=100.0
    )
    risky_today_1 = st.sidebar.number_input(
        "Taxable – Deployed Dollars Today",
        min_value=0.0,
        value=6000.0,
        step=100.0
    )

    # ACCOUNT 2 — Tax-Sheltered
    risky_start_2 = st.sidebar.number_input(
        "Tax-Sheltered – Deployed Dollars at Quarter Start",
        min_value=0.0,
        value=6000.0,
        step=100.0
    )
    risky_today_2 = st.sidebar.number_input(
        "Tax-Sheltered – Deployed Dollars Today",
        min_value=0.0,
        value=6000.0,
        step=100.0
    )

    # ACCOUNT 3 — Joint
    risky_start_3 = st.sidebar.number_input(
        "Joint (Taxable) – Deployed Dollars at Quarter Start",
        min_value=0.0,
        value=6000.0,
        step=100.0
    )
    risky_today_3 = st.sidebar.number_input(
        "Joint (Taxable) – Deployed Dollars Today",
        min_value=0.0,
        value=6000.0,
        step=100.0
    )
    
    if not st.sidebar.button("Run Backtest & Optimize"):
        st.stop()

    risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",")]
    risk_on_weights_list = [float(x) for x in risk_on_weights_str.split(",")]
    risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))

    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))

    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers))
    end_val = end if end.strip() else None

    # Load prices
    prices = load_price_data(all_tickers, start, end_val).dropna(how="any")

    # Run MA optimization
    best_cfg, best_result = run_grid_search(
        prices,
        risk_on_weights,
        risk_off_weights,
        FLIP_COST
    )
    best_len, best_type, best_tol = best_cfg

    sig = best_result["signal"]
    perf = best_result["performance"]
    
    # === Compute MA used for signal (needed throughout UI) ===
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    opt_ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]
    
    # === Trades per Year Calculation (must be defined early) ===
    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252)

    # ============================================
    # ALWAYS-ON RISK-ON PERFORMANCE
    # ============================================

    simple_rets = prices.pct_change().fillna(0)

    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_on_eq = (1 + risk_on_simple).cumprod()
    risk_on_perf = compute_performance(risk_on_simple, risk_on_eq)

    # ============================================
    # SHARPE OPTIMAL PORTFOLIO
    # ============================================

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
    cons = ({ "type": "eq", "fun": lambda w: np.sum(w) - 1 })

    res = minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=cons)
    w_opt = res.x

    sharp_returns = (risk_on_rets * w_opt).sum(axis=1)
    sharp_eq = (1 + sharp_returns).cumprod()
    sharp_perf = compute_performance(sharp_returns, sharp_eq)

    sharp_stats = {
        "CAGR": sharp_perf["CAGR"],
        "Volatility": sharp_perf["Volatility"],
        "Sharpe": sharp_perf["Sharpe"],
        "MaxDD": sharp_perf["MaxDrawdown"],
        "Total": sharp_perf["TotalReturn"],
        "MAR": sharp_perf["CAGR"] / abs(sharp_perf["MaxDrawdown"]) if sharp_perf["MaxDrawdown"] != 0 else np.nan,
        "TID": (sharp_perf["DD_Series"] < 0).mean(),
        "PainGain": sharp_perf["CAGR"] / np.sqrt((sharp_perf["DD_Series"]**2).mean()) if (sharp_perf["DD_Series"]**2).mean() != 0 else np.nan,
        "Skew": sharp_returns.skew(),
        "Kurtosis": sharp_returns.kurt(),
        "Trades/year": 0.0,
    }

    sharp_weights_display = {t: round(w, 4) for t, w in zip(risk_on_px.columns, w_opt)}

    # ============================================
    # HYBRID SIG STRATEGY
    # ============================================

    pure_sig_rw = None   # placeholder: pure SIG weights will be computed later
    pure_sig_sw = None

    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252/len(risk_on_eq)) - 1
    quarterly_target = (1 + bh_cagr)**(1/4) - 1
    
    quarterly_target = (1 + bh_cagr)**(1/4) - 1

    # ============================================================
    # AUTOMATED HYBRID STRATEGY DECISION BLOCK
    # ============================================================

    # 1. MA Regime Today
    latest_signal = sig.iloc[-1]
    regime = "RISK-ON" if latest_signal else "RISK-OFF"
    st.write(f"### Current MA Regime: **{regime}**")

    # 3. Quarterly Target Check
    st.write("### SIG Quarterly Target Check")
    st.write(f"**Quarterly Target Growth:** {quarterly_target:.2%}")

    prog_auto_1 = compute_quarter_progress(risky_start_1, risky_today_1, quarterly_target)
    prog_auto_2 = compute_quarter_progress(risky_start_2, risky_today_2, quarterly_target)
    prog_auto_3 = compute_quarter_progress(risky_start_3, risky_today_3, quarterly_target)

    auto_prog = pd.concat([
        pd.DataFrame.from_dict(prog_auto_1, orient='index', columns=['Taxable']),
        pd.DataFrame.from_dict(prog_auto_2, orient='index', columns=['Tax-Sheltered']),
        pd.DataFrame.from_dict(prog_auto_3, orient='index', columns=['Joint (Taxable)']),
    ], axis=1)

    auto_prog.loc["Gap (%)"] = auto_prog.loc["Gap (%)"].apply(lambda x: f"{x:.2%}")
    st.dataframe(auto_prog)

    def rebalance_text(gap, next_q, days_to_next_q):
        date_str = next_q.strftime("%m/%d/%Y")
        days_str = f"{days_to_next_q} days" if days_to_next_q >= 0 else "0 days"

        if gap > 0:
            return (
                f"Increase deployed sleeve by **${gap:,.2f}** "
                f"on **{date_str}** ({days_str})"
            )
        elif gap < 0:
            return (
                f"Decrease deployed sleeve by **${abs(gap):,.2f}** "
                f"on **{date_str}** ({days_str})"
            )
        else:
            return (
                f"No rebalance needed until **{date_str}** ({days_str})"
            )
    # =====================================================
    # ROLLING 63-DAY REBALANCE LOGIC — NEEDED ABOVE rebalance_text
    # =====================================================

    today = prices.index[-1]
    next_q = strategy_start_date + pd.Timedelta(days=QUARTER_DAYS)

    while next_q <= today:
        next_q += pd.Timedelta(days=QUARTER_DAYS)

    days_to_next_q = (next_q - today).days        

    st.write(f"**Taxable:** {rebalance_text(prog_auto_1['Gap ($)'], next_q, days_to_next_q)}")
    st.write(f"**Tax-Sheltered:** {rebalance_text(prog_auto_2['Gap ($)'], next_q, days_to_next_q)}")
    st.write(f"**Joint (Taxable):** {rebalance_text(prog_auto_3['Gap ($)'], next_q, days_to_next_q)}")

# ============================================================
    
    hybrid_eq, hybrid_rw, hybrid_sw, hybrid_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        sig,
        pure_sig_rw=pure_sig_rw,
        pure_sig_sw=pure_sig_sw,
        flip_cost=FLIP_COST
    )

    hybrid_simple = hybrid_eq.pct_change().fillna(0)
    hybrid_perf = compute_performance(hybrid_simple, hybrid_eq)

    # ============================================
    # ADVANCED METRICS
    # ============================================

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

    # ============================================
    # PURE SIG STRATEGY (NO MA FILTER)
    # ============================================

    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)

    pure_sig_eq, pure_sig_rw, pure_sig_sw, pure_sig_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        pure_sig_signal,
        flip_cost=FLIP_COST
    )

    pure_sig_simple = pure_sig_eq.pct_change().fillna(0)
    pure_sig_perf = compute_performance(pure_sig_simple, pure_sig_eq)

    pure_sig_stats = compute_stats(
        pure_sig_perf,
        pure_sig_simple,
        pure_sig_perf["DD_Series"],
        np.zeros(len(pure_sig_simple), dtype=bool),
        0
    )

    pure_sig_stats["QuarterlyTarget"] = quarterly_target

    strat_stats = compute_stats(
        perf,
        best_result["returns"],
        perf["DD_Series"],
        best_result["flip_mask"],
        trades_per_year
    )

    risk_stats = compute_stats(
        risk_on_perf,
        risk_on_simple,
        risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_simple), dtype=bool),
        0
    )

    hybrid_stats = compute_stats(
        hybrid_perf,
        hybrid_simple,
        hybrid_perf["DD_Series"],
        np.zeros(len(hybrid_simple), dtype=bool),
        0
    )

    # ============================================
    # ALLOCATION HELPERS FOR 3 ACCOUNTS / 4 STRATEGIES
    # ============================================

    def compute_allocations(start_cap, risky_w, safe_w, risk_on_weights, risk_off_weights):
        risky_dollars = start_cap * risky_w
        safe_dollars  = start_cap * safe_w

        alloc = {
            "Total Risky $": risky_dollars,
            "Total Safe $": safe_dollars
        }

        # Risk-On allocations
        for ticker, w in risk_on_weights.items():
            alloc[ticker] = risky_dollars * w

        # Risk-Off allocations
        for ticker, w in risk_off_weights.items():
            alloc[ticker] = safe_dollars * w

        return alloc


    def compute_sharpe_opt_alloc(start_cap, tickers, weights):
        alloc = {}
        for t, w in zip(tickers, weights):
            alloc[t] = start_cap * w
        return alloc


    def compute_ma_allocations(start_cap, weight_row):
        alloc = {}
        for ticker, w in weight_row.items():
            alloc[ticker] = start_cap * w
        return alloc

    def add_percentage_column(alloc_dict):
        df = pd.DataFrame.from_dict(alloc_dict, orient="index", columns=["$"])

        # Case 1: We have Total Risky $ and Total Safe $
        if "Total Risky $" in df.index and "Total Safe $" in df.index:
            total_portfolio = df.loc["Total Risky $","$"] + df.loc["Total Safe $","$"]
        else:
            # Case 2: A strategy like Sharpe-optimal that only lists tickers
            total_portfolio = df["$"].sum()

        # Calculate percentage of total portfolio
        df["% TotalPortfolio"] = df["$"] / total_portfolio * 100

        # Format clean percentages
        df["% TotalPortfolio"] = df["% TotalPortfolio"].apply(lambda x: f"{x:.2f}%")

        return df
    
    avg_safe = hybrid_sw.mean()

    # ============================================
    # ACCOUNT-LEVEL ALLOCATIONS (3 ACCOUNTS × 4 STRATEGIES)
    # ===========================================

    # ============================================================
    # REAL-WORLD ACCOUNT ALLOCATION TABLES
    # Based ONLY on user's actual portfolio today
    # ============================================================

    # These are the real-world total portfolio values (not historical drift)
    real_cap_1 = risky_today_1 / START_RISKY
    real_cap_2 = risky_today_2 / START_RISKY
    real_cap_3 = risky_today_3 / START_RISKY
    
    # ------------------------------------------------------------
    # HYBRID STRATEGY — TODAY’S RECOMMENDED WEIGHTS
    # (NOT historical drift — real world signal)
    # ------------------------------------------------------------
    # Hybrid recommended risky % = START_RISKY when MA signal is ON, else 0%
    hyb_risk_today = START_RISKY if latest_signal else 0.0
    hyb_safe_today = 1 - hyb_risk_today

    hyb_alloc_1 = compute_allocations(real_cap_1, hyb_risk_today, hyb_safe_today, risk_on_weights, risk_off_weights)
    hyb_alloc_2 = compute_allocations(real_cap_2, hyb_risk_today, hyb_safe_today, risk_on_weights, risk_off_weights)
    hyb_alloc_3 = compute_allocations(real_cap_3, hyb_risk_today, hyb_safe_today, risk_on_weights, risk_off_weights)

    # ------------------------------------------------------------
    # PURE SIG STRATEGY — ALWAYS RISK-ON (NO MA FILTER)
    # (Use fixed START_RISKY / START_SAFE, NOT historical drift weights)
    # ------------------------------------------------------------
    pure_risk_today = START_RISKY
    pure_safe_today = START_SAFE

    pure_alloc_1 = compute_allocations(real_cap_1, pure_risk_today, pure_safe_today, risk_on_weights, risk_off_weights)
    pure_alloc_2 = compute_allocations(real_cap_2, pure_risk_today, pure_safe_today, risk_on_weights, risk_off_weights)
    pure_alloc_3 = compute_allocations(real_cap_3, pure_risk_today, pure_safe_today, risk_on_weights, risk_off_weights)

    # ------------------------------------------------------------
    # 100% RISK-ON STRATEGY
    # ------------------------------------------------------------
    riskon_alloc_1 = compute_allocations(real_cap_1, 1.0, 0.0, risk_on_weights, {"SHY": 0})
    riskon_alloc_2 = compute_allocations(real_cap_2, 1.0, 0.0, risk_on_weights, {"SHY": 0})
    riskon_alloc_3 = compute_allocations(real_cap_3, 1.0, 0.0, risk_on_weights, {"SHY": 0})

    # ------------------------------------------------------------
    # SHARPE-OPTIMAL STRATEGY — USING REAL MONEY
    # ------------------------------------------------------------
    sharpe_alloc_1 = compute_sharpe_opt_alloc(real_cap_1, risk_on_px.columns, w_opt)
    sharpe_alloc_2 = compute_sharpe_opt_alloc(real_cap_2, risk_on_px.columns, w_opt)
    sharpe_alloc_3 = compute_sharpe_opt_alloc(real_cap_3, risk_on_px.columns, w_opt)

    # ------------------------------------------------------------
    # MA STRATEGY — TODAY’S SIGNAL, NOT HISTORICAL WEIGHTS
    # ------------------------------------------------------------
    if latest_signal:
        # MA says RISK-ON today
        ma_alloc_1 = compute_allocations(real_cap_1, 1.0, 0.0, risk_on_weights, {"SHY": 0})
        ma_alloc_2 = compute_allocations(real_cap_2, 1.0, 0.0, risk_on_weights, {"SHY": 0})
        ma_alloc_3 = compute_allocations(real_cap_3, 1.0, 0.0, risk_on_weights, {"SHY": 0})
    else:
        # MA says RISK-OFF today
        ma_alloc_1 = compute_allocations(real_cap_1, 0.0, 1.0, {}, risk_off_weights)
        ma_alloc_2 = compute_allocations(real_cap_2, 0.0, 1.0, {}, risk_off_weights)
        ma_alloc_3 = compute_allocations(real_cap_3, 0.0, 1.0, {}, risk_off_weights)
    
    # ============================================
    # METRIC TABLE — 4 COLUMNS
    # ============================================

    st.subheader("Ma Strategy vs. Sharpe-Opt B&H (Limited 9/17/2014) vs. Buy & Hold vs. Hybrid MA/SIG vs. Pure SIG")

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
        sh = sharp_stats.get(key, np.nan)
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
        columns=["Metric", "MA Strategy", "Sharpe-Opt Buy & Hold (Limited: 09/17/2014)", "Buy & Hold", "Hybrid MA/SIG", "SIG"]
    )

    st.dataframe(stat_table, use_container_width=True)

    # =====================================================
    # QUARTERLY PROGRESS TRACKER — EXACT VALUES TODAY
    # =====================================================
    
    # ============================================
    # ACCOUNT ALLOCATION TABLES
    # ============================================

    st.subheader("Account-Level Allocations")

    tab1, tab2, tab3 = st.tabs(["Taxable", "Tax-Sheltered", "Joint (Taxable)"])

    with tab1:
        st.write("### Hybrid SIG Allocation")
        st.dataframe(add_percentage_column(hyb_alloc_1))

        st.write("### Pure SIG Allocation")
        st.dataframe(add_percentage_column(pure_alloc_1))

        st.write("### Risk-ON Allocation")
        st.dataframe(add_percentage_column(riskon_alloc_1))

        st.write("### Sharpe-Optimal Allocation")
        st.dataframe(add_percentage_column(sharpe_alloc_1))
        
        st.write("### MA Strategy Allocation")
        st.dataframe(add_percentage_column(ma_alloc_1))




    with tab2:
        st.write("### Hybrid SIG Allocation")
        st.dataframe(add_percentage_column(hyb_alloc_2))

        st.write("### Pure SIG Allocation")
        st.dataframe(add_percentage_column(pure_alloc_2))
    
        st.write("### Risk-ON Allocation")
        st.dataframe(add_percentage_column(riskon_alloc_2))

        st.write("### Sharpe-Optimal Allocation")
        st.dataframe(add_percentage_column(sharpe_alloc_2))

        st.write("### MA Strategy Allocation")
        st.dataframe(add_percentage_column(ma_alloc_2))

    with tab3:
        st.write("### Hybrid SIG Allocation")
        st.dataframe(add_percentage_column(hyb_alloc_3))

        st.write("### Pure SIG Allocation")
        st.dataframe(add_percentage_column(pure_alloc_3))

        st.write("### Risk-ON Allocation")
        st.dataframe(add_percentage_column(riskon_alloc_3))

        st.write("### Sharpe-Optimal Allocation")
        st.dataframe(add_percentage_column(sharpe_alloc_3))

        st.write("### MA Strategy Allocation")
        st.dataframe(add_percentage_column(ma_alloc_3))

        st.write(f"**Hybrid — Rebalance Events:** {hybrid_rebals}")

    # ============================================
    # EXTERNAL VALIDATION LINK
    # ============================================
    st.subheader("External Sharpe Optimal Validation Link")
    st.markdown(
        f"**Quick Access:** [View the Sharpe Optimal recommended portfolio](https://testfol.io/optimizer?s=9y4FBdfW2oO)"
    )

    # ============================================
    # OPTIMAL SIGNAL PARAMETERS
    # ============================================

    st.subheader("Optimal MA Signal Parameters")
    st.write(f"**Moving Average Type:** {best_type.upper()}")
    st.write(f"**Optimal MA Length:** {best_len}")
    st.write(f"**Optimal Tolerance:** {best_tol:.2%}")

    # ============================================
    # SIGNAL DISTANCE
    # ============================================

    st.subheader("Next MA Signal Information")

    latest_date = opt_ma.dropna().index[-1]
    P = float(portfolio_index.loc[latest_date])
    MA = float(opt_ma.loc[latest_date])

    upper = MA * (1 + best_tol)
    lower = MA * (1 - best_tol)

    if latest_signal:
        delta = (P - lower) / P
        st.write(f"**Drop Required:** {delta:.2%}")
    else:
        delta = (upper - P) / P
        st.write(f"**Gain Required:** {delta:.2%}")

    # ============================================
    # REGIME STATS
    # ============================================

    st.subheader("Regime Statistics")

    sig_int = sig.astype(int)
    flips = sig_int.diff().fillna(0).ne(0)

    segments = []
    current = sig_int.iloc[0]
    start_date = sig_int.index[0]

    for date, sw in flips.iloc[1:].items():
        if sw:
            segments.append((current, start_date, date))
            current = sig_int.loc[date]
            start_date = date

    segments.append((current, start_date, sig_int.index[-1]))

    regime_rows = []
    for r, s, e in segments:
        label = "RISK-ON" if r == 1 else "RISK-OFF"
        regime_rows.append([label, s.date(), e.date(), (e - s).days])

    regime_df = pd.DataFrame(regime_rows, columns=["Regime", "Start", "End", "Duration (days)"])

    avg_on = regime_df[regime_df["Regime"] == "RISK-ON"]["Duration (days)"].mean()
    avg_off = regime_df[regime_df["Regime"] == "RISK-OFF"]["Duration (days)"].mean()

    st.write(f"**Avg RISK-ON:** {avg_on:.1f} days")
    st.write(f"**Avg RISK-OFF:** {avg_off:.1f} days")
    st.dataframe(regime_df, use_container_width=True)

    # ============================================
    # FINAL PLOT
    # ============================================

    st.subheader("Portfolio Strategy vs. Sharpe-Optimal vs. Hybrid vs. Risk-On")

    # === MA LINE FOR PLOTTING ===
    plot_index = build_portfolio_index(prices, risk_on_weights)
    plot_ma = compute_ma_matrix(plot_index, [best_len], best_type)[best_len]

    # Normalize both to 10,000 like your strategy curves
    plot_index_norm = plot_index / plot_index.iloc[0] * 10000
    plot_ma_norm = plot_ma / plot_ma.dropna().iloc[0] * 10000

    fig, ax = plt.subplots(figsize=(12, 6))

    strat_eq_norm  = normalize(best_result["equity_curve"])
    sharp_eq_norm  = normalize(sharp_eq)
    risk_on_norm   = normalize(portfolio_index)
    hybrid_eq_norm = normalize(hybrid_eq)
    pure_sig_norm  = normalize(pure_sig_eq)

    ax.plot(strat_eq_norm,  label="MA Strategy", linewidth=2)
    ax.plot(sharp_eq_norm,  label="Sharpe-Optimal", linewidth=2, color="magenta")
    ax.plot(risk_on_norm,   label="Risk-On Portfolio", alpha=0.65)
    ax.plot(hybrid_eq_norm, label="Hybrid SIG Strategy", linewidth=2, color="blue")
    ax.plot(pure_sig_norm,  label="Pure SIG (No MA Filter)", linewidth=2, color="orange")
    ax.plot(plot_ma_norm, label=f"MA({best_len}) {best_type.upper()}", linestyle="--", color="black", alpha=0.8)

    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)


# ============================================
# LAUNCH APP
# ============================================

if __name__ == "__main__":
    main()
