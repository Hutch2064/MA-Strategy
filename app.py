import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import datetime
from scipy.optimize import minimize

# ============================================================
# CONFIG
# ============================================================

DEFAULT_START_DATE = "1900-01-01"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "BTC-USD": .5,
    "FNGO": .5,
}

RISK_OFF_WEIGHTS = {
    "SHY": 1.0,
}

FLIP_COST = 0.0005

# Starting weights inside the SIG engine (unchanged)
START_RISKY = 0.70
START_SAFE  = 0.30

# FIXED PARAMETERS
FIXED_MA_LENGTH = 200
FIXED_MA_TYPE = "sma"  # or "ema" - you can choose which one to fix
FIXED_TOLERANCE = 0.0  # 2%

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
    
    # Create cumulative product
    cumprod = (1 + idx_rets).cumprod()
    
    # Find first valid (non-NaN, non-zero) index
    valid_mask = cumprod.notna() & (cumprod > 0)
    if not valid_mask.any():
        # All values are NaN or zero - return a constant series
        return pd.Series(1.0, index=cumprod.index)
    
    first_valid_idx = cumprod[valid_mask].index[0]
    
    # Forward fill from first valid value
    cumprod_filled = cumprod.copy()
    cumprod_filled.loc[:first_valid_idx] = 1.0  # Set early values to 1.0
    cumprod_filled = cumprod_filled.ffill()
    
    return cumprod_filled


# ============================================================
# MA MATRIX
# ============================================================

def compute_ma(price_series, length, ma_type):
    """Compute a single MA with fixed parameters"""
    if ma_type == "ema":
        ma = price_series.ewm(span=length, adjust=False).mean()
    else:
        ma = price_series.rolling(window=length, min_periods=1).mean()
    
    return ma.shift(1)


# ============================================================
# TESTFOL SIGNAL LOGIC - ROBUST VERSION
# ============================================================

def generate_testfol_signal_vectorized(price, ma, tol):
    px = price.values
    ma_vals = ma.values
    n = len(px)
    
    # Handle case where all values are NaN
    if np.all(np.isnan(ma_vals)):
        return pd.Series(False, index=ma.index)
    
    upper = ma_vals * (1 + tol)
    lower = ma_vals * (1 - tol)
    
    sig = np.zeros(n, dtype=bool)
    
    # Find first non-NaN index
    non_nan_mask = ~np.isnan(ma_vals)
    if not np.any(non_nan_mask):
        return pd.Series(False, index=ma.index)
    
    first_valid = np.where(non_nan_mask)[0][0]
    if first_valid == 0:
        first_valid = 1
    start_index = first_valid + 1
    
    # Ensure start_index is valid
    if start_index >= n:
        return pd.Series(False, index=ma.index)
    
    for t in range(start_index, n):
        if np.isnan(px[t]) or np.isnan(upper[t]) or np.isnan(lower[t]):
            sig[t] = sig[t-1] if t > 0 else False
        elif not sig[t - 1]:
            sig[t] = px[t] > upper[t]
        else:
            sig[t] = not (px[t] < lower[t])
    
    return pd.Series(sig, index=ma.index).fillna(False)


# ============================================================
# SIG ENGINE — NOW USING CALENDAR QUARTER-ENDS (B1)
# ============================================================

def run_sig_engine(
    risk_on_returns,
    risk_off_returns,
    target_quarter,
    ma_signal,
    pure_sig_rw=None,
    pure_sig_sw=None,
    flip_cost=FLIP_COST,
    quarter_end_dates=None,   # <-- must be mapped_q_ends
    quarterly_multiplier=4.0,  # NEW: 2x for SIG, 2x for Sigma (quarterly part)
    ma_flip_multiplier=4.0     # NEW: 4x for Sigma when MA flips
):

    dates = risk_on_returns.index
    n = len(dates)

    if quarter_end_dates is None:
        raise ValueError("quarter_end_dates must be supplied")

    # Fast lookup
    quarter_end_set = set(quarter_end_dates)

    # MA flip detection
    sig_arr = ma_signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    # Init values
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
    rebalance_dates = []

    for i in range(n):
        date = dates[i]
        r_on = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])
        
        # ============================================
        # FIX: Apply MA flip costs BEFORE checking regime
        # ============================================
        if i > 0 and flip_mask.iloc[i]:  # Skip first day (no diff)
            eq *= (1 - flip_cost * ma_flip_multiplier)  # 4x cost for Sigma
        # ============================================

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

            # Rebalance ON quarter-end date (correct logic)
            if date in quarter_end_set:

                # Identify actual quarter start (previous quarter end)
                prev_qs = [qd for qd in quarter_end_dates if qd < date]

                if prev_qs:
                    prev_q = prev_qs[-1]

                    idx_prev = dates.get_loc(prev_q)

                    # Risky sleeve at the start of this quarter
                    risky_at_qstart = risky_val_series[idx_prev]

                    # Quarterly growth target
                    goal_risky = risky_at_qstart * (1 + target_quarter)

                    # --- Apply SIG logic (unchanged) ---
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

                    # Apply quarterly fee with multiplier (2x for SIG, 2x for Sigma quarterly part)
                    eq *= (1 - flip_cost * quarterly_multiplier)  # Remove target_quarter!

            # Update equity
            eq = risky_val + safe_val
            risky_w = risky_val / eq
            safe_w  = safe_val  / eq

        else:
            # Freeze values on entering RISK-OFF
            if frozen_risky is None:
                frozen_risky = risky_val
                frozen_safe  = safe_val

            # Only safe sleeve earns returns
            eq *= (1 + r_off)
            risky_w = 0.0
            safe_w  = 1.0

        # Store values
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
    if len(eq_curve) == 0 or eq_curve.iloc[0] == 0:
        return {
            "CAGR": 0,
            "Volatility": 0,
            "Sharpe": 0,
            "MaxDrawdown": 0,
            "TotalReturn": 0,
            "DD_Series": pd.Series([], dtype=float)
        }
    
    cagr = (eq_curve.iloc[-1] / eq_curve.iloc[0]) ** (252 / len(eq_curve)) - 1
    vol = simple_returns.std() * np.sqrt(252) if len(simple_returns) > 0 else 0
    sharpe = (simple_returns.mean() * 252 - rf) / vol if vol > 0 else 0
    dd = eq_curve / eq_curve.cummax() - 1

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": dd.min() if len(dd) > 0 else 0,
        "TotalReturn": eq_curve.iloc[-1] / eq_curve.iloc[0] - 1 if eq_curve.iloc[0] != 0 else 0,
        "DD_Series": dd
    }


def backtest(prices, signal, risk_on_weights, risk_off_weights, flip_cost, ma_flip_multiplier=4.0):
    simple = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    # MA flip costs with 4x multiplier for MA Strategy
    flip_costs = np.where(flip_mask, -flip_cost * ma_flip_multiplier, 0.0)
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
# QUARTERLY PROGRESS HELPER (unchanged)
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
    if len(eq) == 0 or eq.iloc[0] == 0:
        return eq
    return eq / eq.iloc[0] * 10000


# ============================================================
# 4-PANEL DIAGNOSTIC PERFORMANCE PLOT
# ============================================================

def plot_diagnostics(hybrid_eq, bh_eq, hybrid_signal):

    # Normalize to common starting value
    hybrid_eq = hybrid_eq / hybrid_eq.iloc[0]
    bh_eq     = bh_eq / bh_eq.iloc[0]

    hybrid_ret = hybrid_eq.pct_change().fillna(0)
    bh_ret = bh_eq.pct_change().fillna(0)

    hybrid_dd = hybrid_eq / hybrid_eq.cummax() - 1
    bh_dd = bh_eq / bh_eq.cummax() - 1

    window = 252
    roll_sharpe_h = hybrid_ret.rolling(window).mean() / hybrid_ret.rolling(window).std() * np.sqrt(252)
    roll_sharpe_b = bh_ret.rolling(window).mean() / bh_ret.rolling(window).std() * np.sqrt(252)

    hybrid_m = hybrid_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)
    bh_m = bh_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # === Panel 1: Cumulative returns + regime shading ===
    ax1.plot(hybrid_eq, label="Sigma", linewidth=2, color="green")
    ax1.plot(bh_eq, label="Buy & Hold", linewidth=2, alpha=0.7)

    in_off = False
    start = None
    for date, on in hybrid_signal.items():
        if not on and not in_off:
            start = date
            in_off = True
        elif on and in_off:
            ax1.axvspan(start, date, color="red", alpha=0.15)
            in_off = False
    if in_off:
        ax1.axvspan(start, hybrid_signal.index[-1], color="red", alpha=0.15)

    ax1.set_title("Cumulative Returns with Regime Shading")
    ax1.set_ylabel("Growth of $1")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # === Panel 2: Drawdowns ===
    ax2.plot(hybrid_dd * 100, label="Sigma", linewidth=1.5, color="green")
    ax2.plot(bh_dd * 100, label="Buy & Hold", linewidth=1.5, alpha=0.7)
    ax2.set_title("Drawdown Comparison (%)")
    ax2.set_ylabel("Drawdown %")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # === Panel 3: Rolling Sharpe ===
    ax3.plot(roll_sharpe_h, label="Sigma", linewidth=1.5, color="green")
    ax3.plot(roll_sharpe_b, label="Buy & Hold", linewidth=1.5, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_title("Rolling 252-Day Sharpe Ratio")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # === Panel 4: Monthly return distribution ===
    bins = np.linspace(
        min(hybrid_m.min(), bh_m.min()),
        max(hybrid_m.max(), bh_m.max()),
        20
    )

    ax4.hist(hybrid_m, bins=bins, alpha=0.7, density=True, label="Sigma")
    ax4.hist(bh_m, bins=bins, alpha=0.5, density=True, label="Buy & Hold")
    ax4.axvline(0, color="black", linestyle="--", linewidth=1)
    ax4.set_title("Monthly Returns Distribution")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    return fig
    
# ============================================================
# STREAMLIT APP
# ============================================================

def main():

    st.set_page_config(page_title="Portfolio MA Regime Strategy", layout="wide")
    st.title("Portfolio Strategy")

    # Backtest inputs unchanged...
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")

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
    
    # REMOVED OPTIMIZATION SETTINGS
    
    st.sidebar.header("Quarterly Portfolio Values")
    qs_cap_1 = st.sidebar.number_input("Taxable – Portfolio Value at Last Rebalance ($)", min_value=0.0, value=75815.26, step=100.0)
    qs_cap_2 = st.sidebar.number_input("Tax-Sheltered – Portfolio Value at Last Rebalance ($)", min_value=0.0, value=10074.83, step=100.0)
    qs_cap_3 = st.sidebar.number_input("Joint – Portfolio Value at Last Rebalance ($)", min_value=0.0, value=4189.76, step=100.0)

    st.sidebar.header("Current Portfolio Values (Today)")
    real_cap_1 = st.sidebar.number_input("Taxable – Portfolio Value Today ($)", min_value=0.0, value=70959.35, step=100.0)
    real_cap_2 = st.sidebar.number_input("Tax-Sheltered – Portfolio Value Today ($)", min_value=0.0, value=8988.32, step=100.0)
    real_cap_3 = st.sidebar.number_input("Joint – Portfolio Value Today ($)", min_value=0.0, value=4064.94, step=100.0)

    # Add fixed parameters display
    st.sidebar.header("Fixed Parameters")
    st.sidebar.write(f"**MA Length:** {FIXED_MA_LENGTH}")
    st.sidebar.write(f"**MA Type:** {FIXED_MA_TYPE.upper()}")
    st.sidebar.write(f"**Tolerance:** {FIXED_TOLERANCE:.1%}")

    run_clicked = st.sidebar.button("Run Backtest")
    if not run_clicked:
        st.stop()

    risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",")]
    risk_on_weights_list = [float(x) for x in risk_on_weights_str.split(",")]
    risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))

    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))

    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers))
    end_val = end if end.strip() else None

    prices = load_price_data(all_tickers, start, end_val).dropna(how="any")
    
    # Check if we have any data
    if len(prices) == 0:
        st.error("No data loaded. Please check your ticker symbols and date range.")
        st.stop()
    
    st.info(f"Loaded {len(prices)} trading days of data from {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # USE FIXED PARAMETERS INSTEAD OF OPTIMIZATION
    best_len, best_type, best_tol = FIXED_MA_LENGTH, FIXED_MA_TYPE, FIXED_TOLERANCE
    
    st.subheader("Fixed MA Parameters")
    st.write(f"**MA Type:** {best_type.upper()}  —  **Length:** {best_len}  —  **Tolerance:** {best_tol:.2%}")
    
    # Generate signal with fixed parameters
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    opt_ma = compute_ma(portfolio_index, best_len, best_type)
    sig = generate_testfol_signal_vectorized(portfolio_index, opt_ma, best_tol)
    
    # Run backtest with fixed parameters
    best_result = backtest(prices, sig, risk_on_weights, risk_off_weights, FLIP_COST, ma_flip_multiplier=4.0)
    
    latest_signal = sig.iloc[-1]
    current_regime = "RISK-ON" if latest_signal else "RISK-OFF"

    st.subheader(f"Current MA Regime: {current_regime}")

    perf = best_result["performance"]

    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252) if len(sig) > 0 else 0

    simple_rets = prices.pct_change().fillna(0)

    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_on_eq = (1 + risk_on_simple).cumprod()
    risk_on_perf = compute_performance(risk_on_simple, risk_on_eq)

    risk_on_px = prices[[t for t in risk_on_tickers if t in prices.columns]].dropna()
    if len(risk_on_px) > 0:
        risk_on_rets = risk_on_px.pct_change().dropna()
    else:
        risk_on_rets = pd.DataFrame()

    if len(risk_on_rets) > 0:
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
    else:
        w_opt = np.array([])
        sharp_returns = pd.Series([], dtype=float)
        sharp_eq = pd.Series([], dtype=float)
        sharp_perf = compute_performance(pd.Series([], dtype=float), pd.Series([], dtype=float))

    # ============================================================
    # REAL CALENDAR QUARTER LOGIC BEGINS HERE
    # ============================================================

    dates = prices.index
    # ============================================================
    # TRUE CALENDAR QUARTER-ENDS (academically correct)
    # ============================================================

    dates = prices.index

    # 1. Generate TRUE calendar quarter-end dates
    true_q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq='Q')

    # 2. Map each to the actual last trading day
    mapped_q_ends = []
    for qd in true_q_ends:
        valid_dates = dates[dates <= qd]
        if len(valid_dates) > 0:
            mapped_q_ends.append(valid_dates.max())

    mapped_q_ends = pd.to_datetime(mapped_q_ends)

    # -----------------------------------------------------------
    # FIXED: TRUE CALENDAR QUARTER LOGIC (never depends on prices)
    # -----------------------------------------------------------

    today_date = pd.Timestamp.today().normalize()

    # 1. Next calendar quarter-end
    true_next_q = pd.date_range(start=today_date, periods=2, freq="Q")[0]
    next_q_end = true_next_q

    # 2. Most recent completed quarter-end
    true_prev_q = pd.date_range(end=today_date, periods=2, freq="Q")[0]
    past_q_end = true_prev_q

    # 3. Days remaining until next rebalance
    days_to_next_q = (next_q_end - today_date).days
    
    # ============================================================
    # Sigma ENGINE USING REAL CALENDAR QUARTERS
    # ============================================================

    # Annualized CAGR → quarterly target unchanged
    if len(risk_on_eq) > 0 and risk_on_eq.iloc[0] != 0:
        bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
        quarterly_target = (1 + bh_cagr) ** (1/4) - 1
    else:
        bh_cagr = 0
        quarterly_target = 0

    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    # SIG (always RISK-ON) - 2x flip costs quarterly
    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)

    pure_sig_eq, pure_sig_rw, pure_sig_sw, pure_sig_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        pure_sig_signal,
        quarter_end_dates=mapped_q_ends,
        quarterly_multiplier=4.0,  # 2x for SIG
        ma_flip_multiplier=4.0     # No MA flips for SIG
    )

    # Sigma (MA Filter) - 2x quarterly + 4x MA flips = 6x total
    hybrid_eq, hybrid_rw, hybrid_sw, hybrid_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        sig,  # <-- CRITICAL: Uses the SAME optimized signal
        pure_sig_rw=pure_sig_rw,
        pure_sig_sw=pure_sig_sw,
        quarter_end_dates=mapped_q_ends,
        quarterly_multiplier=4.0,  # 2x quarterly part
        ma_flip_multiplier=4.0     # 4x when MA flips
    )
    
    # ============================================================
    # CANONICAL STRATEGY PERFORMANCE — Sigma ONLY
    # ============================================================
    hybrid_simple = hybrid_eq.pct_change().fillna(0)
    hybrid_perf = compute_performance(hybrid_simple, hybrid_eq)

    # IMPORTANT: `perf` always means Sigma performance
    perf = hybrid_perf
    
    # ============================================================
    # DISPLAY ACTUAL Sigma REBALANCE DATES (FULL HISTORY)
    # ============================================================
    if len(hybrid_rebals) > 0:
        reb_df = pd.DataFrame({"Rebalance Date": pd.to_datetime(hybrid_rebals)})
        st.subheader("Sigma – Actual Rebalance Dates (Historical)")
        st.dataframe(reb_df)
    else:
        st.subheader("Sigma/MA – Historical Rebalance Dates")
        st.write("No Sigma/MA rebalances occurred during the backtest.")

    # Quarter start should follow the last actual SIG rebalance
    if len(hybrid_rebals) > 0:
        quarter_start_date = hybrid_rebals[-1]
    else:
        quarter_start_date = dates[0] if len(dates) > 0 else None

    st.subheader("Strategy Summary")
    # Display last actual SIG rebalance instead of quarter start
    if len(hybrid_rebals) > 0:
        last_reb = hybrid_rebals[-1]
        st.write(f"**Last Rebalance:** {last_reb.strftime('%Y-%m-%d')}")
    else:
        st.write("**Quarter start (last SIG rebalance):** None yet")
    st.write(f"**Next Rebalance:** {next_q_end.date()} ({days_to_next_q} days)")

    # Quarter-progress calculations
    def get_sig_progress(qs_cap, today_cap):
        if quarter_start_date is not None and len(hybrid_rw) > 0:
            risky_start = qs_cap * float(hybrid_rw.loc[quarter_start_date])
            risky_today = today_cap * float(hybrid_rw.iloc[-1])
            return compute_quarter_progress(risky_start, risky_today, quarterly_target)
        else:
            return compute_quarter_progress(0, 0, 0)

    prog_1 = get_sig_progress(qs_cap_1, real_cap_1)
    prog_2 = get_sig_progress(qs_cap_2, real_cap_2)
    prog_3 = get_sig_progress(qs_cap_3, real_cap_3)

    st.write(f"**Quarterly Target Growth Rate:** {quarterly_target:.2%}")

    prog_df = pd.concat([
        pd.DataFrame.from_dict(prog_1, orient='index', columns=['Taxable']),
        pd.DataFrame.from_dict(prog_2, orient='index', columns=['Tax-Sheltered']),
        pd.DataFrame.from_dict(prog_3, orient='index', columns=['Joint']),
    ], axis=1)

    prog_df.loc["Gap (%)"] = prog_df.loc["Gap (%)"].apply(lambda x: f"{x:.2%}")
    st.dataframe(prog_df)

    def rebalance_text(gap, next_q, days_to_next_q):
        date_str = next_q.strftime("%m/%d/%Y")
        days_str = f"{days_to_next_q} days"
        if gap > 0:
            return f"Increase deployed sleeve by **${gap:,.2f}** on **{date_str}** ({days_str})"
        elif gap < 0:
            return f"Decrease deployed sleeve by **${abs(gap):,.2f}** on **{date_str}** ({days_str})"
        else:
            return f"No rebalance needed until **{date_str}** ({days_str})"

    st.write("### Rebalance Recommendations")
    st.write("**Taxable:** "  + rebalance_text(prog_1["Gap ($)"], next_q_end, days_to_next_q))
    st.write("**Tax-Sheltered:** " + rebalance_text(prog_2["Gap ($)"], next_q_end, days_to_next_q))
    st.write("**Joint:** " + rebalance_text(prog_3["Gap ($)"], next_q_end, days_to_next_q))

    # ADVANCED METRICS (unchanged)
    def time_in_drawdown(dd): return (dd < 0).mean() if len(dd) > 0 else 0
    def mar(c, dd): return c / abs(dd) if dd != 0 else 0
    def ulcer(dd): return np.sqrt((dd**2).mean()) if len(dd) > 0 and (dd**2).mean() != 0 else 0
    def pain_gain(c, dd): return c / ulcer(dd) if ulcer(dd) != 0 else 0

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
            "Skew": returns.skew() if len(returns) > 0 else 0,
            "Kurtosis": returns.kurt() if len(returns) > 0 else 0,
            "Trades/year": tpy,
        }

    hybrid_simple = hybrid_eq.pct_change().fillna(0) if len(hybrid_eq) > 0 else pd.Series([], dtype=float)
    hybrid_perf = compute_performance(hybrid_simple, hybrid_eq)
    
    
    pure_sig_simple = pure_sig_eq.pct_change().fillna(0) if len(pure_sig_eq) > 0 else pd.Series([], dtype=float)
    pure_sig_perf = compute_performance(pure_sig_simple, pure_sig_eq)

    # MA STRATEGY STATS (signal diagnostics ONLY)
    ma_perf = best_result["performance"]

    strat_stats = compute_stats(
        ma_perf,
        best_result["returns"],
        ma_perf["DD_Series"],
        best_result["flip_mask"],
        trades_per_year,
    )

    risk_stats = compute_stats(
        risk_on_perf,
        risk_on_simple,
        risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_simple), dtype=bool) if len(risk_on_simple) > 0 else np.array([], dtype=bool),
        0,
    )

    hybrid_stats = compute_stats(
        hybrid_perf,
        hybrid_simple,
        hybrid_perf["DD_Series"],
        np.zeros(len(hybrid_simple), dtype=bool) if len(hybrid_simple) > 0 else np.array([], dtype=bool),
        0,
    )

    pure_sig_stats = compute_stats(
        pure_sig_perf,
        pure_sig_simple,
        pure_sig_perf["DD_Series"],
        np.zeros(len(pure_sig_simple), dtype=bool) if len(pure_sig_simple) > 0 else np.array([], dtype=bool),
        0,
    )

    # STAT TABLE (updated with Buy & Hold with rebalance)
    st.subheader("MA vs Sharpe vs Buy & Hold (with rebalance) vs Sigma/MA vs SIG")
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
            "MA",
            "Sharpe",
            "Buy & Hold",
            "Sigma",
            "SIG",
        ],
    )

    st.dataframe(stat_table, use_container_width=True)

    # ALLOCATION TABLES (unchanged)
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
        if "Total Risky $" in out.index and "Total Safe $" in out.index:
            total_portfolio = float(out.loc["Total Risky $","$"]) + float(out.loc["Total Safe $","$"])
            out["% Portfolio"] = (out["$"] / total_portfolio * 100).apply(lambda x: f"{x:.2f}%")
            return out
        total = out["$"].sum()
        out["% Portfolio"] = (out["$"] / total * 100).apply(lambda x: f"{x:.2f}%")
        return out

    st.subheader("Account-Level Allocations")

    hyb_r = float(hybrid_rw.iloc[-1]) if len(hybrid_rw) > 0 else 0
    hyb_s = float(hybrid_sw.iloc[-1]) if len(hybrid_sw) > 0 else 0

    pure_r = float(pure_sig_rw.iloc[-1]) if len(pure_sig_rw) > 0 else 0
    pure_s = float(pure_sig_sw.iloc[-1]) if len(pure_sig_sw) > 0 else 0

    latest_signal = sig.iloc[-1] if len(sig) > 0 else False

    tab1, tab2, tab3 = st.tabs(["Taxable", "Tax-Sheltered", "Joint"])

    accounts = [
        ("Taxable", real_cap_1),
        ("Tax-Sheltered", real_cap_2),
        ("Joint", real_cap_3),
    ]

    for (label, cap), tab in zip(accounts, (tab1, tab2, tab3)):
        with tab:
            st.write(f"### {label} — Sigma")
            st.dataframe(add_pct(compute_allocations(cap, hyb_r, hyb_s, risk_on_weights, risk_off_weights)))

            st.write(f"### {label} — SIG")
            st.dataframe(add_pct(compute_allocations(cap, pure_r, pure_s, risk_on_weights, risk_off_weights)))

            if len(w_opt) > 0:
                st.write(f"### {label} — Sharpe")
                st.dataframe(add_pct(compute_sharpe_alloc(cap, risk_on_px.columns, w_opt)))

            st.write(f"### {label} — MA")
            if latest_signal:
                ma_alloc = compute_allocations(cap, 1.0, 0.0, risk_on_weights, {"SHY": 0})
            else:
                ma_alloc = compute_allocations(cap, 0.0, 1.0, {}, risk_off_weights)
            st.dataframe(add_pct(ma_alloc))

    # MA Distance (unchanged)
    st.subheader("Next MA Signal Distance")
    if len(opt_ma) > 0 and len(portfolio_index) > 0:
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
    else:
        st.write("**Insufficient data for MA distance calculation**")

    # Regime stats plot (unchanged)
    st.subheader("Regime Statistics")
    if len(sig) > 0:
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

        on_durations = regime_df[regime_df['Regime']=='RISK-ON']['Duration (days)']
        off_durations = regime_df[regime_df['Regime']=='RISK-OFF']['Duration (days)']
        
        st.write(f"**Avg RISK-ON duration:** {on_durations.mean():.1f} days" if len(on_durations) > 0 else "**Avg RISK-ON duration:** 0 days")
        st.write(f"**Avg RISK-OFF duration:** {off_durations.mean():.1f} days" if len(off_durations) > 0 else "**Avg RISK-OFF duration:** 0 days")
    else:
        st.write("No regime data available")

    st.markdown("---")  # Separator before final plot

    # Final Performance Plot (updated with Buy & Hold with rebalance)
    st.subheader("Portfolio Strategy Performance Comparison")

    plot_index = build_portfolio_index(prices, risk_on_weights)
    plot_ma = compute_ma(plot_index, best_len, best_type)

    plot_index_norm = normalize(plot_index)
    plot_ma_norm = normalize(plot_ma.dropna()) if len(plot_ma.dropna()) > 0 else pd.Series([], dtype=float)

    strat_eq_norm  = normalize(best_result["equity_curve"])
    sharp_eq_norm  = normalize(sharp_eq) if len(sharp_eq) > 0 else pd.Series([], dtype=float)
    hybrid_eq_norm = normalize(hybrid_eq) if len(hybrid_eq) > 0 else pd.Series([], dtype=float)
    pure_sig_norm  = normalize(pure_sig_eq) if len(pure_sig_eq) > 0 else pd.Series([], dtype=float)
    risk_on_norm   = normalize(risk_on_eq) if len(risk_on_eq) > 0 else pd.Series([], dtype=float)

    if len(strat_eq_norm) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(strat_eq_norm, label="MA", linewidth=2)
        if len(sharp_eq_norm) > 0:
            ax.plot(sharp_eq_norm, label="Sharpe", linewidth=2, color="magenta")
        if len(risk_on_norm) > 0:
            ax.plot(risk_on_norm, label="Buy & Hold", alpha=0.65)
        if len(hybrid_eq_norm) > 0:
            ax.plot(hybrid_eq_norm, label="Sigma", linewidth=2, color="blue")
        if len(pure_sig_norm) > 0:
            ax.plot(pure_sig_norm, label="SIG", linewidth=2, color="orange")
        if len(plot_ma_norm) > 0:
            ax.plot(plot_ma_norm, label=f"MA({best_len}) {best_type.upper()}", linestyle="--", color="black", alpha=0.6)

        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Insufficient data for performance plot")

    # ============================================================
    # STRATEGY DIAGNOSTICS
    # ============================================================

    st.subheader("Strategy Diagnostics")

    diag_fig = plot_diagnostics(
        hybrid_eq = hybrid_eq,
        bh_eq     = risk_on_eq,
        hybrid_signal = sig
    )

    st.pyplot(diag_fig)

    # ============================================================
    # IMPLEMENTATION CHECKLIST (Displayed at Bottom)
    # ============================================================
    st.markdown("""
---

## **Implementation Checklist**

- Rotate 100% of portfolio to treasury sleeve whenever the MA regime flips.
- At each calendar quarter-end, input your portfolio value at last rebalance & today's portfolio value.
- Execute the exact dollar adjustment recommended by the model (increase/decrease deployed sleeve) on the rebalance date.
- At each rebalance, re-evaluate the Sharpe portfolio weighting.

Current Sharpe-optimal portfolio: https://testfol.io/optimizer?s=a9xtQg7FUS9

## **Portfolio Allocation & Implementation Notes**

This system implements a dual-account portfolio framework built around
a Moving Average (MA) regime filter combined with a quarterly signal
(target-growth) rebalancing engine.

The strategy is designed to:
1) Concentrate risk during favorable regimes,
2) Systematically de-risk during adverse regimes,
3) Enforce disciplined quarterly capital deployment,
4) Preserve asymmetry while maintaining long-term solvency.

---

### **Taxable & Joint Accounts — Sigma Strategy**

**Regime Filter**
- A fixed moving average (MA) is applied to the risk-on portfolio index.
- When price is ABOVE the MA → **RISK-ON** regime.
- When price is BELOW the MA → **RISK-OFF** regime.

**Base Allocations**
- **Risk-On (Buy & Hold overlay):**
  - 50% QQUP
  - 50% IBIT
- **Risk-Off (Treasury sleeve):**
  - 100% STRC

**Sigma (Quarterly Target-Growth) Logic**
- When RISK-ON, the portfolio follows the Pure Sig rebalancing methodology:
  - Initial allocation: **70% Risk-On / 30% Risk-Off**
  - At each true calendar quarter-end:
    - A target quarterly growth rate is derived from the long-run CAGR
      of the Risk-On portfolio.
    - If Risk-On capital exceeds the target:
      → Excess is trimmed and moved to Risk-Off.
    - If Risk-On capital falls short of the target:
      → Capital is transferred from Risk-Off to Risk-On (subject to availability).
- Rebalancing is strictly calendar-quarter based (true quarter-ends),
  not rolling or price-dependent.

**Risk-Off Behavior**
- Upon entering a RISK-OFF regime:
  - Risk-On capital is frozen.
  - 100% of portfolio exposure is shifted to the Treasury sleeve.
- Upon re-entering RISK-ON:
  - The system resumes Pure Sig allocations using the last valid weights.

**Leverage Rationale**
- Backtesting indicates that **2× exposure via QQUP** provides meaningfully
  higher growth with comparable Sharpe relative to 1× exposure.
- 3× exposure is avoided due to product availability and inferior
  long-term CAGR characteristics (e.g., TQQQ).
- Objective: maximize upside asymmetry while remaining solvent across
  extended drawdown regimes.

**Sharpe-Optimal References**
- Risk-On: 50% QQUP / 50% IBIT  
  https://testfol.io/optimizer?s=a9xtQg7FUS9

---

### **Roth IRA — Modified Risk-On (Buy & Hold)**

The Roth IRA is treated as a permanent risk-on vehicle with no MA-based
de-risking, prioritizing long-term asymmetric growth and tax efficiency.

**Core Allocation (50%)**
- 25% QQUP
- 25% IBIT  
(Represents the Sharpe-optimal risk-on foundation.)

**Asymmetric Growth Sleeve (50%)**

**Bitcoin Treasury Companies**
- Equal-weighted exposure to pure-play Bitcoin treasury firms:
  - MSTR (Strategy)
  - XXI (Twenty One Capital)
  - ASST (Strive)
- Each position weighted at approximately **8.33%**.
- These equities provide leveraged Bitcoin exposure with an embedded
  equity risk premium, historically producing higher Sharpe ratios than
  direct leveraged Bitcoin ETFs.
- Additional treasury companies may be added as new listings emerge.

**Equities (Private Market Exposure)**
- 25% allocation to **DXYZ (Destiny Tech 100)**.
- Provides access to late-stage private companies such as OpenAI,
  SpaceX, and Stripe, capturing pre-IPO growth not available in
  traditional public equity indices.

**Overall Roth Allocation**
- 25% QQUP
- 25% IBIT
- 25% DXYZ
- ~8.33% each to Bitcoin treasury equities

This structure ensures:
- 50% of the portfolio remains in a Sharpe-optimal risk-on base.
- 50% targets convex, asymmetric exposure to equities and Bitcoin.

---

### **Important Notes**
- The system is fully rules-based and non-discretionary.
- All rebalances occur only at true calendar quarter-ends.
- MA parameters and tolerances are fixed to avoid overfitting.
- The objective is long-horizon robustness, asymmetry, and behavioral discipline.

---
""")

# ============================================================
# LAUNCH APP
# ============================================================

if __name__ == "__main__":
    main()
    