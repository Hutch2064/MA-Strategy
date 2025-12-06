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

FLIP_COST = 0.00875

QUARTER_DAYS = 63       # trading days per quarter
START_RISKY = 0.60      # SIG initial risky allocation
START_SAFE = 0.40       # SIG initial safe allocation


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

    idx = (1 + idx_rets).cumprod()
    return idx


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
# NEW: SIG ENGINE — TRUE JASON KELLY LOGIC
# ============================================================

def run_sig_engine(risk_on_returns, risk_off_returns, target_quarter, ma_signal):
    """
    Jason Kelly 3Sig mechanics
    - Risky bucket grows daily using risk-on returns
    - Safe bucket grows using risk-off returns
    - Every quarter:
         If risky bucket growth > target:
             Move excess into safe bucket
    - When MA says risk-off:
         Freeze risky/safe values, move 100% into risk-off portfolio
    """

    dates = risk_on_returns.index
    n = len(dates)

    # Starting state
    risky_val = 6000.0
    safe_val = 4000.0

    risky_w = START_RISKY
    safe_w = START_SAFE

    frozen_risky_val = None
    frozen_safe_val = None

    equity_curve = []
    risky_w_series = []
    safe_w_series = []
    rebalance_events = 0

    for i in range(n):

        # ====================================================
        # CASE 1: MA = RISK-ON → SIG ENGINE ACTIVE
        # ====================================================
        if ma_signal.iloc[i]:

            # Resume SIG state after risk-off
            if frozen_risky_val is not None:
                risky_val = frozen_risky_val
                safe_val = frozen_safe_val
                frozen_risky_val = None
                frozen_safe_val = None

            # Apply daily returns
            risky_val *= (1 + risk_on_returns.iloc[i])
            safe_val  *= (1 + risk_off_returns.iloc[i])
            # Compute total portfolio value during RISK-ON
            total = risky_val + safe_val

            risky_w = risky_val / total if total > 0 else 0
            safe_w  = safe_val  / total if total > 0 else 0
            # ------------------------------
            # TRUE 3SIG QUARTERLY REBALANCE
            # ------------------------------
            if i >= QUARTER_DAYS and (i % QUARTER_DAYS == 0):

                past_total = equity_curve[i - QUARTER_DAYS]
                past_risky = past_total * risky_w_series[i - QUARTER_DAYS]

                goal_risky = past_risky * (1 + target_quarter)

                # SELL DOWN if above the signal line
                if risky_val > goal_risky:
                    excess = risky_val - goal_risky
                    risky_val -= excess
                    safe_val  += excess
                    rebalance_events += 1

                # BUY UP if below the signal line
                elif risky_val < goal_risky:
                    needed = goal_risky - risky_val

                    if safe_val >= needed:
                        safe_val  -= needed
                        risky_val += needed
                    else:
                        risky_val += safe_val
                        safe_val = 0

                    rebalance_events += 1
                    
            # Compute daily total + weights (RISK-ON)
            total = risky_val + safe_val
            risky_w = risky_val / total
            safe_w  = safe_val / total
        
        # ====================================================
        # CASE 2: MA = RISK-OFF → FREEZE SIG ENGINE
        # ====================================================
        else:
             # Freeze values only the first day MA turns OFF
             if frozen_risky_val is None:
                 frozen_risky_val = risky_val
                 frozen_safe_val  = safe_val

             # Risky bucket stays frozen
             risky_val = frozen_risky_val

             # Safe bucket compounds with risk-off returns
             frozen_safe_val *= (1 + risk_off_returns.iloc[i])
             safe_val = frozen_safe_val

             total = risky_val + safe_val
             risky_w = 0.0
             safe_w  = 1.0

        equity_curve.append(total)
        risky_w_series.append(risky_w)
        safe_w_series.append(safe_w)

    return (
        pd.Series(equity_curve, index=dates),
        pd.Series(risky_w_series, index=dates),
        pd.Series(safe_w_series, index=dates),
        rebalance_events
    )


# ============================================
# BACKTEST ENGINE (unchanged from original)
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


def backtest(prices, signal, risk_on_weights, risk_off_weights):
    simple = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    sig_arr = signal.astype(int)

    flip_mask = sig_arr.diff().abs() == 1
    flip_costs = np.where(flip_mask, -FLIP_COST, 0.0)

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

def run_grid_search(prices, risk_on_weights, risk_off_weights):

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
                result = backtest(prices, signal, risk_on_weights, risk_off_weights)

                sig_arr = signal.astype(int)
                switches = sig_arr.diff().abs().sum()
                trades_per_year = switches / (len(sig_arr) / 252)

                sharpe_adj = result["performance"]["Sharpe"]

                idx += 1
                if idx % 200 == 0:
                    progress.progress(idx / total)

                if sharpe_adj > best_sharpe or (sharpe_adj == best_sharpe and trades_per_year < best_trades):
                    best_sharpe = sharpe_adj
                    best_trades = trades_per_year
                    best_cfg = (length, ma_type, tol)
                    best_result = result

    return best_cfg, best_result
    
def normalize(eq):
    return eq / eq.iloc[0] * 10000

# ============================================
# STREAMLIT APP
# ============================================

def main():

    st.set_page_config(page_title="Portfolio MA Regime Strategy", layout="wide")
    st.title("Portfolio MA Strategy")

    st.sidebar.header("Backtest Settings")
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")

    st.sidebar.header("Risk-ON Portfolio")
    risk_on_tickers_str = st.sidebar.text_input("Tickers", ",".join(RISK_ON_WEIGHTS.keys()))
    risk_on_weights_str = st.sidebar.text_input("Weights", ",".join(str(w) for w in RISK_ON_WEIGHTS.values()))

    st.sidebar.header("Risk-OFF Portfolio")
    risk_off_tickers_str = st.sidebar.text_input("Tickers", ",".join(RISK_OFF_WEIGHTS.keys()))
    risk_off_weights_str = st.sidebar.text_input("Weights", ",".join(str(w) for w in RISK_OFF_WEIGHTS.values()))

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
    best_cfg, best_result = run_grid_search(prices, risk_on_weights, risk_off_weights)
    best_len, best_type, best_tol = best_cfg

    sig = best_result["signal"]
    perf = best_result["performance"]

    latest_signal = sig.iloc[-1]
    regime = "RISK-ON" if latest_signal else "RISK-OFF"

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
        "PainGain": sharp_perf["CAGR"] / np.sqrt((sharp_perf["DD_Series"]**2).mean())
                    if (sharp_perf["DD_Series"]**2).mean() != 0 else np.nan,
        "Skew": sharp_returns.skew(),
        "Kurtosis": sharp_returns.kurt(),
        "Trades/year": 0.0,
        "P/L per flip": 0.0,
    }

    sharp_weights_display = {t: round(w, 4) for t, w in zip(risk_on_px.columns, w_opt)}

    # ============================================
    # HYBRID SIG STRATEGY
    # ============================================

    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252/len(risk_on_eq)) - 1
    quarterly_target = (1 + bh_cagr)**(1/4) - 1

    hybrid_eq, hybrid_rw, hybrid_sw, hybrid_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        sig
    )

    hybrid_simple = hybrid_eq.pct_change().fillna(0)
    hybrid_perf = compute_performance(hybrid_simple, hybrid_eq)

    # ============================================
    # ADVANCED METRICS (moved here)
    # ============================================

    def time_in_drawdown(dd):     return (dd < 0).mean()
    def mar(c, dd):              return c / abs(dd) if dd != 0 else np.nan
    def ulcer(dd):               return np.sqrt((dd**2).mean()) if (dd**2).mean() != 0 else np.nan
    def pain_gain(c, dd):        return c / ulcer(dd) if ulcer(dd) != 0 else np.nan

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
            "P/L per flip": float(returns[flips].sum()),
            "Trades/year": tpy,
        }
    
    # ============================================
    # PURE SIG STRATEGY — NO MA FILTER
    # ============================================

    # Signal stays TRUE every day
    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)

    # Run SIG engine in always-on mode
    pure_sig_eq, pure_sig_rw, pure_sig_sw, pure_sig_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        pure_sig_signal
    )

    # Compute Pure SIG performance
    pure_sig_simple = pure_sig_eq.pct_change().fillna(0)
    pure_sig_perf = compute_performance(pure_sig_simple, pure_sig_eq)

    pure_sig_stats = compute_stats(
        pure_sig_perf,
        pure_sig_simple,
        pure_sig_perf["DD_Series"],
        np.zeros(len(pure_sig_simple), dtype=bool),
        0
    )
    # Add SIG quarterly target to stats
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

    avg_safe = hybrid_sw.mean()

    # ============================================
    # METRIC TABLE — 4 COLUMNS
    # ============================================

    st.subheader("Strategy vs. Sharpe-Optimal vs. Risk-ON vs. Hybrid")

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
        ("P/L per flip", "P/L per flip"),
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
        columns=["Metric", "Strategy", "Sharpe-Optimal", "Risk-On", "Hybrid","Pure-SIG"]
    )

    st.dataframe(stat_table, use_container_width=True)

    # ============================================
    # EXTERNAL VALIDATION LINK
    # ============================================

    st.subheader("External Sharpe Optimal Validation Link")
    st.markdown(
        f"**Quick Access:** [View the Sharpe Optimal recommended portfolio](https://testfol.io/optimizer?s=9y4FBdfW2oO)"
    )

    # ============================================
    # HYBRID DIAGNOSTICS
    # ============================================

    st.write(f"**Quarterly Target (based on Buy & Hold CAGR):** {quarterly_target:.2%}")
    st.write(f"**Average Safe Allocation:** {avg_safe:.2%}")
    st.write("**Final Risk-On Allocation:**")
    st.write(hybrid_rw.iloc[-1])

    # ============================================
    # OPTIMAL SIGNAL PARAMETERS
    # ============================================

    st.subheader("Optimal Signal Parameters")
    st.write(f"**Moving Average Type:** {best_type.upper()}")
    st.write(f"**Optimal MA Length:** {best_len}")
    st.write(f"**Optimal Tolerance:** {best_tol:.2%}")

    # ============================================
    # SIGNAL DISTANCE
    # ============================================

    st.subheader("Next Signal Information")

    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    opt_ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]

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

    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize all curves so they start at the same value (10,000)
    strat_eq_norm  = normalize(best_result["equity_curve"])
    sharp_eq_norm  = normalize(sharp_eq)
    risk_on_norm   = normalize(portfolio_index)
    hybrid_eq_norm = normalize(hybrid_eq)

    ax.plot(strat_eq_norm,  label="MA Strategy", linewidth=2)
    ax.plot(sharp_eq_norm,  label="Sharpe-Optimal", linewidth=2, color="magenta")
    ax.plot(risk_on_norm,   label="Risk-On Portfolio", alpha=0.65)
    ax.plot(hybrid_eq_norm, label="Hybrid SIG Strategy", linewidth=2, color="blue")
    pure_sig_norm = normalize(pure_sig_eq)
    ax.plot(pure_sig_norm, label="Pure SIG (No MA Filter)", linewidth=2, color="orange")
    
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)


# ============================================
# LAUNCH APP
# ============================================

if __name__ == "__main__":
    main()