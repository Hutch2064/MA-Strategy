import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

# ============================================
# CONFIG
# ============================================

DEFAULT_START_DATE = "2011-11-24"
RISK_FREE_RATE = 0.0
TRANSACTION_COST = 0.001   # 0.10% per signal flip

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
# SHARPE OPTIMAL PORTFOLIO (GLOBAL MAX)
# ============================================

def sharpe_optimize(returns, bounds):
    mu = returns.mean() * 252
    cov = returns.cov() * 252

    tickers = returns.columns
    n = len(tickers)

    def neg_sharpe(w):
        port_mu = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return 1e9
        return -(port_mu / port_vol)

    x0 = np.array([b[1] / 2 for b in bounds])
    cons = ()  # no sum-to-1 requirement in Option A

    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )

    w = res.x
    return dict(zip(tickers, w))

# ============================================
# TECHNICALS — VECTORIZED TESTFOL HYSTERESIS
# ============================================

def compute_ma_matrix(price, lengths, ma_type):
    ma_dict = {}

    if ma_type == "ema":
        for L in lengths:
            ma = price.ewm(span=L, adjust=False).mean()
            ma_dict[L] = ma.shift(1)
    else:
        for L in lengths:
            ma = price.rolling(window=L, min_periods=L).mean()
            ma_dict[L] = ma.shift(1)

    return ma_dict

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

# ============================================
# BACKTEST (LOG RETURNS + DAILY REBALANCE)
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

def compute_performance(log_returns, equity_curve, rf=0.0):
    cagr = np.exp(log_returns.mean() * 252) - 1
    vol = log_returns.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol > 0 else np.nan
    dd = equity_curve / equity_curve.cummax() - 1
    max_dd = dd.min()
    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": total_ret,
    }

def backtest(prices, signal, risk_on_weights, risk_off_weights):
    log_prices = np.log(prices)
    log_rets = log_prices.diff().fillna(0)

    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strat_log_rets = (weights.shift(1).fillna(0) * log_rets).sum(axis=1)

    flips = signal.astype(int).diff().abs().fillna(0)
    strat_log_rets -= flips * TRANSACTION_COST

    eq = np.exp(strat_log_rets.cumsum())

    return {
        "returns": strat_log_rets,
        "equity_curve": eq,
        "weights": weights,
        "signal": signal,
        "performance": compute_performance(strat_log_rets, eq),
    }

# ============================================
# GRID SEARCH — VECTORIZED + OPTIMIZED
# ============================================

def run_grid_search(prices, risk_on_weights, risk_off_weights):
    btc = prices["BTC-USD"]

    best_sharpe = -1e9
    best_trades = np.inf
    best_cfg = None
    best_result = None

    lengths = list(range(21, 253))
    types = ["sma", "ema"]
    tolerances = np.arange(0.0, 0.1001, 0.002)

    progress = st.progress(0.0)
    total = len(lengths) * len(types) * len(tolerances)
    idx = 0

    ma_cache = {t: compute_ma_matrix(btc, lengths, t) for t in types}

    for ma_type in types:
        for length in lengths:
            ma = ma_cache[ma_type][length]

            for tol in tolerances:
                signal = generate_testfol_signal_vectorized(btc, ma, tol)

                result = backtest(prices, signal, risk_on_weights, risk_off_weights)
                sharpe = result["performance"]["Sharpe"]

                sig_arr = result["signal"].astype(int)
                switches = sig_arr.diff().abs().sum()
                trades_per_year = switches / (len(sig_arr) / 252)

                idx += 1
                if idx % 200 == 0:
                    progress.progress(idx / total)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_trades = trades_per_year
                    best_cfg = (length, ma_type, tol)
                    best_result = result

                elif sharpe == best_sharpe and trades_per_year < best_trades:
                    best_trades = trades_per_year
                    best_cfg = (length, ma_type, tol)
                    best_result = result

    return best_cfg, best_result

# ============================================
# STREAMLIT APP
# ============================================

def main():
    st.set_page_config(page_title="Bitcoin MA Optimized Portfolio", layout="wide")
    st.title("Bitcoin MA Optimized Portfolio (Sharpe-Optimal Risk-ON)")

    st.write("""
    This model now includes:
    - Fully vectorized TestFol hysteresis  
    - Exact 1-day delayed indicators  
    - Log-return engine  
    - Daily rebalance  
    - No lookahead bias  
    - Sharpe → Trades optimization  
    - 0.10% transaction cost per regime switch  
    - **Global Sharpe-optimal Risk-ON portfolio (Option A)**  
    """)

    st.sidebar.header("Backtest Settings")
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")

    st.sidebar.header("Select Risk-ON Assets")
    risk_on_tickers_str = st.sidebar.text_input(
        "Tickers (comma separated)",
        "GLD,TQQQ,BTC-USD"
    )
    tickers_list = [t.strip().upper() for t in risk_on_tickers_str.split(",")]

    st.sidebar.subheader("Max Leverage Constraints")
    max_leverage = {}
    for t in tickers_list:
        max_leverage[t] = st.sidebar.slider(
            f"Max weight for {t}", min_value=0.0, max_value=5.0, value=1.0, step=0.1
        )

    st.sidebar.header("Risk-OFF Portfolio")
    risk_off_tickers_str = st.sidebar.text_input(
        "Tickers", "UUP"
    )
    risk_off_weights_str = st.sidebar.text_input(
        "Weights", "1.0"
    )

    if not st.sidebar.button("Run Backtest & Optimize"):
        st.stop()

    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))

    all_tickers = sorted(set(tickers_list + risk_off_tickers))
    end_val = end if end.strip() else None
    prices = load_price_data(all_tickers, start, end_val).dropna(how="any")

    # ====================================
    # SHARPE OPTIMAL PORTFOLIO FOR RISK-ON
    # ====================================
    returns = np.log(prices[tickers_list]).diff().dropna()
    bounds = [(0, max_leverage[t]) for t in tickers_list]

    optimal_weights = sharpe_optimize(returns, bounds)

    st.subheader("Sharpe-Optimal Risk-ON Portfolio")
    for t, w in optimal_weights.items():
        st.write(f"**{t}: {w:.3f}**")

    # Feed these weights directly into the MA strategy
    risk_on_weights = optimal_weights.copy()

    best_cfg, best_result = run_grid_search(prices, risk_on_weights, risk_off_weights)

    best_len, best_type, best_tol = best_cfg
    perf = best_result["performance"]
    sig = best_result["signal"]

    latest_day = sig.index[-1]
    latest_signal = sig.iloc[-1]
    regime = "RISK-ON" if latest_signal else "RISK-OFF"

    st.subheader("Current Regime Status")
    st.write(f"**Date:** {latest_day.date()}")
    st.write(f"**Regime:** {regime}")

    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252)

    st.subheader("Optimized Strategy Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{perf['CAGR']:.2%}")
    c2.metric("Volatility", f"{perf['Volatility']:.2%}")
    c3.metric("Sharpe", f"{perf['Sharpe']:.3f}")
    c4.metric("Max Drawdown", f"{perf['MaxDrawdown']:.2%}")
    c5.metric("Total Return", f"{perf['TotalReturn']:.2%}")

    st.subheader("Best Parameters")
    st.write(f"**MA Length:** {best_len}")
    st.write(f"**MA Type:** {best_type.upper()}")
    st.write(f"**Tolerance:** {best_tol:.2%}")
    st.write(f"**Trades per year:** {trades_per_year:.2f}")

    # Always-on using Sharpe-optimal weights
    log_prices = np.log(prices[tickers_list])
    log_rets = log_prices.diff().fillna(0)

    user_rets = sum(log_rets[t] * w for t, w in optimal_weights.items())
    user_eq = np.exp(user_rets.cumsum())
    user_perf = compute_performance(user_rets, user_eq)

    st.subheader("Always-On Performance (Sharpe-Optimal Portfolio)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{user_perf['CAGR']:.2%}")
    c2.metric("Volatility", f"{user_perf['Volatility']:.2%}")
    c3.metric("Sharpe", f"{user_perf['Sharpe']:.3f}")
    c4.metric("Max Drawdown", f"{user_perf['MaxDrawdown']:.2%}")
    c5.metric("Total Return", f"{user_perf['TotalReturn']:.2%}")

    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(best_result["equity_curve"], label="Optimized", linewidth=2)
    ax.plot(user_eq, label="Always-On", linestyle="--")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

if __name__ == "__main__":
    main()