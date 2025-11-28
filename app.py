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
TRANSACTION_COST = 0.001   # 0.10% cost per regime switch

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
# TECHNICALS — TESTFOL HYSTERESIS
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
# SHARPE-OPTIMAL PORTFOLIO (USER MAX LEVERAGE)
# ============================================

def compute_max_sharpe_weights(returns, max_leverage_dict, target_leverage):
    """
    returns: DataFrame of daily log returns
    max_leverage_dict: {"GLD": 3.0, "TQQQ": 1.0, ...}
    target_leverage: total portfolio leverage user wants
    """
    tickers = returns.columns.tolist()

    mu = returns.mean().values * 252
    cov = returns.cov().values * 252

    n = len(tickers)

    # Initial equal weights
    x0 = np.ones(n) / n

    # Bounds from user max leverage
    bounds = [(0, max_leverage_dict[t]) for t in tickers]

    # Constraint: sum(weights) = target_leverage
    cons = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - target_leverage
    })

    def neg_sharpe(w):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if port_vol == 0:
            return 1e9
        return -(port_ret / port_vol)

    res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)

    return dict(zip(tickers, res.x))

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

    # Transaction cost applied on flips
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
# GRID SEARCH — MA OPTIMIZATION (USING BTC)
# ============================================

def run_grid_search(prices, risk_on_weights, risk_off_weights):
    btc = prices["BTC-USD"]

    best_sharpe = -1e9
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

                idx += 1
                if idx % 200 == 0:
                    progress.progress(idx / total)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_cfg = (length, ma_type, tol)
                    best_result = result

    return best_cfg, best_result

# ============================================
# STREAMLIT APP
# ============================================

def main():
    st.set_page_config(page_title="MA Strategy + Sharpe Optimizer", layout="wide")
    st.title("Sharpe-Optimized MA Strategy (BTC-Anchored)")

    st.write("""
    This model now includes:
    - Max-Sharpe portfolio optimizer
    - User-specified max leverage per asset
    - Total leverage target L
    - BTC-USD always used as the MA signal anchor
    - 0.10% transaction cost per regime switch
    """)

    # --------------------------
    # USER INPUT
    # --------------------------

    st.sidebar.header("Input Assets")
    tickers_str = st.sidebar.text_input("Assets (comma separated)", "GLD,TQQQ,BTC-USD")
    tickers = [t.strip().upper() for t in tickers_str.split(",")]

    max_lev_str = st.sidebar.text_input("Max leverage per asset (comma separated)", "3,1,1")
    max_leverages = [float(x) for x in max_lev_str.split(",")]
    max_lev_dict = dict(zip(tickers, max_leverages))

    target_leverage = st.sidebar.number_input("Total desired leverage (L)", value=1.0, step=0.1)

    st.sidebar.header("Risk-OFF Portfolio")
    risk_off_tickers_str = st.sidebar.text_input("Risk-OFF tickers", "UUP")
    risk_off_weights_str = st.sidebar.text_input("Risk-OFF weights", "1.0")

    if not st.sidebar.button("Run"):
        st.stop()

    # Parse Risk-OFF
    off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(off_tickers, off_weights_list))

    # Ensure BTC anchor is always included
    if "BTC-USD" not in tickers:
        tickers.append("BTC-USD")

    all_tickers = sorted(set(tickers + off_tickers))

    prices = load_price_data(all_tickers, DEFAULT_START_DATE).dropna(how="any")

    # ------------------------------------
    # SHARPE OPTIMAL PORTFOLIO
    # ------------------------------------
    log_prices = np.log(prices[tickers])
    log_rets = log_prices.diff().fillna(0)

    sharpe_weights = compute_max_sharpe_weights(
        returns=log_rets,
        max_leverage_dict=max_lev_dict,
        target_leverage=target_leverage
    )

    risk_on_weights = sharpe_weights  # overwrite user portfolio

    # ------------------------------------
    # MA OPTIMIZATION
    # ------------------------------------
    best_cfg, best_result = run_grid_search(prices, risk_on_weights, risk_off_weights)
    best_len, best_type, best_tol = best_cfg

    perf = best_result["performance"]
    sig = best_result["signal"]

    latest_day = sig.index[-1]
    regime = "RISK-ON" if sig.iloc[-1] else "RISK-OFF"

    st.subheader("Current Regime")
    st.write(f"**Date:** {latest_day.date()}")
    st.write(f"**Regime:** {regime}")

    # ------------------------------------
    # SHOW RESULTS
    # ------------------------------------
    st.subheader("Optimized Strategy Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{perf['CAGR']:.2%}")
    c2.metric("Volatility", f"{perf['Volatility']:.2%}")
    c3.metric("Sharpe", f"{perf['Sharpe']:.3f}")
    c4.metric("Max DD", f"{perf['MaxDrawdown']:.2%}")
    c5.metric("Total Return", f"{perf['TotalReturn']:.2%}")

    st.subheader("Best Parameters")
    st.write(f"MA Length: {best_len}")
    st.write(f"Type: {best_type.upper()}")
    st.write(f"Tolerance: {best_tol:.2%}")

    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(best_result["equity_curve"], label="Optimized", linewidth=2)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    main()