import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# ============================================
# CONFIG
# ============================================

DEFAULT_START_DATE = "2011-11-24"
RISK_FREE_RATE = 0.0
TRANSACTION_COST = 0.001   # 0.10% per regime flip

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
# SHARPE-OPTIMAL WEIGHTS (NO SCIPY)
# ============================================

def compute_sharpe_optimal_weights(prices, max_leverages):
    """
    Closed-form unconstrained Sharpe maximization:
        w ∝ Σ^{-1} μ
    Then clamp weights to user-specified max leverage per asset.
    Short selling allowed. Total leverage may exceed 1.
    """

    log_returns = np.log(prices).diff().dropna()

    mu = log_returns.mean().values * 252                # annualized mean
    cov = np.cov(log_returns.values.T)                 # covariance matrix

    inv_cov = np.linalg.pinv(cov)
    raw_w = inv_cov @ mu                               # Σ^{-1} μ

    if np.sum(np.abs(raw_w)) > 0:
        raw_w = raw_w / np.sum(np.abs(raw_w))          # normalize by L1

    tickers = list(prices.columns)
    w = pd.Series(raw_w, index=tickers)

    # Clamp with per-asset max leverage
    for t in tickers:
        limit = max_leverages.get(t, 1.0)
        w[t] = np.clip(w[t], -limit, limit)

    return w / w.abs().sum()     # keep direction—Option A allows >1 total

# ============================================
# MA COMPUTATION
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

# ============================================
# TESTFOL HYSTERESIS
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

# ============================================
# BACKTEST ENGINE
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

    # Transaction cost per flip
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
# GRID SEARCH
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

                if sharpe > best_sharpe or (sharpe == best_sharpe and trades_per_year < best_trades):
                    best_sharpe = sharpe
                    best_trades = trades_per_year
                    best_cfg = (length, ma_type, tol)
                    best_result = result

    return best_cfg, best_result

# ============================================
# STREAMLIT UI
# ============================================

def main():
    st.set_page_config(page_title="MA Strategy + Sharpe Optimizer", layout="wide")
    st.title("Sharpe-Optimal MA Strategy (Fully Integrated Version)")

    st.sidebar.header("Backtest Settings")
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")

    st.sidebar.header("Risk-ON Assets (User Choice)")
    risk_on_tickers_str = st.sidebar.text_input("Tickers", "GLD,TQQQ,BTC-USD")

    st.sidebar.subheader("Max Leverage per Asset")
    max_lev_str = st.sidebar.text_input("Max Leverage (same order)", "1.0,3.0,2.0")

    st.sidebar.header("Risk-OFF Portfolio")
    risk_off_tickers_str = st.sidebar.text_input("Risk-OFF Tickers", "UUP")
    risk_off_weights_str = st.sidebar.text_input("Risk-OFF Weights", "1.0")

    if not st.sidebar.button("RUN FULL OPTIMIZATION"):
        st.stop()

    # Parse tickers
    risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",")]
    max_lev_list = [float(x) for x in max_lev_str.split(",")]
    max_leverages = dict(zip(risk_on_tickers, max_lev_list))

    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))

    # Load prices
    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers + ["BTC-USD"]))
    end_val = end if end.strip() else None
    prices = load_price_data(all_tickers, start, end_val)

    # --------------------------------------------
    # SHARPE-OPTIMAL RISK-ON WEIGHTS
    # --------------------------------------------
    st.subheader("Sharpe-Optimal Risk-ON Portfolio Weights")
    risk_on_prices = prices[risk_on_tickers]

    optimal_weights = compute_sharpe_optimal_weights(risk_on_prices, max_leverages)
    st.write(optimal_weights)

    # Convert to dict for backtest
    risk_on_weights = optimal_weights.to_dict()

    # --------------------------------------------
    # MA OPTIMIZATION
    # --------------------------------------------
    best_cfg, best_result = run_grid_search(prices, risk_on_weights, risk_off_weights)
    best_len, best_type, best_tol = best_cfg

    perf = best_result["performance"]
    sig = best_result["signal"]

    # --------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------

    latest_day = sig.index[-1]
    latest_signal = sig.iloc[-1]
    regime = "RISK-ON" if latest_signal else "RISK-OFF"

    st.subheader("Current Regime")
    st.write(f"**Date:** {latest_day.date()} — **{regime}**")

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
    st.write(f"**Trades/year:** {trades_per_year:.2f}")

    # --------------------------------------------
    # Equity Curve Plot
    # --------------------------------------------
    log_prices = np.log(prices)
    log_rets = log_prices.diff().fillna(0)

    always_on = (log_rets[risk_on_tickers] @ optimal_weights).cumsum()
    always_on_eq = np.exp(always_on)

    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(best_result["equity_curve"], label="Optimized Strategy", linewidth=2)
    ax.plot(always_on_eq, "--", label="Always Risk-ON (Sharpe-optimal)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

if __name__ == "__main__":
    main()