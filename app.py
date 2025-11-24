import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass

# ============================================
# CONFIG
# ============================================

DEFAULT_START_DATE = "2011-11-24"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "GLD": 3/3,
    "TQQQ": 1/3,
    "BTC-USD": 1/3,
}

RISK_OFF_WEIGHTS = {
    "UUP": 1.0,
}

# ============================================
# DATA
# ============================================

@st.cache_data(show_spinner=True)
def load_price_data(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # 1. ALWAYS FLATTEN MULTIINDEX
    if isinstance(data.columns, pd.MultiIndex):
        # Example: ('Adj Close','BTC-USD') -> 'BTC-USD'
        data.columns = data.columns.get_level_values(-1)

    # 2. Pick adjusted or close
    if "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
    elif "Close" in data.columns:
        px = data["Close"].copy()
    else:
        raise RuntimeError(f"No price columns found. Columns = {list(data.columns)}")

    # 3. Ensure DataFrame, not Series
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    # 4. Keep only requested tickers
    px = px[[c for c in px.columns if c in tickers]]

    # 5. Drop empty rows
    return px.dropna(how="all")

# ============================================
# TECHNICALS
# ============================================

def compute_ma(series, length, ma_type):
    if ma_type == "ema":
        return series.ewm(span=length, adjust=False).mean()
    return series.rolling(window=length, min_periods=length).mean()

def generate_signal(price, length, ma_type, tol):
    ma = compute_ma(price, length, ma_type)
    sig = price > ma * (1 + tol)
    return sig.fillna(False)

# ============================================
# BACKTEST
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

def compute_performance(returns, equity_curve, rf=0.0):
    n = len(returns)
    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (1 + total_ret)**(252/n) - 1

    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol > 0 else np.nan

    dd = equity_curve / equity_curve.cummax() - 1
    max_dd = dd.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": total_ret,
    }

def backtest(prices, signal, risk_on_weights, risk_off_weights):
    rets = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strat_rets = (weights.shift(1).fillna(0) * rets).sum(axis=1)
    eq = (1 + strat_rets).cumprod()

    return {
        "returns": strat_rets,
        "equity_curve": eq,
        "weights": weights,
        "signal": signal,
        "performance": compute_performance(strat_rets, eq),
    }

# ============================================
# GRID SEARCH (DETERMINISTIC)
# ============================================

def run_grid_search(prices, risk_on_weights, risk_off_weights):
    btc = prices["BTC-USD"]

    best_sharpe = -1e9
    best = None
    best_cfg = None

    lengths = range(21, 253)
    types = ["sma", "ema"]
    tolerances = np.arange(0.0, 0.1001, 0.001)  # 0.0% to 10.0% by 0.1%

    progress = st.progress(0.0)
    total = len(lengths) * len(types) * len(tolerances)
    idx = 0

    for length in lengths:
        for ma_type in types:
            for tol in tolerances:

                signal = generate_signal(btc, length, ma_type, tol)
                result = backtest(prices, signal, risk_on_weights, risk_off_weights)
                sharpe = result["performance"]["Sharpe"]

                idx += 1
                progress.progress(idx / total)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_cfg = (length, ma_type, tol)
                    best = result

    return best_cfg, best

# ============================================
# STREAMLIT APP
# ============================================

def main():
    st.set_page_config(page_title="BTC Trend Optimized Portfolio", layout="wide")
    st.title("Bitcoin Trend – Optimized Risk-On Portfolio")

    st.write("""
    Deterministic brute-force moving-average regime model.
    Searches all combinations of:
    - MA Length: 21–252  
    - Type: SMA/EMA  
    - Tolerance: 0.0%–10.0% (0.1% steps)  
    """)

    # User inputs
    st.sidebar.header("Backtest Settings")
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")

    # Risk-ON portfolio
    st.sidebar.header("Risk-ON Portfolio")
    risk_on_tickers_str = st.sidebar.text_input(
        "Tickers", ",".join(RISK_ON_WEIGHTS.keys())
    )
    risk_on_weights_str = st.sidebar.text_input(
        "Weights", ",".join(str(w) for w in RISK_ON_WEIGHTS.values())
    )

    # Risk-OFF portfolio
    st.sidebar.header("Risk-OFF Portfolio")
    risk_off_tickers_str = st.sidebar.text_input(
        "Tickers", ",".join(RISK_OFF_WEIGHTS.keys())
    )
    risk_off_weights_str = st.sidebar.text_input(
        "Weights", ",".join(str(w) for w in RISK_OFF_WEIGHTS.values())
    )

    if not st.sidebar.button("Run Backtest & Optimize"):
        st.stop()

    # Parse tickers + weights
    risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",")]
    risk_on_weights_list = [float(x) for x in risk_on_weights_str.split(",")]
    risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))

    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))

    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers))
    if "BTC-USD" not in all_tickers:
        all_tickers.append("BTC-USD")

    end_val = end if end.strip() != "" else None
    prices = load_price_data(all_tickers, start, end_val)
    prices = prices.dropna(how="any")

    # GRID SEARCH
    best_cfg, best_result = run_grid_search(
        prices, risk_on_weights, risk_off_weights
    )

    best_len, best_type, best_tol = best_cfg

    perf = best_result["performance"]
    sig = best_result["signal"]

    # Trade count
    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252)

    # =====================================
    # OUTPUT
    # =====================================

    st.subheader("Optimized Strategy Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("CAGR", f"{perf['CAGR']:.2%}")
    col2.metric("Volatility", f"{perf['Volatility']:.2%}")
    col3.metric("Sharpe", f"{perf['Sharpe']:.3f}")
    col4.metric("Max Drawdown", f"{perf['MaxDrawdown']:.2%}")
    col5.metric("Total Return", f"{perf['TotalReturn']:.2%}")

    st.subheader("Optimized MA Configuration")
    st.write(f"**MA Length:** {best_len}")
    st.write(f"**MA Type:** {best_type.upper()}")
    st.write(f"**Tolerance:** {best_tol:.2%}")

    st.subheader("Trading Frequency")
    st.write(f"**Total Trades:** {int(switches)}")
    st.write(f"**Trades per year:** {trades_per_year:.2f}")

    # User always-on
    rets = prices.pct_change().fillna(0)
    user_rets = pd.Series(0, index=rets.index)
    for a, w in risk_on_weights.items():
        if a in rets.columns:
            user_rets += rets[a] * w
    user_eq = (1 + user_rets).cumprod()
    user_perf = compute_performance(user_rets, user_eq)

    st.subheader("Always-On Portfolio Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{user_perf['CAGR']:.2%}")
    c2.metric("Volatility", f"{user_perf['Volatility']:.2%}")
    c3.metric("Sharpe", f"{user_perf['Sharpe']:.3f}")
    c4.metric("Max Drawdown", f"{user_perf['MaxDrawdown']:.2%}")
    c5.metric("Total Return", f"{user_perf['TotalReturn']:.2%}")

    # Plot equity curve
    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(best_result["equity_curve"], label="Optimized Strategy", linewidth=2)
    ax.plot(user_eq, label="Always-On Risk-ON", linestyle="--", linewidth=2)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

if __name__ == "__main__":
    main()

