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

RISK_ON_WEIGHTS = {
    "GLD": 1/3,
    "TQQQ": 1/3,
    "BTC-USD": 1/3,
}

RISK_OFF_WEIGHTS = {
    "UUP": 1.0,
}

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
# TECHNICALS — TRUE TESTFOL HYSTERESIS LOGIC
# ============================================

def compute_ma(series, length, ma_type):
    if ma_type == "ema":
        return series.ewm(span=length, adjust=False).mean()
    else:
        return series.rolling(window=length, min_periods=length).mean()

def generate_testfol_signal(price, length, ma_type, tol):
    """
    Implements TestFol-style hysteresis + 1-day delay:
      - price[t] compared to SMA[t], but with full buffer:
        If previously OFF → turn ON only when price > SMA*(1+tol)
        If previously ON  → turn OFF only when price < SMA*(1-tol)
    """

    # Compute MA with 1-day delay (TestFol)
    ma = compute_ma(price, length, ma_type).shift(1)
    px = price.shift(1)  # delayed 1 day to remove lookahead bias

    # Initialize signal array
    sig = pd.Series(False, index=price.index)

    # Hysteresis rules
    for t in range(1, len(price)):
        prev = sig.iloc[t-1]

        upper = ma.iloc[t] * (1 + tol)
        lower = ma.iloc[t] * (1 - tol)

        if prev is False:
            # OFF → ON only if full break above buffer
            if px.iloc[t] > upper:
                sig.iloc[t] = True
            else:
                sig.iloc[t] = False

        else:
            # ON → OFF only if full break below buffer
            if px.iloc[t] < lower:
                sig.iloc[t] = False
            else:
                sig.iloc[t] = True

    return sig.fillna(False)

# ============================================
# BACKTEST (LOG-RETURNS, DAILY REBALANCE)
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
    n = len(log_returns)
    cagr = np.exp(log_returns.mean() * 252) - 1
    vol = log_returns.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol > 0 else np.nan

    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = drawdown.min()

    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": total_ret,
    }

def backtest(prices, signal, risk_on_weights, risk_off_weights):
    # Log returns
    log_prices = np.log(prices)
    log_rets = log_prices.diff().fillna(0)

    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    # Apply 1-day delay (TestFol)
    strat_log_rets = (weights.shift(1).fillna(0) * log_rets).sum(axis=1)

    # Convert log returns to equity curve
    eq = np.exp(strat_log_rets.cumsum())

    return {
        "returns": strat_log_rets,
        "equity_curve": eq,
        "weights": weights,
        "signal": signal,
        "performance": compute_performance(strat_log_rets, eq),
    }

# ============================================
# GRID SEARCH (Sharpe → Trades)
# ============================================

def run_grid_search(prices, risk_on_weights, risk_off_weights):
    btc = prices["BTC-USD"]

    best_sharpe = -1e9
    best_trades = np.inf
    best_cfg = None
    best_result = None

    lengths = range(21, 253)
    types = ["sma", "ema"]
    tolerances = np.arange(0.0, 0.1001, 0.001)

    progress = st.progress(0.0)
    total = len(lengths) * len(types) * len(tolerances)
    idx = 0

    for length in lengths:
        for ma_type in types:
            for tol in tolerances:

                signal = generate_testfol_signal(btc, length, ma_type, tol)
                result = backtest(prices, signal, risk_on_weights, risk_off_weights)
                sharpe = result["performance"]["Sharpe"]

                sig = result["signal"]
                switches = sig.astype(int).diff().abs().sum()
                trades_per_year = switches / (len(sig) / 252)

                idx += 1
                progress.progress(idx / total)

                # Lexicographic optimization
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
    st.set_page_config(page_title="BTC MA Optimized Portfolio", layout="wide")
    st.title("BTC MA Optimized Portfolio (TestFol-Accurate Version)")

    st.write("""
    This version implements:
    - True TestFol hysteresis
    - Full 1-day delay
    - Log-return engine
    - Daily rebalance
    - No lookahead bias
    - Sharpe → Trades lexicographic optimization  
    """)

    st.sidebar.header("Backtest Settings")
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)")

    st.sidebar.header("Risk-ON Portfolio")
    risk_on_tickers_str = st.sidebar.text_input(
        "Tickers", ",".join(RISK_ON_WEIGHTS.keys())
    )
    risk_on_weights_str = st.sidebar.text_input(
        "Weights", ",".join(str(w) for w in RISK_ON_WEIGHTS.values())
    )

    st.sidebar.header("Risk-OFF Portfolio")
    risk_off_tickers_str = st.sidebar.text_input(
        "Tickers", ",".join(RISK_OFF_WEIGHTS.keys())
    )
    risk_off_weights_str = st.sidebar.text_input(
        "Weights", ",".join(str(w) for w in RISK_OFF_WEIGHTS.values())
    )

    if not st.sidebar.button("Run Backtest & Optimize"):
        st.stop()

    # Parse
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

    # Run optimization
    best_cfg, best_result = run_grid_search(prices, risk_on_weights, risk_off_weights)

    best_len, best_type, best_tol = best_cfg
    perf = best_result["performance"]
    sig = best_result["signal"]

    # Current regime
    latest_day = sig.index[-1]
    latest_signal = sig.iloc[-1]
    current_regime = "RISK-ON" if latest_signal else "RISK-OFF"

    st.subheader("Current Regime")
    st.write(f"**Date:** {latest_day.date()}")
    st.write(f"**Status:** {current_regime}")

    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252)

    st.subheader("Optimized Strategy Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{perf['CAGR']:.2%}")
    c2.metric("Volatility", f"{perf['Volatility']:.2%}")
    c3.metric("Sharpe", f"{perf['Sharpe']:.3f}")
    c4.metric("Max Drawdown", f"{perf['MaxDrawdown']:.2%}")
    c5.metric("Total Return", f"{perf['TotalReturn']:.2%}")

    st.subheader("Best Parameters Found")
    st.write(f"**Length:** {best_len}")
    st.write(f"**Type:** {best_type.upper()}")
    st.write(f"**Tolerance:** {best_tol:.2%}")
    st.write(f"**Trades per year:** {trades_per_year:.2f}")

    # Always-on portfolio in log space
    log_prices = np.log(prices)
    log_rets = log_prices.diff().fillna(0)

    user_rets = pd.Series(0, index=log_rets.index)
    for a, w in risk_on_weights.items():
        if a in log_rets.columns:
            user_rets += log_rets[a] * w

    user_eq = np.exp(user_rets.cumsum())
    user_perf = compute_performance(user_rets, user_eq)

    st.subheader("Always-On Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{user_perf['CAGR']:.2%}")
    c2.metric("Volatility", f"{user_perf['Volatility']:.2%}")
    c3.metric("Sharpe", f"{user_perf['Sharpe']:.3f}")
    c4.metric("Max Drawdown", f"{user_perf['MaxDrawdown']:.2%}")
    c5.metric("Total Return", f"{user_perf['TotalReturn']:.2%}")

    st.subheader("Equity Curve Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(best_result["equity_curve"], label="Optimized", linewidth=2)
    ax.plot(user_eq, label="Always-On Risk-ON", linestyle="--")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

if __name__ == "__main__":
    main()

