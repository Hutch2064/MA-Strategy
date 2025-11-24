import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass
import streamlit as st

# =========================
# CONFIG
# =========================

DEFAULT_START_DATE = "2011-11-24"
DEFAULT_END_DATE = None

TICKERS = ["BTC-USD", "GLD", "TQQQ", "UUP"]

RISK_ON_WEIGHTS = {
    "GLD": 3.0 / 3.0,
    "TQQQ": 1.0 / 3.0,
    "BTC-USD": 1.0 / 3.0,
}

RISK_OFF_WEIGHTS = {
    "UUP": 1.0,
}

RISK_FREE_RATE = 0.0


# =========================
# DATA LOADING
# =========================

@st.cache_data(show_spinner=True)
def load_price_data(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    px = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px.dropna(how="all")


# =========================
# MOVING AVERAGES
# =========================

def compute_ma(series, length, ma_type):
    if ma_type == "ema":
        return series.ewm(span=length, adjust=False).mean()
    return series.rolling(window=length, min_periods=length).mean()


# =========================
# SIGNAL
# =========================

def generate_signal(price, length, ma_type, tolerance):
    ma = compute_ma(price, length, ma_type)
    return (price > ma * (1 + tolerance)).fillna(False)


# =========================
# BACKTEST ENGINE
# =========================

def build_weight_df(prices, signal, risk_on_w, risk_off_w):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for a, w in risk_on_w.items():
        if a in weights.columns:
            weights.loc[signal, a] = w
    for a, w in risk_off_w.items():
        if a in weights.columns:
            weights.loc[~signal, a] = w
    return weights


def compute_performance(returns, equity_curve):
    n = len(returns)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (1 + total_return) ** (252 / n) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else -np.inf
    dd = (equity_curve / equity_curve.cummax()) - 1
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": dd.min(),
        "TotalReturn": total_return,
    }


def backtest(prices, signal, ron_w, roff_w):
    rets = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, ron_w, roff_w)
    strat_rets = (weights.shift(1).fillna(0) * rets).sum(axis=1)
    eq = (1 + strat_rets).cumprod()
    return strat_rets, eq


# =========================
# DETERMINISTIC GRID SEARCH
# =========================

def run_grid_search(prices, risk_on_w, risk_off_w, progress_bar=None):
    btc = prices["BTC-USD"]

    lengths = range(21, 253)  # 21 to 252
    types = ["sma", "ema"]
    tolerances = np.round(np.arange(0.0, 0.1001, 0.001), 3)

    best = None
    best_params = None
    best_sharpe = -np.inf

    total_tests = len(lengths) * len(types) * len(tolerances)
    test_count = 0

    for ma_type in types:
        for L in lengths:
            for tol in tolerances:
                test_count += 1
                if progress_bar:
                    progress_bar.progress(test_count / total_tests)

                sig = generate_signal(btc, L, ma_type, tol)
                strat_rets, eq = backtest(prices, sig, risk_on_w, risk_off_w)
                perf = compute_performance(strat_rets, eq)

                if perf["Sharpe"] > best_sharpe:
                    best_sharpe = perf["Sharpe"]
                    best = {
                        "returns": strat_rets,
                        "equity_curve": eq,
                        "signal": sig,
                        "performance": perf,
                    }
                    best_params = (L, ma_type, tol)

    return best_params, best


# =========================
# STREAMLIT APP
# =========================

def main():
    st.set_page_config(page_title="BTC Trend Optimized Portfolio", layout="wide")
    st.title("Bitcoin Trend – Deterministic MA Optimizer (1 MA)")

    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date", "")
    end = end if end.strip() else None

    risk_on_t = ",".join(RISK_ON_WEIGHTS.keys())
    risk_on_w = ",".join(str(x) for x in RISK_ON_WEIGHTS.values())

    ron_tickers = [t.strip().upper() for t in risk_on_t.split(",")]
    ron_weights = [float(x) for x in risk_on_w.split(",")]
    ron_dict = dict(zip(ron_tickers, ron_weights))

    roff_tickers = ["UUP"]
    roff_weights = [1.0]
    roff_dict = dict(zip(roff_tickers, roff_weights))

    if st.sidebar.button("Run"):
        prices = load_price_data(sorted(set(ron_tickers + roff_tickers + ["BTC-USD"])), start, end)
        prices = prices.dropna()

        progress = st.progress(0.0)

        (L, ma_type, tol), result = run_grid_search(prices, ron_dict, roff_dict, progress)

        perf = result["performance"]
        sig = result["signal"]
        eq = result["equity_curve"]

        # Output
        st.subheader("Optimized Parameters")
        st.write(f"MA Length: {L}")
        st.write(f"MA Type: {ma_type.upper()}")
        st.write(f"Tolerance: {tol:.3f}")

        st.subheader("Performance")
        for k, v in perf.items():
            st.write(f"{k}: {v:.4f}")

        # Trading frequency
        flips = sig.astype(int).diff().abs().fillna(0).sum()
        st.write(f"Trades: {int(flips)}")

        # Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(eq, label="Optimized Strategy", linewidth=2)
        ax.grid(alpha=0.3)
        st.pyplot(fig)


if __name__ == "__main__":
    main()


