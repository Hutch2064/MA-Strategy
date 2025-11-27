import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import optuna

# ============================================
# CONFIG
# ============================================

DEFAULT_START_DATE = "2011-11-24"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "GLD": 1.0,
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
# TECHNICALS — VECTORIZED TESTFOL HYSTERESIS
# ============================================

def compute_ma(price, L, ma_type):
    if ma_type == "ema":
        return price.ewm(span=L, adjust=False).mean()
    return price.rolling(window=L, min_periods=L).mean()


def generate_testfol_signal(price, ma, tol, delay):
    px = price.shift(1 + delay).values
    ma_vals = ma.shift(delay).values
    n = len(px)

    upper = ma_vals * (1 + tol)
    lower = ma_vals * (1 - tol)

    sig = np.zeros(n, dtype=bool)

    first_valid = np.nanargmin(np.isnan(ma_vals))
    if first_valid < 2:
        first_valid = 2

    for t in range(first_valid, n):
        if not sig[t-1]:
            sig[t] = px[t] > upper[t]
        else:
            sig[t] = not (px[t] < lower[t])

    return pd.Series(sig, index=price.index).fillna(False)

# ============================================
# BACKTEST — LOG RETURNS + DAILY REBALANCE
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
    sharpe = (cagr - rf) / vol if vol > 0 else -999
    return sharpe


def run_backtest(prices, L, ma_type, tol, delay, risk_on_weights, risk_off_weights):
    log_px = np.log(prices)
    log_rets = log_px.diff().fillna(0)

    btc = prices["BTC-USD"]
    ma = compute_ma(btc, L, ma_type)

    sig = generate_testfol_signal(btc, ma, tol, delay)
    weights = build_weight_df(prices, sig, risk_on_weights, risk_off_weights)

    strat_log = (weights.shift(1).fillna(0) * log_rets).sum(axis=1)
    eq = np.exp(strat_log.cumsum())

    sharpe = compute_performance(strat_log, eq)
    return sharpe, sig, eq, strat_log

# ============================================
# OPTUNA OBJECTIVE
# ============================================

def build_objective(prices, risk_on_weights, risk_off_weights):

    def objective(trial):

        L = trial.suggest_int("length", 21, 252)
        ma_type = trial.suggest_categorical("ma_type", ["sma", "ema"])
        tol = trial.suggest_float("tolerance", 0.0, 0.10, step=0.001)
        delay = trial.suggest_int("delay", 0, 21)

        sharpe, _, _, _ = run_backtest(
            prices, L, ma_type, tol, delay, risk_on_weights, risk_off_weights
        )

        return sharpe

    return objective

# ============================================
# STREAMLIT APP
# ============================================

def main():
    st.set_page_config(page_title="BTC MA Optimized Portfolio", layout="wide")
    st.title("BTC MA Strategy — Optuna Sharpe Optimization")

    st.sidebar.header("Backtest Settings")
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")
    n_trials = st.sidebar.number_input("Optuna Trials", 150, 3000, 300)

    st.sidebar.header("Risk-ON Portfolio")
    ron = st.sidebar.text_input("Tickers", ",".join(RISK_ON_WEIGHTS.keys()))
    ron_w = st.sidebar.text_input("Weights", ",".join(str(w) for w in RISK_ON_WEIGHTS.values()))

    st.sidebar.header("Risk-OFF Portfolio")
    rof = st.sidebar.text_input("Tickers", ",".join(RISK_OFF_WEIGHTS.keys()))
    rof_w = st.sidebar.text_input("Weights", ",".join(str(w) for w in RISK_OFF_WEIGHTS.values()))

    if not st.sidebar.button("Run Optimization"):
        st.stop()

    ron_tickers = [t.strip().upper() for t in ron.split(",")]
    ron_weights_list = [float(x) for x in ron_w.split(",")]
    ron_weights = dict(zip(ron_tickers, ron_weights_list))

    rof_tickers = [t.strip().upper() for t in rof.split(",")]
    rof_weights_list = [float(x) for x in rof_w.split(",")]
    rof_weights = dict(zip(rof_tickers, rof_weights_list))

    all_tickers = sorted(set(ron_tickers + rof_tickers))
    if "BTC-USD" not in all_tickers:
        all_tickers.append("BTC-USD")

    end_val = end if end.strip() else None
    prices = load_price_data(all_tickers, start, end_val)

    # Run Optuna
    st.write("### Running Optuna…")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        build_objective(prices, ron_weights, rof_weights),
        n_trials=n_trials,
        n_jobs=1
    )

    best = study.best_params

    st.success("Optimization Complete")
    st.write(best)

    # Full backtest for best parameters
    sharpe, sig, eq, strat_log = run_backtest(
        prices,
        best["length"],
        best["ma_type"],
        best["tolerance"],
        best["delay"],
        ron_weights,
        rof_weights
    )

    st.subheader("Sharpe")
    st.metric("Sharpe", f"{sharpe:.3f}")

    st.subheader("Best Parameters")
    st.write(best)

    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(eq, label="Optimized Strategy", linewidth=2)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()