import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import streamlit as st

# =========================
# CONFIG
# =========================

DEFAULT_START_DATE = "2011-11-16"
DEFAULT_END_DATE = None  # None = up to today

TICKERS = ["BTC-USD", "GLD", "TQQQ", "UUP"]

RISK_ON_WEIGHTS = {
    "GLD": 3.0 / 3.0,
    "TQQQ": 1.0 / 3.0,
    "BTC-USD": 1.0 / 3.0,
}

RISK_OFF_WEIGHTS = {
    "UUP": 1.0,
}

DEFAULT_N_RANDOM_SEARCH_ITER = 1000
RISK_FREE_RATE = 0.0


# =========================
# DATA LOADING
# =========================

@st.cache_data(show_spinner=True)
def load_price_data(tickers, start_date, end_date=None):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False
    )

    if "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
    else:
        px = data["Close"].copy()

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    return px.dropna(how="all")


# =========================
# MOVING AVERAGES & SIGNALS
# =========================

def compute_ma(series, length, ma_type="ema"):
    ma_type = ma_type.lower()
    if ma_type == "ema":
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == "sma":
        return series.rolling(window=length, min_periods=length).mean()
    else:
        raise ValueError(f"Unknown ma_type: {ma_type}")


def generate_risk_on_signal(
    price_btc: pd.Series,
    ma_lengths,
    ma_types,
    ma_tolerances,
    min_ma_above: int,
    confirm_days: int = 1,
):
    conditions = []
    for length, ma_type, tol in zip(ma_lengths, ma_types, ma_tolerances):
        ma = compute_ma(price_btc, length, ma_type)
        cond = price_btc > ma * (1.0 + tol)
        conditions.append(cond)

    cond_df = pd.concat(conditions, axis=1)
    count_above = cond_df.sum(axis=1)
    base_signal = count_above >= min_ma_above

    if confirm_days <= 1:
        risk_on_raw = base_signal
    else:
        risk_on_raw = (
            base_signal
            .rolling(window=confirm_days, min_periods=confirm_days)
            .apply(lambda x: np.all(x == 1.0), raw=True)
            .astype(bool)
        )

    return risk_on_raw.reindex(price_btc.index).fillna(False)


# =========================
# BACKTEST ENGINE
# =========================

def build_weight_df(prices, risk_on_signal, risk_on_weights, risk_off_weights):
    cols = prices.columns
    weights = pd.DataFrame(index=prices.index, columns=cols, data=0.0)

    for asset, w in risk_on_weights.items():
        if asset in cols:
            weights.loc[risk_on_signal, asset] = w

    for asset, w in risk_off_weights.items():
        if asset in cols:
            weights.loc[~risk_on_signal, asset] = w

    return weights


def compute_performance(returns, equity_curve, rf_rate=0.0):
    n_days = len(returns)
    if n_days == 0:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "MaxDrawdown": np.nan,
            "TotalReturn": np.nan,
        }

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - rf_rate) / vol if vol != 0 else np.nan

    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    max_dd = dd.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": total_return,
    }


def backtest_strategy(prices, risk_on_signal, risk_on_weights, risk_off_weights):
    rets = prices.pct_change().fillna(0.0)
    weights = build_weight_df(prices, risk_on_signal, risk_on_weights, risk_off_weights)

    weights_shifted = weights.shift(1).fillna(0.0)
    strat_rets = (weights_shifted * rets).sum(axis=1)
    equity_curve = (1 + strat_rets).cumprod()

    return {
        "returns": strat_rets,
        "equity_curve": equity_curve,
        "weights": weights,
        "risk_on_signal": risk_on_signal,
        "performance": compute_performance(strat_rets, equity_curve),
    }


# =========================
# PARAM DATACLASS
# =========================

@dataclass
class StrategyParams:
    ma_lengths: list
    ma_types: list
    ma_tolerances: list
    min_ma_above: int
    confirm_days: int


# =========================
# RANDOM SEARCH OPTIMIZER
# =========================

def random_param_sample(rng):
    min_length = 21
    max_length = 252

    n_ma = rng.integers(1, 5)
    ma_lengths = list(rng.integers(min_length, max_length + 1, size=n_ma))
    ma_types = [rng.choice(["sma", "ema"]) for _ in range(n_ma)]
    ma_tolerances = list(rng.uniform(0.0, 0.05, size=n_ma))
    min_ma_above = rng.integers(1, n_ma + 1)
    confirm_days = rng.integers(2, 6)

    return StrategyParams(
        ma_lengths=ma_lengths,
        ma_types=ma_types,
        ma_tolerances=ma_tolerances,
        min_ma_above=min_ma_above,
        confirm_days=confirm_days,
    )


def run_random_search(
    prices,
    n_iter,
    risk_on_weights,
    risk_off_weights,
    rf_rate=0.0,
    seed=42,
    progress_bar=None,
):
    btc = prices["BTC-USD"]
    rng = np.random.default_rng(seed)
    best_params = None
    best_result = None
    best_sharpe = -np.inf

    for i in range(n_iter):
        params = random_param_sample(rng)

        risk_on = generate_risk_on_signal(
            price_btc=btc,
            ma_lengths=params.ma_lengths,
            ma_types=params.ma_types,
            ma_tolerances=params.ma_tolerances,
            min_ma_above=params.min_ma_above,
            confirm_days=params.confirm_days,
        )

        result = backtest_strategy(prices, risk_on, risk_on_weights, risk_off_weights)
        sharpe = result["performance"]["Sharpe"]

        if sharpe is not None and not np.isnan(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
            best_result = result

        if progress_bar is not None:
            progress_bar.progress((i + 1) / n_iter)

    return best_params, best_result


# =========================
# STREAMLIT APP
# =========================

def format_params(params: StrategyParams) -> pd.DataFrame:
    rows = []
    for i, (L, t, tol) in enumerate(
        zip(params.ma_lengths, params.ma_types, params.ma_tolerances), start=1
    ):
        rows.append({
            "MA #": i,
            "Length (days)": int(L),
            "Type": t.upper(),
            "Tolerance": f"{tol:.2%}",
        })
    df = pd.DataFrame(rows)
    meta = pd.DataFrame({
        "Setting": ["min_ma_above", "confirm_days"],
        "Value": [params.min_ma_above, params.confirm_days],
    })
    return df, meta


def main():
    st.set_page_config(page_title="Moving Average Strategy", layout="wide")

    st.title("Moving Average Strategy")

    st.sidebar.header("Backtest Settings")

    start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", DEFAULT_START_DATE)
    end_date = st.sidebar.text_input("End Date (YYYY-MM-DD) or empty for today", "")
    end_date_val = end_date if end_date.strip() != "" else None

    st.sidebar.markdown(f"Optimization iterations: **{DEFAULT_N_RANDOM_SEARCH_ITER}**")
    n_iter = DEFAULT_N_RANDOM_SEARCH_ITER
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    st.sidebar.header("Risk-ON Portfolio")
    default_risk_on_tickers = ",".join(RISK_ON_WEIGHTS.keys())
    default_risk_on_weights = ",".join(str(w) for w in RISK_ON_WEIGHTS.values())

    risk_on_tickers_str = st.sidebar.text_input("Risk-ON Tickers", value=default_risk_on_tickers)
    risk_on_weights_str = st.sidebar.text_input("Risk-ON Weights", value=default_risk_on_weights)

    st.sidebar.header("Risk-OFF Portfolio")
    default_risk_off_tickers = ",".join(RISK_OFF_WEIGHTS.keys())
    default_risk_off_weights = ",".join(str(w) for w in RISK_OFF_WEIGHTS.values())

    risk_off_tickers_str = st.sidebar.text_input("Risk-OFF Tickers", value=default_risk_off_tickers)
    risk_off_weights_str = st.sidebar.text_input("Risk-OFF Weights", value=default_risk_off_weights)

    run_button = st.sidebar.button("Run Backtest & Optimize")

    if not run_button:
        st.info("Set your parameters and click Run Backtest & Optimize.")
        return

    risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",") if t.strip()]
    risk_on_weights_list = [float(w.strip()) for w in risk_on_weights_str.split(",")]
    risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))

    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",") if t.strip()]
    risk_off_weights_list = [float(w.strip()) for w in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))

    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers + ["BTC-USD"]))

    with st.spinner("Downloading data and optimizing..."):
        prices = load_price_data(all_tickers, start_date, end_date_val)
        prices = prices.dropna(how="any")

        progress_bar = st.progress(0.0)
        best_params, best_result = run_random_search(
            prices=prices,
            n_iter=n_iter,
            risk_on_weights=risk_on_weights,
            risk_off_weights=risk_off_weights,
            rf_rate=RISK_FREE_RATE,
            seed=seed,
            progress_bar=progress_bar,
        )

        perf = best_result["performance"]
        risk_on_signal = best_result["risk_on_signal"]

        rets = prices.pct_change().fillna(0.0)
        user_rets = sum(rets[a] * w for a, w in risk_on_weights.items() if a in rets)
        user_curve = (1 + user_rets).cumprod()
        user_perf = compute_performance(user_rets, user_curve)

    st.subheader("Optimized Strategy Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{perf['CAGR']:.2%}")
    c2.metric("Volatility", f"{perf['Volatility']:.2%}")
    c3.metric("Sharpe", f"{perf['Sharpe']:.3f}")
    c4.metric("Max Drawdown", f"{perf['MaxDrawdown']:.2%}")
    c5.metric("Total Return", f"{perf['TotalReturn']:.2%}")

    st.subheader("Always-On Portfolio Performance")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("CAGR", f"{user_perf['CAGR']:.2%}")
    d2.metric("Volatility", f"{user_perf['Volatility']:.2%}")
    d3.metric("Sharpe", f"{user_perf['Sharpe']:.3f}")
    d4.metric("Max Drawdown", f"{user_perf['MaxDrawdown']:.2%}")
    d5.metric("Total Return", f"{user_perf['TotalReturn']:.2%}")

    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Background shading
    signal = risk_on_signal.astype(int)
    changes = signal.diff().fillna(0)
    segments = []
    current_state = signal.iloc[0]
    start_idx = signal.index[0]

    for date, change in changes.items():
        if change != 0:
            segments.append((start_idx, date, current_state))
            start_idx = date
            current_state = 1 - current_state
    segments.append((start_idx, signal.index[-1], current_state))

    for start, end, state in segments:
        color = "blue" if state == 1 else "red"
        ax.axvspan(start, end, color=color, alpha=0.07)

    ax.plot(best_result["equity_curve"], label="Optimized Strategy", linewidth=2)
    ax.plot(user_curve, label="Always-On Portfolio", linestyle="--", linewidth=2)
    ax.set_title("Equity Curves with Risk-On/Off Regimes")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)


if __name__ == "__main__":
    main()


