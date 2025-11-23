import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import streamlit as st

# =========================
# CONFIG
# =========================

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = None  # None = up to today

TICKERS = ["BTC-USD", "SHNY", "TQQQ", "UUP"]

RISK_ON_WEIGHTS = {
    "SHNY": 1.0 / 3.0,
    "TQQQ": 1.0 / 3.0,
    "BTC-USD": 1.0 / 3.0,
}

RISK_OFF_WEIGHTS = {
    "UUP": 1.0,
}

DEFAULT_N_RANDOM_SEARCH_ITER = 300
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
    """
    price_btc: BTC-USD price series
    ma_lengths: list[int]
    ma_types: list['sma' | 'ema']
    ma_tolerances: list[float], price > MA * (1 + tol)
    min_ma_above: how many MAs must be satisfied
    confirm_days: condition must hold this many consecutive days
    """
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

    # REAL-LIFE EXECUTION: use yesterday's weights on today's returns
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

def random_param_sample():
    """
    Sample one set of strategy parameters.

    MA lengths: any integer from 21 to 252
    Number of MAs: 1 to 6
    MA type: SMA or EMA
    Tolerance: 0% to 5%
    min_ma_above: 1 .. n_ma
    confirm_days: one of [1, 3, 5, 10, 20]
    """
    min_length = 21
    max_length = 252

    n_ma = np.random.randint(1, 7)  # 1–6 MAs

    ma_lengths = list(np.random.randint(min_length, max_length + 1, size=n_ma))
    ma_types = [np.random.choice(["sma", "ema"]) for _ in range(n_ma)]
    ma_tolerances = list(np.random.choice([0.0, 0.005, 0.01, 0.02, 0.05], size=n_ma))

    min_ma_above = np.random.randint(1, n_ma + 1)
    confirm_days = int(np.random.choice([1, 3, 5, 10, 20]))

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
):
    btc = prices["BTC-USD"]

    np.random.seed(seed)

    best_params = None
    best_result = None
    best_sharpe = -np.inf

    for _ in range(n_iter):
        params = random_param_sample()

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

    return best_params, best_result


# =========================
# STREAMLIT APP
# =========================

def format_params(params: StrategyParams) -> pd.DataFrame:
    rows = []
    for i, (L, t, tol) in enumerate(
        zip(params.ma_lengths, params.ma_types, params.ma_tolerances), start=1
    ):
        rows.append(
            {
                "MA #": i,
                "Length (days)": int(L),
                "Type": t.upper(),
                "Tolerance": f"{tol:.2%}",
            }
        )
    df = pd.DataFrame(rows)
    meta = pd.DataFrame(
        {
            "Setting": ["min_ma_above", "confirm_days"],
            "Value": [params.min_ma_above, params.confirm_days],
        }
    )
    return df, meta


def main():
    st.set_page_config(page_title="BTC Trend Optimized Portfolio", layout="wide")

    st.title("Bitcoin Trend – Optimized Risk-On Portfolio")
    st.write(
        "This app optimizes a BTC-based risk-on / risk-off strategy using moving averages "
        "and backtests the resulting portfolio:\n\n"
        "- **Risk-On**: 33.33% SHNY, 33.33% TQQQ, 33.33% BTC\n"
        "- **Risk-Off**: 100% UUP\n\n"
        "The optimizer searches over MA lengths (21–252 days), number of MAs, SMA vs EMA, "
        "tolerances, confirmation window, and confirmation count, maximizing Sharpe ratio."
    )

    # Sidebar controls
    st.sidebar.header("Backtest Settings")

    start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", DEFAULT_START_DATE)
    end_date = st.sidebar.text_input("End Date (YYYY-MM-DD) or empty for today", "")
    end_date_val = end_date if end_date.strip() != "" else None

    n_iter = st.sidebar.slider(
        "Optimization iterations", min_value=50, max_value=1000,
        value=DEFAULT_N_RANDOM_SEARCH_ITER, step=50
    )

    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    run_button = st.sidebar.button("Run Backtest & Optimize")

    if not run_button:
        st.info("Set your parameters in the sidebar and click **Run Backtest & Optimize**.")
        return

    with st.spinner("Downloading data and running optimization..."):
        prices = load_price_data(TICKERS, start_date, end_date_val)

        # Keep only the core assets and drop rows with missing data
        core_assets = [t for t in ["BTC-USD", "SHNY", "TQQQ", "UUP"] if t in prices.columns]
        if len(core_assets) < 4:
            st.error(f"Missing one or more tickers in data. Found: {core_assets}")
            return

        prices = prices[core_assets].dropna(how="any")

        best_params, best_result = run_random_search(
            prices=prices,
            n_iter=n_iter,
            risk_on_weights=RISK_ON_WEIGHTS,
            risk_off_weights=RISK_OFF_WEIGHTS,
            rf_rate=RISK_FREE_RATE,
            seed=seed,
        )

        perf = best_result["performance"]
        risk_on_signal = best_result["risk_on_signal"]

        # Pure risk-on benchmark (always 33/33/33)
        rets = prices.pct_change().fillna(0.0)
        cols = ["SHNY", "TQQQ", "BTC-USD"]
        pure_risk_on_rets = (rets[cols] * np.array([1 / 3, 1 / 3, 1 / 3])).sum(axis=1)
        pure_risk_on_curve = (1 + pure_risk_on_rets).cumprod()

    # ====== Display Results ======

    st.subheader("Optimized Strategy Performance")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("CAGR", f"{perf['CAGR']:.2%}")
    col2.metric("Volatility", f"{perf['Volatility']:.2%}")
    col3.metric("Sharpe", f"{perf['Sharpe']:.3f}")
    col4.metric("Max Drawdown", f"{perf['MaxDrawdown']:.2%}")
    col5.metric("Total Return", f"{perf['TotalReturn']:.2%}")

    st.subheader("Optimized Moving Average Configuration")

    df_ma, df_meta = format_params(best_params)
    st.write("**Moving Averages**")
    st.dataframe(df_ma, use_container_width=True)

    st.write("**Global Settings**")
    st.dataframe(df_meta, use_container_width=True)

    # Current regime
    is_risk_on_today = bool(risk_on_signal.iloc[-1])
    regime_text = "RISK-ON (33/33/33 SHNY / TQQQ / BTC)" if is_risk_on_today else "RISK-OFF (100% UUP)"
    regime_color = "🟢" if is_risk_on_today else "🔴"

    st.subheader("Current Regime")
    st.markdown(f"### {regime_color} Today the model is in: **{regime_text}**")

    # Equity curves
    st.subheader("Equity Curve")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(best_result["equity_curve"], label="Optimized Strategy", linewidth=2)
    ax.plot(pure_risk_on_curve, label="Pure Risk-On 33/33/33", linestyle="--", linewidth=2)
    ax.set_title("Equity Curves: Optimized Strategy vs Pure Risk-On")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.caption(
        "Execution assumes trades occur on the next day's open based on today's signal "
        "(weights are shifted by one day), matching realistic manual trading behavior."
    )


if __name__ == "__main__":
    main()
