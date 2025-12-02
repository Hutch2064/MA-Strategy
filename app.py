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

DEFAULT_START_DATE = "2011-11-24"
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
# BUILD PORTFOLIO INDEX FOR SIGNAL
# ============================================

def build_portfolio_index(prices, weights_dict):
    log_px = np.log(prices)
    log_rets = log_px.diff().fillna(0)

    idx_rets = pd.Series(0.0, index=log_rets.index)
    for a, w in weights_dict.items():
        if a in log_rets.columns:
            idx_rets += log_rets[a] * w

    idx = np.exp(idx_rets.cumsum())
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
# BACKTEST ENGINE — WITH FLIP COST
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
        "DD_Series": dd
    }


def backtest(prices, signal, risk_on_weights, risk_off_weights):

    log_px = np.log(prices)
    log_rets = log_px.diff().fillna(0)

    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)
    strat_log_rets = (weights.shift(1).fillna(0) * log_rets).sum(axis=1)

    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    friction_series = np.where(flip_mask, -FLIP_COST, 0.0)
    strat_log_rets_adj = strat_log_rets + friction_series

    eq = np.exp(strat_log_rets_adj.cumsum())

    return {
        "returns": strat_log_rets_adj,
        "equity_curve": eq,
        "signal": signal,
        "weights": weights,
        "performance": compute_performance(strat_log_rets_adj, eq),
        "flip_mask": flip_mask,
    }



# ============================================
# GRID SEARCH
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

    prices = load_price_data(all_tickers, start, end_val).dropna(how="any")

    best_cfg, best_result = run_grid_search(prices, risk_on_weights, risk_off_weights)
    best_len, best_type, best_tol = best_cfg

    sig = best_result["signal"]
    perf = best_result["performance"]

    latest_signal = sig.iloc[-1]
    regime = "RISK-ON" if latest_signal else "RISK-OFF"

    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252)



    # ============================================
    # RISK-ON ALWAYS-ON PERFORMANCE
    # ============================================

    log_px = np.log(prices)
    log_rets = log_px.diff().fillna(0)

    risk_on_log = pd.Series(0.0, index=log_rets.index)
    for a, w in risk_on_weights.items():
        if a in log_rets.columns:
            risk_on_log += log_rets[a] * w

    risk_on_eq = np.exp(risk_on_log.cumsum())
    risk_on_perf = compute_performance(risk_on_log, risk_on_eq)



    # ============================================
    # SHARPE-OPTIMAL PORTFOLIO (RISK-ON ONLY)
    # ============================================

    risk_on_px = prices[[t for t in risk_on_tickers if t in prices.columns]].copy()
    common_start = risk_on_px.dropna().index[0]
    risk_on_px = risk_on_px.loc[common_start:]

    risk_on_rets = risk_on_px.pct_change().dropna(how="any")

    mu_vec = risk_on_rets.mean().values
    cov_mat = risk_on_rets.cov().values

    cov_mat += np.eye(cov_mat.shape[0]) * 1e-10

    def neg_sharpe(w):
        ret = np.dot(mu_vec, w)
        vol = np.sqrt(np.dot(w.T, cov_mat @ w))
        if vol == 0:
            return 1e9
        return -(ret / vol)

    n = len(mu_vec)
    bounds = [(0, 1)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})

    res = minimize(neg_sharpe, np.ones(n) / n, bounds=bounds, constraints=constraints)
    w_opt = res.x

    sharp_returns = (risk_on_rets * w_opt).sum(axis=1)
    sharp_eq = (1 + sharp_returns).cumprod()
    sharp_perf = compute_performance(np.log(1 + sharp_returns), sharp_eq)

    sharp_stats = {
        "CAGR": sharp_perf["CAGR"],
        "Volatility": sharp_perf["Volatility"],
        "Sharpe": sharp_perf["Sharpe"],
        "MaxDD": sharp_perf["MaxDrawdown"],
        "Total": sharp_perf["TotalReturn"]
    }

    sharp_weights_display = {t: round(w, 4) for t, w in zip(risk_on_px.columns, w_opt)}



    # ============================================
    # ADVANCED METRICS & STRATEGY STATS
    # ============================================

    def time_in_drawdown(dd):
        return (dd < 0).mean()

    def pain_to_gain(dd, cagr):
        ulcer = np.sqrt((dd**2).mean())
        return cagr / ulcer if ulcer != 0 else np.nan

    def mar_ratio(cagr, max_dd):
        return cagr / abs(max_dd) if max_dd != 0 else np.nan

    def pl_per_flip(returns, flip_mask):
        return float(returns[flip_mask].sum())

    def compute_stats(perf_obj, returns, dd_series, flip_mask, trades_per_year):
        cagr = perf_obj["CAGR"]
        vol = perf_obj["Volatility"]
        sharpe = perf_obj["Sharpe"]
        maxdd = perf_obj["MaxDrawdown"]
        total = perf_obj["TotalReturn"]

        return {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "MaxDD": maxdd,
            "Total": total,
            "MAR": mar_ratio(cagr, maxdd),
            "TID": time_in_drawdown(dd_series),
            "PainGain": pain_to_gain(dd_series, cagr),
            "Skew": returns.skew(),
            "Kurtosis": returns.kurt(),
            "P/L per flip": pl_per_flip(returns, flip_mask),
            "Trades/year": trades_per_year,
        }

    strat_stats = compute_stats(
        perf,
        best_result["returns"],
        perf["DD_Series"],
        best_result["flip_mask"],
        trades_per_year
    )

    risk_stats = compute_stats(
        risk_on_perf,
        risk_on_log,
        risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_log), dtype=bool),
        0
    )



    # ============================================
    # CONSOLIDATED METRIC TABLE (NOW 3-COLUMN)
    # ============================================

    st.subheader("Strategy vs. Sharpe-Optimal vs. Risk-ON Statistics")

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

    def fmt_pct(x):
        return f"{x:.2%}" if pd.notna(x) else "—"

    def fmt_dec(x):
        return f"{x:.3f}" if pd.notna(x) else "—"

    def fmt_num(x):
        return f"{x:,.2f}" if pd.notna(x) else "—"

    table_rows = []
    for label, key in rows:

        sv = strat_stats[key]
        shv = sharp_stats[key] if key in sharp_stats else None
        rv = risk_stats[key] if key in risk_stats else None

        if key in ["CAGR", "Volatility", "MaxDD", "Total", "TID"]:
            sv_fmt = fmt_pct(sv)
            sh_fmt = fmt_pct(shv)
            rv_fmt = fmt_pct(rv)
        elif key in ["Sharpe", "MAR", "PainGain", "Skew", "Kurtosis"]:
            sv_fmt = fmt_dec(sv)
            sh_fmt = fmt_dec(shv)
            rv_fmt = fmt_dec(rv)
        else:
            sv_fmt = fmt_num(sv)
            sh_fmt = fmt_num(shv)
            rv_fmt = fmt_num(rv)

        table_rows.append([label, sv_fmt, sh_fmt, rv_fmt])

    table = pd.DataFrame(table_rows, columns=["Metric", "Strategy", "Sharpe-Optimal", "Risk-On"])
    st.dataframe(table, use_container_width=True)



    # ============================================
    # SHOW SHARPE-OPTIMAL WEIGHTS
    # ============================================

    st.subheader("Sharpe-Optimal Weights (Risk-ON Universe)")
    st.write(sharp_weights_display)



    # ============================================
    # OPTIMAL SIGNAL PARAMETERS
    # ============================================

    st.subheader("Optimal Signal Parameters")
    st.write(f"**Moving Average Type:** {best_type.upper()}")
    st.write(f"**Optimal MA Length:** {best_len} days")
    st.write(f"**Optimal Tolerance:** {best_tol:.2%}")



    # ============================================
    # SIGNAL DISTANCE
    # ============================================

    st.subheader("Next Signal Information")

    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    ma_opt_dict = compute_ma_matrix(portfolio_index, [best_len], best_type)
    ma_opt_series = ma_opt_dict[best_len]

    latest_date = ma_opt_series.dropna().index[-1]
    P = float(portfolio_index.loc[latest_date])
    MA = float(ma_opt_series.loc[latest_date])
    tol = best_tol

    upper = MA * (1 + tol)
    lower = MA * (1 - tol)

    if latest_signal:
        pct_to_flip = (P - lower) / P
        direction = "RISK-ON → RISK-OFF"
        distance_str = f"Drop Required: {pct_to_flip:.2%}"
    else:
        pct_to_flip = (upper - P) / P
        direction = "RISK-OFF → RISK-ON"
        distance_str = f"Gain Required: {pct_to_flip:.2%}"

    st.write(f"**Latest Signal Change Direction:** {direction}")
    st.write(f"**Portfolio Index:** {P:,.2f}")
    st.write(f"**MA({best_len}) Value:** {MA:,.2f}")
    st.write(f"**Tolerance Bands:** Lower={lower:,.2f} | Upper={upper:,.2f}")
    st.write(f"**{distance_str}**")



    # ============================================
    # REGIME AGING STATS
    # ============================================

    st.subheader("Regime Statistics")

    sig_series = sig.astype(int)
    switch_points = sig_series.diff().fillna(0).ne(0)

    segments = []
    current_regime = sig_series.iloc[0]
    start_date = sig_series.index[0]

    for date, sw in switch_points.iloc[1:].items():
        if sw:
            end_date = date
            segments.append((current_regime, start_date, end_date))
            current_regime = sig_series.loc[date]
            start_date = date

    segments.append((current_regime, start_date, sig_series.index[-1]))

    regime_rows = []
    for r, s, e in segments:
        length_days = (e - s).days
        label = "RISK-ON" if r == 1 else "RISK-OFF"
        regime_rows.append([label, s.date(), e.date(), length_days])

    regime_df = pd.DataFrame(regime_rows, columns=["Regime", "Start", "End", "Duration (days)"])

    avg_on = regime_df[regime_df["Regime"] == "RISK-ON"]["Duration (days)"].mean()
    avg_off = regime_df[regime_df["Regime"] == "RISK-OFF"]["Duration (days)"].mean()

    st.write(f"**Average RISK-ON Duration:** {avg_on:.1f} days")
    st.write(f"**Average RISK-OFF Duration:** {avg_off:.1f} days")
    st.dataframe(regime_df, use_container_width=True)



    # ============================================
    # FINAL PLOT
    # ============================================

    st.subheader("Portfolio Strategy vs. Risk-On Graph")

    regime_color = "green" if latest_signal else "red"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(best_result["equity_curve"], label=f"Strategy ({regime})", linewidth=2, color=regime_color)
    ax.plot(portfolio_index, label="Portfolio Index (Risk-On Basket)", alpha=0.65)
    ax.plot(ma_opt_series, label=f"Optimal {best_type.upper()}({best_len}) MA", linewidth=2)

    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)



# ============================================
# LAUNCH APP
# ============================================

if __name__ == "__main__":
    main()