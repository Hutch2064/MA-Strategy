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

DEFAULT_START_DATE = "1999-01-01"
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
QUARTER_DAYS = 63   # approx trading days per quarter
START_RISKY = 0.60  # SIG starts at 60% risky
START_SAFE = 0.40   # SIG starts at 40% risk-off bucket


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
# BUILD PORTFOLIO INDEX — SIMPLE RETURNS
# ============================================

def build_portfolio_index(prices, weights_dict):
    simple_rets = prices.pct_change().fillna(0)

    idx_rets = pd.Series(0.0, index=simple_rets.index)
    for a, w in weights_dict.items():
        if a in simple_rets.columns:
            idx_rets += simple_rets[a] * w

    idx = (1 + idx_rets).cumprod()
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
# TESTFOL SIGNAL LOGIC
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


# ============================================================
# NEW: SIG ENGINE (3Sig mechanics)
# ============================================================

def run_sig_engine(risk_on_returns, risk_off_returns, target_quarter, ma_signal):
    """
    risk_on_returns : Series (daily simple returns of risk-on portfolio)
    risk_off_returns: Series (daily simple returns of risk-off portfolio)
    target_quarter  : float (average quarterly BH return of the risk-on portfolio)
    ma_signal       : Series of bool (True = risk-on, False = risk-off)
    """

    dates = risk_on_returns.index
    n = len(dates)

    # Starting SIG buckets
    risky_w = START_RISKY
    safe_w = START_SAFE

    risky_val = 6000.0
    safe_val = 4000.0
    total_val = risky_val + safe_val

    equity_curve = []
    risky_weights_series = []
    safe_weights_series = []
    rebalance_events = 0

    # To resume correctly after risk-off:
    frozen_risky_w = None
    frozen_safe_w = None

    for i in range(n):

        if ma_signal.iloc[i]:  
            # ------------------------------
            # RISK-ON (SIG engine active)
            # ------------------------------

            # If this is a re-entry into risk-on, resume frozen weights
            if frozen_risky_w is not None:
                risky_w = frozen_risky_w
                safe_w = frozen_safe_w
                risky_val = total_val * risky_w
                safe_val = total_val * safe_w
                frozen_risky_w = None
                frozen_safe_w = None

            # Apply daily returns to both buckets
            risky_val *= (1 + risk_on_returns.iloc[i])
            safe_val  *= (1 + risk_off_returns.iloc[i])

            total_val = risky_val + safe_val

            # Quarterly rebalance — only once we have enough history
            if i % QUARTER_DAYS == 0 and i > 0:

                # Not enough data yet for a quarterly comparison
                if i - QUARTER_DAYS < 0:
                    pass
                else:
                    start_val = equity_curve[i - QUARTER_DAYS]
                    quarter_growth = (total_val / start_val) - 1

                    if quarter_growth > target_quarter:
                    excess = (quarter_growth - target_quarter) * total_val
                    safe_val += excess
                    risky_val -= excess
                    rebalance_events += 1

                total_val = risky_val + safe_val
                risky_w = risky_val / total_val
                safe_w = safe_val / total_val
                
                quarter_growth = (total_val / start_val) - 1

                if quarter_growth > target_quarter:
                    excess = (quarter_growth - target_quarter) * total_val
                    safe_val += excess
                    risky_val -= excess
                    rebalance_events += 1

                total_val = risky_val + safe_val

            # update drifted weights
            risky_w = risky_val / total_val
            safe_w = safe_val / total_val

        else:
            # ------------------------------
            # RISK-OFF (Freeze SIG and go 100% risk-off)
            # ------------------------------

            # Save SIG state (frozen once)
            if frozen_risky_w is None:
                frozen_risky_w = risky_w
                frozen_safe_w = safe_w

            # Portfolio becomes 100% risk-off returns
            total_val *= (1 + risk_off_returns.iloc[i])

            # SIG engine does NOT drift or rebalance here
            risky_w = 0.0
            safe_w = 1.0

        equity_curve.append(total_val)
        risky_weights_series.append(risky_w)
        safe_weights_series.append(safe_w)

    return (
        pd.Series(equity_curve, index=dates),
        pd.Series(risky_weights_series, index=dates),
        pd.Series(safe_weights_series, index=dates),
        rebalance_events
    )
# ============================================
# GRID SEARCH (unchanged)
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

    # --- Load prices
    prices = load_price_data(all_tickers, start, end_val).dropna(how="any")

    # --- Run MA grid search
    best_cfg, best_result = run_grid_search(prices, risk_on_weights, risk_off_weights)
    best_len, best_type, best_tol = best_cfg

    sig = best_result["signal"]
    perf = best_result["performance"]

    latest_signal = sig.iloc[-1]
    regime = "RISK-ON" if latest_signal else "RISK-OFF"

    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252)

    # ============================================
    # ALWAYS-ON RISK-ON PERFORMANCE (Simple Returns)
    # ============================================

    simple_rets = prices.pct_change().fillna(0)

    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_on_eq = (1 + risk_on_simple).cumprod()
    risk_on_perf = compute_performance(risk_on_simple, risk_on_eq)

    # ============================================
    # SHARPE OPTIMAL PORTFOLIO
    # (UNCHANGED — required for your validation column)
    # ============================================

    risk_on_px = prices[[t for t in risk_on_tickers if t in prices.columns]].copy()
    risk_on_px = risk_on_px.dropna()

    risk_on_rets = risk_on_px.pct_change().dropna()

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
    sharp_perf = compute_performance(sharp_returns, sharp_eq)

    sharp_stats = {
        "CAGR": sharp_perf["CAGR"],
        "Volatility": sharp_perf["Volatility"],
        "Sharpe": sharp_perf["Sharpe"],
        "MaxDD": sharp_perf["MaxDrawdown"],
        "Total": sharp_perf["TotalReturn"],
        "MAR": sharp_perf["CAGR"] / abs(sharp_perf["MaxDrawdown"]) if sharp_perf["MaxDrawdown"] != 0 else np.nan,
        "TID": (sharp_perf["DD_Series"] < 0).mean(),
        "PainGain": sharp_perf["CAGR"] / np.sqrt((sharp_perf["DD_Series"]**2).mean())
                    if (sharp_perf["DD_Series"]**2).mean() != 0 else np.nan,
        "Skew": sharp_returns.skew(),
        "Kurtosis": sharp_returns.kurt(),
        "P/L per flip": 0.0,
        "Trades/year": 0.0,
    }

    sharp_weights_display = {t: round(w, 4) for t, w in zip(risk_on_px.columns, w_opt)}


    # ============================================================
    # NEW SECTION: BUILD SIG-HYBRID STRATEGY
    # ============================================================

    # Daily risk-on portfolio returns
    risk_on_daily = risk_on_simple.copy()

    # Daily risk-off portfolio returns
    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    # Compute quarterly target based on risk-on CAGR
    bh_cagr = np.power(risk_on_eq.iloc[-1] / risk_on_eq.iloc[0], 252/len(risk_on_eq)) - 1
    quarterly_target = (1 + bh_cagr)**(1/4) - 1   # same as (annual → quarterly)

    # Run hybrid engine
    hybrid_eq, hybrid_rw, hybrid_sw, hybrid_rebals = run_sig_engine(
        risk_on_daily,
        risk_off_daily,
        quarterly_target,
        sig
    )

    # Hybrid performance calculations
    hybrid_simple = hybrid_eq.pct_change().fillna(0)
    hybrid_perf = compute_performance(hybrid_simple, hybrid_eq)
    # ============================================================
    # ADVANCED METRICS & STATS (Unchanged)
    # ============================================================

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
        risk_on_simple,
        risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_simple), dtype=bool),
        0
    )

    hybrid_stats = compute_stats(
        hybrid_perf,
        hybrid_simple,
        hybrid_perf["DD_Series"],
        np.zeros(len(hybrid_simple), dtype=bool),
        0
    )

    avg_safe_exposure = hybrid_sw.mean()


    # ============================================
    # METRIC TABLE — NOW 4 COLUMNS
    # ============================================

    st.subheader("Strategy vs. Sharpe-Optimal vs. Risk-ON vs. Hybrid")

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

    def fmt_pct(x): return f"{x:.2%}" if pd.notna(x) else "—"
    def fmt_dec(x): return f"{x:.3f}" if pd.notna(x) else "—"
    def fmt_num(x): return f"{x:,.2f}" if pd.notna(x) else "—"

    table_rows = []
    for label, key in rows:
        sv = strat_stats[key]
        shv = sharp_stats[key]
        rv = risk_stats[key]
        hv = hybrid_stats[key]

        if key in ["CAGR", "Volatility", "MaxDD", "Total", "TID"]:
            sv_fmt, sh_fmt, rv_fmt, hv_fmt = fmt_pct(sv), fmt_pct(shv), fmt_pct(rv), fmt_pct(hv)
        elif key in ["Sharpe", "MAR", "PainGain", "Skew", "Kurtosis"]:
            sv_fmt, sh_fmt, rv_fmt, hv_fmt = fmt_dec(sv), fmt_dec(shv), fmt_dec(rv), fmt_dec(hv)
        else:
            sv_fmt, sh_fmt, rv_fmt, hv_fmt = fmt_num(sv), fmt_num(shv), fmt_num(rv), fmt_num(hv)

        table_rows.append([label, sv_fmt, sh_fmt, rv_fmt, hv_fmt])

    table = pd.DataFrame(
        table_rows,
        columns=["Metric", "Strategy", "Sharpe-Optimal", "Risk-On", "Hybrid"]
    )
    st.dataframe(table, use_container_width=True)


    # ============================================
    # EXTERNAL VALIDATION LINK (unchanged)
    # ============================================

    link_url = "https://testfol.io/optimizer?s=9y4FBdfW2oO"
    link_text = "View the Sharpe Optimal recommended portfolio"

    st.subheader("External Sharpe Optimal Validation Link")
    st.markdown(
        f"**Quick Access:** [{link_text}]({link_url})"
    )


    # ============================================
    # SHOW SIG DRIFT + SAFE BUCKET
    # ============================================

    st.subheader("Hybrid SIG Engine Diagnostics")
    st.write(f"**Average Safe Allocation:** {avg_safe_exposure:.2%}")
    st.write("**Final Risk-On Allocation:**")
    st.write(hybrid_rw.iloc[-1])


    # ============================================
    # OPTIMAL SIGNAL PARAMETERS
    # ============================================

    st.subheader("Optimal Signal Parameters")
    st.write(f"**Moving Average Type:** {best_type.upper()}")
    st.write(f"**Optimal MA Length:** {best_len} days")
    st.write(f"**Optimal Tolerance:** {best_tol:.2%}")


    # ============================================
    # SIGNAL DISTANCE (unchanged)
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

    st.write(f"**Portfolio Index:** {P:,.2f}")
    st.write(f"**MA({best_len}) Value:** {MA:,.2f}")
    st.write(f"**Tolerance Bands:** Lower={lower:,.2f} | Upper={upper:,.2f}")
    st.write(f"**{distance_str}**")


    # ============================================
    # REGIME AGING STATS (unchanged)
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
    # FINAL PLOT — includes Hybrid
    # ============================================

    st.subheader("Portfolio Strategy vs. Sharpe-Optimal vs. Hybrid vs. Risk-On")

    fig, ax = plt.subplots(figsize=(12, 6))

    regime_color = "green" if latest_signal else "red"

    ax.plot(best_result["equity_curve"], label=f"Strategy ({regime})", linewidth=2, color=regime_color)
    ax.plot(sharp_eq, label="Sharpe-Optimal Portfolio", linewidth=2, color="magenta")
    ax.plot(portfolio_index, label="Portfolio Index (Risk-On Basket)", alpha=0.65)
    ax.plot(hybrid_eq, label="Hybrid SIG Strategy", linewidth=2, color="blue")

    ax.plot(ma_opt_series, label=f"Optimal {best_type.upper()}({best_len}) MA", linewidth=2, linestyle="--")

    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)


# ============================================
# LAUNCH APP
# ============================================

if __name__ == "__main__":
    main()