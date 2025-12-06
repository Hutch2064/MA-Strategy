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

RISK_ON_WEIGHTS = {"UGL": .22, "QLD": .39, "BTC-USD": .39}
RISK_OFF_WEIGHTS = {"SHY": 1.0}
FLIP_COST = 0.00875
QUARTER_DAYS = 63  # 3Sig rebal cadence


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
    simple = prices.pct_change().fillna(0)
    idx_rets = pd.Series(0.0, index=simple.index)
    for a, w in weights_dict.items():
        if a in simple.columns:
            idx_rets += simple[a] * w
    return (1 + idx_rets).cumprod()


# ============================================
# MA MATRIX
# ============================================

def compute_ma_matrix(price_series, lengths, ma_type):
    out = {}
    if ma_type == "ema":
        for L in lengths:
            out[L] = price_series.ewm(span=L, adjust=False).mean().shift(1)
    else:
        for L in lengths:
            out[L] = price_series.rolling(L, min_periods=L).mean().shift(1)
    return out


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

    for t in range(first_valid + 1, n):
        if not sig[t - 1]:
            sig[t] = px[t] > upper[t]
        else:
            sig[t] = not (px[t] < lower[t])

    return pd.Series(sig, index=ma.index).fillna(False)


# ============================================
# BACKTEST ENGINE
# ============================================

def build_weight_df(prices, signal, risk_on_weights, risk_off_weights):
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for a, wt in risk_on_weights.items():
        if a in prices.columns:
            w.loc[signal, a] = wt
    for a, wt in risk_off_weights.items():
        if a in prices.columns:
            w.loc[~signal, a] = wt
    return w

def compute_performance(simple_returns, eq, rf=0.0):
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1
    vol = simple_returns.std() * np.sqrt(252)
    sharpe = (simple_returns.mean()*252 - rf) / vol if vol>0 else np.nan
    dd = eq / eq.cummax() - 1
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": dd.min(),
        "TotalReturn": eq.iloc[-1] / eq.iloc[0] - 1,
        "DD_Series": dd
    }

def backtest(prices, signal, ron_w, roff_w):
    simple = prices.pct_change().fillna(0)
    w = build_weight_df(prices, signal, ron_w, roff_w)
    strat = (w.shift(1).fillna(0) * simple).sum(axis=1)

    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1
    strat += np.where(flip_mask, -FLIP_COST, 0)

    eq = (1 + strat).cumprod()
    return {
        "returns": strat, "equity_curve": eq, "signal": signal,
        "weights": w, "performance": compute_performance(strat, eq),
        "flip_mask": flip_mask
    }


# ============================================
# GRID SEARCH
# ============================================

def run_grid_search(prices, ron_w, roff_w):
    best_sharpe = -1e9
    best_cfg = None
    best_res = None
    best_trades = np.inf

    px_idx = build_portfolio_index(prices, ron_w)

    lengths = list(range(21, 253))
    types = ["sma", "ema"]
    tolerances = np.arange(0, 0.0501, 0.002)

    progress = st.progress(0.0)
    total = len(lengths)*len(types)*len(tolerances)
    k = 0

    ma_cache = {t: compute_ma_matrix(px_idx, lengths, t) for t in types}

    for ma_type in types:
        for L in lengths:
            ma = ma_cache[ma_type][L]
            for tol in tolerances:
                sig = generate_testfol_signal_vectorized(px_idx, ma, tol)
                res = backtest(prices, sig, ron_w, roff_w)

                switches = sig.astype(int).diff().abs().sum()
                trades_year = switches / (len(sig)/252)
                sh = res["performance"]["Sharpe"]

                k += 1
                if k % 200 == 0:
                    progress.progress(k/total)

                if sh > best_sharpe or (sh == best_sharpe and trades_year < best_trades):
                    best_sharpe = sh
                    best_trades = trades_year
                    best_cfg = (L, ma_type, tol)
                    best_res = res

    return best_cfg, best_res


# ============================================
# HYBRID 3SIG MA-GATED STRATEGY
# ============================================

def run_hybrid_sig_strategy(prices, signal, ron_w, roff_w):
    simple = prices.pct_change().fillna(0)

    ron = pd.Series(0.0, index=simple.index)
    for a, w in ron_w.items():
        if a in simple.columns:
            ron += simple[a] * w

    roff = pd.Series(0.0, index=simple.index)
    for a, w in roff_w.items():
        if a in simple.columns:
            roff += simple[a] * w

    # Starting buckets
    stock_bucket = 6000.0
    safe_bucket = 4000.0  # now risk-off portfolio
    day_count = 0
    target = 10000.0

    # Compute target CAGR of buy & hold risk-on portfolio
    bh_eq = (1 + ron).cumprod()
    bh_cagr = (bh_eq.iloc[-1] / bh_eq.iloc[0]) ** (252/len(bh_eq)) - 1

    hybrid_vals = []
    hybrid_r = []
    stock_hist = []
    safe_hist = []
    exposure_hist = []
    buy_count = 0
    sell_count = 0

    sig_vals = signal.values
    ron_vals = ron.values
    roff_vals = roff.values
    idx = ron.index

    for i in range(len(idx)):
        is_on = sig_vals[i]
        r_on = ron_vals[i]
        r_off = roff_vals[i]

        total_val = stock_bucket + safe_bucket

        if is_on:
            exposure = stock_bucket / total_val
            r = exposure*r_on + (1-exposure)*r_off

            stock_bucket *= (1 + r_on)
            safe_bucket *= (1 + r_off)

            day_count += 1
            if day_count == QUARTER_DAYS:
                target *= (1 + bh_cagr/4)

                if stock_bucket > target:
                    diff = stock_bucket - target
                    stock_bucket = target
                    safe_bucket += diff
                    sell_count += 1
                else:
                    diff = target - stock_bucket
                    buy_amt = min(diff, safe_bucket)
                    stock_bucket += buy_amt
                    safe_bucket -= buy_amt
                    if buy_amt > 0:
                        buy_count += 1

                day_count = 0

        else:
            exposure = 0.0
            r = r_off  # 100% risk-off portfolio returns
            # buckets freeze

        hybrid_vals.append(total_val * (1 + r))
        hybrid_r.append(r)
        stock_hist.append(stock_bucket)
        safe_hist.append(safe_bucket)
        exposure_hist.append(exposure)

    hybrid_eq = pd.Series(hybrid_vals, index=idx)
    hybrid_r = pd.Series(hybrid_r, index=idx)
    avg_safe = np.mean(np.array(safe_hist) / (np.array(stock_hist) + np.array(safe_hist)))

    return {
        "equity_curve": hybrid_eq,
        "returns": hybrid_r,
        "stock_bucket": pd.Series(stock_hist, index=idx),
        "safe_bucket": pd.Series(safe_hist, index=idx),
        "exposure": pd.Series(exposure_hist, index=idx),
        "avg_safe_pct": avg_safe,
        "buy_count": buy_count,
        "sell_count": sell_count,
    }


# ============================================
# STREAMLIT APP
# ============================================

def main():

    st.set_page_config(page_title="Portfolio MA Strategy", layout="wide")
    st.title("Portfolio MA Strategy")

    st.sidebar.header("Backtest Settings")
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")

    st.sidebar.header("Risk-ON Portfolio")
    ron_tickers = st.sidebar.text_input("Tickers", ",".join(RISK_ON_WEIGHTS.keys()))
    ron_w_str = st.sidebar.text_input("Weights", ",".join(str(w) for w in RISK_ON_WEIGHTS.values()))

    st.sidebar.header("Risk-OFF Portfolio")
    roff_tickers = st.sidebar.text_input("Tickers", ",".join(RISK_OFF_WEIGHTS.keys()))
    roff_w_str = st.sidebar.text_input("Weights", ",".join(str(w) for w in RISK_OFF_WEIGHTS.values()))

    if not st.sidebar.button("Run Backtest & Optimize"):
        st.stop()

    ron_tickers = [t.strip().upper() for t in ron_tickers.split(",")]
    ron_w = dict(zip(ron_tickers, [float(x) for x in ron_w_str.split(",")]))

    roff_tickers = [t.strip().upper() for t in roff_tickers.split(",")]
    roff_w = dict(zip(roff_tickers, [float(x) for x in roff_w_str.split(",")]))

    all_tickers = sorted(set(ron_tickers + roff_tickers))
    prices = load_price_data(all_tickers, start, end if end.strip() else None).dropna(how="any")

    # RUN MA GRID SEARCH
    best_cfg, best_result = run_grid_search(prices, ron_w, roff_w)
    best_len, best_type, best_tol = best_cfg

    sig = best_result["signal"]
    perf = best_result["performance"]

    # ALWAYS-ON RISK-ON
    simple = prices.pct_change().fillna(0)
    ron_simple = pd.Series(0.0, index=simple.index)
    for a,w in ron_w.items():
        if a in simple.columns:
            ron_simple += simple[a]*w
    ron_eq = (1+ron_simple).cumprod()
    ron_perf = compute_performance(ron_simple, ron_eq)

    # SHARPE OPTIMAL
    ron_px = prices[[t for t in ron_tickers if t in prices.columns]]
    ron_rets = ron_px.pct_change().dropna()
    mu, cov = ron_rets.mean().values, ron_rets.cov().values
    cov += np.eye(cov.shape[0])*1e-10

    def neg_sharpe(w):
        r = np.dot(mu,w)
        v = np.sqrt(w.T @ cov @ w)
        return -(r/v if v>0 else 1e9)

    n = len(mu)
    res = minimize(neg_sharpe, np.ones(n)/n, bounds=[(0,1)]*n,
                   constraints={"type":"eq","fun":lambda w:w.sum()-1})
    w_opt = res.x
    sharp_r = (ron_rets*w_opt).sum(axis=1)
    sharp_eq = (1+sharp_r).cumprod()
    sharp_perf = compute_performance(sharp_r, sharp_eq)

    # HYBRID 3SIG
    hybrid = run_hybrid_sig_strategy(prices, sig, ron_w, roff_w)
    hybrid_eq = hybrid["equity_curve"]
    hybrid_r = hybrid["returns"]
    hybrid_perf = compute_performance(hybrid_r, hybrid_eq)

    # TABLES + PLOTS
    st.subheader("Hybrid MA-Gated 3Sig")
    st.write(f"Hybrid Final Value: ${hybrid_eq.iloc[-1]:,.2f}")
    st.write(f"Hybrid CAGR: {hybrid_perf['CAGR']*100:.2f}%")
    st.write(f"Hybrid Sharpe: {hybrid_perf['Sharpe']:.3f}")
    st.write(f"Average Safe Sleeve: {hybrid['avg_safe_pct']*100:.2f}%")
    st.write(f"Buys: {hybrid['buy_count']} | Sells: {hybrid['sell_count']}")

    # Comparison plot
    st.subheader("Comparison of Strategies")
    fig,ax = plt.subplots(figsize=(12,6))
    ax.plot(hybrid_eq, label="Hybrid 3Sig", linewidth=2)
    ax.plot(best_result["equity_curve"], label="MA Strategy")
    ax.plot(sharp_eq, label="Sharpe Optimal")
    ax.plot(build_portfolio_index(prices, ron_w), label="Risk-On Buy & Hold")
    ax.legend(); ax.grid(alpha=.3)
    st.pyplot(fig)

    # Buckets
    st.subheader("3Sig Buckets Through Time")
    bucket_df = pd.DataFrame({
        "Stock Bucket": hybrid["stock_bucket"],
        "Safe Bucket": hybrid["safe_bucket"],
        "Exposure": hybrid["exposure"]
    })
    st.line_chart(bucket_df)


if __name__ == "__main__":
    main()