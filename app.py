# ============================================
# IMPORTS
# ============================================

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
QUARTER_DAYS = 63  # True 3Sig calendar approximation

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

# ============================================
# BACKTEST ENGINE — SIMPLE RETURNS (unchanged)
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

def compute_performance(simple_returns, eq_curve, rf=0.0):
    cagr = (eq_curve.iloc[-1] / eq_curve.iloc[0]) ** (252 / len(eq_curve)) - 1
    vol = simple_returns.std() * np.sqrt(252)
    sharpe = (simple_returns.mean() * 252 - rf) / vol if vol > 0 else np.nan
    dd = eq_curve / eq_curve.cummax() - 1
    max_dd = dd.min()
    total_ret = eq_curve.iloc[-1] / eq_curve.iloc[0] - 1
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": total_ret,
        "DD_Series": dd
    }

def backtest(prices, signal, risk_on_weights, risk_off_weights):
    simple = prices.pct_change().fillna(0)

    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)
    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)

    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1
    flip_costs = np.where(flip_mask, -FLIP_COST, 0.0)

    strat_adj = strategy_simple + flip_costs
    eq = (1 + strat_adj).cumprod()

    return {
        "returns": strat_adj,
        "equity_curve": eq,
        "signal": signal,
        "weights": weights,
        "performance": compute_performance(strat_adj, eq),
        "flip_mask": flip_mask,
    }

# ============================================
# PLACEHOLDER FOR HYBRID 3SIG ENGINE
# ============================================
# (Block 2 will insert the full hybrid code here)
# ============================================
# HYBRID MA-GATED 3SIG ENGINE
# ============================================

def run_hybrid_sig_strategy(prices, signal, risk_on_weights, risk_off_weights):
    """
    prices: DataFrame of all tickers
    signal: MA-gated risk-on boolean Series
    risk_on_weights: dict of user-selected risk-on weights
    risk_off_weights: dict of user-selected risk-off weights

    Returns dictionary with:
        - hybrid equity curve
        - hybrid daily returns
        - buckets through time
        - buy_count, sell_count
        - avg_safe_bucket_pct
        - exposures
    """

    # Daily returns for both portfolios
    simple_rets = prices.pct_change().fillna(0)

    # Risk-ON portfolio returns
    ron = pd.Series(0.0, index=prices.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            ron += simple_rets[a] * w

    # Risk-OFF portfolio returns
    roff = pd.Series(0.0, index=prices.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            roff += simple_rets[a] * w

    # --- Initialize 3Sig buckets ---
    stock_bucket = 6000.0     # 60% stock sleeve
    safe_bucket  = 4000.0     # 40% "cash" sleeve → NOW RISK-OFF PORTFOLIO
    target       = 10000.0    # starting target
    day_count    = 0

    hybrid_vals = []
    hybrid_daily = []
    stock_hist = []
    safe_hist = []
    exposure_hist = []
    sell_count = 0
    buy_count = 0

    # Pre-calc returns into arrays for speed
    ron_vals = ron.values
    roff_vals = roff.values
    sig_vals = signal.values
    idx = ron.index

    for i in range(len(idx)):
        r_on = ron_vals[i]
        r_off = roff_vals[i]
        is_on = sig_vals[i]

        total_val = stock_bucket + safe_bucket

        # --- RISK-ON MODE ---
        if is_on:

            exposure_stock = stock_bucket / total_val
            hybrid_r = exposure_stock * r_on + (1 - exposure_stock) * r_off

            # update buckets normally
            stock_bucket *= (1 + r_on)
            safe_bucket  *= (1 + r_off)

            # quarterly update
            day_count += 1
            if day_count == QUARTER_DAYS:
                target *= (1 + 0.25 * (ron.mean() * 252))   # TARGET BASED ON BH CAGR

                if stock_bucket > target:
                    surplus = stock_bucket - target
                    stock_bucket = target
                    safe_bucket += surplus
                    sell_count += 1
                else:
                    deficit = target - stock_bucket
                    buy_amt = min(deficit, safe_bucket)
                    stock_bucket += buy_amt
                    safe_bucket -= buy_amt
                    if buy_amt > 0:
                        buy_count += 1

                day_count = 0

        # --- RISK-OFF MODE ---
        else:
            exposure_stock = 0.0
            hybrid_r = r_off

            # Freeze engine state → no change to buckets
            pass

        hybrid_vals.append(total_val * (1 + hybrid_r))
        hybrid_daily.append(hybrid_r)

        stock_hist.append(stock_bucket)
        safe_hist.append(safe_bucket)
        exposure_hist.append(exposure_stock)

        # Update total after applying return
        if is_on:
            total_val = stock_bucket + safe_bucket
        else:
            total_val = hybrid_vals[-1]

    hybrid_vals = np.array(hybrid_vals)
    hybrid_daily = np.array(hybrid_daily)
    stock_hist = np.array(stock_hist)
    safe_hist = np.array(safe_hist)
    exposure_hist = np.array(exposure_hist)

    avg_safe_pct = np.mean(safe_hist / (stock_hist + safe_hist))

    return {
        "equity_curve": pd.Series(hybrid_vals, index=idx),
        "returns": pd.Series(hybrid_daily, index=idx),
        "stock_bucket": pd.Series(stock_hist, index=idx),
        "safe_bucket": pd.Series(safe_hist, index=idx),
        "exposure": pd.Series(exposure_hist, index=idx),
        "avg_safe_pct": avg_safe_pct,
        "buy_count": buy_count,
        "sell_count": sell_count,
    }
    
# ============================================
# HYBRID MA-GATED 3SIG STRATEGY EXECUTION
# ============================================

st.subheader("Hybrid MA-Gated 3Sig Strategy")

# Run hybrid strategy
hybrid = run_hybrid_sig_strategy(
    prices=prices,
    signal=sig,
    risk_on_weights=risk_on_weights,
    risk_off_weights=risk_off_weights
)

hybrid_eq = hybrid["equity_curve"]
hybrid_returns = hybrid["returns"]

# Compute hybrid performance
hybrid_perf = compute_performance(hybrid_returns, hybrid_eq)

# Display hybrid stats in Streamlit
st.write(f"**Hybrid Final Value:** ${hybrid_eq.iloc[-1]:,.2f}")
st.write(f"**Hybrid CAGR:** {hybrid_perf['CAGR']*100:.2f}%")
st.write(f"**Hybrid Sharpe:** {hybrid_perf['Sharpe']:.3f}")
st.write(f"**Hybrid Max Drawdown:** {hybrid_perf['MaxDrawdown']*100:.2f}%")
st.write(f"**Hybrid Avg Safe Sleeve:** {hybrid['avg_safe_pct']*100:.2f}%")
st.write(f"**Hybrid Buy Count:** {hybrid['buy_count']}")
st.write(f"**Hybrid Sell Count:** {hybrid['sell_count']}")

# ============================================
# ADD HYBRID STRATEGY TO METRIC TABLE
# ============================================

hybrid_stats_row = {
    "CAGR": hybrid_perf["CAGR"],
    "Volatility": hybrid_perf["Volatility"],
    "Sharpe": hybrid_perf["Sharpe"],
    "MaxDD": hybrid_perf["MaxDrawdown"],
    "Total": hybrid_perf["TotalReturn"],
    "MAR": hybrid_perf["CAGR"] / abs(hybrid_perf["MaxDrawdown"]) if hybrid_perf["MaxDrawdown"] != 0 else np.nan,
    "TID": (hybrid_perf["DD_Series"] < 0).mean(),
    "PainGain": hybrid_perf["CAGR"] / np.sqrt((hybrid_perf["DD_Series"]**2).mean())
                if (hybrid_perf["DD_Series"]**2).mean() != 0 else np.nan,
    "Skew": hybrid_returns.skew(),
    "Kurtosis": hybrid_returns.kurt(),
    "Trades/year": hybrid["sell_count"] + hybrid["buy_count"],  # approximate
    "P/L per flip": 0.0,   # flips are internal to 3Sig, not priced like TestFol
}

# Insert Hybrid strategy into the main stats table
st.subheader("Comparative Statistics Including Hybrid 3Sig Strategy")

combined_rows = []
for label, key in rows:
    sv = strat_stats[key]
    shv = sharp_stats[key]
    rv = risk_stats[key]
    hv = hybrid_stats_row[key]

    # Format appropriately
    if key in ["CAGR", "Volatility", "MaxDD", "Total", "TID"]:
        sv_fmt = fmt_pct(sv)
        sh_fmt = fmt_pct(shv)
        rv_fmt = fmt_pct(rv)
        hv_fmt = fmt_pct(hv)
    elif key in ["Sharpe", "MAR", "PainGain", "Skew", "Kurtosis"]:
        sv_fmt = fmt_dec(sv)
        sh_fmt = fmt_dec(shv)
        rv_fmt = fmt_dec(rv)
        hv_fmt = fmt_dec(hv)
    else:
        sv_fmt = fmt_num(sv)
        sh_fmt = fmt_num(shv)
        rv_fmt = fmt_num(rv)
        hv_fmt = fmt_num(hv)

    combined_rows.append([label, sv_fmt, sh_fmt, rv_fmt, hv_fmt])

combined_table = pd.DataFrame(
    combined_rows,
    columns=["Metric", "Strategy", "Sharpe-Optimal", "Risk-On", "Hybrid 3Sig"]
)

st.dataframe(combined_table, use_container_width=True)

# ============================================
# PLOT HYBRID VS OTHERS
# ============================================

st.subheader("Hybrid MA-Gated 3Sig Strategy vs Other Strategies")

fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(hybrid_eq, label="Hybrid MA-3Sig", linewidth=2, color="blue")
ax2.plot(best_result["equity_curve"], label="MA Strategy", alpha=0.7)
ax2.plot(sharp_eq, label="Sharpe Optimal", alpha=0.7)
ax2.plot(portfolio_index, label="Risk-On Buy & Hold", alpha=0.7)
ax2.grid(alpha=0.3)
ax2.legend()
st.pyplot(fig2)

# ============================================
# SHOW BUCKET BEHAVIOR (STOCK + SAFE)
# ============================================

st.subheader("3Sig Engine Buckets Through Time")

bucket_df = pd.DataFrame({
    "Stock Bucket": hybrid["stock_bucket"],
    "Safe Bucket": hybrid["safe_bucket"],
    "Exposure %": hybrid["exposure"]
})

st.line_chart(bucket_df)
# ============================================
# REGIME AGING STATS (Unchanged)
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
# EXTERNAL VALIDATION LINK (unchanged)
# ============================================

link_url = "https://testfol.io/optimizer?s=9y4FBdfW2oO"
link_text = "View the Sharpe Optimal recommended portfolio"

st.subheader("External Sharpe Optimal Validation Link")
st.markdown(
    f"**Quick Access:** View extended-history Sharpe-optimal weights here: "
    f"[{link_text}]({link_url})"
)


# ============================================
# DISPLAY SHARPE-OPTIMAL WEIGHTS
# ============================================

st.subheader("Sharpe-Optimal Weights (Risk-ON Universe)")
st.write(sharp_weights_display)


# ============================================
# OPTIMAL SIGNAL PARAMETERS
# ============================================

st.subheader("Optimal MA Signal Parameters")
st.write(f"**MA Type:** {best_type.upper()}")
st.write(f"**MA Length:** {best_len} days")
st.write(f"**Tolerance:** {best_tol:.2%}")


# ============================================
# SIGNAL DISTANCE PANEL
# ============================================

st.subheader("Next Signal Information")

ma_opt_dict = compute_ma_matrix(portfolio_index, [best_len], best_type)
ma_opt_series = ma_opt_dict[best_len]

latest_date = ma_opt_series.dropna().index[-1]
P = float(portfolio_index.loc[latest_date])
MA = float(ma_opt_series.loc[latest_date])
tol = best_tol

upper = MA * (1 + tol)
lower = MA * (1 - tol)

latest_signal = sig.iloc[-1]

if latest_signal:
    pct_move = (P - lower) / P
    direction = "RISK-ON → RISK-OFF"
    msg = f"Drop Required: {pct_move:.2%}"
else:
    pct_move = (upper - P) / P
    direction = "RISK-OFF → RISK-ON"
    msg = f"Gain Required: {pct_move:.2%}"

st.write(f"**Portfolio Index:** {P:,.2f}")
st.write(f"**MA({best_len}) Value:** {MA:,.2f}")
st.write(f"**Lower Band:** {lower:,.2f} | **Upper Band:** {upper:,.2f}")
st.write(f"**{msg}**")


# ============================================
# FINAL PLOT — ORIGINAL STRATEGY VS RISK-ON
# ============================================

st.subheader("Original MA Strategy vs Risk-On Portfolio")

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(best_result["equity_curve"], label="MA Strategy", linewidth=2)
ax3.plot(portfolio_index, label="Risk-On Buy & Hold", alpha=0.7)
ax3.plot(ma_opt_series, label=f"{best_type.upper()}({best_len}) MA", alpha=0.6)
ax3.legend()
ax3.grid(alpha=0.3)
st.pyplot(fig3)


# ============================================
# LAUNCH APP
# ============================================

if __name__ == "__main__":
    main()