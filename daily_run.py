import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib

# ============================================
# CONFIG — MATCH STREAMLIT EXACTLY
# ============================================

DEFAULT_START_DATE = "2011-11-24"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "GLD": 0.9,
    "TQQQ": 0.3,
    "BTC-USD": 0.4,
}

RISK_OFF_WEIGHTS = {
    "SHY": 1.0,
}

# Slippage + tax drag applied on flip days
FLIP_COST = 0.00875


# ============================================
# DATA LOADING (no cache)
# ============================================

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
# BUILD PORTFOLIO INDEX FOR SIGNAL — MATCH STREAMLIT
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
# MA MATRIX — MATCH STREAMLIT
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
# TESTFOL HYSTERESIS — MATCH STREAMLIT
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
        if not sig[t - 1]:
            sig[t] = px[t] > upper[t]
        else:
            sig[t] = not (px[t] < lower[t])

    return pd.Series(sig, index=ma.index).fillna(False)


# ============================================
# BACKTEST ENGINE — WITH FLIP COST (STREAMLIT)
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
# GRID SEARCH — IDENTICAL TO STREAMLIT
# ============================================

def run_grid_search(prices, risk_on_weights, risk_off_weights):
    best_sharpe = -1e9
    best_cfg = None
    best_result = None
    best_trades = np.inf

    portfolio_index = build_portfolio_index(prices, risk_on_weights)

    lengths = list(range(21, 253))
    types = ["sma", "ema"]
    tolerances = np.arange(0.0, 0.0501, 0.002)

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

                if (sharpe_adj > best_sharpe or
                    (sharpe_adj == best_sharpe and trades_per_year < best_trades)):
                    best_sharpe = sharpe_adj
                    best_trades = trades_per_year
                    best_cfg = (length, ma_type, tol)
                    best_result = result

    return best_cfg, best_result


# ============================================
# ADVANCED METRICS — MATCH STREAMLIT
# ============================================

def time_in_drawdown(dd):
    return (dd < 0).mean()


def pain_to_gain(dd, cagr):
    ulcer = np.sqrt((dd ** 2).mean())
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


# ============================================
# EMAIL HELPERS
# ============================================

def attach_file(msg, filepath):
    with open(filepath, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(filepath)}"')
    msg.attach(part)


def fmt_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "—"


def fmt_dec(x):
    return f"{x:.3f}" if pd.notna(x) else "—"


def fmt_num(x):
    return f"{x:,.2f}" if pd.notna(x) else "—"


def send_email(
    regime,
    best_cfg,
    strat_stats,
    risk_stats,
    direction,
    pct_to_flip,
    P,
    MA,
    lower,
    upper,
):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    SEND_TO = os.getenv("SEND_TO")

    best_len, best_type, best_tol = best_cfg

    # Build HTML for stats table (Strategy vs Risk-On)
    rows = [
        ("CAGR", "CAGR"),
        ("Volatility", "Volatility"),
        ("Sharpe", "Sharpe"),
        ("Max Drawdown", "MaxDD"),
        ("Total Return", "Total"),
        ("MAR Ratio", "MAR"),
        ("Time in Drawdown", "TID"),
        ("Pain-to-Gain", "PainGain"),
        ("Skew", "Skew"),
        ("Kurtosis", "Kurtosis"),
        ("Trades per year", "Trades/year"),
        ("P/L per flip", "P/L per flip"),
    ]

    table_rows_html = ""
    for label, key in rows:
        sval = strat_stats.get(key, np.nan)
        rval = risk_stats.get(key, np.nan)

        if key in ["CAGR", "Volatility", "MaxDD", "Total", "TID"]:
            sval_fmt = fmt_pct(sval)
            rval_fmt = fmt_pct(rval)
        elif key in ["Sharpe", "MAR", "PainGain", "Skew", "Kurtosis"]:
            sval_fmt = fmt_dec(sval)
            rval_fmt = fmt_dec(rval)
        else:
            sval_fmt = fmt_num(sval)
            rval_fmt = fmt_num(rval)

        table_rows_html += f"""
        <tr>
          <td>{label}</td>
          <td style="text-align:right;">{sval_fmt}</td>
          <td style="text-align:right;">{rval_fmt}</td>
        </tr>
        """

    direction_str = "Drop Required" if "→ RISK-OFF" in direction else "Gain Required"

    html = f"""
    <html>
      <body>
        <h2>Portfolio MA Optimized Regime Strategy — Daily Signal</h2>

        <p><b>Current Regime:</b> {regime}</p>

        <h3>Optimal Signal Parameters</h3>
        <ul>
          <li><b>Moving Average Type:</b> {best_type.upper()}</li>
          <li><b>Optimal MA Length:</b> {best_len} days</li>
          <li><b>Optimal Tolerance:</b> {best_tol:.2%}</li>
        </ul>

        <h3>Distance Until Next Signal</h3>
        <ul>
          <li><b>Latest Signal Change Direction:</b> {direction}</li>
          <li><b>Portfolio Index (P):</b> {P:,.2f}</li>
          <li><b>MA({best_len}) Value:</b> {MA:,.2f}</li>
          <li><b>Tolerance Bands:</b> Lower = {lower:,.2f}, Upper = {upper:,.2f}</li>
          <li><b>{direction_str}:</b> {pct_to_flip:.2%}</li>
        </ul>

        <h3>Full Strategy Statistics (Strategy vs Always-On Risk-On)</h3>
        <table border="1" cellspacing="0" cellpadding="4">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Strategy</th>
              <th>Risk-On</th>
            </tr>
          </thead>
          <tbody>
            {table_rows_html}
          </tbody>
        </table>

        <p>The attached chart shows the optimized strategy equity curve
        (colored by current regime), the risk-on portfolio index, and the
        optimal moving average.</p>
      </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["Subject"] = f"Portfolio MA Regime Signal — {regime}"
    msg["From"] = EMAIL_USER
    msg["To"] = SEND_TO
    msg.attach(MIMEText(html, "html"))

    attach_file(msg, "equity_curve.png")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, [SEND_TO], msg.as_string())


# ============================================
# MAIN — HEADLESS DAILY ENGINE
# ============================================

if __name__ == "__main__":
    # Same tickers as Streamlit (risk-on + risk-off universe)
    tickers = sorted(set(list(RISK_ON_WEIGHTS.keys()) + list(RISK_OFF_WEIGHTS.keys())))
    prices = load_price_data(tickers, DEFAULT_START_DATE).dropna(how="any")

    # Optimized strategy (same grid search as Streamlit)
    best_cfg, best_result = run_grid_search(prices, RISK_ON_WEIGHTS, RISK_OFF_WEIGHTS)
    best_len, best_type, best_tol = best_cfg

    sig = best_result["signal"]
    perf = best_result["performance"]
    eq = best_result["equity_curve"]
    flip_mask = best_result["flip_mask"]

    latest_signal = bool(sig.iloc[-1])
    regime = "RISK-ON" if latest_signal else "RISK-OFF"

    sig_arr = sig.astype(int)
    switches = sig_arr.diff().abs().sum()
    trades_per_year = switches / (len(sig_arr) / 252)

    # Always-ON Risk-ON performance — identical math
    log_px = np.log(prices)
    log_rets = log_px.diff().fillna(0)

    risk_on_log = pd.Series(0.0, index=log_rets.index)
    for a, w in RISK_ON_WEIGHTS.items():
        if a in log_rets.columns:
            risk_on_log += log_rets[a] * w

    risk_on_eq = np.exp(risk_on_log.cumsum())
    risk_on_perf = compute_performance(risk_on_log, risk_on_eq)

    # Build stats dicts (same as Streamlit)
    strat_stats = compute_stats(
        perf,
        best_result["returns"],
        perf["DD_Series"],
        flip_mask,
        trades_per_year
    )

    risk_stats = compute_stats(
        risk_on_perf,
        risk_on_log,
        risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_log), dtype=bool),
        0
    )

    # Distance until next signal — same logic
    portfolio_index = build_portfolio_index(prices, RISK_ON_WEIGHTS)
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
    else:
        pct_to_flip = (upper - P) / P
        direction = "RISK-OFF → RISK-ON"

    # Final plot — match Streamlit visual
    regime_color = "green" if latest_signal else "red"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(eq, label=f"Strategy ({regime})", linewidth=2, color=regime_color)
    ax.plot(portfolio_index, label="Portfolio Index (Risk-On Basket)", alpha=0.65)
    ax.plot(ma_opt_series, label=f"Optimal {best_type.upper()}({best_len}) MA", linewidth=2)

    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.close()

    # Send email with all stats + chart
    send_email(
        regime,
        best_cfg,
        strat_stats,
        risk_stats,
        direction,
        pct_to_flip,
        P,
        MA,
        lower,
        upper,
    )