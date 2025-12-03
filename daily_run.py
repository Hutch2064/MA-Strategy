import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib

# ============================================
# CONFIG — MATCH STREAMLIT EXACTLY
# ============================================

DEFAULT_START_DATE = "1999-01-01"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "UGL": 0.22,
    "QLD": 0.39,
    "BTC-USD": 0.39,
}

RISK_OFF_WEIGHTS = {
    "SHY": 1.0,
}

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
# BACKTEST ENGINE — SIMPLE RETURNS + FLIP COST
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
    """
    Simple-return math, identical to Streamlit main app:
    - CAGR from equity curve
    - Vol, Sharpe from simple daily returns
    """
    if len(eq_curve) < 2:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "MaxDrawdown": np.nan,
            "TotalReturn": np.nan,
            "DD_Series": pd.Series(index=eq_curve.index, dtype=float),
        }

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
        "DD_Series": dd,
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
        "performance": compute_performance(strat_adj, eq, rf=RISK_FREE_RATE),
        "flip_mask": flip_mask,
    }


# ============================================
# GRID SEARCH — IDENTICAL LOGIC TO STREAMLIT
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
# REGIME AGING STATS — MATCH STREAMLIT
# ============================================

def compute_regime_stats(sig):
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

    # Close final segment
    segments.append((current_regime, start_date, sig_series.index[-1]))

    regime_rows = []
    for r, s, e in segments:
        length_days = (e - s).days
        label = "RISK-ON" if r == 1 else "RISK-OFF"
        regime_rows.append([label, s.date(), e.date(), length_days])

    regime_df = pd.DataFrame(regime_rows, columns=["Regime", "Start", "End", "Duration (days)"])

    avg_on = regime_df[regime_df["Regime"] == "RISK-ON"]["Duration (days)"].mean()
    avg_off = regime_df[regime_df["Regime"] == "RISK-OFF"]["Duration (days)"].mean()

    return regime_df, avg_on, avg_off


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
    sharp_stats,
    risk_stats,
    sharp_weights_display,
    direction,
    pct_to_flip,
    P,
    MA,
    lower,
    upper,
    regime_df,
    avg_on,
    avg_off,
):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    SEND_TO = os.getenv("SEND_TO")

    best_len, best_type, best_tol = best_cfg

    # 3-column stats table (Strategy vs Sharpe-Optimal vs Risk-On) — match Streamlit
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

    table_rows_html = ""
    for label, key in rows:
        sval = strat_stats.get(key, np.nan)
        shv = sharp_stats.get(key, np.nan)
        rval = risk_stats.get(key, np.nan)

        if key in ["CAGR", "Volatility", "MaxDD", "Total", "TID"]:
            sval_fmt = fmt_pct(sval)
            shv_fmt = fmt_pct(shv)
            rval_fmt = fmt_pct(rval)
        elif key in ["Sharpe", "MAR", "PainGain", "Skew", "Kurtosis"]:
            sval_fmt = fmt_dec(sval)
            shv_fmt = fmt_dec(shv)
            rval_fmt = fmt_dec(rval)
        else:
            sval_fmt = fmt_num(sval)
            shv_fmt = fmt_num(shv)
            rval_fmt = fmt_num(rval)

        table_rows_html += f"""
        <tr>
          <td>{label}</td>
          <td style="text-align:right;">{sval_fmt}</td>
          <td style="text-align:right;">{shv_fmt}</td>
          <td style="text-align:right;">{rval_fmt}</td>
        </tr>
        """

    direction_str = "Drop Required" if "→ RISK-OFF" in direction else "Gain Required"

    # Regime aging table
    regime_rows_html = ""
    for _, row in regime_df.iterrows():
        regime_rows_html += f"""
        <tr>
          <td>{row['Regime']}</td>
          <td>{row['Start']}</td>
          <td>{row['End']}</td>
          <td style="text-align:right;">{row['Duration (days)']}</td>
        </tr>
        """

    # Sharpe-optimal weights list
    sharp_weights_items = ""
    for t, w in sharp_weights_display.items():
        sharp_weights_items += f"<li><b>{t}:</b> {w:.2%}</li>"

    # External validation link (Testfol) — same as main app
    link_url = "https://testfol.io/optimizer?s=9y4FBdfW2oO"
    link_text = "View the Sharpe Optimal recommended portfolio"

    html = f"""
    <html>
      <body>
        <h2>Portfolio MA Strategy — Daily Signal</h2>

        <p><b>Current Regime:</b> {regime}</p>

        <h3>External Sharpe-Optimal Validation</h3>
        <p>
          <b>Quick Access:</b> You can view the portfolio’s Sharpe-Optimal weights using extended history here:
          <a href="{link_url}">{link_text}</a>
        </p>

        <h3>Optimal Signal Parameters</h3>
        <ul>
          <li><b>Moving Average Type:</b> {best_type.upper()}</li>
          <li><b>Optimal MA Length:</b> {best_len} days</li>
          <li><b>Optimal Tolerance:</b> {best_tol:.2%}</li>
        </ul>

        <h3>Sharpe-Optimal Weights (Risk-ON Universe)</h3>
        <ul>
          {sharp_weights_items}
        </ul>

        <h3>Next Signal Information</h3>
        <ul>
          <li><b>Latest Signal Change Direction:</b> {direction}</li>
          <li><b>Portfolio Index (P):</b> {P:,.2f}</li>
          <li><b>MA({best_len}) Value:</b> {MA:,.2f}</li>
          <li><b>Tolerance Bands:</b> Lower = {lower:,.2f}, Upper = {upper:,.2f}</li>
          <li><b>{direction_str}:</b> {pct_to_flip:.2%}</li>
        </ul>

        <h3>Strategy vs. Sharpe-Optimal vs. Risk-ON Statistics</h3>
        <table border="1" cellspacing="0" cellpadding="4">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Strategy</th>
              <th>Sharpe-Optimal</th>
              <th>Risk-On</th>
            </tr>
          </thead>
          <tbody>
            {table_rows_html}
          </tbody>
        </table>

        <h3>Regime Statistics</h3>
        <ul>
          <li><b>Average RISK-ON Duration:</b> {avg_on:.1f} days</li>
          <li><b>Average RISK-OFF Duration:</b> {avg_off:.1f} days</li>
        </ul>

        <table border="1" cellspacing="0" cellpadding="4">
          <thead>
            <tr>
              <th>Regime</th>
              <th>Start</th>
              <th>End</th>
              <th>Duration (days)</th>
            </tr>
          </thead>
          <tbody>
            {regime_rows_html}
          </tbody>
        </table>

        <p>The attached chart shows the optimized strategy equity curve
        (colored by current regime), the Sharpe-Optimal portfolio, the risk-on
        portfolio index, and the optimal moving average.</p>
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

    # Always-ON Risk-ON performance — SIMPLE RETURN MATH (identical to Streamlit)
    simple_rets = prices.pct_change().fillna(0)

    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in RISK_ON_WEIGHTS.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_on_eq = (1 + risk_on_simple).cumprod()
    risk_on_perf = compute_performance(risk_on_simple, risk_on_eq, rf=RISK_FREE_RATE)

    # Sharpe-Optimal portfolio (Simple Return Math) — match Streamlit
    risk_on_px = prices[[t for t in RISK_ON_WEIGHTS.keys() if t in prices.columns]].copy()
    risk_on_px = risk_on_px.dropna()
    risk_on_rets = risk_on_px.pct_change().dropna()

    mu_vec = risk_on_rets.mean().values
    cov_mat = risk_on_rets.cov().values

    # tiny ridge to ensure positive definite
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
    sharp_perf = compute_performance(sharp_returns, sharp_eq, rf=RISK_FREE_RATE)

    sharp_stats = {
        "CAGR": sharp_perf["CAGR"],
        "Volatility": sharp_perf["Volatility"],
        "Sharpe": sharp_perf["Sharpe"],
        "MaxDD": sharp_perf["MaxDrawdown"],
        "Total": sharp_perf["TotalReturn"],
        "MAR": sharp_perf["CAGR"] / abs(sharp_perf["MaxDrawdown"])
               if sharp_perf["MaxDrawdown"] != 0 else np.nan,
        "TID": (sharp_perf["DD_Series"] < 0).mean(),
        "PainGain": sharp_perf["CAGR"] / np.sqrt((sharp_perf["DD_Series"] ** 2).mean())
                    if (sharp_perf["DD_Series"] ** 2).mean() != 0 else np.nan,
        "Skew": sharp_returns.skew(),
        "Kurtosis": sharp_returns.kurt(),
        "P/L per flip": 0.0,
        "Trades/year": 0.0,
    }

    sharp_weights_display = {t: round(w, 4) for t, w in zip(risk_on_px.columns, w_opt)}

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
        risk_on_simple,
        risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_simple), dtype=bool),
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

    # Regime aging stats (same logic as Streamlit)
    regime_df, avg_on, avg_off = compute_regime_stats(sig)

    # Final plot — match Streamlit visual
    regime_color = "green" if latest_signal else "red"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(eq, label=f"Strategy ({regime})", linewidth=2, color=regime_color)
    ax.plot(sharp_eq, label="Sharpe-Optimal Portfolio", linewidth=2, color="magenta")
    ax.plot(portfolio_index, label="Portfolio Index (Risk-On Basket)", alpha=0.65)
    ax.plot(ma_opt_series, label=f"Optimal {best_type.upper()}({best_len}) MA", linewidth=2)

    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.close()

    # Send email with all stats + chart + regime aging
    send_email(
        regime,
        best_cfg,
        strat_stats,
        sharp_stats,
        risk_stats,
        sharp_weights_display,
        direction,
        pct_to_flip,
        P,
        MA,
        lower,
        upper,
        regime_df,
        avg_on,
        avg_off,
    )