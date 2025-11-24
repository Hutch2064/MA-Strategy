import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # headless backend for GitHub Actions
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dataclasses import dataclass

# =========================
# CONFIG (identical)
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

def load_price_data_raw(tickers, start_date, end_date=None):
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
# RANDOM SEARCH OPTIMIZER (patched to Streamlit-match)
# =========================

def random_param_sample(rng):
    min_length = 21
    max_length = 252

    n_ma = rng.integers(1, 5)  # 1–4 MAs

    ma_lengths = list(rng.integers(min_length, max_length + 1, size=n_ma))
    ma_types = [rng.choice(["sma", "ema"]) for _ in range(n_ma)]
    ma_tolerances = list(rng.uniform(0.0, 0.05, size=n_ma))

    min_ma_above = rng.integers(1, n_ma + 1)
    confirm_days = np.random.randint(1, 6)

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
    rng = np.random.default_rng(seed)

    best_params = None
    best_result = None
    best_sharpe = -np.inf

    for _ in range(n_iter):
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

    return best_params, best_result

# =========================
# PLOT
# =========================

def plot_equity_curve(equity_curve, prices, filename="equity_curve.png"):
    rets = prices.pct_change().fillna(0.0)
    cols = ["GLD", "TQQQ", "BTC-USD"]
    pure_risk_on_rets = (rets[cols] * np.array([1/3, 1/3, 1/3])).sum(axis=1)
    pure_risk_on_curve = (1 + pure_risk_on_rets).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve.index, equity_curve.values,
             label="Optimized Strategy", linewidth=2)
    plt.plot(pure_risk_on_curve.index, pure_risk_on_curve.values,
             label="Pure Risk-On (33/33/33 BTC/GLD/TQQQ)",
             linestyle="--", linewidth=2)

    plt.title("Equity Curve – Optimized Strategy vs Pure Risk-On")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (normalized)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# =========================
# EMAIL
# =========================

def attach_file(msg, filepath, mime_subtype="octet-stream"):
    if not os.path.exists(filepath):
        return

    with open(filepath, "rb") as f:
        part = MIMEBase("application", mime_subtype)
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{os.path.basename(filepath)}"',
    )
    msg.attach(part)

def send_email(regime, params: StrategyParams, perf: dict):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    SEND_TO = os.getenv("SEND_TO")

    if not EMAIL_USER or not EMAIL_PASS or not SEND_TO:
        raise RuntimeError("Missing EMAIL_USER, EMAIL_PASS, or SEND_TO environment variables.")

    ma_rows = ""
    for i, (L, t, tol) in enumerate(zip(params.ma_lengths, params.ma_types, params.ma_tolerances), start=1):
        ma_rows += f"""
        <tr>
          <td>{i}</td>
          <td>{int(L)}</td>
          <td>{t.upper()}</td>
          <td>{tol:.2%}</td>
        </tr>
        """

    html = f"""
    <html>
      <body>
        <h2>Optimized MA Signal</h2>
        <p><b>Today's Regime:</b> {regime}</p>

        <h3>Performance (Optimized Strategy)</h3>
        <ul>
          <li><b>CAGR:</b> {perf['CAGR']:.2%}</li>
          <li><b>Volatility:</b> {perf['Volatility']:.2%}</li>
          <li><b>Sharpe:</b> {perf['Sharpe']:.3f}</li>
          <li><b>Max Drawdown:</b> {perf['MaxDrawdown']:.2%}</li>
          <li><b>Total Return:</b> {perf['TotalReturn']:.2%}</li>
        </ul>

        <h3>Optimized MA Configuration</h3>
        <table border="1" cellpadding="4" cellspacing="0">
          <tr>
            <th>#</th>
            <th>Length (days)</th>
            <th>Type</th>
            <th>Tolerance</th>
          </tr>
          {ma_rows}
        </table>

        <h3>Global Settings</h3>
        <ul>
          <li><b>min_ma_above:</b> {params.min_ma_above}</li>
          <li><b>confirm_days:</b> {params.confirm_days}</li>
        </ul>

        <p>Equity curve and risk-on/off history are attached as images.</p>
      </body>
    </html>
    """

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f" MA Strategy Report – {regime}"
    msg["From"] = EMAIL_USER
    msg["To"] = SEND_TO

    msg_alt = MIMEMultipart("alternative")
    msg.attach(msg_alt)
    msg_alt.attach(MIMEText(html, "html"))

    attach_file(msg, "equity_curve.png", mime_subtype="octet-stream")
    attach_file(msg, "risk_on_history.png", mime_subtype="octet-stream")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, SEND_TO, msg.as_string())

# =========================
# MAIN DAILY JOB
# =========================

if __name__ == "__main__":
    prices = load_price_data_raw(TICKERS, DEFAULT_START_DATE, DEFAULT_END_DATE)

    core_assets = [t for t in ["BTC-USD", "GLD", "TQQQ", "UUP"] if t in prices.columns]
    if len(core_assets) < 4:
        raise RuntimeError(f"Missing one or more tickers in data. Found: {core_assets}")

    prices = prices[core_assets].dropna(how="any")

    best_params, best_result = run_random_search(
        prices=prices,
        n_iter=DEFAULT_N_RANDOM_SEARCH_ITER,
        risk_on_weights=RISK_ON_WEIGHTS,
        risk_off_weights=RISK_OFF_WEIGHTS,
        rf_rate=RISK_FREE_RATE,
        seed=42,
    )

    perf = best_result["performance"]
    risk_on_signal = best_result["risk_on_signal"]
    equity_curve = best_result["equity_curve"]

    plot_equity_curve(equity_curve, prices, "equity_curve.png")

    is_risk_on_today = bool(risk_on_signal.iloc[-1])
    regime_text = "RISK-ON (33/33/33 3XGLD / TQQQ / BTC)" if is_risk_on_today else "RISK-OFF (100% UUP)"

    send_email(regime_text, best_params, perf)
