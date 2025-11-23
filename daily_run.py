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
# CONFIG (same as app.py)
# =========================

DEFAULT_START_DATE = "2011-11-24"
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

DEFAULT_N_RANDOM_SEARCH_ITER = 300
RISK_FREE_RATE = 0.0


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
# PLOTTING HELPERS
# =========================

def plot_equity_curve(equity_curve, filename="equity_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve.index, equity_curve.values, label="Optimized Strategy", linewidth=2)
    plt.title("Equity Curve – Optimized Strategy")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (normalized)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_risk_on_history(risk_on_signal, filename="risk_on_history.png"):
    plt.figure(figsize=(10, 2.5))
    # Convert bool to 0/1
    y = risk_on_signal.astype(int)
    plt.step(risk_on_signal.index, y.values, where="post", linewidth=2)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ["Risk-Off", "Risk-On"])
    plt.title("Risk-On / Risk-Off History (Optimized)")
    plt.xlabel("Date")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =========================
# EMAIL SENDER
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

    # Build HTML body
    ma_rows = ""
    for i, (L, t, tol) in enumerate(
        zip(params.ma_lengths, params.ma_types, params.ma_tolerances), start=1
    ):
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
        <h2>Daily BTC Optimized MA Signal</h2>
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
    msg["Subject"] = f"BTC MA Model – {regime} (Daily Optimized)"
    msg["From"] = EMAIL_USER
    msg["To"] = SEND_TO

    msg_alt = MIMEMultipart("alternative")
    msg.attach(msg_alt)
    msg_alt.attach(MIMEText(html, "html"))

    # Attach plots
    attach_file(msg, "equity_curve.png", mime_subtype="octet-stream")
    attach_file(msg, "risk_on_history.png", mime_subtype="octet-stream")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, SEND_TO, msg.as_string())


# =========================
# MAIN DAILY JOB
# =========================

if __name__ == "__main__":
    # 1) Download data
    prices = yf.download(
        TICKERS,
        start=DEFAULT_START_DATE,
        end=DEFAULT_END_DATE,
        progress=False
    )

    if "Adj Close" in prices.columns:
        prices = prices["Adj Close"].copy()
    else:
        prices = prices["Close"].copy()

    prices = prices.dropna(how="any")

    # 2) Run optimizer (same as app)
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

    # 3) Generate plots
    plot_equity_curve(equity_curve, "equity_curve.png")
    plot_risk_on_history(risk_on_signal, "risk_on_history.png")

    # 4) Today's regime
    is_risk_on_today = bool(risk_on_signal.iloc[-1])
    regime_text = "RISK-ON (33/33/33 3XGLD / TQQQ / BTC)" if is_risk_on_today else "RISK-OFF (100% UUP)"

    # 5) Send email with all requested info
    send_email(regime_text, best_params, perf)


