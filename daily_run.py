import numpy as np
import pandas as pd
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dataclasses import dataclass

# ===========================================================
# CONFIG (same as Streamlit)
# ===========================================================

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


# ===========================================================
# MOVING AVERAGES & SIGNALS (identical to Streamlit)
# ===========================================================

def compute_ma(series, length, ma_type="ema"):
    ma_type = ma_type.lower()
    if ma_type == "ema":
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == "sma":
        return series.rolling(window=length, min_periods=length).mean()
    else:
        raise ValueError(f"Unknown ma_type: {ma_type}")


def generate_risk_on_signal(price_btc, ma_lengths, ma_types, ma_tolerances,
                            min_ma_above, confirm_days=1):
    conditions = []
    for length, ma_type, tol in zip(ma_lengths, ma_types, ma_tolerances):
        ma = compute_ma(price_btc, length, ma_type)
        cond = price_btc > ma * (1.0 + tol)
        conditions.append(cond)

    cond_df = pd.concat(conditions, axis=1)
    count_above = cond_df.sum(axis=1)
    base_signal = count_above >= min_ma_above

    if confirm_days <= 1:
        return base_signal
    else:
        return (
            base_signal.rolling(window=confirm_days)
                       .apply(lambda x: (x == 1.0).all(), raw=True)
                       .astype(bool)
        )


# ===========================================================
# BACKTEST ENGINE (unchanged)
# ===========================================================

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
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    n_days = len(returns)
    cagr = (1 + total_return) ** (252 / n_days) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - rf_rate) / vol if vol != 0 else np.nan
    dd = equity_curve / equity_curve.cummax() - 1
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
    strat_rets = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    equity_curve = (1 + strat_rets).cumprod()

    return {
        "returns": strat_rets,
        "equity_curve": equity_curve,
        "weights": weights,
        "risk_on_signal": risk_on_signal,
        "performance": compute_performance(strat_rets, equity_curve),
    }


# ===========================================================
# PARAM DATACLASS (same as Streamlit)
# ===========================================================

@dataclass
class StrategyParams:
    ma_lengths: list
    ma_types: list
    ma_tolerances: list
    min_ma_above: int
    confirm_days: int


# ===========================================================
# RANDOM SEARCH OPTIMIZER (FULL VERSION)
# ===========================================================

def random_param_sample():
    n_ma = np.random.randint(1, 7)
    ma_lengths = list(np.random.randint(21, 253, size=n_ma))
    ma_types = [np.random.choice(["sma", "ema"]) for _ in range(n_ma)]
    ma_tolerances = list(np.random.choice([0.0, 0.005, 0.01, 0.02, 0.05], size=n_ma))
    min_ma_above = np.random.randint(1, n_ma + 1)
    confirm_days = int(np.random.choice([1, 3, 5, 10, 20]))

    return StrategyParams(
        ma_lengths, ma_types, ma_tolerances, min_ma_above, confirm_days
    )


def run_random_search(prices, n_iter, risk_on_weights, risk_off_weights, seed=42):
    np.random.seed(seed)
    btc = prices["BTC-USD"]

    best_params = None
    best_result = None
    best_sharpe = -np.inf

    for _ in range(n_iter):
        params = random_param_sample()

        signal = generate_risk_on_signal(
            btc, params.ma_lengths, params.ma_types,
            params.ma_tolerances, params.min_ma_above,
            params.confirm_days
        )

        result = backtest_strategy(prices, signal, risk_on_weights, risk_off_weights)
        sharpe = result["performance"]["Sharpe"]

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
            best_result = result

    return best_params, best_result


# ===========================================================
# EMAIL NOTIFIER
# ===========================================================

def send_email(msg_html):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    SEND_TO = os.getenv("SEND_TO")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Daily BTC Optimized MA Signal"
    msg["From"] = EMAIL_USER
    msg["To"] = SEND_TO

    msg.attach(MIMEText(msg_html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, SEND_TO, msg.as_string())


# ===========================================================
# MAIN DAILY JOB
# ===========================================================

if __name__ == "__main__":
    prices = yf.download(TICKERS, start=DEFAULT_START_DATE)["Adj Close"].dropna()

    best_params, best_result = run_random_search(
        prices, DEFAULT_N_RANDOM_SEARCH_ITER,
        RISK_ON_WEIGHTS, RISK_OFF_WEIGHTS
    )

    # TODAY'S SIGNAL
    today_signal = best_result["risk_on_signal"].iloc[-1]
    regime = "RISK-ON" if today_signal else "RISK-OFF"

    msg = f"""
    <h2>Daily Optimized BTC Trend Signal</h2>
    <p><b>Status Today:</b> {regime}</p>
    <br>
    <h3>Best Parameters Found:</h3>
    <p>{best_params}</p>
    """

    send_email(msg)

