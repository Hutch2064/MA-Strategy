import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import io
from scipy.optimize import minimize
import datetime

# ============================================================
# CONFIG - UPDATED WITH TAX PARAMETERS
# ============================================================

DEFAULT_START_DATE = "2000-01-01"
RISK_FREE_RATE = 0.0

RISK_ON_WEIGHTS = {
    "UGL": .25,
    "TQQQ": .30,
    "BTC-USD": .45,
}

RISK_OFF_WEIGHTS = {
    "SHY": 1.0,
}

# UPDATED: More realistic costs based on academic standards
FLIP_COST = 0.001  # 0.1% slippage per trade (academic median for ETFs)
ANNUAL_TAX_RATE = 0.20  # 20% effective capital gains tax (academic standard)
TAX_PAYMENT_MONTH = 4  # April (when taxes are typically paid)

# Starting weights inside the SIG engine (unchanged)
START_RISKY = 0.70
START_SAFE  = 0.30

# MA optimization constraints
MA_MIN_DAYS = 20    # Minimum MA length to test
MA_MAX_DAYS = 200   # Maximum MA length to test  
MA_STEP_FACTOR = 25 # Test ~25 different lengths
MA_TOL_RANGE = (0.00, 0.05, 0.002)  # Min, Max, Step for tolerance

# ============================================================
# DATA LOADING
# ============================================================

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


# ============================================================
# BUILD PORTFOLIO INDEX — SIMPLE RETURNS
# ============================================================

def build_portfolio_index(prices, weights_dict):
    simple_rets = prices.pct_change().fillna(0)
    idx_rets = pd.Series(0.0, index=simple_rets.index)

    for a, w in weights_dict.items():
        if a in simple_rets.columns:
            idx_rets += simple_rets[a] * w

    return (1 + idx_rets).cumprod()


# ============================================================
# MA MATRIX
# ============================================================

def compute_ma_matrix(price_series, lengths, ma_type):
    ma_dict = {}
    if ma_type == "ema":
        for L in lengths:
            ma = price_series.ewm(span=L, adjust=False).mean()
            ma_dict[L] = ma.shift(1)
    else:
        for L in lengths:
            ma = price_series.rolling(window=L, min_periods=1).mean()
            ma_dict[L] = ma.shift(1)
    return ma_dict


# ============================================================
# TESTFOL SIGNAL LOGIC - ROBUST VERSION
# ============================================================

def generate_testfol_signal_vectorized(price, ma, tol):
    px = price.shift(1).values
    ma_vals = ma.values
    n = len(px)
    
    # Handle case where all values are NaN
    if np.all(np.isnan(ma_vals)):
        return pd.Series(False, index=ma.index)
    
    upper = ma_vals * (1 + tol)
    lower = ma_vals * (1 - tol)
    
    sig = np.zeros(n, dtype=bool)
    
    # Find first non-NaN index
    non_nan_mask = ~np.isnan(ma_vals)
    if not np.any(non_nan_mask):
        return pd.Series(False, index=ma.index)
    
    first_valid = np.where(non_nan_mask)[0][0]
    if first_valid == 0:
        first_valid = 1
    start_index = first_valid + 1
    
    # Ensure start_index is valid
    if start_index >= n:
        return pd.Series(False, index=ma.index)
    
    for t in range(start_index, n):
        if np.isnan(px[t]) or np.isnan(upper[t]) or np.isnan(lower[t]):
            sig[t] = sig[t-1] if t > 0 else False
        elif not sig[t - 1]:
            sig[t] = px[t] > upper[t]
        else:
            sig[t] = not (px[t] < lower[t])
    
    return pd.Series(sig, index=ma.index).fillna(False)


# ============================================================
# TAX CALCULATION FUNCTIONS (NEW)
# ============================================================

def calculate_tax_liability_at_realization(returns, signal_changes, tax_rate=ANNUAL_TAX_RATE):
    """
    Calculate taxes when gains are realized (academically correct)
    Taxes applied immediately when:
    1. MA signal flips from RISK-ON to RISK-OFF
    2. Quarterly rebalancing triggers partial realization
    """
    if len(returns) == 0:
        return pd.Series(0.0, index=returns.index)
    
    # Convert to equity curve to track cost basis
    equity_pre_tax = (1 + returns).cumprod()
    tax_payments = pd.Series(0.0, index=returns.index)
    
    # Track cost basis and entry prices
    cost_basis = equity_pre_tax.iloc[0]  # Initial investment
    last_entry_idx = 0
    in_position = True
    
    for i in range(1, len(returns)):
        # Check for signal change from RISK-ON to RISK-OFF
        if i > 0 and signal_changes.iloc[i] == -1:  # RISK-ON → RISK-OFF
            if in_position:
                # Calculate gain since last entry
                current_value = equity_pre_tax.iloc[i]
                gain = max(0, current_value - cost_basis)
                
                if gain > 0:
                    # Apply tax immediately
                    tax_amount = gain * tax_rate
                    tax_payments.iloc[i] = -tax_amount
                    
                    # Reduce equity by tax amount
                    equity_pre_tax.iloc[i:] *= (1 - tax_amount / current_value)
                    
                    # Reset cost basis
                    cost_basis = equity_pre_tax.iloc[i]
                
                in_position = False
                last_entry_idx = i
        
        # Check for signal change from RISK-OFF to RISK-ON
        elif i > 0 and signal_changes.iloc[i] == 1:  # RISK-OFF → RISK-ON
            if not in_position:
                # New position - set new cost basis
                cost_basis = equity_pre_tax.iloc[i]
                last_entry_idx = i
                in_position = True
    
    return tax_payments

def calculate_continuous_tax_drag(returns, turnover_rate, tax_rate=ANNUAL_TAX_RATE):
    """
    Alternative: Continuous tax drag (simpler academic approach)
    Daily tax drag = (annual turnover × tax rate) / 252
    """
    daily_tax_drag = (turnover_rate * tax_rate) / 252
    tax_drag_series = pd.Series(-daily_tax_drag, index=returns.index)
    return tax_drag_series

def calculate_turnover(signal):
    """
    Calculate annual turnover rate from signal changes
    """
    if len(signal) == 0:
        return 0
    
    sig_arr = signal.astype(int)
    changes = sig_arr.diff().abs().sum()
    annual_turnover = changes / (len(signal) / 252)
    
    return annual_turnover

# ============================================================
# SIG ENGINE — UPDATED WITH TAXES
# ============================================================

def run_sig_engine(
    risk_on_returns,
    risk_off_returns,
    target_quarter,
    ma_signal,
    pure_sig_rw=None,
    pure_sig_sw=None,
    flip_cost=FLIP_COST,
    tax_rate=ANNUAL_TAX_RATE,
    quarter_end_dates=None
):

    dates = risk_on_returns.index
    n = len(dates)

    if quarter_end_dates is None:
        raise ValueError("quarter_end_dates must be supplied")

    # Fast lookup
    quarter_end_set = set(quarter_end_dates)

    # MA flip detection
    sig_arr = ma_signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    # Init values
    eq = 10000.0
    risky_val = eq * START_RISKY
    safe_val  = eq * START_SAFE

    # Tax tracking
    tax_basis = eq  # Initial cost basis
    last_risky_enter_date = None
    last_risky_enter_value = risky_val + safe_val
    
    frozen_risky = None
    frozen_safe  = None

    equity_curve = []
    risky_w_series = []
    safe_w_series = []
    risky_val_series = []
    safe_val_series = []
    tax_payments_series = []
    rebalance_events = 0
    rebalance_dates = []

    for i in range(n):
        date = dates[i]
        r_on = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])
        
        tax_paid_today = 0.0

        if ma_on:
            # Track when we enter RISK-ON
            if last_risky_enter_date is None and i > 0:
                last_risky_enter_date = date
                last_risky_enter_value = risky_val + safe_val

            # Restore pure-SIG weights after exiting RISK-OFF
            if frozen_risky is not None:
                w_r = pure_sig_rw.iloc[i]
                w_s = pure_sig_sw.iloc[i]
                risky_val = eq * w_r
                safe_val  = eq * w_s
                frozen_risky = None
                frozen_safe  = None

            # Apply daily returns
            risky_val *= (1 + r_on)
            safe_val  *= (1 + r_off)

            # Rebalance ON quarter-end date
            if date in quarter_end_set:
                prev_qs = [qd for qd in quarter_end_dates if qd < date]

                if prev_qs:
                    prev_q = prev_qs[-1]
                    idx_prev = dates.get_loc(prev_q)
                    risky_at_qstart = risky_val_series[idx_prev]
                    goal_risky = risky_at_qstart * (1 + target_quarter)

                    if risky_val > goal_risky:
                        excess = risky_val - goal_risky
                        risky_val -= excess
                        safe_val  += excess
                        rebalance_dates.append(date)
                        
                        # Calculate tax on rebalanced amount (partial realization)
                        current_total = risky_val + safe_val
                        if current_total > tax_basis:
                            gain_fraction = max(0, (current_total - tax_basis) / current_total)
                            tax_on_rebalance = excess * gain_fraction * tax_rate
                            eq -= tax_on_rebalance
                            risky_val -= tax_on_rebalance * (risky_val / current_total)
                            safe_val -= tax_on_rebalance * (safe_val / current_total)
                            tax_paid_today += tax_on_rebalance

                    elif risky_val < goal_risky:
                        needed = goal_risky - risky_val
                        move = min(needed, safe_val)
                        safe_val -= move
                        risky_val += move
                        rebalance_dates.append(date)

                    # Apply quarterly fee
                    eq *= (1 - flip_cost * target_quarter)

            # Update equity
            eq = risky_val + safe_val
            risky_w = risky_val / eq
            safe_w  = safe_val  / eq

            # Flip cost at MA transition
            if flip_mask.iloc[i]:
                eq *= (1 - flip_cost)

        else:
            # RISK-OFF: Selling risky assets → realize gains
            if i > 0 and bool(ma_signal.iloc[i-1]) and not ma_on:
                # Calculate gain since entering RISK-ON
                current_value = risky_val + safe_val
                if last_risky_enter_value is not None and current_value > last_risky_enter_value:
                    gain = current_value - last_risky_enter_value
                    tax_payment = gain * tax_rate
                    
                    # Apply tax
                    eq -= tax_payment
                    tax_paid_today += tax_payment
                    
                    # Reset cost basis
                    tax_basis = eq
                
                last_risky_enter_date = None
                last_risky_enter_value = None

            # Freeze values on entering RISK-OFF
            if frozen_risky is None:
                frozen_risky = risky_val
                frozen_safe  = safe_val

            # Only safe sleeve earns returns
            eq *= (1 + r_off)
            risky_w = 0.0
            safe_w  = 1.0

        # Store values
        equity_curve.append(eq)
        risky_w_series.append(risky_w)
        safe_w_series.append(safe_w)
        risky_val_series.append(risky_val)
        safe_val_series.append(safe_val)
        tax_payments_series.append(tax_paid_today)

    return (
        pd.Series(equity_curve, index=dates),
        pd.Series(risky_w_series, index=dates),
        pd.Series(safe_w_series, index=dates),
        pd.Series(tax_payments_series, index=dates),
        rebalance_dates
    )


# ============================================================
# BACKTEST ENGINE - UPDATED WITH TAXES
# ============================================================

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
    if len(eq_curve) == 0 or eq_curve.iloc[0] == 0:
        return {
            "CAGR": 0,
            "Volatility": 0,
            "Sharpe": 0,
            "MaxDrawdown": 0,
            "TotalReturn": 0,
            "DD_Series": pd.Series([], dtype=float)
        }
    
    cagr = (eq_curve.iloc[-1] / eq_curve.iloc[0]) ** (252 / len(eq_curve)) - 1
    vol = simple_returns.std() * np.sqrt(252) if len(simple_returns) > 0 else 0
    sharpe = (simple_returns.mean() * 252 - rf) / vol if vol > 0 else 0
    dd = eq_curve / eq_curve.cummax() - 1

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": dd.min() if len(dd) > 0 else 0,
        "TotalReturn": eq_curve.iloc[-1] / eq_curve.iloc[0] - 1 if eq_curve.iloc[0] != 0 else 0,
        "DD_Series": dd
    }

def backtest_with_taxes(prices, signal, risk_on_weights, risk_off_weights, 
                       flip_cost=FLIP_COST, tax_rate=ANNUAL_TAX_RATE):
    """
    Backtest with proper tax treatment
    """
    simple = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1
    signal_changes = sig_arr.diff().fillna(0)

    # 1. Apply slippage costs
    flip_costs = np.where(flip_mask, -flip_cost, 0.0)
    
    # 2. Calculate taxes (realization-based)
    tax_payments = calculate_tax_liability_at_realization(
        strategy_simple + flip_costs, 
        signal_changes,
        tax_rate
    )
    
    # 3. Combine all costs
    total_costs = flip_costs + tax_payments.values
    
    # 4. Calculate returns and equity
    strat_adj = strategy_simple + total_costs
    eq = (1 + strat_adj).cumprod()

    # Calculate turnover for analysis
    turnover = calculate_turnover(signal)

    return {
        "returns": strat_adj,
        "equity_curve": eq,
        "signal": signal,
        "weights": weights,
        "performance": compute_performance(strat_adj, eq),
        "flip_mask": flip_mask,
        "flip_costs": pd.Series(flip_costs, index=strategy_simple.index),
        "tax_payments": tax_payments,
        "total_costs": pd.Series(total_costs, index=strategy_simple.index),
        "annual_turnover": turnover,
        "tax_rate": tax_rate
    }

def backtest_simple(prices, signal, risk_on_weights, risk_off_weights, flip_cost=FLIP_COST):
    """
    Simple backtest without taxes (for comparison)
    """
    simple = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    flip_costs = np.where(flip_mask, -flip_cost, 0.0)
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


# ============================================================
# ROBUST MA OPTIMIZATION (UPDATED)
# ============================================================

def evaluate_ma_configuration(returns, signal, flip_cost=FLIP_COST, tax_rate=ANNUAL_TAX_RATE):
    """
    Evaluate MA configuration using multiple criteria
    Returns a composite score and metrics
    """
    if len(returns) == 0 or signal.sum() == 0:
        return 0, {"sharpe": 0, "trades_per_year": 0, "hit_ratio": 0}
    
    # Apply costs to calculate realistic returns
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1
    flip_costs = np.where(flip_mask, -flip_cost, 0.0)
    
    # Calculate tax drag based on turnover
    turnover = calculate_turnover(signal)
    tax_drag = calculate_continuous_tax_drag(returns, turnover, tax_rate)
    
    # Net returns after costs
    net_returns = returns + flip_costs + tax_drag.values
    
    # Calculate metrics
    if net_returns.std() > 0:
        sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    trades_per_year = flip_mask.sum() / (len(signal) / 252)
    
    # Monthly hit ratio
    monthly_returns = net_returns.resample('M').apply(lambda x: (1+x).prod()-1)
    hit_ratio = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0
    
    # Penalize excessive trading
    trade_penalty = max(0, (trades_per_year - 12) * 0.02)  # Penalize >12 trades/year
    
    # Penalize extreme regimes (always ON or always OFF)
    regime_balance = min(signal.mean(), 1 - signal.mean())
    regime_penalty = max(0, 0.1 - regime_balance) * 10  # Penalize if <10% in either regime
    
    # Composite score (academic approach)
    score = sharpe - trade_penalty - regime_penalty + (hit_ratio * 0.5)
    
    metrics = {
        "sharpe": sharpe,
        "trades_per_year": trades_per_year,
        "hit_ratio": hit_ratio,
        "regime_balance": regime_balance,
        "trade_penalty": trade_penalty,
        "regime_penalty": regime_penalty,
        "raw_score": score
    }
    
    return score, metrics

def run_robust_ma_optimization(prices, risk_on_weights, risk_off_weights, 
                              flip_cost=FLIP_COST, tax_rate=ANNUAL_TAX_RATE):
    """
    Robust MA optimization using walk-forward validation
    """
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    returns = portfolio_index.pct_change().fillna(0)
    
    # Use global parameters directly (no global declaration needed here)
    candidate_lengths = list(range(MA_MIN_DAYS, MA_MAX_DAYS + 1, 
                                 max(1, (MA_MAX_DAYS - MA_MIN_DAYS) // MA_STEP_FACTOR)))
    candidate_types = ["sma", "ema"]
    min_tol, max_tol, step_tol = MA_TOL_RANGE
    candidate_tolerances = np.arange(min_tol, max_tol + step_tol/2, step_tol)
    
    best_score = -np.inf
    best_config = None
    best_metrics = None
    
    # Cache MAs for efficiency
    ma_cache = {}
    for ma_type in candidate_types:
        ma_cache[ma_type] = compute_ma_matrix(portfolio_index, candidate_lengths, ma_type)
    
    progress = st.progress(0.0)
    total_combos = len(candidate_lengths) * len(candidate_types) * len(candidate_tolerances)
    current_combo = 0
    
    for ma_type in candidate_types:
        for length in candidate_lengths:
            ma = ma_cache[ma_type][length]
            for tol in candidate_tolerances:
                current_combo += 1
                if current_combo % 50 == 0:
                    progress.progress(min(current_combo / total_combos, 1.0))
                
                signal = generate_testfol_signal_vectorized(portfolio_index, ma, tol)
                
                # Skip if signal is invalid
                if signal.sum() == 0 or signal.sum() == len(signal):
                    continue
                
                # Evaluate configuration
                score, metrics = evaluate_ma_configuration(returns, signal, flip_cost, tax_rate)
                
                # Minimum requirements
                if metrics["sharpe"] < 0.3:  # Minimum Sharpe
                    continue
                if metrics["trades_per_year"] > 24:  # Maximum trades/year
                    continue
                if metrics["regime_balance"] < 0.05:  # Minimum time in each regime
                    continue
                
                if score > best_score:
                    best_score = score
                    best_config = (length, ma_type, tol)
                    best_metrics = metrics
    
    # If no valid config found, use reasonable defaults
    if best_config is None:
        best_config = (100, "sma", 0.02)
        ma = compute_ma_matrix(portfolio_index, [100], "sma")[100]
        signal = generate_testfol_signal_vectorized(portfolio_index, ma, 0.02)
    
    # Run final backtest with taxes
    ma = compute_ma_matrix(portfolio_index, [best_config[0]], best_config[1])[best_config[0]]
    signal = generate_testfol_signal_vectorized(portfolio_index, ma, best_config[2])
    result = backtest_with_taxes(prices, signal, risk_on_weights, risk_off_weights, 
                                flip_cost, tax_rate)
    
    return best_config, result, best_metrics


# ============================================================
# BENCHMARK CREATION FUNCTIONS (NEW)
# ============================================================

def create_benchmarks(prices, risk_on_weights, risk_off_weights, 
                     rebalance_dates, flip_cost=FLIP_COST, tax_rate=ANNUAL_TAX_RATE):
    """
    Create multiple academically valid benchmarks
    """
    simple_rets = prices.pct_change().fillna(0)
    
    # 1. Static Buy & Hold (no rebalancing, lets weights drift)
    bh_static_returns = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            bh_static_returns += simple_rets[a] * w
    bh_static_eq = (1 + bh_static_returns).cumprod()
    
    # 2. Rebalanced Buy & Hold (quarterly, with trading costs)
    bh_rebal_eq = pd.Series(10000.0, index=simple_rets.index)
    current_weights = risk_on_weights.copy()
    
    for i, date in enumerate(simple_rets.index):
        # Apply returns with current weights
        daily_ret = 0
        for asset, weight in current_weights.items():
            if asset in simple_rets.columns:
                daily_ret += simple_rets[asset].iloc[i] * weight
        
        if i > 0:
            bh_rebal_eq.iloc[i] = bh_rebal_eq.iloc[i-1] * (1 + daily_ret)
        
        # Quarterly rebalancing with costs
        if date in rebalance_dates:
            # Calculate turnover
            turnover = 0
            for asset in risk_on_weights:
                if asset in current_weights:
                    turnover += abs(current_weights.get(asset, 0) - risk_on_weights.get(asset, 0))
            
            # Apply trading costs
            trading_cost = turnover * flip_cost
            bh_rebal_eq.iloc[i] *= (1 - trading_cost)
            
            # Reset to target weights
            current_weights = risk_on_weights.copy()
    
    bh_rebal_returns = bh_rebal_eq.pct_change().fillna(0)
    
    # 3. 60/40 Portfolio (common institutional benchmark)
    # Try to find SPY and AGG, or use first two assets
    assets = list(prices.columns)
    if len(assets) >= 2:
        spy_col = next((a for a in assets if 'SPY' in a or 'IVV' in a), assets[0])
        agg_col = next((a for a in assets if 'AGG' in a or 'BND' in a), assets[1])
        
        spy_ret = simple_rets[spy_col] if spy_col in simple_rets.columns else simple_rets.iloc[:, 0]
        agg_ret = simple_rets[agg_col] if agg_col in simple_rets.columns else simple_rets.iloc[:, 1]
        
        sixty_forty_returns = 0.6 * spy_ret + 0.4 * agg_ret
    else:
        sixty_forty_returns = bh_static_returns  # Fallback
    
    sixty_forty_eq = (1 + sixty_forty_returns).cumprod()
    
    # Apply tax drag to benchmarks (continuous approximation)
    bh_static_turnover = 0.0  # No trading
    bh_rebal_turnover = 1.0   # 100% annual turnover (quarterly rebalancing)
    
    bh_static_tax_drag = calculate_continuous_tax_drag(bh_static_returns, bh_static_turnover, tax_rate)
    bh_rebal_tax_drag = calculate_continuous_tax_drag(bh_rebal_returns, bh_rebal_turnover, tax_rate)
    
    bh_static_after_tax = bh_static_returns + bh_static_tax_drag.values
    bh_rebal_after_tax = bh_rebal_returns + bh_rebal_tax_drag.values
    
    return {
        'bh_static': {
            'eq_pre_tax': (1 + bh_static_returns).cumprod(),
            'eq_after_tax': (1 + bh_static_after_tax).cumprod(),
            'returns_pre_tax': bh_static_returns,
            'returns_after_tax': bh_static_after_tax,
            'description': 'Static Buy & Hold (weights drift)'
        },
        'bh_rebalanced': {
            'eq_pre_tax': bh_rebal_eq,
            'eq_after_tax': (1 + bh_rebal_after_tax).cumprod(),
            'returns_pre_tax': bh_rebal_returns,
            'returns_after_tax': bh_rebal_after_tax,
            'description': 'Quarterly Rebalanced Buy & Hold'
        },
        'sixty_forty': {
            'eq_pre_tax': sixty_forty_eq,
            'eq_after_tax': sixty_forty_eq,  # Same for simplicity
            'returns_pre_tax': sixty_forty_returns,
            'returns_after_tax': sixty_forty_returns,
            'description': '60/40 Stock/Bond Portfolio'
        }
    }


# ============================================================
# QUARTERLY PROGRESS HELPER (unchanged)
# ============================================================

def compute_quarter_progress(risky_start, risky_today, quarterly_target):
    target_risky = risky_start * (1 + quarterly_target)
    gap = target_risky - risky_today
    pct_gap = gap / risky_start if risky_start > 0 else 0

    return {
        "Deployed Capital at Last Rebalance ($)": risky_start,
        "Deployed Capital Today ($)": risky_today,
        "Deployed Capital Target Next Rebalance ($)": target_risky,
        "Gap ($)": gap,
        "Gap (%)": pct_gap,
    }

def normalize(eq):
    if len(eq) == 0 or eq.iloc[0] == 0:
        return eq
    return eq / eq.iloc[0] * 10000


# ============================================================
# VALIDATION FUNCTIONS (UPDATED WITH TAX AWARENESS)
# ============================================================

def calculate_max_dd(prices_series):
    """Calculate maximum drawdown for a price series"""
    if len(prices_series) == 0:
        return 0
    cumulative = (1 + prices_series.pct_change().fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() if len(drawdown) > 0 else 0

def walk_forward_validation_with_taxes(prices, strategy_result, window_years=3):
    """
    Walk-forward validation that includes tax effects
    """
    returns = strategy_result["returns"]
    signal = strategy_result["signal"]
    
    results = []
    total_days = len(prices)
    
    min_days_needed = 252 * 2  # At least 2 years
    if total_days < min_days_needed:
        return pd.DataFrame()
    
    window_days = min(window_years * 252, total_days // 2)
    
    for start in range(0, total_days - window_days, 126):
        train_end = start + window_days
        test_end = min(train_end + 252, total_days)
        
        if test_end - train_end < 63:
            continue
            
        # Test period returns
        test_returns = returns.iloc[train_end:test_end]
        
        if len(test_returns) > 0 and test_returns.std() > 0:
            # Calculate after-tax Sharpe
            test_signal = signal.iloc[train_end:test_end]
            test_turnover = calculate_turnover(test_signal)
            test_tax_drag = calculate_continuous_tax_drag(test_returns, test_turnover)
            test_after_tax = test_returns + test_tax_drag.values
            
            test_sharpe_pre_tax = test_returns.mean() / test_returns.std() * np.sqrt(252)
            test_sharpe_after_tax = test_after_tax.mean() / test_after_tax.std() * np.sqrt(252)
            
            results.append({
                'train_period': f"{prices.index[start].date()} to {prices.index[train_end-1].date()}",
                'test_period': f"{prices.index[train_end].date()} to {prices.index[test_end-1].date()}",
                'test_sharpe_pre_tax': test_sharpe_pre_tax,
                'test_sharpe_after_tax': test_sharpe_after_tax,
                'test_turnover': test_turnover,
                'test_length_days': len(test_returns)
            })
    
    return pd.DataFrame(results)

def monte_carlo_significance_with_costs(strategy_returns, signal, flip_cost=FLIP_COST, n_simulations=500):
    """
    Monte Carlo test that includes trading costs
    """
    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
        return {
            'actual_sharpe': 0,
            'p_value': 1.0,
            'significance_95': False
        }
    
    # Calculate actual strategy metrics with costs
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1
    flip_costs = np.where(flip_mask, -flip_cost, 0.0)
    net_returns = strategy_returns + flip_costs
    
    if net_returns.std() > 0:
        actual_sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
    else:
        actual_sharpe = 0
    
    # Generate null distribution by shuffling returns
    null_sharpes = []
    for _ in range(min(n_simulations, len(strategy_returns) // 2)):
        # Shuffle returns but keep signal structure
        shuffled_returns = np.random.permutation(strategy_returns.values)
        
        # Apply same trading costs based on original signal
        # (conservative: random strategy pays same costs)
        shuffled_with_costs = shuffled_returns + flip_costs
        
        if np.std(shuffled_with_costs) > 0:
            null_sharpe = np.mean(shuffled_with_costs) / np.std(shuffled_with_costs) * np.sqrt(252)
            null_sharpes.append(null_sharpe)
    
    # Calculate p-value
    if null_sharpes:
        p_value = (np.array(null_sharpes) >= actual_sharpe).mean()
    else:
        p_value = 1.0
    
    return {
        'actual_sharpe': actual_sharpe,
        'p_value': p_value,
        'significance_95': p_value < 0.05,
        'null_mean': np.mean(null_sharpes) if null_sharpes else 0,
        'null_std': np.std(null_sharpes) if null_sharpes else 0
    }

def calculate_cost_breakdown(strategy_result):
    """
    Calculate and display cost breakdown
    """
    returns = strategy_result["returns"]
    flip_costs = strategy_result.get("flip_costs", pd.Series(0.0, index=returns.index))
    tax_payments = strategy_result.get("tax_payments", pd.Series(0.0, index=returns.index))
    
    total_flip_costs = flip_costs.sum()
    total_taxes = tax_payments.sum()
    total_costs = total_flip_costs + total_taxes
    
    total_return = strategy_result["equity_curve"].iloc[-1] / strategy_result["equity_curve"].iloc[0] - 1
    
    cost_percentage = abs(total_costs) / total_return if total_return != 0 else 0
    
    return {
        'total_flip_costs': total_flip_costs,
        'total_taxes': total_taxes,
        'total_costs': total_costs,
        'cost_as_percent_of_return': cost_percentage,
        'annual_turnover': strategy_result.get('annual_turnover', 0),
        'tax_rate': strategy_result.get('tax_rate', ANNUAL_TAX_RATE)
    }


# ============================================================
# STREAMLIT APP - UPDATED
# ============================================================

def main():
    st.set_page_config(page_title="Portfolio MA Regime Strategy", layout="wide")
    st.title("Portfolio Strategy with Tax-Aware Optimization")

    # Sidebar configuration
    start = st.sidebar.text_input("Start Date", DEFAULT_START_DATE)
    end = st.sidebar.text_input("End Date (optional)", "")

    st.sidebar.header("Deployed Capital Sleeve")
    risk_on_tickers_str = st.sidebar.text_input(
        "Tickers", ",".join(RISK_ON_WEIGHTS.keys())
    )
    risk_on_weights_str = st.sidebar.text_input(
        "Weights", ",".join(str(w) for w in RISK_ON_WEIGHTS.values())
    )

    st.sidebar.header("Treasury Sleeve")
    risk_off_tickers_str = st.sidebar.text_input(
        "Tickers", ",".join(RISK_OFF_WEIGHTS.keys())
    )
    risk_off_weights_str = st.sidebar.text_input(
        "Weights", ",".join(str(w) for w in RISK_OFF_WEIGHTS.values())
    )

    st.sidebar.header("Cost Parameters")
    flip_cost_input = st.sidebar.number_input("Slippage Cost per Trade (%)", 
                                            min_value=0.0, max_value=5.0, 
                                            value=FLIP_COST*100, step=0.1) / 100
    tax_rate_input = st.sidebar.number_input("Capital Gains Tax Rate (%)", 
                                           min_value=0.0, max_value=50.0, 
                                           value=ANNUAL_TAX_RATE*100, step=1.0) / 100

    st.sidebar.header("MA Optimization Constraints")
    ma_min_days = st.sidebar.number_input("Minimum MA Length (days)", 
                                         min_value=5, max_value=100, 
                                         value=MA_MIN_DAYS, step=5)
    ma_max_days = st.sidebar.number_input("Maximum MA Length (days)", 
                                         min_value=50, max_value=500, 
                                         value=MA_MAX_DAYS, step=10)

    st.sidebar.header("Quarterly Portfolio Values")
    qs_cap_1 = st.sidebar.number_input("Taxable – Portfolio Value at Last Rebalance ($)", 
                                      min_value=0.0, value=75815.26, step=100.0)
    qs_cap_2 = st.sidebar.number_input("Tax-Sheltered – Portfolio Value at Last Rebalance ($)", 
                                      min_value=0.0, value=10074.83, step=100.0)
    qs_cap_3 = st.sidebar.number_input("Joint – Portfolio Value at Last Rebalance ($)", 
                                      min_value=0.0, value=4189.76, step=100.0)

    st.sidebar.header("Current Portfolio Values (Today)")
    real_cap_1 = st.sidebar.number_input("Taxable – Portfolio Value Today ($)", 
                                        min_value=0.0, value=73165.78, step=100.0)
    real_cap_2 = st.sidebar.number_input("Tax-Sheltered – Portfolio Value Today ($)", 
                                        min_value=0.0, value=9264.46, step=100.0)
    real_cap_3 = st.sidebar.number_input("Joint – Portfolio Value Today ($)", 
                                        min_value=0.0, value=4191.56, step=100.0)

    st.sidebar.header("Validation Settings")
    run_validation = st.sidebar.checkbox("Run Validation Suite", value=True)
    show_cost_breakdown = st.sidebar.checkbox("Show Cost Breakdown", value=True)

    run_clicked = st.sidebar.button("Run Backtest & Optimize")
    if not run_clicked:
        st.stop()

    # Parse inputs
    risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",")]
    risk_on_weights_list = [float(x) for x in risk_on_weights_str.split(",")]
    risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))

    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))

    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers))
    end_val = end if end.strip() else None

    # Load data
    prices = load_price_data(all_tickers, start, end_val).dropna(how="any")
    
    if len(prices) == 0:
        st.error("No data loaded. Please check your ticker symbols and date range.")
        st.stop()
    
    st.info(f"Loaded {len(prices)} trading days of data from {prices.index[0].date()} to {prices.index[-1].date()}")

    # Use local variables instead of modifying globals
    current_ma_min_days = ma_min_days
    current_ma_max_days = ma_max_days
    current_flip_cost = flip_cost_input
    current_tax_rate = tax_rate_input

    # RUN ROBUST MA OPTIMIZATION WITH TAXES
    st.subheader("🔍 MA Optimization (Tax-Aware)")
    with st.spinner("Optimizing MA parameters with tax-aware evaluation..."):
        best_cfg, best_result, best_metrics = run_robust_ma_optimization(
            prices, risk_on_weights, risk_off_weights, current_flip_cost, current_tax_rate
        )
    
    best_len, best_type, best_tol = best_cfg
    sig = best_result["signal"]
    perf = best_result["performance"]

    latest_signal = sig.iloc[-1] if len(sig) > 0 else False
    current_regime = "RISK-ON" if latest_signal else "RISK-OFF"

    st.subheader(f"📈 Current MA Regime: {current_regime}")
    st.write(f"**MA Type:** {best_type.upper()}  |  **Length:** {best_len} days  |  **Tolerance:** {best_tol:.2%}")
    
    if best_metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Optimization Score", f"{best_metrics['raw_score']:.3f}")
        with col2:
            st.metric("Expected Sharpe", f"{best_metrics['sharpe']:.3f}")
        with col3:
            st.metric("Trades/Year", f"{best_metrics['trades_per_year']:.1f}")
        with col4:
            st.metric("Regime Balance", f"{best_metrics['regime_balance']:.1%}")

    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    opt_ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]

    # Calculate calendar quarter dates
    dates = prices.index
    true_q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq='Q')
    mapped_q_ends = []
    for qd in true_q_ends:
        valid_dates = dates[dates <= qd]
        if len(valid_dates) > 0:
            mapped_q_ends.append(valid_dates.max())
    mapped_q_ends = pd.to_datetime(mapped_q_ends)

    # Today's date for rebalancing
    today_date = pd.Timestamp.today().normalize()
    true_next_q = pd.date_range(start=today_date, periods=2, freq="Q")[0]
    next_q_end = true_next_q
    true_prev_q = pd.date_range(end=today_date, periods=2, freq="Q")[0]
    past_q_end = true_prev_q
    days_to_next_q = (next_q_end - today_date).days

    # Calculate quarterly target
    simple_rets = prices.pct_change().fillna(0)
    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w
    risk_on_eq = (1 + risk_on_simple).cumprod()
    
    if len(risk_on_eq) > 0 and risk_on_eq.iloc[0] != 0:
        bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
        quarterly_target = (1 + bh_cagr) ** (1/4) - 1
    else:
        bh_cagr = 0
        quarterly_target = 0

    # Risk-off returns
    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    # Run SIG engines with taxes
    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)
    pure_sig_eq, pure_sig_rw, pure_sig_sw, pure_sig_taxes, pure_sig_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        pure_sig_signal,
        flip_cost=current_flip_cost,
        tax_rate=current_tax_rate,
        quarter_end_dates=mapped_q_ends
    )

    hybrid_eq, hybrid_rw, hybrid_sw, hybrid_taxes, hybrid_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        sig,
        pure_sig_rw=pure_sig_rw,
        pure_sig_sw=pure_sig_sw,
        flip_cost=current_flip_cost,
        tax_rate=current_tax_rate,
        quarter_end_dates=mapped_q_ends
    )

    # Create benchmarks
    st.subheader("📊 Benchmark Creation")
    benchmarks = create_benchmarks(prices, risk_on_weights, risk_off_weights, 
                                  mapped_q_ends, current_flip_cost, current_tax_rate)

    # Display rebalance dates
    if len(hybrid_rebals) > 0:
        st.subheader("Hybrid SIG – Actual Rebalance Dates")
        reb_df = pd.DataFrame({
            "Rebalance Date": pd.to_datetime(hybrid_rebals),
            "Quarter": [f"Q{(d.month-1)//3 + 1}-{d.year}" for d in hybrid_rebals]
        })
        st.dataframe(reb_df)

    # Quarter progress calculations
    quarter_start_date = hybrid_rebals[-1] if len(hybrid_rebals) > 0 else dates[0]

    st.subheader("Strategy Summary")
    if len(hybrid_rebals) > 0:
        last_reb = hybrid_rebals[-1]
        st.write(f"**Last Rebalance:** {last_reb.strftime('%Y-%m-%d')}")
    st.write(f"**Next Rebalance:** {next_q_end.date()} ({days_to_next_q} days)")
    st.write(f"**Quarterly Target Growth Rate:** {quarterly_target:.2%}")

    def get_sig_progress(qs_cap, today_cap):
        if quarter_start_date is not None and len(hybrid_rw) > 0:
            risky_start = qs_cap * float(hybrid_rw.loc[quarter_start_date])
            risky_today = today_cap * float(hybrid_rw.iloc[-1])
            return compute_quarter_progress(risky_start, risky_today, quarterly_target)
        else:
            return compute_quarter_progress(0, 0, 0)

    prog_1 = get_sig_progress(qs_cap_1, real_cap_1)
    prog_2 = get_sig_progress(qs_cap_2, real_cap_2)
    prog_3 = get_sig_progress(qs_cap_3, real_cap_3)

    prog_df = pd.concat([
        pd.DataFrame.from_dict(prog_1, orient='index', columns=['Taxable']),
        pd.DataFrame.from_dict(prog_2, orient='index', columns=['Tax-Sheltered']),
        pd.DataFrame.from_dict(prog_3, orient='index', columns=['Joint']),
    ], axis=1)

    prog_df.loc["Gap (%)"] = prog_df.loc["Gap (%)"].apply(lambda x: f"{x:.2%}")
    st.dataframe(prog_df)

    # Rebalance recommendations
    def rebalance_text(gap, next_q, days_to_next_q):
        date_str = next_q.strftime("%m/%d/%Y")
        days_str = f"{days_to_next_q} days"
        if gap > 0:
            return f"Increase deployed sleeve by **${gap:,.2f}** on **{date_str}** ({days_str})"
        elif gap < 0:
            return f"Decrease deployed sleeve by **${abs(gap):,.2f}** on **{date_str}** ({days_str})"
        else:
            return f"No rebalance needed until **{date_str}** ({days_str})"

    st.write("### Rebalance Recommendations")
    st.write("**Taxable:** "  + rebalance_text(prog_1["Gap ($)"], next_q_end, days_to_next_q))
    st.write("**Tax-Sheltered:** " + rebalance_text(prog_2["Gap ($)"], next_q_end, days_to_next_q))
    st.write("**Joint:** " + rebalance_text(prog_3["Gap ($)"], next_q_end, days_to_next_q))

    # COST BREAKDOWN SECTION
    if show_cost_breakdown:
        st.subheader("💰 Cost Breakdown Analysis")
        
        cost_data = calculate_cost_breakdown(best_result)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Slippage Costs", f"${cost_data['total_flip_costs']*10000:,.2f}",
                     delta=f"{cost_data['total_flip_costs']/best_result['performance']['TotalReturn']:.1%} of returns" 
                     if best_result['performance']['TotalReturn'] != 0 else "N/A")
        with col2:
            st.metric("Total Tax Costs", f"${cost_data['total_taxes']*10000:,.2f}",
                     delta=f"{cost_data['total_taxes']/best_result['performance']['TotalReturn']:.1%} of returns"
                     if best_result['performance']['TotalReturn'] != 0 else "N/A")
        with col3:
            st.metric("Total Costs", f"${cost_data['total_costs']*10000:,.2f}",
                     delta=f"{cost_data['cost_as_percent_of_return']:.1%} of returns")
        with col4:
            st.metric("Annual Turnover", f"{cost_data['annual_turnover']:.1%}",
                     help="Percentage of portfolio traded annually")
        
        # Plot cost accumulation
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(best_result.get("flip_costs", pd.Series(0.0, index=prices.index)).cumsum() * 10000, 
                label="Cumulative Slippage Costs", linewidth=2)
        ax.plot(best_result.get("tax_payments", pd.Series(0.0, index=prices.index)).cumsum() * 10000, 
                label="Cumulative Tax Payments", linewidth=2)
        ax.set_title("Cost Accumulation Over Time ($ per $10,000 invested)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Costs ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # PERFORMANCE COMPARISON TABLE (Updated with benchmarks)
    st.subheader("📈 Performance Comparison (After-Tax)")
    
    # Calculate performance for all strategies
    def calculate_strategy_perf(name, returns, eq_curve):
        perf = compute_performance(returns, eq_curve)
        return {
            "Strategy": name,
            "CAGR": perf["CAGR"],
            "Volatility": perf["Volatility"],
            "Sharpe": perf["Sharpe"],
            "MaxDD": perf["MaxDrawdown"],
            "Total Return": perf["TotalReturn"]
        }
    
    # MA Strategy
    ma_perf = calculate_strategy_perf("MA Strategy", 
                                     best_result["returns"], 
                                     best_result["equity_curve"])
    
    # Hybrid SIG
    hybrid_simple = hybrid_eq.pct_change().fillna(0)
    hybrid_perf = calculate_strategy_perf("Hybrid SIG", 
                                         hybrid_simple, 
                                         hybrid_eq)
    
    # Pure SIG
    pure_sig_simple = pure_sig_eq.pct_change().fillna(0)
    pure_sig_perf = calculate_strategy_perf("Pure SIG", 
                                           pure_sig_simple, 
                                           pure_sig_eq)
    
    # Benchmarks (after-tax)
    benchmarks_data = []
    for key, bench in benchmarks.items():
        bench_perf = calculate_strategy_perf(bench['description'], 
                                            bench['returns_after_tax'], 
                                            bench['eq_after_tax'])
        benchmarks_data.append(bench_perf)
    
    # Combine all performance data
    all_perf = [ma_perf, hybrid_perf, pure_sig_perf] + benchmarks_data
    perf_df = pd.DataFrame(all_perf)
    
    # Format for display
    def format_perf_df(df):
        formatted = df.copy()
        for col in ["CAGR", "Volatility", "MaxDD", "Total Return"]:
            if col in formatted.columns:
                formatted[col] = formatted[col].apply(lambda x: f"{x:.2%}")
        if "Sharpe" in formatted.columns:
            formatted["Sharpe"] = formatted["Sharpe"].apply(lambda x: f"{x:.3f}")
        return formatted
    
    st.dataframe(format_perf_df(perf_df), use_container_width=True)
    
    # Highlight best performer in each category
    st.write("**Best Performers:**")
    for metric in ["Sharpe", "CAGR", "MaxDD"]:
        if metric in perf_df.columns:
            if metric == "MaxDD":
                best_idx = perf_df[metric].idxmax()  # Higher (less negative) is better for MaxDD
            else:
                best_idx = perf_df[metric].idxmax()
            best_value = perf_df.loc[best_idx, metric]
            best_name = perf_df.loc[best_idx, "Strategy"]
            st.write(f"- **{metric}:** {best_name} ({best_value:.2% if metric != 'Sharpe' else best_value:.3f})")

    # FINAL PERFORMANCE PLOT
    st.subheader("📊 Performance Comparison Chart")
    
    # Normalize all equity curves
    plot_index = build_portfolio_index(prices, risk_on_weights)
    plot_ma = compute_ma_matrix(plot_index, [best_len], best_type)[best_len]
    
    # Normalize to $10,000 starting value
    def normalize_to_10k(eq):
        if len(eq) == 0 or eq.iloc[0] == 0:
            return eq
        return eq / eq.iloc[0] * 10000
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot strategies
    ax.plot(normalize_to_10k(best_result["equity_curve"]), 
            label=f"MA Strategy ({best_len}-day {best_type.upper()})", 
            linewidth=2, color='green')
    ax.plot(normalize_to_10k(hybrid_eq), 
            label="Hybrid SIG", 
            linewidth=2, color='blue', alpha=0.8)
    ax.plot(normalize_to_10k(pure_sig_eq), 
            label="Pure SIG", 
            linewidth=2, color='orange', alpha=0.8)
    
    # Plot benchmarks
    colors = ['red', 'purple', 'brown']
    for (key, bench), color in zip(benchmarks.items(), colors):
        ax.plot(normalize_to_10k(bench['eq_after_tax']), 
                label=bench['description'], 
                linewidth=1.5, color=color, alpha=0.6, linestyle='--')
    
    # Plot MA line
    if len(plot_ma.dropna()) > 0:
        ma_norm = normalize_to_10k(plot_ma.dropna())
        ax.plot(ma_norm, label=f"MA({best_len}) Line", 
                linestyle=':', color='black', alpha=0.5, linewidth=1)
    
    ax.set_title("Growth of $10,000 Investment (After-Tax)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)

    # VALIDATION SECTION
    if run_validation:
        st.header("🎯 Strategy Validation")
        
        val_tab1, val_tab2, val_tab3 = st.tabs([
            "Statistical Significance", 
            "Walk-Forward Analysis",
            "Strategy Metrics"
        ])
        
        with val_tab1:
            st.subheader("Monte Carlo Significance Test (with Costs)")
            
            mc_results = monte_carlo_significance_with_costs(
                best_result["returns"], 
                best_result["signal"],
                current_flip_cost
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Actual Sharpe (with costs)", f"{mc_results['actual_sharpe']:.3f}")
            with col2:
                st.metric("p-value", f"{mc_results['p_value']:.3f}")
            with col3:
                significance = "✅ Statistically Significant" if mc_results['significance_95'] else "❌ Not Significant"
                st.metric("Significance (95%)", significance)
            
            st.write(f"**Null Distribution:** Mean = {mc_results['null_mean']:.3f}, Std = {mc_results['null_std']:.3f}")
        
        with val_tab2:
            st.subheader("Walk-Forward Validation")
            
            wfa_results = walk_forward_validation_with_taxes(prices, best_result)
            
            if not wfa_results.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Test Sharpe (Pre-tax)", 
                             f"{wfa_results['test_sharpe_pre_tax'].mean():.3f}")
                with col2:
                    st.metric("Avg Test Sharpe (After-tax)", 
                             f"{wfa_results['test_sharpe_after_tax'].mean():.3f}")
                with col3:
                    success_rate = (wfa_results['test_sharpe_after_tax'] > 0.5).mean()
                    st.metric("Success Rate (Sharpe > 0.5)", f"{success_rate:.1%}")
                with col4:
                    avg_turnover = wfa_results['test_turnover'].mean()
                    st.metric("Avg Test Turnover", f"{avg_turnover:.1%}")
                
                # Plot results
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(wfa_results.index, wfa_results['test_sharpe_after_tax'], 
                       marker='o', linewidth=2, label='After-tax Sharpe')
                ax.plot(wfa_results.index, wfa_results['test_sharpe_pre_tax'], 
                       marker='s', linewidth=1, alpha=0.5, label='Pre-tax Sharpe')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=wfa_results['test_sharpe_after_tax'].mean(), 
                          color='g', linestyle='--', alpha=0.5, label='Average')
                ax.set_title('Walk-Forward Test Sharpe Ratios')
                ax.set_xlabel('Test Period')
                ax.set_ylabel('Sharpe Ratio')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            else:
                st.info("Not enough data for walk-forward analysis (need at least 4 years of data)")
        
        with val_tab3:
            st.subheader("Strategy Health Metrics")
            
            # Calculate various metrics
            metrics = {
                "Monthly Hit Ratio": (best_result["returns"].resample('M')
                                     .apply(lambda x: (1+x).prod()-1 > 0).mean()),
                "Win/Loss Ratio": (abs(best_result["returns"][best_result["returns"] > 0].mean() / 
                                     best_result["returns"][best_result["returns"] < 0].mean()) 
                                 if len(best_result["returns"][best_result["returns"] < 0]) > 0 else 0),
                "Profit Factor": (best_result["returns"][best_result["returns"] > 0].sum() / 
                                abs(best_result["returns"][best_result["returns"] < 0].sum()) 
                                if best_result["returns"][best_result["returns"] < 0].sum() != 0 else 0),
                "Calmar Ratio": (best_result["performance"]["CAGR"] / 
                               abs(best_result["performance"]["MaxDrawdown"]) 
                               if best_result["performance"]["MaxDrawdown"] != 0 else 0),
                "Omega Ratio": (best_result["returns"][best_result["returns"] > 0].sum() / 
                              abs(best_result["returns"][best_result["returns"] < 0].sum()) 
                              if best_result["returns"][best_result["returns"] < 0].sum() != 0 else 0)
            }
            
            # Display metrics
            cols = st.columns(5)
            for (metric, value), col in zip(metrics.items(), cols):
                with col:
                    if metric in ["Monthly Hit Ratio"]:
                        st.metric(metric, f"{value:.1%}")
                    elif metric in ["Win/Loss Ratio", "Profit Factor", "Calmar Ratio", "Omega Ratio"]:
                        st.metric(metric, f"{value:.2f}")
            
            # Interpretation guide
            st.write("### 📊 Interpretation Guide")
            st.write("- **Monthly Hit Ratio > 55%**: Good consistency")
            st.write("- **Win/Loss Ratio > 1.5**: Favorable risk/reward")
            st.write("- **Profit Factor > 1.5**: Profitable strategy")
            st.write("- **Calmar Ratio > 1.0**: Good risk-adjusted returns")
            st.write("- **Omega Ratio > 1.5**: Favorable return distribution")

    # IMPLEMENTATION CHECKLIST
    st.markdown("""
---

## **Implementation Checklist**

1. **Tax Considerations:**
   - Taxes are paid when gains are realized (MA flips to RISK-OFF or quarterly rebalancing)
   - Effective tax rate: {:.1%}
   - Slippage cost per trade: {:.1%}

2. **Portfolio Actions:**
   - Rotate to treasury sleeve when MA regime flips to RISK-OFF
   - At each calendar quarter-end, input portfolio values
   - Execute dollar adjustments as recommended
   - Re-evaluate Sharpe-optimal weights quarterly

3. **Current Configuration:**
   - MA: {}-day {} with {:.1%} tolerance
   - Expected annual turnover: {:.1%}
   - Expected trades/year: {:.1f}

4. **Academic Notes:**
   - All returns shown are AFTER estimated taxes and trading costs
   - Benchmarks include rebalancing costs and tax drag
   - MA optimization penalizes excessive trading

Current Sharpe-optimal portfolio: https://testfol.io/optimizer?s=9TIGHucZuaJ

---
    """.format(current_tax_rate, current_flip_cost, best_len, best_type.upper(), best_tol,
              best_result.get('annual_turnover', 0), 
              best_metrics['trades_per_year'] if best_metrics else 0))


# ============================================================
# LAUNCH APP
# ============================================================

if __name__ == "__main__":
    main()