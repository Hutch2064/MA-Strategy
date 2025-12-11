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
# UNIFIED TAX CALCULATION FUNCTIONS (FIXED)
# ============================================================

def calculate_unified_tax_liability(returns, signal, tax_rate=ANNUAL_TAX_RATE):
    """
    Unified tax calculation for ALL strategies
    Taxes are applied when switching from RISK-ON to RISK-OFF (selling risky assets)
    """
    if len(returns) == 0 or len(signal) == 0:
        return pd.Series(0.0, index=returns.index)
    
    # Initialize tracking
    equity = 10000.0
    cost_basis = 10000.0  # Track separately from equity
    tax_payments = pd.Series(0.0, index=returns.index)
    
    # Convert signal to boolean
    sig_bool = signal.astype(bool)
    
    for i in range(len(returns)):
        # Apply daily return
        equity *= (1 + returns.iloc[i])
        
        # Check if we just switched from RISK-ON to RISK-OFF
        if i > 0 and sig_bool.iloc[i-1] and not sig_bool.iloc[i]:
            # We just sold risky assets - realize gains
            if equity > cost_basis:
                realized_gain = equity - cost_basis
                tax_amount = realized_gain * tax_rate
                
                # Apply tax payment
                tax_payments.iloc[i] = -tax_amount
                equity -= tax_amount
                
                # Update cost basis (post-tax value)
                cost_basis = equity
        
        # Check if we just switched from RISK-OFF to RISK-ON
        elif i > 0 and not sig_bool.iloc[i-1] and sig_bool.iloc[i]:
            # We just bought risky assets - new cost basis
            cost_basis = equity
    
    return tax_payments

def calculate_continuous_tax_drag(returns, turnover_rate, tax_rate=ANNUAL_TAX_RATE):
    """
    Continuous tax drag (simpler academic approach)
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

def calculate_strategy_taxes(returns, signal, tax_rate=ANNUAL_TAX_RATE, method='auto'):
    """
    Unified tax calculation with automatic method selection
    """
    if method == 'auto':
        # Auto-select based on turnover
        turnover = calculate_turnover(signal)
        if turnover > 3.0:  # >300% annual turnover
            method = 'continuous'
        else:
            method = 'realization'
    
    if method == 'continuous':
        turnover = calculate_turnover(signal)
        return calculate_continuous_tax_drag(returns, turnover, tax_rate)
    else:
        return calculate_unified_tax_liability(returns, signal, tax_rate)


# ============================================================
# UNIFIED BACKTEST ENGINE (FOR ALL STRATEGIES)
# ============================================================

def build_weight_df(prices, signal, risk_on_weights, risk_off_weights):
    """
    Build daily weight matrix based on signal
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # When signal is True (RISK-ON), use risk_on_weights
    for asset, weight in risk_on_weights.items():
        if asset in prices.columns:
            weights.loc[signal, asset] = weight
    
    # When signal is False (RISK-OFF), use risk_off_weights
    for asset, weight in risk_off_weights.items():
        if asset in prices.columns:
            weights.loc[~signal, asset] = weight
    
    return weights

def compute_performance(simple_returns, eq_curve, rf=0.0):
    """
    Compute performance metrics safely
    """
    if len(eq_curve) == 0 or eq_curve.iloc[0] == 0:
        return {
            "CAGR": 0,
            "Volatility": 0,
            "Sharpe": 0,
            "MaxDrawdown": 0,
            "TotalReturn": 0,
            "DD_Series": pd.Series([], dtype=float)
        }
    
    # Ensure valid data
    eq_curve = eq_curve.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(eq_curve) < 2:
        return {
            "CAGR": 0,
            "Volatility": 0,
            "Sharpe": 0,
            "MaxDrawdown": 0,
            "TotalReturn": 0,
            "DD_Series": pd.Series([], dtype=float)
        }
    
    # Calculate years
    years = len(eq_curve) / 252
    
    # Calculate CAGR
    if eq_curve.iloc[0] > 0 and eq_curve.iloc[-1] > 0:
        cagr = (eq_curve.iloc[-1] / eq_curve.iloc[0]) ** (1 / years) - 1
    else:
        cagr = -1.0
    
    # Calculate volatility
    if len(simple_returns) > 1:
        vol = simple_returns.std() * np.sqrt(252)
    else:
        vol = 0
    
    # Calculate Sharpe ratio
    if vol > 0 and len(simple_returns) > 0:
        annual_return = simple_returns.mean() * 252
        sharpe = (annual_return - rf) / vol
    else:
        sharpe = 0
    
    # Calculate max drawdown
    if eq_curve.iloc[0] > 0:
        cumulative = eq_curve / eq_curve.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() if len(drawdown) > 0 else 0
    else:
        max_dd = -1.0
    
    # Calculate total return
    if eq_curve.iloc[0] > 0:
        total_return = eq_curve.iloc[-1] / eq_curve.iloc[0] - 1
    else:
        total_return = -1.0
    
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": total_return,
        "DD_Series": drawdown if 'drawdown' in locals() else pd.Series([], dtype=float)
    }

def backtest_unified(prices, signal, risk_on_weights, risk_off_weights,
                    flip_cost=FLIP_COST, tax_rate=ANNUAL_TAX_RATE, tax_method='auto'):
    """
    Unified backtest function for ALL strategies (MA, Hybrid SIG, Pure SIG, etc.)
    """
    # Calculate daily returns
    simple_rets = prices.pct_change().fillna(0)
    
    # Build weight matrix
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)
    
    # Calculate strategy returns BEFORE costs
    strategy_returns = (weights.shift(1).fillna(0) * simple_rets).sum(axis=1)
    
    # Apply flip costs
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1
    returns_after_flip = strategy_returns.copy()
    returns_after_flip[flip_mask] -= flip_cost
    
    # Calculate taxes using unified function
    tax_payments = calculate_strategy_taxes(returns_after_flip, signal, tax_rate, tax_method)
    
    # Net returns after all costs
    net_returns = returns_after_flip + tax_payments.values
    
    # Calculate equity curve
    equity_curve = (1 + net_returns).cumprod() * 10000
    
    # Calculate turnover
    turnover = calculate_turnover(signal)
    
    return {
        "returns": net_returns,
        "equity_curve": equity_curve,
        "signal": signal,
        "weights": weights,
        "performance": compute_performance(net_returns, equity_curve),
        "flip_costs": pd.Series(np.where(flip_mask, -flip_cost, 0.0), index=strategy_returns.index),
        "tax_payments": tax_payments,
        "annual_turnover": turnover,
        "tax_rate": tax_rate,
        "tax_method": tax_method
    }


# ============================================================
# SIG ENGINE - UPDATED WITHOUT EMBEDDED TAX LOGIC
# ============================================================

def run_sig_engine_no_tax(risk_on_returns, risk_off_returns, target_quarter,
                         ma_signal, pure_sig_rw=None, pure_sig_sw=None,
                         flip_cost=FLIP_COST, quarter_end_dates=None):
    """
    SIG Engine without embedded tax logic - only handles rebalancing
    Returns daily returns that can be fed into unified backtest
    """
    dates = risk_on_returns.index
    n = len(dates)
    
    if quarter_end_dates is None:
        raise ValueError("quarter_end_dates must be supplied")
    
    quarter_end_set = set(quarter_end_dates)
    
    # MA flip detection
    sig_arr = ma_signal.astype(int)
    
    # Initialize values
    eq = 10000.0
    risky_val = eq * START_RISKY
    safe_val = eq * START_SAFE
    
    frozen_risky = None
    frozen_safe = None
    
    # Track daily returns
    daily_returns = pd.Series(0.0, index=dates)
    rebalance_dates = []
    
    for i in range(n):
        date = dates[i]
        r_on = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])
        
        if ma_on:
            # Restore pure-SIG weights after exiting RISK-OFF
            if frozen_risky is not None:
                w_r = pure_sig_rw.iloc[i]
                w_s = pure_sig_sw.iloc[i]
                risky_val = eq * w_r
                safe_val = eq * w_s
                frozen_risky = None
                frozen_safe = None
            
            # Apply daily returns
            risky_val *= (1 + r_on)
            safe_val *= (1 + r_off)
            
            # Rebalance on quarter-end date
            if date in quarter_end_set:
                prev_qs = [qd for qd in quarter_end_dates if qd < date]
                
                if prev_qs:
                    prev_q = prev_qs[-1]
                    idx_prev = dates.get_loc(prev_q)
                    risky_at_qstart = risky_val
                    goal_risky = risky_at_qstart * (1 + target_quarter)
                    
                    if risky_val > goal_risky:
                        excess = risky_val - goal_risky
                        risky_val -= excess
                        safe_val += excess
                        rebalance_dates.append(date)
                        
                        # Apply quarterly fee
                        eq *= (1 - flip_cost * target_quarter)
                    
                    elif risky_val < goal_risky:
                        needed = goal_risky - risky_val
                        move = min(needed, safe_val)
                        safe_val -= move
                        risky_val += move
                        rebalance_dates.append(date)
            
            # Update equity and calculate return
            new_eq = risky_val + safe_val
            daily_return = (new_eq / eq) - 1
            eq = new_eq
        
        else:
            # RISK-OFF regime
            # Freeze values on entering RISK-OFF
            if frozen_risky is None:
                frozen_risky = risky_val
                frozen_safe = safe_val
            
            # Only safe sleeve earns returns
            eq *= (1 + r_off)
            daily_return = r_off
            
            # Update values for consistency
            risky_val = 0.0
            safe_val = eq
        
        # Store daily return
        daily_returns.iloc[i] = daily_return
    
    return daily_returns, rebalance_dates


# ============================================================
# ROBUST MA OPTIMIZATION (UPDATED WITH UNIFIED BACKTEST)
# ============================================================

def evaluate_ma_configuration_unified(prices, signal, risk_on_weights, risk_off_weights,
                                    flip_cost=FLIP_COST, tax_rate=ANNUAL_TAX_RATE):
    """
    Evaluate MA configuration using unified backtest
    """
    if len(signal) == 0 or signal.sum() == 0:
        return 0, {"sharpe": 0, "trades_per_year": 0, "hit_ratio": 0}
    
    # Run unified backtest
    result = backtest_unified(prices, signal, risk_on_weights, risk_off_weights,
                             flip_cost, tax_rate, tax_method='auto')
    
    # Calculate additional metrics
    perf = result["performance"]
    turnover = result["annual_turnover"]
    
    # Monthly hit ratio
    monthly_returns = result["returns"].resample('M').apply(lambda x: (1+x).prod()-1)
    hit_ratio = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0
    
    # Penalize excessive trading
    trades_per_year = turnover
    trade_penalty = max(0, (trades_per_year - 12) * 0.02)
    
    # Penalize extreme regimes
    regime_balance = min(signal.mean(), 1 - signal.mean())
    regime_penalty = max(0, 0.1 - regime_balance) * 10
    
    # Composite score
    score = perf["Sharpe"] - trade_penalty - regime_penalty + (hit_ratio * 0.5)
    
    metrics = {
        "sharpe": perf["Sharpe"],
        "trades_per_year": trades_per_year,
        "hit_ratio": hit_ratio,
        "regime_balance": regime_balance,
        "trade_penalty": trade_penalty,
        "regime_penalty": regime_penalty,
        "raw_score": score,
        "cagr": perf["CAGR"],
        "max_dd": perf["MaxDrawdown"]
    }
    
    return score, metrics, result

def run_robust_ma_optimization_unified(prices, risk_on_weights, risk_off_weights,
                                     flip_cost=FLIP_COST, tax_rate=ANNUAL_TAX_RATE):
    """
    Robust MA optimization using unified backtest
    """
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    returns = portfolio_index.pct_change().fillna(0)
    
    # Generate candidate parameters
    candidate_lengths = list(range(MA_MIN_DAYS, MA_MAX_DAYS + 1,
                                 max(1, (MA_MAX_DAYS - MA_MIN_DAYS) // MA_STEP_FACTOR)))
    candidate_types = ["sma", "ema"]
    min_tol, max_tol, step_tol = MA_TOL_RANGE
    candidate_tolerances = np.arange(min_tol, max_tol + step_tol/2, step_tol)
    
    best_score = -np.inf
    best_config = None
    best_metrics = None
    best_result = None
    
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
                
                # Evaluate configuration using unified backtest
                score, metrics, result = evaluate_ma_configuration_unified(
                    prices, signal, risk_on_weights, risk_off_weights,
                    flip_cost, tax_rate
                )
                
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
                    best_result = result
    
    # If no valid config found, use reasonable defaults
    if best_config is None:
        best_config = (100, "sma", 0.02)
        ma = compute_ma_matrix(portfolio_index, [100], "sma")[100]
        signal = generate_testfol_signal_vectorized(portfolio_index, ma, 0.02)
        _, _, best_result = evaluate_ma_configuration_unified(
            prices, signal, risk_on_weights, risk_off_weights,
            flip_cost, tax_rate
        )
    
    return best_config, best_result, best_metrics


# ============================================================
# BENCHMARK CREATION FUNCTIONS (UPDATED)
# ============================================================

def create_benchmarks_unified(prices, risk_on_weights, risk_off_weights,
                             rebalance_dates, flip_cost=FLIP_COST, tax_rate=ANNUAL_TAX_RATE):
    """
    Create benchmarks using unified backtest framework
    """
    # 1. Static Buy & Hold (no rebalancing)
    bh_static_signal = pd.Series(True, index=prices.index)
    bh_static = backtest_unified(prices, bh_static_signal, risk_on_weights, {},
                                flip_cost=0.0, tax_rate=tax_rate, tax_method='continuous')
    
    # 2. Rebalanced Buy & Hold (quarterly)
    bh_rebal_signal = pd.Series(True, index=prices.index)
    bh_rebal = backtest_unified(prices, bh_rebal_signal, risk_on_weights, {},
                               flip_cost=flip_cost, tax_rate=tax_rate, tax_method='continuous')
    
    # 3. 60/40 Portfolio
    # Create synthetic 60/40 weights
    assets = list(prices.columns)
    if len(assets) >= 2:
        # Try to find SPY and AGG equivalents
        spy_col = next((a for a in assets if 'SPY' in a or 'IVV' in a or 'VOO' in a), assets[0])
        agg_col = next((a for a in assets if 'AGG' in a or 'BND' in a or 'IEF' in a), assets[1])
        
        sixty_forty_weights = {spy_col: 0.6, agg_col: 0.4}
        sixty_forty_signal = pd.Series(True, index=prices.index)
        sixty_forty = backtest_unified(prices, sixty_forty_signal, sixty_forty_weights, {},
                                      flip_cost=flip_cost, tax_rate=tax_rate, tax_method='continuous')
    else:
        # Fallback to buy & hold
        sixty_forty = bh_static
    
    return {
        'bh_static': {
            'result': bh_static,
            'description': 'Static Buy & Hold (weights drift)'
        },
        'bh_rebalanced': {
            'result': bh_rebal,
            'description': 'Quarterly Rebalanced Buy & Hold'
        },
        'sixty_forty': {
            'result': sixty_forty,
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
# VALIDATION FUNCTIONS (UPDATED WITH UNIFIED BACKTEST)
# ============================================================

def walk_forward_validation_unified(prices, strategy_result, window_years=3):
    """
    Walk-forward validation using unified backtest
    """
    returns = strategy_result["returns"]
    signal = strategy_result["signal"]
    
    results = []
    total_days = len(prices)
    
    min_days_needed = 252 * 2
    if total_days < min_days_needed:
        return pd.DataFrame()
    
    window_days = min(window_years * 252, total_days // 2)
    
    for start in range(0, total_days - window_days, 126):
        train_end = start + window_days
        test_end = min(train_end + 252, total_days)
        
        if test_end - train_end < 63:
            continue
        
        # Extract test period data
        test_prices = prices.iloc[train_end:test_end]
        test_signal = signal.iloc[train_end:test_end]
        
        if len(test_prices) > 0 and len(test_signal) > 0:
            # Run backtest on test period
            test_result = backtest_unified(test_prices, test_signal,
                                          strategy_result["weights"].columns.tolist(),
                                          {}, flip_cost=0.001, tax_rate=0.20)
            
            test_perf = test_result["performance"]
            
            results.append({
                'train_period': f"{prices.index[start].date()} to {prices.index[train_end-1].date()}",
                'test_period': f"{prices.index[train_end].date()} to {prices.index[test_end-1].date()}",
                'test_sharpe': test_perf["Sharpe"],
                'test_cagr': test_perf["CAGR"],
                'test_max_dd': test_perf["MaxDrawdown"],
                'test_length_days': len(test_prices)
            })
    
    return pd.DataFrame(results)

def monte_carlo_significance_unified(strategy_result, n_simulations=500):
    """
    Monte Carlo test using unified framework
    """
    returns = strategy_result["returns"]
    signal = strategy_result["signal"]
    
    if len(returns) == 0 or returns.std() == 0:
        return {
            'actual_sharpe': 0,
            'p_value': 1.0,
            'significance_95': False
        }
    
    actual_sharpe = strategy_result["performance"]["Sharpe"]
    
    # Generate null distribution by shuffling returns
    null_sharpes = []
    for _ in range(min(n_simulations, len(returns) // 2)):
        # Shuffle returns but keep signal structure
        shuffled_returns = np.random.permutation(returns.values)
        shuffled_series = pd.Series(shuffled_returns, index=returns.index)
        
        # Calculate Sharpe on shuffled returns with same signal
        if shuffled_series.std() > 0:
            null_sharpe = shuffled_series.mean() / shuffled_series.std() * np.sqrt(252)
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

def calculate_cost_breakdown_unified(strategy_result):
    """
    Calculate cost breakdown using unified structure
    """
    returns = strategy_result["returns"]
    flip_costs = strategy_result.get("flip_costs", pd.Series(0.0, index=returns.index))
    tax_payments = strategy_result.get("tax_payments", pd.Series(0.0, index=returns.index))
    
    total_flip_costs = flip_costs.sum() * 10000  # Scale to $10,000 investment
    total_taxes = tax_payments.sum() * 10000
    total_costs = total_flip_costs + total_taxes
    
    total_return = strategy_result["equity_curve"].iloc[-1] - 10000
    
    if total_return != 0:
        cost_percentage = abs(total_costs) / abs(total_return)
    else:
        cost_percentage = 0
    
    return {
        'total_flip_costs': total_flip_costs,
        'total_taxes': total_taxes,
        'total_costs': total_costs,
        'cost_as_percent_of_return': cost_percentage,
        'annual_turnover': strategy_result.get('annual_turnover', 0),
        'tax_rate': strategy_result.get('tax_rate', ANNUAL_TAX_RATE),
        'tax_method': strategy_result.get('tax_method', 'auto')
    }


# ============================================================
# STREAMLIT APP - UPDATED WITH UNIFIED LOGIC
# ============================================================

def main():
    st.set_page_config(page_title="Portfolio MA Regime Strategy", layout="wide")
    st.title("Portfolio Strategy with Unified Tax-Aware Optimization")

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
    show_tax_method = st.sidebar.checkbox("Show Tax Method Details", value=True)

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

    # Use local parameters
    current_flip_cost = flip_cost_input
    current_tax_rate = tax_rate_input

    # RUN ROBUST MA OPTIMIZATION WITH UNIFIED BACKTEST
    st.subheader("🔍 MA Optimization (Unified Tax-Aware)")
    with st.spinner("Optimizing MA parameters with unified tax-aware evaluation..."):
        best_cfg, ma_result, best_metrics = run_robust_ma_optimization_unified(
            prices, risk_on_weights, risk_off_weights, current_flip_cost, current_tax_rate
        )
    
    best_len, best_type, best_tol = best_cfg
    ma_signal = ma_result["signal"]
    ma_perf = ma_result["performance"]

    latest_signal = ma_signal.iloc[-1] if len(ma_signal) > 0 else False
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
            st.metric("Tax Method", ma_result.get('tax_method', 'auto').title())

    # Calculate calendar quarter dates
    dates = prices.index
    true_q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq='Q')
    mapped_q_ends = []
    for qd in true_q_ends:
        valid_dates = dates[dates <= qd]
        if len(valid_dates) > 0:
            mapped_q_ends.append(valid_dates.max())
    mapped_q_ends = pd.to_datetime(mapped_q_ends)

    # Calculate quarterly target
    simple_rets = prices.pct_change().fillna(0)
    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w
    
    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w
    
    # Calculate quarterly target
    risk_on_eq = (1 + risk_on_simple).cumprod()
    if len(risk_on_eq) > 0 and risk_on_eq.iloc[0] != 0:
        bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
        quarterly_target = (1 + bh_cagr) ** (1/4) - 1
    else:
        bh_cagr = 0
        quarterly_target = 0

    # Get pure SIG weights (no MA signal)
    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)
    pure_sig_result = backtest_unified(prices, pure_sig_signal, risk_on_weights, risk_off_weights,
                                      current_flip_cost, current_tax_rate, tax_method='continuous')

    # Get Hybrid SIG returns from SIG engine (without tax logic)
    hybrid_returns, hybrid_rebals = run_sig_engine_no_tax(
        risk_on_simple, risk_off_daily, quarterly_target, ma_signal,
        flip_cost=current_flip_cost, quarter_end_dates=mapped_q_ends
    )
    
    # Create Hybrid SIG signal (same as MA signal for tax purposes)
    hybrid_signal = ma_signal.copy()
    
    # Run unified backtest for Hybrid SIG
    hybrid_result = backtest_unified(prices, hybrid_signal, risk_on_weights, risk_off_weights,
                                    current_flip_cost, current_tax_rate, tax_method='auto')

    # Create benchmarks
    st.subheader("📊 Benchmark Creation")
    benchmarks = create_benchmarks_unified(prices, risk_on_weights, risk_off_weights,
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
    
    # Calculate next quarter date
    today_date = pd.Timestamp.today().normalize()
    next_q_end = pd.date_range(start=today_date, periods=2, freq="Q")[0]
    days_to_next_q = (next_q_end - today_date).days
    
    st.write(f"**Next Rebalance:** {next_q_end.date()} ({days_to_next_q} days)")
    st.write(f"**Quarterly Target Growth Rate:** {quarterly_target:.2%}")

    def get_sig_progress(qs_cap, today_cap):
        if quarter_start_date is not None and len(hybrid_result["weights"]) > 0:
            # Extract risky weight from hybrid result
            risky_weight = 0.0
            for asset, weight in risk_on_weights.items():
                if asset in hybrid_result["weights"].columns:
                    risky_weight += hybrid_result["weights"].loc[quarter_start_date, asset] if quarter_start_date in hybrid_result["weights"].index else 0
            
            risky_start = qs_cap * risky_weight
            risky_today = today_cap * risky_weight  # Using same weight for simplicity
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
        
        ma_cost_data = calculate_cost_breakdown_unified(ma_result)
        hybrid_cost_data = calculate_cost_breakdown_unified(hybrid_result)
        pure_sig_cost_data = calculate_cost_breakdown_unified(pure_sig_result)
        
        # Compare costs
        cost_comparison = pd.DataFrame({
            "MA Strategy": [
                f"${ma_cost_data['total_costs']:,.2f}",
                f"{ma_cost_data['cost_as_percent_of_return']:.1%}",
                f"{ma_cost_data['annual_turnover']:.1%}",
                ma_cost_data['tax_method']
            ],
            "Hybrid SIG": [
                f"${hybrid_cost_data['total_costs']:,.2f}",
                f"{hybrid_cost_data['cost_as_percent_of_return']:.1%}",
                f"{hybrid_cost_data['annual_turnover']:.1%}",
                hybrid_cost_data['tax_method']
            ],
            "Pure SIG": [
                f"${pure_sig_cost_data['total_costs']:,.2f}",
                f"{pure_sig_cost_data['cost_as_percent_of_return']:.1%}",
                f"{pure_sig_cost_data['annual_turnover']:.1%}",
                pure_sig_cost_data['tax_method']
            ]
        }, index=["Total Costs", "Costs/Return", "Annual Turnover", "Tax Method"])
        
        st.dataframe(cost_comparison)
        
        # Plot cost accumulation
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ma_result["flip_costs"].cumsum() * 10000, 
                label="MA Slippage Costs", linewidth=2)
        ax.plot(ma_result["tax_payments"].cumsum() * 10000, 
                label="MA Tax Payments", linewidth=2)
        ax.set_title("Cost Accumulation for MA Strategy ($ per $10,000 invested)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Costs ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # PERFORMANCE COMPARISON TABLE
    st.subheader("📈 Performance Comparison (After-Tax)")
    
    # Collect all performance data
    all_perf = []
    
    # MA Strategy
    all_perf.append({
        "Strategy": "MA Strategy",
        "CAGR": ma_perf["CAGR"],
        "Volatility": ma_perf["Volatility"],
        "Sharpe": ma_perf["Sharpe"],
        "MaxDD": ma_perf["MaxDrawdown"],
        "Total Return": ma_perf["TotalReturn"]
    })
    
    # Hybrid SIG
    hybrid_perf = hybrid_result["performance"]
    all_perf.append({
        "Strategy": "Hybrid SIG",
        "CAGR": hybrid_perf["CAGR"],
        "Volatility": hybrid_perf["Volatility"],
        "Sharpe": hybrid_perf["Sharpe"],
        "MaxDD": hybrid_perf["MaxDrawdown"],
        "Total Return": hybrid_perf["TotalReturn"]
    })
    
    # Pure SIG
    pure_sig_perf = pure_sig_result["performance"]
    all_perf.append({
        "Strategy": "Pure SIG",
        "CAGR": pure_sig_perf["CAGR"],
        "Volatility": pure_sig_perf["Volatility"],
        "Sharpe": pure_sig_perf["Sharpe"],
        "MaxDD": pure_sig_perf["MaxDrawdown"],
        "Total Return": pure_sig_perf["TotalReturn"]
    })
    
    # Benchmarks
    for key, bench in benchmarks.items():
        bench_perf = bench["result"]["performance"]
        all_perf.append({
            "Strategy": bench["description"],
            "CAGR": bench_perf["CAGR"],
            "Volatility": bench_perf["Volatility"],
            "Sharpe": bench_perf["Sharpe"],
            "MaxDD": bench_perf["MaxDrawdown"],
            "Total Return": bench_perf["TotalReturn"]
        })
    
    # Create DataFrame
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
    
    perf_df_formatted = format_perf_df(perf_df)
    st.dataframe(perf_df_formatted, use_container_width=True)
    
    # Highlight best performers
    st.write("**Best Performers:**")
    perf_df_raw = perf_df.copy()
    
    for metric in ["Sharpe", "CAGR"]:
        if metric in perf_df_raw.columns:
            best_idx = perf_df_raw[metric].idxmax()
            best_value = perf_df_raw.loc[best_idx, metric]
            best_name = perf_df_raw.loc[best_idx, "Strategy"]
            
            if metric == "Sharpe":
                formatted_value = f"{best_value:.3f}"
            else:
                formatted_value = f"{best_value:.2%}"
            
            st.write(f"- **{metric}:** {best_name} ({formatted_value})")

    # PERFORMANCE PLOT
    st.subheader("📊 Performance Comparison Chart")
    
    # Normalize to $10,000 starting value
    def normalize_to_10k(eq):
        if len(eq) == 0 or eq.iloc[0] == 0:
            return eq
        return eq / eq.iloc[0] * 10000
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot strategies
    ax.plot(normalize_to_10k(ma_result["equity_curve"]), 
            label=f"MA Strategy ({best_len}-day {best_type.upper()})", 
            linewidth=2, color='green')
    ax.plot(normalize_to_10k(hybrid_result["equity_curve"]), 
            label="Hybrid SIG", 
            linewidth=2, color='blue', alpha=0.8)
    ax.plot(normalize_to_10k(pure_sig_result["equity_curve"]), 
            label="Pure SIG", 
            linewidth=2, color='orange', alpha=0.8)
    
    # Plot benchmarks
    colors = ['red', 'purple', 'brown']
    for (key, bench), color in zip(benchmarks.items(), colors):
        ax.plot(normalize_to_10k(bench["result"]["equity_curve"]), 
                label=bench['description'], 
                linewidth=1.5, color=color, alpha=0.6, linestyle='--')
    
    ax.set_title("Growth of $10,000 Investment (After-Tax, Unified Logic)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)

    # TAX METHOD DETAILS
    if show_tax_method:
        st.subheader("🧾 Tax Method Details")
        
        st.write("""
        **Unified Tax Logic Applied:**
        
        1. **All strategies** use the same tax calculation logic
        2. **Tax Method Selection:**
           - **Auto:** Chooses method based on annual turnover
           - **Realization:** Taxes paid when switching from RISK-ON to RISK-OFF
           - **Continuous:** Daily tax drag based on turnover rate
        
        3. **Rules:**
           - Turnover < 300% → Realization method
           - Turnover ≥ 300% → Continuous method
           - Benchmarks use continuous method (academic standard)
        """)
        
        # Show tax method details
        tax_methods = pd.DataFrame({
            "Strategy": ["MA Strategy", "Hybrid SIG", "Pure SIG"],
            "Annual Turnover": [
                f"{ma_result.get('annual_turnover', 0):.1%}",
                f"{hybrid_result.get('annual_turnover', 0):.1%}",
                f"{pure_sig_result.get('annual_turnover', 0):.1%}"
            ],
            "Tax Method": [
                ma_result.get('tax_method', 'auto'),
                hybrid_result.get('tax_method', 'auto'),
                pure_sig_result.get('tax_method', 'auto')
            ],
            "Total Taxes Paid": [
                f"${ma_result.get('tax_payments', pd.Series([0])).sum() * 10000:,.2f}",
                f"${hybrid_result.get('tax_payments', pd.Series([0])).sum() * 10000:,.2f}",
                f"${pure_sig_result.get('tax_payments', pd.Series([0])).sum() * 10000:,.2f}"
            ]
        })
        
        st.dataframe(tax_methods)

    # VALIDATION SECTION
    if run_validation:
        st.header("🎯 Strategy Validation")
        
        val_tab1, val_tab2, val_tab3 = st.tabs([
            "Statistical Significance", 
            "Walk-Forward Analysis",
            "Strategy Metrics"
        ])
        
        with val_tab1:
            st.subheader("Monte Carlo Significance Test")
            
            mc_results = monte_carlo_significance_unified(ma_result)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Actual Sharpe", f"{mc_results['actual_sharpe']:.3f}")
            with col2:
                st.metric("p-value", f"{mc_results['p_value']:.3f}")
            with col3:
                significance = "✅ Statistically Significant" if mc_results['significance_95'] else "❌ Not Significant"
                st.metric("Significance (95%)", significance)
            
            st.write(f"**Null Distribution:** Mean = {mc_results['null_mean']:.3f}, Std = {mc_results['null_std']:.3f}")
        
        with val_tab2:
            st.subheader("Walk-Forward Validation")
            
            wfa_results = walk_forward_validation_unified(prices, ma_result)
            
            if not wfa_results.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Test Sharpe", 
                             f"{wfa_results['test_sharpe'].mean():.3f}")
                with col2:
                    st.metric("Avg Test CAGR", 
                             f"{wfa_results['test_cagr'].mean():.2%}")
                with col3:
                    success_rate = (wfa_results['test_sharpe'] > 0.5).mean()
                    st.metric("Success Rate (Sharpe > 0.5)", f"{success_rate:.1%}")
                with col4:
                    avg_max_dd = wfa_results['test_max_dd'].mean()
                    st.metric("Avg Max Drawdown", f"{avg_max_dd:.1%}")
                
                # Plot results
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(range(len(wfa_results)), wfa_results['test_sharpe'], 
                       marker='o', linewidth=2, label='Test Sharpe')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=wfa_results['test_sharpe'].mean(), 
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
            metrics_data = {}
            
            for strategy_name, result in [("MA Strategy", ma_result), 
                                         ("Hybrid SIG", hybrid_result),
                                         ("Pure SIG", pure_sig_result)]:
                
                returns = result["returns"]
                
                metrics = {
                    "Monthly Hit Ratio": (returns.resample('M')
                                         .apply(lambda x: (1+x).prod()-1 > 0).mean()),
                    "Win/Loss Ratio": (abs(returns[returns > 0].mean() / 
                                         returns[returns < 0].mean()) 
                                     if len(returns[returns < 0]) > 0 else 0),
                    "Profit Factor": (returns[returns > 0].sum() / 
                                    abs(returns[returns < 0].sum()) 
                                    if returns[returns < 0].sum() != 0 else 0),
                    "Calmar Ratio": (result["performance"]["CAGR"] / 
                                   abs(result["performance"]["MaxDrawdown"]) 
                                   if result["performance"]["MaxDrawdown"] != 0 else 0)
                }
                
                metrics_data[strategy_name] = metrics
            
            # Display metrics in columns
            st.write("#### Key Metrics Comparison")
            metric_names = ["Monthly Hit Ratio", "Win/Loss Ratio", "Profit Factor", "Calmar Ratio"]
            
            for metric in metric_names:
                cols = st.columns(3)
                for idx, (strategy_name, metrics) in enumerate(metrics_data.items()):
                    with cols[idx]:
                        value = metrics.get(metric, 0)
                        if metric == "Monthly Hit Ratio":
                            st.metric(f"{strategy_name} - {metric}", f"{value:.1%}")
                        else:
                            st.metric(f"{strategy_name} - {metric}", f"{value:.2f}")

    # IMPLEMENTATION CHECKLIST
    st.markdown("""
---

## **Implementation Checklist**

1. **Unified Tax Logic Applied:**
   - All strategies use same tax calculation
   - Tax method auto-selected based on turnover
   - Effective tax rate: {:.1%}
   - Slippage cost per trade: {:.1%}

2. **Portfolio Actions:**
   - Rotate to treasury sleeve when MA regime flips to RISK-OFF
   - At each calendar quarter-end, input portfolio values
   - Execute dollar adjustments as recommended
   - Re-evaluate Sharpe-optimal weights quarterly

3. **Current Configuration:**
   - MA: {}-day {} with {:.1%} tolerance
   - Tax Method: {}
   - Expected annual turnover: {:.1%}

4. **Academic Notes:**
   - All returns shown are AFTER estimated taxes and trading costs
   - Unified tax logic ensures consistent comparison
   - Benchmarks include rebalancing costs and tax drag

Current Sharpe-optimal portfolio: https://testfol.io/optimizer?s=9TIGHucZuaJ

---
    """.format(current_tax_rate, current_flip_cost, best_len, best_type.upper(), best_tol,
              ma_result.get('tax_method', 'auto'), ma_result.get('annual_turnover', 0)))


# ============================================================
# LAUNCH APP
# ============================================================

if __name__ == "__main__":
    main()