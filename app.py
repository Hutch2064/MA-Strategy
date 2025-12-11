import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import io
from scipy.optimize import minimize
import datetime
from statsmodels.tsa.stattools import adfuller

# ============================================================
# CONFIG
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

FLIP_COST = 0.0075

# Starting weights inside the SIG engine (unchanged)
START_RISKY = 0.70
START_SAFE  = 0.30


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
    px = price.values
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
# DATA-ADAPTIVE MA SELECTION (NEW)
# ============================================================

def calculate_data_quality_metrics(prices, portfolio_index):
    """
    Assess data quality to determine appropriate MA bounds
    """
    metrics = {}
    
    # 1. Check data completeness
    daily_returns = portfolio_index.pct_change().dropna()
    if len(daily_returns) > 0:
        days_diff = (daily_returns.index[-1] - daily_returns.index[0]).days
        if days_diff > 0:
            trading_days_per_year = len(daily_returns) / (days_diff / 365.25)
            metrics['completeness'] = min(1.0, trading_days_per_year / 252)  # 252 = typical trading days
        else:
            metrics['completeness'] = 0.5
    else:
        metrics['completeness'] = 0.5
    
    # 2. Check volatility stability
    if len(daily_returns) > 500:
        # Split into 3 periods
        split_point1 = len(daily_returns) // 3
        split_point2 = 2 * len(daily_returns) // 3
        
        vol1 = daily_returns.iloc[:split_point1].std()
        vol2 = daily_returns.iloc[split_point1:split_point2].std()
        vol3 = daily_returns.iloc[split_point2:].std()
        
        # Simple volatility stability measure
        vol_stability = 1 - (max(vol1, vol2, vol3) - min(vol1, vol2, vol3)) / daily_returns.std()
        metrics['vol_stability'] = max(0, min(1, vol_stability))
    else:
        metrics['vol_stability'] = 0.7  # Default assumption
    
    # 3. Check trend strength (ADF test for stationarity)
    try:
        if len(daily_returns) > 100:
            adf_result = adfuller(daily_returns.dropna())
            metrics['stationarity_p'] = adf_result[1]
        else:
            metrics['stationarity_p'] = 0.05  # Default
    except:
        metrics['stationarity_p'] = 0.05  # Default
    
    # 4. Calculate volatility
    if len(daily_returns) > 0:
        metrics['annual_vol'] = daily_returns.std() * np.sqrt(252)
    else:
        metrics['annual_vol'] = 0.2  # Default
    
    return metrics


def regime_adaptive_ma_selection(prices, risk_on_weights, flip_cost, regime_history=None):
    """
    Select MA parameters based on current regime characteristics
    """
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    
    # Calculate data quality metrics
    metrics = calculate_data_quality_metrics(prices, portfolio_index)
    
    # Determine adaptive MA length bounds
    total_days = len(portfolio_index)
    min_ma_days = max(20, int(0.1 * total_days))  # At least 10% of data
    max_ma_days = min(300, int(0.5 * total_days))  # At most 50% of data
    
    # Adjust based on data quality
    if metrics['completeness'] < 0.8:
        # Incomplete data: use shorter, more responsive MA
        max_ma_days = min(max_ma_days, 150)
    
    if metrics['stationarity_p'] < 0.05:
        # Stationary returns: longer MA for smoother signals
        min_ma_days = max(min_ma_days, 50)
    
    # If we have regime history, use it to inform selection
    if regime_history is not None and len(regime_history) > 0:
        avg_regime_duration = regime_history['Duration (days)'].mean()
        # Set MA length proportional to typical regime duration
        target_ma = min(max_ma_days, max(min_ma_days, int(avg_regime_duration * 0.5)))
    else:
        # Default to middle of range
        target_ma = (min_ma_days + max_ma_days) // 2
    
    # Determine MA type based on volatility
    if metrics['vol_stability'] < 0.6:
        # Volatile periods: EMA responds faster
        ma_type = "ema"
    else:
        # Stable periods: SMA is more robust
        ma_type = "sma"
    
    # Simple tolerance: wider in volatile markets
    if metrics['annual_vol'] > 0.25:  # >25% annual vol
        tolerance = 0.04
    elif metrics['annual_vol'] > 0.15:
        tolerance = 0.03
    else:
        tolerance = 0.02
    
    return target_ma, ma_type, tolerance, metrics


def robust_ma_validation(prices, risk_on_weights, risk_off_weights, flip_cost, 
                         candidate_params, n_folds=5):
    """
    Robust cross-validation for MA parameters
    """
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    
    # Time-series cross-validation (blocked to preserve sequence)
    fold_size = len(prices) // n_folds
    if fold_size < 180:  # Minimum fold size (approx 9 months)
        n_folds = max(2, len(prices) // 180)
        fold_size = len(prices) // n_folds
    
    cv_scores = {}
    
    for L, ma_type, tol in candidate_params:
        sharpe_scores = []
        
        for fold in range(n_folds):
            # Create expanding window: train on all data before test fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(prices)
            
            # Training data: everything before test
            train_indices = list(range(test_start))
            if len(train_indices) < 180:  # Need minimum training data
                continue
            
            train_portfolio = portfolio_index.iloc[train_indices]
            train_prices = prices.iloc[train_indices]
            
            # Test on the fold
            test_indices = list(range(test_start, test_end))
            test_portfolio = portfolio_index.iloc[test_indices]
            test_prices = prices.iloc[test_indices]
            
            # Generate signal on full sequence up to test end
            full_ma = compute_ma_matrix(portfolio_index.iloc[:test_end], [L], ma_type)[L]
            full_signal = generate_testfol_signal_vectorized(
                portfolio_index.iloc[:test_end], full_ma, tol
            )
            
            # Use only the test portion
            test_signal = full_signal.iloc[test_indices]
            
            if test_signal.sum() == 0 or test_signal.sum() == len(test_signal):
                continue  # Skip constant signals
            
            # Backtest on test fold
            result = backtest(test_prices, test_signal, risk_on_weights, 
                            risk_off_weights, flip_cost, ma_flip_multiplier=4.0)
            sharpe_scores.append(result["performance"]["Sharpe"])
        
        if sharpe_scores:
            cv_scores[(L, ma_type, tol)] = {
                'mean_sharpe': np.mean(sharpe_scores),
                'std_sharpe': np.std(sharpe_scores),
                'n_folds': len(sharpe_scores),
                'robustness': np.mean(sharpe_scores) / max(0.001, np.std(sharpe_scores))
            }
    
    # Select best with robustness penalty
    best_score = -1e9
    best_params = None
    
    for params, scores in cv_scores.items():
        # Penalize parameters with high variance
        robustness_penalty = 1.0 / (1.0 + scores['std_sharpe'])
        adjusted_score = scores['mean_sharpe'] * robustness_penalty
        
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_params = params
    
    return best_params, cv_scores


def adaptive_ma_optimization(prices, risk_on_weights, risk_off_weights, flip_cost):
    """
    Main adaptive MA optimization that handles limited data
    """
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    total_days = len(prices)
    total_years = total_days / 252
    
    # STAGE 1: Generate candidate parameters based on data characteristics
    regime_history = None  # Could load from your CSV file here
    
    # Get adaptive bounds and initial guess
    adaptive_ma, ma_type, tolerance, metrics = regime_adaptive_ma_selection(
        prices, risk_on_weights, flip_cost, regime_history
    )
    
    # Create candidate parameter space
    candidate_params = []
    
    # Center around adaptive MA
    base_lengths = [adaptive_ma]
    if adaptive_ma > 50:
        base_lengths.extend([adaptive_ma - 25, adaptive_ma + 25])
    if adaptive_ma > 100:
        base_lengths.extend([adaptive_ma - 50, adaptive_ma + 50])
    
    # Ensure reasonable bounds
    min_len = max(20, int(0.1 * total_days))
    max_len = min(300, int(0.5 * total_days))
    
    base_lengths = [l for l in base_lengths if min_len <= l <= max_len]
    
    # Add both MA types
    for L in base_lengths:
        candidate_params.append((L, "sma", tolerance))
        candidate_params.append((L, "ema", tolerance))
    
    # Add tolerance variations
    tolerance_variants = []
    for L, ma_type, _ in candidate_params[:10]:  # Limit to first 10 to keep manageable
        for tol in [0.01, 0.02, 0.03, 0.04]:
            tolerance_variants.append((L, ma_type, tol))
    
    candidate_params.extend(tolerance_variants[:min(10, len(tolerance_variants))])
    
    # STAGE 2: Robust validation
    if total_years >= 2:  # Need at least 2 years for meaningful CV
        best_params, cv_scores = robust_ma_validation(
            prices, risk_on_weights, risk_off_weights, flip_cost, 
            candidate_params, n_folds=min(5, int(total_years))
        )
    else:
        # Very limited data: use simplest reasonable parameters
        best_params = (adaptive_ma, ma_type, tolerance)
        cv_scores = {}
    
    # STAGE 3: Out-of-sample stability check (if enough data)
    stability_check = {}
    if total_years >= 4:
        # Reserve last year for out-of-sample test
        split_idx = int(0.75 * total_days)
        train_prices = prices.iloc[:split_idx]
        test_prices = prices.iloc[split_idx:]
        
        # Train on first 75%
        train_portfolio = build_portfolio_index(train_prices, risk_on_weights)
        L, ma_type, tol = best_params
        
        train_ma = compute_ma_matrix(train_portfolio, [L], ma_type)[L]
        train_signal = generate_testfol_signal_vectorized(train_portfolio, train_ma, tol)
        
        # Test on last 25%
        test_portfolio = build_portfolio_index(test_prices, risk_on_weights)
        full_portfolio = build_portfolio_index(prices, risk_on_weights)
        
        full_ma = compute_ma_matrix(full_portfolio, [L], ma_type)[L]
        full_signal = generate_testfol_signal_vectorized(full_portfolio, full_ma, tol)
        
        test_signal = full_signal.iloc[split_idx:]
        
        train_result = backtest(train_prices, train_signal, risk_on_weights, 
                              risk_off_weights, flip_cost, ma_flip_multiplier=4.0)
        test_result = backtest(test_prices, test_signal, risk_on_weights, 
                             risk_off_weights, flip_cost, ma_flip_multiplier=4.0)
        
        stability_check = {
            'train_sharpe': train_result["performance"]["Sharpe"],
            'test_sharpe': test_result["performance"]["Sharpe"],
            'oos_ratio': test_result["performance"]["Sharpe"] / max(0.001, train_result["performance"]["Sharpe"]),
            'train_days': len(train_prices),
            'test_days': len(test_prices)
        }
    
    # STAGE 4: Final backtest with selected parameters
    L, ma_type, tol = best_params
    full_ma = compute_ma_matrix(portfolio_index, [L], ma_type)[L]
    full_signal = generate_testfol_signal_vectorized(portfolio_index, full_ma, tol)
    final_result = backtest(prices, full_signal, risk_on_weights, risk_off_weights,
                          flip_cost, ma_flip_multiplier=4.0)
    
    # Create results summary
    results_summary = {
        'selected_params': best_params,
        'data_metrics': metrics,
        'cv_scores': cv_scores,
        'stability_check': stability_check,
        'final_performance': final_result["performance"]
    }
    
    return best_params, final_result, results_summary


# ============================================================
# SIG ENGINE — NOW USING CALENDAR QUARTER-ENDS (B1)
# ============================================================

def run_sig_engine(
    risk_on_returns,
    risk_off_returns,
    target_quarter,
    ma_signal,
    pure_sig_rw=None,
    pure_sig_sw=None,
    flip_cost=FLIP_COST,
    quarter_end_dates=None,   # <-- must be mapped_q_ends
    quarterly_multiplier=2.0,  # NEW: 2x for Pure SIG, 2x for Hybrid SIG (quarterly part)
    ma_flip_multiplier=4.0     # NEW: 4x for Hybrid SIG when MA flips
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

    frozen_risky = None
    frozen_safe  = None

    equity_curve = []
    risky_w_series = []
    safe_w_series = []
    risky_val_series = []
    safe_val_series = []
    rebalance_events = 0
    rebalance_dates = []

    for i in range(n):
        date = dates[i]
        r_on = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])
        
        # ============================================
        # FIX: Apply MA flip costs BEFORE checking regime
        # ============================================
        if i > 0 and flip_mask.iloc[i]:  # Skip first day (no diff)
            eq *= (1 - flip_cost * ma_flip_multiplier)  # 4x cost for Hybrid SIG
        # ============================================

        if ma_on:

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

            # Rebalance ON quarter-end date (correct logic)
            if date in quarter_end_set:

                # Identify actual quarter start (previous quarter end)
                prev_qs = [qd for qd in quarter_end_dates if qd < date]

                if prev_qs:
                    prev_q = prev_qs[-1]

                    idx_prev = dates.get_loc(prev_q)

                    # Risky sleeve at the start of this quarter
                    risky_at_qstart = risky_val_series[idx_prev]

                    # Quarterly growth target
                    goal_risky = risky_at_qstart * (1 + target_quarter)

                    # --- Apply SIG logic (unchanged) ---
                    if risky_val > goal_risky:
                        excess = risky_val - goal_risky
                        risky_val -= excess
                        safe_val  += excess
                        rebalance_dates.append(date)

                    elif risky_val < goal_risky:
                        needed = goal_risky - risky_val
                        move = min(needed, safe_val)
                        safe_val -= move
                        risky_val += move
                        rebalance_dates.append(date)

                    # Apply quarterly fee with multiplier (2x for Pure SIG, 2x for Hybrid SIG quarterly part)
                    eq *= (1 - flip_cost * quarterly_multiplier)  # Remove target_quarter!

            # Update equity
            eq = risky_val + safe_val
            risky_w = risky_val / eq
            safe_w  = safe_val  / eq

        else:
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

    return (
        pd.Series(equity_curve, index=dates),
        pd.Series(risky_w_series, index=dates),
        pd.Series(safe_w_series, index=dates),
        rebalance_dates
    )


# ============================================================
# BACKTEST ENGINE
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


def backtest(prices, signal, risk_on_weights, risk_off_weights, flip_cost, ma_flip_multiplier=4.0):
    simple = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    # MA flip costs with 4x multiplier for MA Strategy
    flip_costs = np.where(flip_mask, -flip_cost * ma_flip_multiplier, 0.0)
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
# BUY & HOLD ENGINE WITH QUARTERLY REBALANCE
# ============================================================

def buy_and_hold_with_rebalance(prices, weights_dict, flip_cost, quarter_end_dates):
    """Buy & Hold with quarterly rebalancing to target weights (1x flip cost)"""
    simple = prices.pct_change().fillna(0)
    
    # Initial investment
    eq = 10000.0
    equity_curve = [eq]
    
    # Fast lookup for quarter ends
    quarter_end_set = set(quarter_end_dates)
    
    for i in range(1, len(prices)):
        date = prices.index[i]
        
        # Calculate portfolio return for the day
        daily_return = 0.0
        for asset, weight in weights_dict.items():
            if asset in simple.columns:
                daily_return += simple.iloc[i][asset] * weight
        
        # Apply daily return
        eq *= (1 + daily_return)
        
        # Apply quarterly rebalancing cost (1x flip cost)
        if date in quarter_end_set:
            eq *= (1 - flip_cost)
        
        equity_curve.append(eq)
    
    eq_series = pd.Series(equity_curve, index=prices.index)
    returns = eq_series.pct_change().fillna(0)
    
    return {
        "returns": returns,
        "equity_curve": eq_series,
        "performance": compute_performance(returns, eq_series)
    }


# ============================================================
# WALK-FORWARD OPTIMIZATION (UPDATED)
# ============================================================

def run_grid_search(prices, risk_on_weights, risk_off_weights, flip_cost):
    """
    Updated main optimization - uses adaptive, robust MA selection
    """
    return adaptive_ma_optimization(prices, risk_on_weights, risk_off_weights, flip_cost)


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
# VALIDATION FUNCTIONS
# ============================================================

def calculate_max_dd(prices_series):
    """Calculate maximum drawdown for a price series"""
    if len(prices_series) == 0:
        return 0
    cumulative = (1 + prices_series.pct_change().fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() if len(drawdown) > 0 else 0

def walk_forward_validation(prices, strategy_returns, window_years=3):
    """
    Simple walk-forward validation
    """
    results = []
    total_days = len(prices)
    
    # Adaptive window based on available data
    min_days_needed = 252  # At least 1 year
    if total_days < min_days_needed * 2:  # Need at least 2x for train/test
        return pd.DataFrame()  # Not enough data
    
    window_days = min(window_years * 252, total_days // 2)
    
    for start in range(0, total_days - window_days, 126):  # Slide every 6 months
        train_end = start + window_days
        test_end = min(train_end + 252, total_days)  # 1 year test or less
        
        if test_end - train_end < 63:  # Skip if test too short
            continue
            
        train_data = prices.iloc[start:train_end]
        test_data = prices.iloc[train_end:test_end]
        
        # Simple: just track how strategy performs out-of-sample
        test_returns = strategy_returns.iloc[train_end:test_end]
        
        if len(test_returns) > 0 and test_returns.std() > 0:
            test_sharpe = test_returns.mean() / test_returns.std() * np.sqrt(252)
            test_cagr = (1 + test_returns).prod() ** (252/len(test_returns)) - 1
            
            results.append({
                'train_period': f"{train_data.index[0].date()} to {train_data.index[-1].date()}",
                'test_period': f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
                'test_sharpe': test_sharpe,
                'test_cagr': test_cagr,
                'test_length_days': len(test_returns)
            })
    
    return pd.DataFrame(results)

def calculate_consistency_metrics(strategy_returns):
    """
    Calculate hit ratio and other consistency metrics
    """
    if len(strategy_returns) == 0:
        return {
            'monthly_hit_ratio': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'win_loss_ratio': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_consecutive_wins': 0,
            'avg_consecutive_losses': 0
        }
    
    # Monthly hit ratio
    monthly_returns = strategy_returns.resample('M').apply(lambda x: (1+x).prod()-1)
    hit_ratio = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0
    
    # Win/Loss metrics
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    return {
        'monthly_hit_ratio': hit_ratio,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'max_consecutive_wins': 0,  # Simplified
        'max_consecutive_losses': 0,
        'avg_consecutive_wins': 0,
        'avg_consecutive_losses': 0
    }

def parameter_sensitivity_analysis(prices, base_params, risk_on_weights, risk_off_weights, flip_cost):
    """
    Test how sensitive results are to parameter changes
    """
    if len(prices) == 0:
        return {
            'length_sensitivity': pd.DataFrame(),
            'tolerance_sensitivity': pd.DataFrame()
        }
    
    base_len, base_type, base_tol = base_params
    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    
    # Test different MA lengths (adaptive based on data)
    max_possible = min(300, len(portfolio_index) - 1)
    test_lengths = [max(20, min(l, max_possible)) for l in [50, 100, 150, 200, 250, 300] if min(l, max_possible) >= 20]
    
    length_results = []
    for length in test_lengths:
        ma = compute_ma_matrix(portfolio_index, [length], base_type)[length]
        signal = generate_testfol_signal_vectorized(portfolio_index, ma, base_tol)
        result = backtest(prices, signal, risk_on_weights, risk_off_weights, flip_cost, ma_flip_multiplier=4.0)
        perf = result["performance"]
        length_results.append({
            'length': length,
            'sharpe': perf['Sharpe'],
            'cagr': perf['CAGR'],
            'max_dd': perf['MaxDrawdown']
        })
    
    # Test different tolerances
    tol_results = []
    ma = compute_ma_matrix(portfolio_index, [min(base_len, max_possible)], base_type)[min(base_len, max_possible)]
    for tol in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        signal = generate_testfol_signal_vectorized(portfolio_index, ma, tol)
        result = backtest(prices, signal, risk_on_weights, risk_off_weights, flip_cost, ma_flip_multiplier=4.0)
        perf = result["performance"]
        tol_results.append({
            'tolerance': tol,
            'sharpe': perf['Sharpe'],
            'cagr': perf['CAGR'],
            'max_dd': perf['MaxDrawdown']
        })
    
    return {
        'length_sensitivity': pd.DataFrame(length_results),
        'tolerance_sensitivity': pd.DataFrame(tol_results)
    }


# ============================================================
# STREAMLIT APP
# ============================================================

def main():

    st.set_page_config(page_title="Portfolio MA Regime Strategy", layout="wide")
    st.title("Portfolio Strategy")

    # Backtest inputs unchanged...
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

    st.sidebar.header("Quarterly Portfolio Values")
    qs_cap_1 = st.sidebar.number_input("Taxable – Portfolio Value at Last Rebalance ($)", min_value=0.0, value=75815.26, step=100.0)
    qs_cap_2 = st.sidebar.number_input("Tax-Sheltered – Portfolio Value at Last Rebalance ($)", min_value=0.0, value=10074.83, step=100.0)
    qs_cap_3 = st.sidebar.number_input("Joint – Portfolio Value at Last Rebalance ($)", min_value=0.0, value=4189.76, step=100.0)

    st.sidebar.header("Current Portfolio Values (Today)")
    real_cap_1 = st.sidebar.number_input("Taxable – Portfolio Value Today ($)", min_value=0.0, value=73165.78, step=100.0)
    real_cap_2 = st.sidebar.number_input("Tax-Sheltered – Portfolio Value Today ($)", min_value=0.0, value=9264.46, step=100.0)
    real_cap_3 = st.sidebar.number_input("Joint – Portfolio Value Today ($)", min_value=0.0, value=4191.56, step=100.0)

    # Add validation toggle to sidebar
    st.sidebar.header("Validation Settings")
    run_validation = st.sidebar.checkbox("Run Validation Suite", value=True)

    run_clicked = st.sidebar.button("Run Backtest & Optimize")
    if not run_clicked:
        st.stop()

    risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",")]
    risk_on_weights_list = [float(x) for x in risk_on_weights_str.split(",")]
    risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))

    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))

    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers))
    end_val = end if end.strip() else None

    prices = load_price_data(all_tickers, start, end_val).dropna(how="any")
    
    # Check if we have any data
    if len(prices) == 0:
        st.error("No data loaded. Please check your ticker symbols and date range.")
        st.stop()
    
    st.info(f"Loaded {len(prices)} trading days of data from {prices.index[0].date()} to {prices.index[-1].date()}")

    # RUN ADAPTIVE MA OPTIMIZATION
    best_cfg, best_result, optimization_summary = adaptive_ma_optimization(
        prices, risk_on_weights, risk_off_weights, FLIP_COST
    )
    best_len, best_type, best_tol = best_cfg
    sig = best_result["signal"]
    perf = best_result["performance"]

    latest_signal = sig.iloc[-1]
    current_regime = "RISK-ON" if latest_signal else "RISK-OFF"

    st.subheader(f"Current MA Regime: {current_regime}")
    
    # Display optimization details
    st.write(f"**Optimization Method:** Adaptive MA Selection with Robust Validation")
    st.write(f"**MA Type:** {best_type.upper()}  —  **Length:** {best_len}  —  **Tolerance:** {best_tol:.2%}")
    
    # Show data quality metrics
    with st.expander("View Data Quality & Optimization Details"):
        st.subheader("Data Quality Metrics")
        metrics_df = pd.DataFrame([optimization_summary['data_metrics']]).T
        metrics_df.columns = ['Value']
        st.dataframe(metrics_df)
        
        if optimization_summary['cv_scores']:
            st.subheader("Cross-Validation Results")
            cv_df = pd.DataFrame(optimization_summary['cv_scores']).T
            st.dataframe(cv_df)
        
        if optimization_summary['stability_check']:
            st.subheader("Out-of-Sample Stability")
            st.write(f"Train Sharpe: {optimization_summary['stability_check']['train_sharpe']:.3f}")
            st.write(f"Test Sharpe: {optimization_summary['stability_check']['test_sharpe']:.3f}")
            st.write(f"OOS Ratio: {optimization_summary['stability_check']['oos_ratio']:.2f}")
            if optimization_summary['stability_check']['oos_ratio'] < 0.7:
                st.warning("Out-of-sample performance is significantly lower than in-sample. Consider simpler parameters.")

    portfolio_index = build_portfolio_index(prices, risk_on_weights)
    opt_ma = compute_ma_matrix(portfolio_index, [best_len], best_type)[best_len]

    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252) if len(sig) > 0 else 0

    simple_rets = prices.pct_change().fillna(0)

    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w

    risk_on_eq = (1 + risk_on_simple).cumprod()
    risk_on_perf = compute_performance(risk_on_simple, risk_on_eq)

    risk_on_px = prices[[t for t in risk_on_tickers if t in prices.columns]].dropna()
    if len(risk_on_px) > 0:
        risk_on_rets = risk_on_px.pct_change().dropna()
    else:
        risk_on_rets = pd.DataFrame()

    if len(risk_on_rets) > 0:
        mu = risk_on_rets.mean().values
        cov = risk_on_rets.cov().values + np.eye(len(mu)) * 1e-10

        def neg_sharpe(w):
            r = np.dot(mu, w)
            v = np.sqrt(np.dot(w.T, cov @ w))
            return -(r / v) if v > 0 else 1e9

        n = len(mu)
        bounds = [(0, 1)] * n
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        res = minimize(neg_sharpe, np.ones(n) / n, bounds=bounds, constraints=cons)
        w_opt = res.x

        sharp_returns = (risk_on_rets * w_opt).sum(axis=1)
        sharp_eq = (1 + sharp_returns).cumprod()
        sharp_perf = compute_performance(sharp_returns, sharp_eq)
    else:
        w_opt = np.array([])
        sharp_returns = pd.Series([], dtype=float)
        sharp_eq = pd.Series([], dtype=float)
        sharp_perf = compute_performance(pd.Series([], dtype=float), pd.Series([], dtype=float))

    # ============================================================
    # REAL CALENDAR QUARTER LOGIC BEGINS HERE
    # ============================================================

    dates = prices.index
    # ============================================================
    # TRUE CALENDAR QUARTER-ENDS (academically correct)
    # ============================================================

    dates = prices.index

    # 1. Generate TRUE calendar quarter-end dates
    true_q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq='Q')

    # 2. Map each to the actual last trading day
    mapped_q_ends = []
    for qd in true_q_ends:
        valid_dates = dates[dates <= qd]
        if len(valid_dates) > 0:
            mapped_q_ends.append(valid_dates.max())

    mapped_q_ends = pd.to_datetime(mapped_q_ends)

    # -----------------------------------------------------------
    # FIXED: TRUE CALENDAR QUARTER LOGIC (never depends on prices)
    # -----------------------------------------------------------

    today_date = pd.Timestamp.today().normalize()

    # 1. Next calendar quarter-end
    true_next_q = pd.date_range(start=today_date, periods=2, freq="Q")[0]
    next_q_end = true_next_q

    # 2. Most recent completed quarter-end
    true_prev_q = pd.date_range(end=today_date, periods=2, freq="Q")[0]
    past_q_end = true_prev_q

    # 3. Days remaining until next rebalance
    days_to_next_q = (next_q_end - today_date).days
    
    # ============================================================
    # HYBRID SIG ENGINE USING REAL CALENDAR QUARTERS
    # ============================================================

    # Annualized CAGR → quarterly target unchanged
    if len(risk_on_eq) > 0 and risk_on_eq.iloc[0] != 0:
        bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
        quarterly_target = (1 + bh_cagr) ** (1/4) - 1
    else:
        bh_cagr = 0
        quarterly_target = 0

    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w

    # PURE SIG (always RISK-ON) - 2x flip costs quarterly
    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)

    pure_sig_eq, pure_sig_rw, pure_sig_sw, pure_sig_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        pure_sig_signal,
        quarter_end_dates=mapped_q_ends,
        quarterly_multiplier=2.0,  # 2x for Pure SIG
        ma_flip_multiplier=0.0     # No MA flips for Pure SIG
    )

    # HYBRID SIG (MA Filter) - 2x quarterly + 4x MA flips = 6x total
    hybrid_eq, hybrid_rw, hybrid_sw, hybrid_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        sig,  # <-- CRITICAL: Uses the SAME optimized signal
        pure_sig_rw=pure_sig_rw,
        pure_sig_sw=pure_sig_sw,
        quarter_end_dates=mapped_q_ends,
        quarterly_multiplier=2.0,  # 2x quarterly part
        ma_flip_multiplier=4.0     # 4x when MA flips
    )
    
    
    # ============================================================
    # BUY & HOLD WITH QUARTERLY REBALANCE (1x flip costs)
    # ============================================================
    
    # Combine risk_on and risk_off weights for Buy & Hold
    bh_weights = risk_on_weights.copy()
    bh_weights.update(risk_off_weights)
    
    # Run Buy & Hold with quarterly rebalance (1x flip costs)
    bh_rebalance_result = buy_and_hold_with_rebalance(
        prices, 
        bh_weights, 
        FLIP_COST, 
        mapped_q_ends
    )
    
    # ============================================================
    # DISPLAY ACTUAL HYBRID SIG REBALANCE DATES (FULL HISTORY)
    # ============================================================
    if len(hybrid_rebals) > 0:
        reb_df = pd.DataFrame({"Rebalance Date": pd.to_datetime(hybrid_rebals)})
        st.subheader("Hybrid SIG – Actual Rebalance Dates (Historical)")
        st.dataframe(reb_df)
    else:
        st.subheader("Hybrid SIG – Actual Rebalance Dates (Historical)")
        st.write("No hybrid SIG rebalances occurred during the backtest.")

    # Quarter start should follow the last actual SIG rebalance
    if len(hybrid_rebals) > 0:
        quarter_start_date = hybrid_rebals[-1]
    else:
        quarter_start_date = dates[0] if len(dates) > 0 else None

    st.subheader("Strategy Summary")
    # Display last actual SIG rebalance instead of quarter start
    if len(hybrid_rebals) > 0:
        last_reb = hybrid_rebals[-1]
        st.write(f"**Last Rebalance:** {last_reb.strftime('%Y-%m-%d')}")
    else:
        st.write("**Quarter start (last SIG rebalance):** None yet")
    st.write(f"**Next Rebalance:** {next_q_end.date()} ({days_to_next_q} days)")

    # Quarter-progress calculations
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

    st.write(f"**Quarterly Target Growth Rate:** {quarterly_target:.2%}")

    prog_df = pd.concat([
        pd.DataFrame.from_dict(prog_1, orient='index', columns=['Taxable']),
        pd.DataFrame.from_dict(prog_2, orient='index', columns=['Tax-Sheltered']),
        pd.DataFrame.from_dict(prog_3, orient='index', columns=['Joint']),
    ], axis=1)

    prog_df.loc["Gap (%)"] = prog_df.loc["Gap (%)"].apply(lambda x: f"{x:.2%}")
    st.dataframe(prog_df)

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

    # ADVANCED METRICS (unchanged)
    def time_in_drawdown(dd): return (dd < 0).mean() if len(dd) > 0 else 0
    def mar(c, dd): return c / abs(dd) if dd != 0 else 0
    def ulcer(dd): return np.sqrt((dd**2).mean()) if len(dd) > 0 and (dd**2).mean() != 0 else 0
    def pain_gain(c, dd): return c / ulcer(dd) if ulcer(dd) != 0 else 0

    def compute_stats(perf, returns, dd, flips, tpy):
        return {
            "CAGR": perf["CAGR"],
            "Volatility": perf["Volatility"],
            "Sharpe": perf["Sharpe"],
            "MaxDD": perf["MaxDrawdown"],
            "Total": perf["TotalReturn"],
            "MAR": mar(perf["CAGR"], perf["MaxDrawdown"]),
            "TID": time_in_drawdown(dd),
            "PainGain": pain_gain(perf["CAGR"], dd),
            "Skew": returns.skew() if len(returns) > 0 else 0,
            "Kurtosis": returns.kurt() if len(returns) > 0 else 0,
            "Trades/year": tpy,
        }

    hybrid_simple = hybrid_eq.pct_change().fillna(0) if len(hybrid_eq) > 0 else pd.Series([], dtype=float)
    hybrid_perf = compute_performance(hybrid_simple, hybrid_eq)

    pure_sig_simple = pure_sig_eq.pct_change().fillna(0) if len(pure_sig_eq) > 0 else pd.Series([], dtype=float)
    pure_sig_perf = compute_performance(pure_sig_simple, pure_sig_eq)

    strat_stats = compute_stats(
        perf,
        best_result["returns"],
        perf["DD_Series"],
        best_result["flip_mask"],
        trades_per_year,
    )

    risk_stats = compute_stats(
        risk_on_perf,
        risk_on_simple,
        risk_on_perf["DD_Series"],
        np.zeros(len(risk_on_simple), dtype=bool) if len(risk_on_simple) > 0 else np.array([], dtype=bool),
        0,
    )

    hybrid_stats = compute_stats(
        hybrid_perf,
        hybrid_simple,
        hybrid_perf["DD_Series"],
        np.zeros(len(hybrid_simple), dtype=bool) if len(hybrid_simple) > 0 else np.array([], dtype=bool),
        0,
    )

    pure_sig_stats = compute_stats(
        pure_sig_perf,
        pure_sig_simple,
        pure_sig_perf["DD_Series"],
        np.zeros(len(pure_sig_simple), dtype=bool) if len(pure_sig_simple) > 0 else np.array([], dtype=bool),
        0,
    )
    
    # Buy & Hold stats (with quarterly rebalance)
    bh_stats = compute_stats(
        bh_rebalance_result["performance"],
        bh_rebalance_result["returns"],
        bh_rebalance_result["performance"]["DD_Series"],
        np.zeros(len(bh_rebalance_result["returns"]), dtype=bool) if len(bh_rebalance_result["returns"]) > 0 else np.array([], dtype=bool),
        0,
    )

    # STAT TABLE (updated with Buy & Hold with rebalance)
    st.subheader("MA vs Sharpe-Optimal vs Buy & Hold (with rebalance) vs Hybrid SIG/MA vs Pure SIG")
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
    ]

    def fmt_pct(x): return f"{x:.2%}" if pd.notna(x) else "—"
    def fmt_dec(x): return f"{x:.3f}" if pd.notna(x) else "—"
    def fmt_num(x): return f"{x:,.2f}" if pd.notna(x) else "—"

    table_data = []
    for label, key in rows:
        sv = strat_stats.get(key, np.nan)
        sh = sharp_perf.get(key, np.nan)
        bh = bh_stats.get(key, np.nan)
        rv = risk_stats.get(key, np.nan)
        hv = hybrid_stats.get(key, np.nan)
        ps = pure_sig_stats.get(key, np.nan)

        if key in ["CAGR", "Volatility", "MaxDD", "Total", "TID"]:
            row = [label, fmt_pct(sv), fmt_pct(sh), fmt_pct(bh), fmt_pct(rv), fmt_pct(hv), fmt_pct(ps)]
        elif key in ["Sharpe", "MAR", "PainGain", "Skew", "Kurtosis"]:
            row = [label, fmt_dec(sv), fmt_dec(sh), fmt_dec(bh), fmt_dec(rv), fmt_dec(hv), fmt_dec(ps)]
        else:
            row = [label, fmt_num(sv), fmt_num(sh), fmt_num(bh), fmt_num(rv), fmt_num(hv), fmt_num(ps)]

        table_data.append(row)

    stat_table = pd.DataFrame(
        table_data,
        columns=[
            "Metric",
            "MA Strategy",
            "Sharpe-Optimal",
            "Buy & Hold (Rebalanced)",
            "Buy & Hold (No Rebalance)",
            "Hybrid SIG",
            "Pure SIG",
        ],
    )

    st.dataframe(stat_table, use_container_width=True)

    # ALLOCATION TABLES (unchanged)
    def compute_allocations(account_value, risky_w, safe_w, ron_w, roff_w):
        risky_dollars = account_value * risky_w
        safe_dollars  = account_value * safe_w
        alloc = {"Total Risky $": risky_dollars, "Total Safe $": safe_dollars}
        for t, w in ron_w.items():
            alloc[t] = risky_dollars * w
        for t, w in roff_w.items():
            alloc[t] = safe_dollars * w
        return alloc

    def compute_sharpe_alloc(account_value, tickers, weights):
        return {t: account_value * w for t, w in zip(tickers, weights)}

    def add_pct(df_dict):
        out = pd.DataFrame.from_dict(df_dict, orient="index", columns=["$"])
        if "Total Risky $" in out.index and "Total Safe $" in out.index:
            total_portfolio = float(out.loc["Total Risky $","$"]) + float(out.loc["Total Safe $","$"])
            out["% Portfolio"] = (out["$"] / total_portfolio * 100).apply(lambda x: f"{x:.2f}%")
            return out
        total = out["$"].sum()
        out["% Portfolio"] = (out["$"] / total * 100).apply(lambda x: f"{x:.2f}%")
        return out

    st.subheader("Account-Level Allocations")

    hyb_r = float(hybrid_rw.iloc[-1]) if len(hybrid_rw) > 0 else 0
    hyb_s = float(hybrid_sw.iloc[-1]) if len(hybrid_sw) > 0 else 0

    pure_r = float(pure_sig_rw.iloc[-1]) if len(pure_sig_rw) > 0 else 0
    pure_s = float(pure_sig_sw.iloc[-1]) if len(pure_sig_sw) > 0 else 0

    latest_signal = sig.iloc[-1] if len(sig) > 0 else False

    tab1, tab2, tab3 = st.tabs(["Taxable", "Tax-Sheltered", "Joint"])

    accounts = [
        ("Taxable", real_cap_1),
        ("Tax-Sheltered", real_cap_2),
        ("Joint", real_cap_3),
    ]

    for (label, cap), tab in zip(accounts, (tab1, tab2, tab3)):
        with tab:
            st.write(f"### {label} — Hybrid SIG")
            st.dataframe(add_pct(compute_allocations(cap, hyb_r, hyb_s, risk_on_weights, risk_off_weights)))

            st.write(f"### {label} — Pure SIG")
            st.dataframe(add_pct(compute_allocations(cap, pure_r, pure_s, risk_on_weights, risk_off_weights)))

            st.write(f"### {label} — 100% Risk-On Portfolio")
            st.dataframe(add_pct(compute_allocations(cap, 1.0, 0.0, risk_on_weights, {"SHY": 0})))

            if len(w_opt) > 0:
                st.write(f"### {label} — Sharpe-Optimal")
                st.dataframe(add_pct(compute_sharpe_alloc(cap, risk_on_px.columns, w_opt)))

            st.write(f"### {label} — MA Strategy")
            if latest_signal:
                ma_alloc = compute_allocations(cap, 1.0, 0.0, risk_on_weights, {"SHY": 0})
            else:
                ma_alloc = compute_allocations(cap, 0.0, 1.0, {}, risk_off_weights)
            st.dataframe(add_pct(ma_alloc))

    # MA Distance (unchanged)
    st.subheader("Next MA Signal Distance")
    if len(opt_ma) > 0 and len(portfolio_index) > 0:
        latest_date = opt_ma.dropna().index[-1]
        P = float(portfolio_index.loc[latest_date])
        MA = float(opt_ma.loc[latest_date])

        upper = MA * (1 + best_tol)
        lower = MA * (1 - best_tol)

        if latest_signal:
            delta = (P - lower) / P
            st.write(f"**Drop Required for RISK-OFF:** {delta:.2%}")
        else:
            delta = (upper - P) / P
            st.write(f"**Gain Required for RISK-ON:** {delta:.2%}")
    else:
        st.write("**Insufficient data for MA distance calculation**")

    # Regime stats plot (unchanged)
    st.subheader("Regime Statistics")
    if len(sig) > 0:
        sig_int = sig.astype(int)
        flips = sig_int.diff().fillna(0).ne(0)

        segments = []
        current = sig_int.iloc[0]
        seg_start = sig_int.index[0]

        for date, sw in flips.iloc[1:].items():
            if sw:
                segments.append((current, seg_start, date))
                current = sig_int.loc[date]
                seg_start = date

        segments.append((current, seg_start, sig_int.index[-1]))

        regime_rows = []
        for r, s, e in segments:
            regime_rows.append([
                "RISK-ON" if r == 1 else "RISK-OFF",
                s.date(), e.date(),
                (e - s).days
            ])

        regime_df = pd.DataFrame(regime_rows, columns=["Regime", "Start", "End", "Duration (days)"])
        st.dataframe(regime_df)

        on_durations = regime_df[regime_df['Regime']=='RISK-ON']['Duration (days)']
        off_durations = regime_df[regime_df['Regime']=='RISK-OFF']['Duration (days)']
        
        st.write(f"**Avg RISK-ON duration:** {on_durations.mean():.1f} days" if len(on_durations) > 0 else "**Avg RISK-ON duration:** 0 days")
        st.write(f"**Avg RISK-OFF duration:** {off_durations.mean():.1f} days" if len(off_durations) > 0 else "**Avg RISK-OFF duration:** 0 days")
    else:
        st.write("No regime data available")

    # ============================================================
    # STRATEGY VALIDATION DASHBOARD
    # ============================================================
    
    if run_validation:
        st.header("🎯 Strategy Validation")
        
        # Create tabs for different validation tests
        val_tab1, val_tab2, val_tab3, val_tab4 = st.tabs([
            "Statistical Significance", 
            "Walk-Forward Analysis",
            "Parameter Sensitivity", 
            "Consistency Metrics"
        ])
        
        with val_tab1:
            st.subheader("Hybrid SIG vs Buy & Hold Analysis")
            
            # Get returns for comparison
            portfolio_index = build_portfolio_index(prices, risk_on_weights)
            bh_returns = portfolio_index.pct_change().dropna()  # Buy & Hold (always risk-on)
            hybrid_returns = hybrid_eq.pct_change().fillna(0)   # Hybrid SIG strategy
            
            # Get hybrid signal (when in RISK-ON vs RISK-OFF)
            hybrid_signal = sig  # Use the same MA signal that hybrid SIG uses
            
            # Align dates
            common_idx = bh_returns.index.intersection(hybrid_returns.index)
            if len(common_idx) > 0:
                bh_returns_aligned = bh_returns.loc[common_idx]
                hybrid_returns_aligned = hybrid_returns.loc[common_idx]
                hybrid_signal_aligned = hybrid_signal.loc[common_idx]
                
                # 1. Risk-Adjusted Metrics
                st.write("### 1. Risk-Adjusted Performance")
                
                # Calculate metrics
                bh_sharpe = bh_returns_aligned.mean() / bh_returns_aligned.std() * np.sqrt(252) if bh_returns_aligned.std() > 0 else 0
                hybrid_sharpe = hybrid_returns_aligned.mean() / hybrid_returns_aligned.std() * np.sqrt(252) if hybrid_returns_aligned.std() > 0 else 0
                
                # Sortino Ratio (downside risk only)
                bh_downside = bh_returns_aligned[bh_returns_aligned < 0]
                hybrid_downside = hybrid_returns_aligned[hybrid_returns_aligned < 0]
                
                bh_sortino = bh_returns_aligned.mean() * 252 / (bh_downside.std() * np.sqrt(252)) if len(bh_downside) > 0 else 0
                hybrid_sortino = hybrid_returns_aligned.mean() * 252 / (hybrid_downside.std() * np.sqrt(252)) if len(hybrid_downside) > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("**Sharpe Ratio**", 
                             f"{hybrid_sharpe:.3f}",
                             delta=f"{hybrid_sharpe - bh_sharpe:+.3f} vs B&H")
                with col2:
                    st.metric("**Sortino Ratio**",
                             f"{hybrid_sortino:.3f}",
                             delta=f"{hybrid_sortino - bh_sortino:+.3f} vs B&H")
                with col3:
                    bh_vol = bh_returns_aligned.std() * np.sqrt(252)
                    hybrid_vol = hybrid_returns_aligned.std() * np.sqrt(252)
                    st.metric("**Volatility**",
                             f"{hybrid_vol:.2%}",
                             delta=f"{hybrid_vol - bh_vol:+.2%}")
                with col4:
                    bh_cagr = (1 + bh_returns_aligned).prod() ** (252/len(bh_returns_aligned)) - 1
                    hybrid_cagr = (1 + hybrid_returns_aligned).prod() ** (252/len(hybrid_returns_aligned)) - 1
                    st.metric("**CAGR**",
                             f"{hybrid_cagr:.2%}",
                             delta=f"{hybrid_cagr - bh_cagr:+.2%}")
                
                # 2. Drawdown Analysis
                st.write("### 2. Drawdown Protection")
                
                bh_eq = (1 + bh_returns_aligned).cumprod()
                hybrid_eq_aligned = (1 + hybrid_returns_aligned).cumprod()
                
                bh_dd = bh_eq / bh_eq.cummax() - 1
                hybrid_dd = hybrid_eq_aligned / hybrid_eq_aligned.cummax() - 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("**Max Drawdown**",
                             f"{hybrid_dd.min():.2%}",
                             delta=f"{hybrid_dd.min() - bh_dd.min():+.2%}")
                with col2:
                    st.metric("**Avg Drawdown**",
                             f"{hybrid_dd.mean():.2%}",
                             delta=f"{hybrid_dd.mean() - bh_dd.mean():+.2%}")
                with col3:
                    st.metric("**Time in Drawdown**",
                             f"{(hybrid_dd < 0).mean():.1%}",
                             delta=f"{(hybrid_dd < 0).mean() - (bh_dd < 0).mean():+.1%}")
                
                # 3. Regime Performance
                st.write("### 3. Regime-Specific Performance")
                
                risk_on_mask = hybrid_signal_aligned == True
                risk_off_mask = hybrid_signal_aligned == False
                
                col1, col2 = st.columns(2)
                with col1:
                    if risk_on_mask.any():
                        risk_on_days = risk_on_mask.sum()
                        hybrid_risk_on = hybrid_returns_aligned[risk_on_mask].mean() * 252
                        bh_risk_on = bh_returns_aligned[risk_on_mask].mean() * 252
                        
                        st.metric("**RISK-ON Periods**",
                                 f"{risk_on_days} days ({risk_on_days/len(hybrid_signal_aligned):.1%})",
                                 help="When MA signal is ON (invested in risky assets)")
                        st.metric("**RISK-ON Returns**",
                                 f"{hybrid_risk_on:.2%}",
                                 delta=f"{hybrid_risk_on - bh_risk_on:+.2%} vs B&H")
                
                with col2:
                    if risk_off_mask.any():
                        risk_off_days = risk_off_mask.sum()
                        hybrid_risk_off = hybrid_returns_aligned[risk_off_mask].mean() * 252
                        bh_risk_off = bh_returns_aligned[risk_off_mask].mean() * 252
                        
                        st.metric("**RISK-OFF Periods**",
                                 f"{risk_off_days} days ({risk_off_days/len(hybrid_signal_aligned):.1%})",
                                 help="When MA signal is OFF (in treasuries/cash)")
                        st.metric("**RISK-OFF Protection**",
                                 f"{bh_risk_off:.2%}",
                                 help="Losses avoided during RISK-OFF periods")
                
                # 4. Bear Market Performance
                st.write("### 4. Bear Market Protection")
                
                bear_periods = bh_returns_aligned < 0
                if bear_periods.any():
                    hybrid_bear = hybrid_returns_aligned[bear_periods]
                    bh_bear = bh_returns_aligned[bear_periods]
                    
                    bear_outperformance = (hybrid_bear.mean() - bh_bear.mean()) * 252
                    bear_win_rate = (hybrid_bear > bh_bear).mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("**Bear Market Days**",
                                 f"{bear_periods.sum()} ({bear_periods.mean():.1%})")
                    with col2:
                        st.metric("**Bear Outperformance**",
                                 f"{bear_outperformance:.2%}")
                    with col3:
                        st.metric("**Bear Win Rate**",
                                 f"{bear_win_rate:.1%}")
                
                # 5. Visualization
                st.write("### 5. Performance Visualization")
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
                
                # Plot 1: Cumulative returns with regimes
                ax1.plot(hybrid_eq_aligned, label='Hybrid SIG', linewidth=2, color='green')
                ax1.plot(bh_eq, label='Buy & Hold', linewidth=2, color='blue', alpha=0.7)
                
                # Shade RISK-OFF periods
                risk_off_starts = hybrid_signal_aligned.diff() == -1
                risk_on_starts = hybrid_signal_aligned.diff() == 1
                
                risk_off_periods = []
                current_start = None
                for i, (date, signal) in enumerate(hybrid_signal_aligned.items()):
                    if signal == False and current_start is None:
                        current_start = date
                    elif signal == True and current_start is not None:
                        risk_off_periods.append((current_start, date))
                        current_start = None
                
                if current_start is not None:
                    risk_off_periods.append((current_start, hybrid_signal_aligned.index[-1]))
                
                for start, end in risk_off_periods:
                    ax1.axvspan(start, end, alpha=0.2, color='red', label='RISK-OFF' if start == risk_off_periods[0][0] else "")
                
                ax1.set_title('Cumulative Returns with Regime Shading')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Growth of $1')
                ax1.legend()
                ax1.grid(alpha=0.3)
                
                # Plot 2: Drawdown comparison
                ax2.plot(hybrid_dd * 100, label='Hybrid SIG', linewidth=1.5, color='green')
                ax2.plot(bh_dd * 100, label='Buy & Hold', linewidth=1.5, color='blue', alpha=0.7)
                ax2.set_title('Drawdown Comparison (%)')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Drawdown %')
                ax2.legend()
                ax2.grid(alpha=0.3)
                
                # Plot 3: Rolling Sharpe (1-year)
                window = min(252, len(hybrid_returns_aligned))
                if window >= 63:
                    rolling_sharpe_hybrid = hybrid_returns_aligned.rolling(window).mean() / hybrid_returns_aligned.rolling(window).std() * np.sqrt(252)
                    rolling_sharpe_bh = bh_returns_aligned.rolling(window).mean() / bh_returns_aligned.rolling(window).std() * np.sqrt(252)
                    
                    ax3.plot(rolling_sharpe_hybrid, label='Hybrid SIG', linewidth=1.5, color='green')
                    ax3.plot(rolling_sharpe_bh, label='Buy & Hold', linewidth=1.5, color='blue', alpha=0.7)
                    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    ax3.set_title(f'Rolling {window}-Day Sharpe Ratio')
                    ax3.set_xlabel('Date')
                    ax3.set_ylabel('Sharpe Ratio')
                    ax3.legend()
                    ax3.grid(alpha=0.3)
                
                # Plot 4: Monthly returns distribution
                hybrid_monthly = hybrid_returns_aligned.resample('M').apply(lambda x: (1+x).prod()-1)
                bh_monthly = bh_returns_aligned.resample('M').apply(lambda x: (1+x).prod()-1)
                
                bins = np.linspace(min(hybrid_monthly.min(), bh_monthly.min()), 
                                 max(hybrid_monthly.max(), bh_monthly.max()), 20)
                
                ax4.hist(hybrid_monthly, bins=bins, alpha=0.7, label='Hybrid SIG', color='green', density=True)
                ax4.hist(bh_monthly, bins=bins, alpha=0.5, label='Buy & Hold', color='blue', density=True)
                ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
                ax4.set_title('Monthly Returns Distribution')
                ax4.set_xlabel('Monthly Return')
                ax4.set_ylabel('Frequency')
                ax4.legend()
                ax4.grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 6. Key Takeaways
                st.write("### 6. Key Takeaways for Hybrid SIG Strategy")
                
                takeaways = []
                
                if hybrid_sharpe > bh_sharpe:
                    takeaways.append("✅ **Higher Sharpe Ratio** - Better risk-adjusted returns")
                
                if hybrid_dd.min() > bh_dd.min():
                    takeaways.append("✅ **Smaller Maximum Drawdown** - Better capital preservation")
                
                if hybrid_vol < bh_vol:
                    takeaways.append("✅ **Lower Volatility** - Smoother ride")
                
                if bear_periods.any() and hybrid_bear.mean() > bh_bear.mean():
                    takeaways.append("✅ **Bear Market Protection** - Outperforms during downturns")
                
                if hybrid_cagr < bh_cagr:
                    takeaways.append("⚠️ **Lower CAGR** - Expected trade-off for regime strategies")
                    takeaways.append("   *Goal is risk reduction, not maximum returns*")
                
                # Additional metrics specific to Hybrid SIG
                st.write("### 7. Hybrid SIG Specific Metrics")
                
                # Quarterly rebalancing impact
                if len(hybrid_rebals) > 0:
                    st.metric("**Quarterly Rebalances**", 
                             f"{len(hybrid_rebals)} rebalances",
                             help="Number of times SIG engine rebalanced between quarters")
                
                # Current allocation
                current_risky_pct = float(hybrid_rw.iloc[-1]) * 100
                current_safe_pct = float(hybrid_sw.iloc[-1]) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("**Current Risky Allocation**",
                             f"{current_risky_pct:.1f}%")
                with col2:
                    st.metric("**Current Safe Allocation**",
                             f"{current_safe_pct:.1f}%")
                
                for takeaway in takeaways:
                    st.write(takeaway)
                
            else:
                st.info("Insufficient overlapping data for comparison")
        
        with val_tab2:
            st.subheader("Walk-Forward Validation (3-Year Train, 1-Year Test)")
            
            portfolio_index = build_portfolio_index(prices, risk_on_weights)
            wfa_results = walk_forward_validation(portfolio_index, best_result["returns"])
            
            if not wfa_results.empty:
                # Display summary stats
                avg_test_sharpe = wfa_results['test_sharpe'].mean()
                std_test_sharpe = wfa_results['test_sharpe'].std()
                success_rate = (wfa_results['test_sharpe'] > 0).mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Test Sharpe", f"{avg_test_sharpe:.3f}")
                with col2:
                    st.metric("Test Sharpe Std", f"{std_test_sharpe:.3f}")
                with col3:
                    st.metric("Success Rate", f"{success_rate:.1%}")
                
                # Plot walk-forward results
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(wfa_results.index, wfa_results['test_sharpe'], marker='o', linewidth=2)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=avg_test_sharpe, color='g', linestyle='--', alpha=0.5)
                ax.set_title('Walk-Forward Test Sharpe Ratios')
                ax.set_xlabel('Test Period')
                ax.set_ylabel('Sharpe Ratio')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                
                # Show detailed table
                with st.expander("Show Detailed Walk-Forward Results"):
                    st.dataframe(wfa_results)
            else:
                st.info("Not enough data for walk-forward analysis (need at least 2 years)")
        
        with val_tab3:
            st.subheader("Parameter Sensitivity Analysis")
            
            sens_results = parameter_sensitivity_analysis(
                prices, best_cfg, risk_on_weights, risk_off_weights, FLIP_COST
            )
            
            if not sens_results['length_sensitivity'].empty and not sens_results['tolerance_sensitivity'].empty:
                # Plot MA length sensitivity
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(sens_results['length_sensitivity']['length'], 
                        sens_results['length_sensitivity']['sharpe'], 
                        marker='o', linewidth=2)
                ax1.axvline(x=best_len, color='r', linestyle='--', label=f'Optimal ({best_len})')
                ax1.set_title('Sharpe Ratio vs MA Length')
                ax1.set_xlabel('MA Length (days)')
                ax1.set_ylabel('Sharpe Ratio')
                ax1.legend()
                ax1.grid(alpha=0.3)
                
                # Plot tolerance sensitivity
                ax2.plot(sens_results['tolerance_sensitivity']['tolerance'], 
                        sens_results['tolerance_sensitivity']['sharpe'], 
                        marker='o', linewidth=2)
                ax2.axvline(x=best_tol, color='r', linestyle='--', label=f'Optimal ({best_tol:.2%})')
                ax2.set_title('Sharpe Ratio vs Tolerance')
                ax2.set_xlabel('Tolerance')
                ax2.set_ylabel('Sharpe Ratio')
                ax2.legend()
                ax2.grid(alpha=0.3)
                
                st.pyplot(fig)
                
                # Calculate robustness scores
                length_std = sens_results['length_sensitivity']['sharpe'].std()
                length_range = sens_results['length_sensitivity']['sharpe'].max() - sens_results['length_sensitivity']['sharpe'].min()
                
                tol_std = sens_results['tolerance_sensitivity']['sharpe'].std()
                tol_range = sens_results['tolerance_sensitivity']['sharpe'].max() - sens_results['tolerance_sensitivity']['sharpe'].min()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MA Length Robustness", f"{1 - (length_std/length_range if length_range > 0 else 0):.2%}")
                with col2:
                    st.metric("Tolerance Robustness", f"{1 - (tol_std/tol_range if tol_range > 0 else 0):.2%}")
            else:
                st.info("Insufficient data for parameter sensitivity analysis")
        
        with val_tab4:
            st.subheader("Strategy Consistency Metrics")
            
            consistency = calculate_consistency_metrics(best_result["returns"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Monthly Hit Ratio", f"{consistency['monthly_hit_ratio']:.1%}")
                st.metric("Avg Win %", f"{consistency['avg_win_pct']:.2%}")
            with col2:
                st.metric("Win/Loss Ratio", f"{consistency['win_loss_ratio']:.2f}")
                st.metric("Avg Loss %", f"{consistency['avg_loss_pct']:.2%}")
            with col3:
                st.metric("Max Consecutive Wins", consistency['max_consecutive_wins'])
                st.metric("Max Consecutive Losses", consistency['max_consecutive_losses'])
            
            # Create consistency visualization
            if len(best_result["returns"]) > 0:
                monthly_returns = best_result["returns"].resample('M').apply(lambda x: (1+x).prod()-1)
                
                if len(monthly_returns) > 0:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['green' if x > 0 else 'red' for x in monthly_returns]
                    ax.bar(range(len(monthly_returns)), monthly_returns.values * 100, color=colors, alpha=0.7)
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    ax.set_title('Monthly Returns (%)')
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Return %')
                    ax.grid(alpha=0.3, axis='y')
                    
                    # Add cumulative line
                    ax2 = ax.twinx()
                    cumulative = (1 + monthly_returns).cumprod() - 1
                    ax2.plot(range(len(monthly_returns)), cumulative.values * 100, color='blue', linewidth=2)
                    ax2.set_ylabel('Cumulative Return %', color='blue')
                    ax2.tick_params(axis='y', labelcolor='blue')
                    
                    st.pyplot(fig)
            
            # Interpretation
            st.write("### 📊 Interpretation Guide")
            st.write("- **Hit Ratio > 55%**: Good consistency")
            st.write("- **Win/Loss Ratio > 1.5**: Good risk/reward")  
            st.write("- **Max Consecutive Losses < 4**: Manageable drawdowns")
    
    st.markdown("---")  # Separator before final plot

    # Final Performance Plot (updated with Buy & Hold with rebalance)
    st.subheader("Portfolio Strategy Performance Comparison")

    plot_index = build_portfolio_index(prices, risk_on_weights)
    plot_ma = compute_ma_matrix(plot_index, [best_len], best_type)[best_len]

    plot_index_norm = normalize(plot_index)
    plot_ma_norm = normalize(plot_ma.dropna()) if len(plot_ma.dropna()) > 0 else pd.Series([], dtype=float)

    strat_eq_norm  = normalize(best_result["equity_curve"])
    sharp_eq_norm  = normalize(sharp_eq) if len(sharp_eq) > 0 else pd.Series([], dtype=float)
    hybrid_eq_norm = normalize(hybrid_eq) if len(hybrid_eq) > 0 else pd.Series([], dtype=float)
    pure_sig_norm  = normalize(pure_sig_eq) if len(pure_sig_eq) > 0 else pd.Series([], dtype=float)
    risk_on_norm   = normalize(risk_on_eq) if len(risk_on_eq) > 0 else pd.Series([], dtype=float)
    bh_rebalance_norm = normalize(bh_rebalance_result["equity_curve"]) if len(bh_rebalance_result["equity_curve"]) > 0 else pd.Series([], dtype=float)

    if len(strat_eq_norm) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(strat_eq_norm, label="MA Strategy", linewidth=2)
        if len(sharp_eq_norm) > 0:
            ax.plot(sharp_eq_norm, label="Sharpe-Optimal", linewidth=2, color="magenta")
        if len(bh_rebalance_norm) > 0:
            ax.plot(bh_rebalance_norm, label="Buy & Hold (Rebalanced)", linewidth=2, color="purple", linestyle=":")
        ax.plot(risk_on_norm, label="Buy & Hold (No Rebalance)", alpha=0.65)
        if len(hybrid_eq_norm) > 0:
            ax.plot(hybrid_eq_norm, label="Hybrid SIG", linewidth=2, color="blue")
        if len(pure_sig_norm) > 0:
            ax.plot(pure_sig_norm, label="Pure SIG", linewidth=2, color="orange")
        if len(plot_ma_norm) > 0:
            ax.plot(plot_ma_norm, label=f"MA({best_len}) {best_type.upper()}", linestyle="--", color="black", alpha=0.6)

        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Insufficient data for performance plot")

    # ============================================================
    # IMPLEMENTATION CHECKLIST (Displayed at Bottom)
    # ============================================================
    st.markdown("""
---

## **Implementation Checklist**

- Rotate 100% of portfolio to treasury sleeve whenever the MA regime flips.
- At each calendar quarter-end, input your portfolio value at last rebalance & today's portfolio value.
- Execute the exact dollar adjustment recommended by the model (increase/decrease deployed sleeve) on the rebalance date.
- At each rebalance, re-evaluate the Sharpe-optimal portfolio weighting.

Current Sharpe-optimal portfolio: https://testfol.io/optimizer?s=9TIGHucZuaJ

---
    """)


# ============================================================
# LAUNCH APP
# ============================================================

if __name__ == "__main__":
    main()