# patterns.py
import numpy as np
import pandas as pd

def detect_simple_flag(df):
    """
    Placeholder: detect brief consolidation followed by breakout.
    df: pandas DataFrame with Open/High/Low/Close/Volume indexed by datetime.
    Return True/False and meta info for debugging.
    """
    # Simple heuristic: decreasing range for N bars then breakout
    lookback = 20
    if len(df) < lookback+5:
        return False, {}
    rngs = df['High'] - df['Low']
    if rngs[-lookback:].std() < (rngs[-lookback:].mean()*0.5):
        # check breakout in last 3 bars
        if df['Close'][-1] > df['High'][-lookback-1:-1].max():
            return True, {"type":"flag"}
    return False, {}

def detect_rsi_divergence(df, rsi_col='rsi'):
    """
    Very simple divergence detector: price makes lower low while rsi makes higher low (bullish).
    """
    if len(df) < 6:
        return None
    lows_idx = df['Low'].rolling(5).apply(lambda x: np.argmin(x) == 2).astype(bool)
    # Simplified check: compare two local lows
    try:
        last_low = df['Low'].iloc[-6:-1].min()
        prev_low = df['Low'].iloc[-12:-6].min()
        rsi_last = df[rsi_col].iloc[-6:-1].min()
        rsi_prev = df[rsi_col].iloc[-12:-6].min()
        if last_low < prev_low and rsi_last > rsi_prev:
            return {"divergence":"bullish"}
    except Exception:
        pass
    return None
