"""Feature engineering shared by training (train_model.py) and live inference (strategy.py).

Every feature here is causal — computed only from data up to and including the current
bar — so the same function is safe to use for both backtesting and live trading with no
look-ahead leakage.
"""

import numpy as np
import ta

# Column names the model is trained and served on. Keep train_model.py and strategy.py
# in sync by always importing this list rather than hardcoding column names elsewhere.
FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "ema_diff", "rsi_14", "macd_diff", "bb_pct",
    "atr_pct", "vol_zscore", "volatility_20", "momentum_10",
]


def compute_features(df):
    """Given a chronologically sorted OHLCV DataFrame, return a copy with feature columns
    added. Early rows will contain NaN until each indicator's warmup window is satisfied."""
    out = df.copy()
    close, high, low, volume = out["close"], out["high"], out["low"], out["volume"]

    # Momentum / recent returns over several lookback windows.
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["ret_12"] = close.pct_change(12)
    out["ret_24"] = close.pct_change(24)
    out["momentum_10"] = close / close.shift(10) - 1

    # Trend: normalized EMA spread (fast vs slow), so the value is comparable across price levels.
    ema_fast = ta.trend.ema_indicator(close, window=12)
    ema_slow = ta.trend.ema_indicator(close, window=26)
    out["ema_diff"] = (ema_fast - ema_slow) / close

    # SMA(20)/SMA(50) trend confirmation — used by the rule-based confluence engine
    # (signals.py) alongside the ML model. Not part of FEATURE_COLUMNS, so adding
    # these doesn't require retraining existing per-coin models.
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["sma_diff"] = (out["sma_20"] - out["sma_50"]) / close

    # Oscillators.
    out["rsi_14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    out["macd_diff"] = macd.macd_diff() / close

    # Volatility / bands.
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    band_width = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
    out["bb_pct"] = (close - bb.bollinger_lband()) / band_width
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    out["atr_pct"] = atr / close
    out["volatility_20"] = close.pct_change().rolling(20).std()

    # Volume anomaly vs its own recent history.
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    out["vol_zscore"] = (volume - vol_mean) / vol_std

    return out


def latest_feature_row(df):
    """Compute features and return the most recent row as a single-row 2D array plus a
    validity flag. Returns (None, False) if the latest row has any NaN feature (not
    enough warmup history yet)."""
    feat = compute_features(df)
    row = feat.iloc[[-1]][FEATURE_COLUMNS]
    if row.isnull().any(axis=None):
        return None, False
    return row.values, True
