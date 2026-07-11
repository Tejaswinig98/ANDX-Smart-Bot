"""Rule-based multi-indicator confluence engine.

Scores each bar using four classic, independent signals:
  - SMA(20) vs SMA(50):    trend direction (golden-cross style)
  - RSI(14):               momentum / overbought-oversold
  - Bollinger Bands(20,2): mean-reversion positioning within the bands
  - MACD histogram:        trend momentum confirmation

Each contributes -1 (bearish), 0 (neutral), or +1 (bullish). The total (-4..+4) is
combined with the ML model's up-probability (if a trained model exists for the coin)
into a single confidence score used by smart_bot.py to decide entries/exits.

This is intentionally simple and inspectable — you should be able to look at any bar's
score and see exactly which indicators agree or disagree, rather than trusting a black
box. Tune the thresholds below based on backtest.py results per coin.
"""

from dataclasses import dataclass

import features

# Indicator thresholds — tune per-coin via backtesting if a coin behaves very differently.
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
BB_LOW = 0.20     # price in the bottom 20% of the Bollinger band -> mean-reversion buy zone
BB_HIGH = 0.80    # price in the top 20% -> mean-reversion sell zone


@dataclass
class Confluence:
    sma_signal: int       # -1 / 0 / +1
    rsi_signal: int
    bb_signal: int
    macd_signal: int
    total: int             # sum, range -4..+4
    prob_up: float          # ML model's probability, 0.5 if no model / not confident
    ml_valid: bool
    combined: float         # -1..+1 blended score used for the actual decision


def score_row(row, prob_up=0.5, ml_valid=False):
    """Score one feature row (a dict-like with sma_diff, rsi_14, bb_pct, macd_diff)."""
    sma_diff = row["sma_diff"]
    rsi = row["rsi_14"]
    bb_pct = row["bb_pct"]
    macd_diff = row["macd_diff"]

    sma_signal = 1 if sma_diff > 0 else (-1 if sma_diff < 0 else 0)

    if rsi <= RSI_OVERSOLD:
        rsi_signal = 1        # oversold -> bullish (mean reversion up)
    elif rsi >= RSI_OVERBOUGHT:
        rsi_signal = -1       # overbought -> bearish
    else:
        rsi_signal = 0

    if bb_pct <= BB_LOW:
        bb_signal = 1         # near lower band -> bullish (mean reversion up)
    elif bb_pct >= BB_HIGH:
        bb_signal = -1        # near upper band -> bearish
    else:
        bb_signal = 0

    macd_signal = 1 if macd_diff > 0 else (-1 if macd_diff < 0 else 0)

    total = sma_signal + rsi_signal + bb_signal + macd_signal

    # Blend technical confluence (-4..+4, normalized to -1..+1) with the ML probability
    # (0..1, recentered to -1..+1). Equal weight by default; if there's no trained model
    # for this coin, fall back to technicals alone rather than dragging the score toward
    # a meaningless 0.5.
    technical_component = total / 4.0
    if ml_valid:
        ml_component = (prob_up - 0.5) * 2
        combined = 0.5 * technical_component + 0.5 * ml_component
    else:
        combined = technical_component

    return Confluence(sma_signal, rsi_signal, bb_signal, macd_signal, total,
                       prob_up, ml_valid, combined)


def latest_confluence(df, clf=None):
    """Compute features for df and score the latest bar. clf is an optional trained
    sklearn classifier (see model.py) — pass None to score on technicals alone."""
    feat = features.compute_features(df)
    needed = ["sma_diff", "rsi_14", "bb_pct", "macd_diff"]
    last = feat.iloc[-1]
    if last[needed].isnull().any():
        return None, False   # not enough warmup history yet

    prob_up, ml_valid = 0.5, False
    if clf is not None:
        row, valid = features.latest_feature_row(df)
        if valid:
            prob_up = float(clf.predict_proba(row)[0, 1])
            ml_valid = True

    conf = score_row(last, prob_up, ml_valid)
    return conf, True
