"""ml_trend strategy: buy when the trained classifier gives a high probability of an
up-move AND price is in a confirmed uptrend (EMA filter); size the trade by how
confident the model is; exit on an ATR take-profit / stop-loss, or early if the model
itself turns bearish while the position is open.

This keeps the same "flat/long, ATR-based TP-SL" skeleton as sample-2's ema_rsi bot,
but replaces the fixed RSI-oversold entry rule with a learned probability, and adds
confidence-based position sizing plus a model-based early exit.
"""

from dataclasses import dataclass

import ta

import features

WARMUP = 150          # bars of history needed before indicators/features are trustworthy

PARAMS = dict(
    ema_fast=12, ema_slow=26,          # trend filter, same spirit as sample-2
    atr_period=14, tp_mult=2.5, sl_mult=1.0,   # tighter stop-loss (was 1.5) — smaller loss per losing trade
    buy_prob=0.56,                      # min model probability to open a long
    exit_prob=0.42,                     # model probability below which an open long exits early
    base_trade_usd=5.0,                 # trade size at the buy_prob threshold (was 7.0)
    max_trade_fraction=0.08,            # never risk more than this fraction of equity on one entry (was 0.15)
    min_order_usd=5.0,                  # ANDX competition minimum order size
)


@dataclass
class Decision:
    action: str             # BUY / SELL / HOLD
    price: float             # latest price, for logging
    trade_usd: float = 0.0   # sizing for a BUY (ignored for SELL/HOLD)
    tp_price: float = 0.0    # take-profit, set on a BUY
    sl_price: float = 0.0    # stop-loss, set on a BUY
    prob_up: float = 0.5     # model's latest up-probability, for logging
    note: str = ""


def position_size(prob_up, equity_quote):
    """Scale trade size with model confidence above the buy threshold, capped by a
    fraction of account equity and floored at the exchange's minimum order size."""
    confidence = max(0.0, (prob_up - PARAMS["buy_prob"]) / (1 - PARAMS["buy_prob"]))
    confidence = min(confidence, 1.0)
    raw = PARAMS["base_trade_usd"] * (1 + confidence)     # up to 2x base at max confidence
    cap = equity_quote * PARAMS["max_trade_fraction"]
    trade_usd = min(raw, cap)
    return trade_usd


def decide(df, state, clf, equity_quote):
    """Return a BUY / SELL / HOLD Decision for the latest bar.
    `clf` is the trained sklearn classifier (see model.py / train_model.py).
    `equity_quote` is the account's current value in the quote currency (for sizing)."""
    close, high, low = df["close"], df["high"], df["low"]
    price = float(close.iloc[-1])

    row, valid = features.latest_feature_row(df)
    prob_up = float(clf.predict_proba(row)[0, 1]) if valid else 0.5

    # Holding -> exit on TP/SL, or early if the model itself has turned bearish.
    if state.get("position") == "LONG":
        tp_price = float(state.get("tp_price") or 0)
        sl_price = float(state.get("sl_price") or 0)
        if price >= tp_price:
            return Decision("SELL", price, prob_up=prob_up, note="take-profit hit")
        if price <= sl_price:
            return Decision("SELL", price, prob_up=prob_up, note="stop-loss hit")
        if valid and prob_up <= PARAMS["exit_prob"]:
            return Decision("SELL", price, prob_up=prob_up, note=f"model turned bearish (p={prob_up:.2f})")
        return Decision("HOLD", price, prob_up=prob_up, note="holding within exit bands")

    # Flat -> require both a confirmed uptrend and a confident model signal.
    if not valid:
        return Decision("HOLD", price, prob_up=prob_up, note="warming up (not enough history for features)")

    ema_fast = ta.trend.ema_indicator(close, window=PARAMS["ema_fast"]).iloc[-1]
    ema_slow = ta.trend.ema_indicator(close, window=PARAMS["ema_slow"]).iloc[-1]
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close,
                                         window=PARAMS["atr_period"]).average_true_range().iloc[-1]
    uptrend = ema_fast > ema_slow

    if uptrend and prob_up >= PARAMS["buy_prob"] and atr > 0:
        trade_usd = max(position_size(prob_up, equity_quote), PARAMS["min_order_usd"])
        tp_price = price + PARAMS["tp_mult"] * atr
        sl_price = price - PARAMS["sl_mult"] * atr
        return Decision("BUY", price, trade_usd, tp_price, sl_price, prob_up,
                         note=f"uptrend + model p(up)={prob_up:.2f}")

    return Decision("HOLD", price, prob_up=prob_up,
                     note=f"no edge (uptrend={uptrend}, p(up)={prob_up:.2f})")
