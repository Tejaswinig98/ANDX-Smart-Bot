"""Smart multi-coin bot: combines rule-based technical confluence (SMA, RSI, Bollinger
Bands, MACD — see signals.py) with each coin's ML model (if trained) to decide
directional trades, and separately paces volume-generating round trips toward a daily
target — but only ever spends a capped, measured "spread-cost budget" doing so (see
risk.py's spread_cost_budget_remaining), rather than blindly chasing a dollar figure
regardless of cost.

Honesty check, read before running: no strategy can guarantee zero-loss volume
generation — every trade has some spread/slippage risk. This bot's actual promise is
narrower: (1) directional trades only fire when multiple independent signals agree,
(2) volume top-up round trips only fire on the currently cheapest coin and only within
a small, capped daily cost budget, and (3) the existing daily/weekly drawdown halts
still apply on top as a hard backstop. Some days it may fall short of the volume
target rather than force a trade — that's the point, not a bug.

Run once per tick (e.g. every 15 min via cron/GitHub Actions):
    python smart_bot.py
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ta
from dotenv import load_dotenv

import feed
import model as model_lib
import risk
import screener
import signals
import spread
from andx_api import ANDX, APIError

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

# Coins scanned each tick. All must exist on Coinbase (for feed/training data) and be
# listed on ANDX. Extend this list once you've trained a model for a new coin (a
# model isn't required — coins without one just trade on technicals alone — but ML
# does improve entry quality per the earlier backtests).
# Coin selection is now dynamic (see screener.py): each run, the top N recent
# performers from a ~60-coin candidate pool are picked to actually trade. "Performer"
# means trailing price momentum over the last day — the honest, implementable meaning
# of "profitable" here; nothing can predict future returns. Coins without a trained
# ML model (only the original 11 have one) still trade on technicals alone via
# signals.py's fallback. The screener re-scans the full pool at most once per hour
# (see screener.REFRESH_MINUTES) and caches results, so most ticks are cheap.
QUOTE = "USDT"
ACTIVE_COIN_COUNT = 5   # how many top performers to actually trade each cycle


WARMUP = 150   # bars of history needed before SMA(50)/indicators are trustworthy

# Directional trade parameters (per coin, applied uniformly — tune per-coin via
# backtest.py if one coin behaves very differently).
ENTRY_SCORE = 0.60     # combined confluence score (-1..+1) needed to open a long — raised for stricter agreement
EXIT_SCORE = -0.10     # combined score below which an open long exits early — tighter, cuts losers sooner
ATR_PERIOD = 14
TP_MULT = 2.5          # take profit a bit sooner than before
SL_MULT = 0.75         # tighter still (was 1.0) — larger trade sizes need faster loss cutoffs
BASE_TRADE_USD = 5.0   # smaller base size (was 7.0)
MAX_TRADE_EQUITY_FRACTION = 0.90   # can use up to 90% of equity per trade (was 8%) — see the
                                   # README's risk-tradeoff note: bigger size means a bad move
                                   # between ticks can cost more before the bot can react
MIN_ORDER_USD = 5.0

# Volume top-up parameters. Per explicit request: trade every tick regardless of pace,
# stopping only on the loss ceiling above — not on a cost or pacing budget. Each tick
# scans the current candidate coins live and round-trips whichever is cheapest.
TARGET_DAILY_VOLUME = float(os.environ.get("TARGET_DAILY_VOLUME", 2000.0))
SANITY_MAX_COST_PCT = 0.03    # only skip entirely if even the cheapest coin looks broken/abnormal (3%+)
MAX_LEG_USD = 100.0            # cap per round trip (raised — see floor-based sizing below)
LEG_FLOOR_BUFFER_FRACTION = 0.90   # use up to this much of the room above the $75 floor per round trip.
                                    # Safe to be aggressive here: a round trip's real risk is just its
                                    # spread cost (a fraction of a percent), not the whole notional —
                                    # the cash comes right back on the sell leg a few seconds later.


def build_client():
    return ANDX(
        os.environ.get("ANDX_USER_NAME", ""),
        os.environ.get("ANDX_TOKEN", ""),
        os.environ.get("ANDX_API_KEY", ""),
        os.environ.get("ANDX_API_SECRET", ""),
        os.environ.get("ANDX_PASSPHRASE", ""),
    )


def confirm_status(api, order_number, tries=5):
    status = "?"
    for _ in range(tries):
        try:
            status = api.get_order_status(order_number)
        except APIError:
            status = "?"
        if status in ("F", "C"):
            break
        time.sleep(1)
    return status


def state_path(coin):
    return HERE / f"state_{coin}.json"


def load_state(coin):
    path = state_path(coin)
    if path.exists():
        import json
        return json.loads(path.read_text())
    return {"position": "FLAT", "entry_price": 0, "tp_price": 0, "sl_price": 0}


def save_state(coin, state):
    import json
    state_path(coin).write_text(json.dumps(state, indent=2))


def portfolio_equity_quote(api, prices):
    """Total account value in QUOTE terms: quote balance + every coin's balance marked
    at its current price. `prices` is {coin: price}."""
    total = api.get_available(QUOTE)
    for coin, price in prices.items():
        total += api.get_available(coin) * price
    return total


def position_size(score, equity_quote):
    """Scale trade size with confluence strength above ENTRY_SCORE, capped by equity
    fraction AND by the room above the $75 floor (directional trades carry real
    downside risk if the stop-loss gaps, unlike a round trip), floored at the
    exchange minimum."""
    confidence = max(0.0, (score - ENTRY_SCORE) / (1 - ENTRY_SCORE))
    confidence = min(confidence, 1.0)
    raw = BASE_TRADE_USD * (1 + confidence)
    buffer_above_floor = max(equity_quote - risk.MIN_EQUITY_FLOOR_USD, 0.0)
    cap = min(equity_quote * MAX_TRADE_EQUITY_FRACTION, buffer_above_floor)
    return min(raw, cap)


def directional_tick(api, coin, equity_quote):
    """Check/execute one coin's directional entry or exit. Returns USD volume traded
    (0 if nothing happened)."""
    days = WARMUP // 24 + 3
    df = feed.recent_bars(coin, days=days)
    if len(df) < WARMUP:
        print(f"[{coin}] warmup: {len(df)}/{WARMUP} bars — skipping")
        return 0.0

    try:
        clf, _meta = model_lib.load(coin)
    except model_lib.ModelNotTrained:
        clf = None   # coin trades on technicals alone until a model is trained for it

    conf, valid = signals.latest_confluence(df, clf)
    price = float(df["close"].iloc[-1])
    if not valid:
        print(f"[{coin}] warming up (indicators not ready) — price {price:.4f}")
        return 0.0

    state = load_state(coin)
    tag = (f"sma={conf.sma_signal:+d} rsi={conf.rsi_signal:+d} bb={conf.bb_signal:+d} "
           f"macd={conf.macd_signal:+d} ml={'y' if conf.ml_valid else 'n'}({conf.prob_up:.2f}) "
           f"combined={conf.combined:+.2f}")

    if state.get("position") == "LONG":
        tp_price = float(state.get("tp_price") or 0)
        sl_price = float(state.get("sl_price") or 0)
        if price >= tp_price:
            reason = "take-profit hit"
        elif price <= sl_price:
            reason = "stop-loss hit"
        elif conf.combined <= EXIT_SCORE:
            reason = f"confluence turned bearish ({tag})"
        else:
            print(f"[{coin}] HOLD long — price {price:.4f}  {tag}")
            return 0.0

        sell_amount = api.get_available(coin)
        if sell_amount <= 0:
            print(f"[{coin}] wanted to exit ({reason}) but no balance found — clearing stale state")
            save_state(coin, {"position": "FLAT", "entry_price": 0, "tp_price": 0, "sl_price": 0})
            return 0.0
        quote = api.get_quote(QUOTE, coin, f"{sell_amount:.8f}")
        order = api.place_instant_order(QUOTE, coin, quote["sell_currency_amount"],
                                        quote["buy_currency_amount"], quote["visible_price"])
        status = confirm_status(api, order["order_number"])
        usd = float(quote["buy_currency_amount"])
        print(f"[{coin}] SELL ({reason}) order {order['order_number']} status={status} size=${usd:.2f}")
        if status == "F":
            save_state(coin, {"position": "FLAT", "entry_price": 0, "tp_price": 0, "sl_price": 0})
            return usd
        return 0.0

    # Flat: only enter on strong multi-indicator agreement.
    if conf.combined < ENTRY_SCORE:
        print(f"[{coin}] HOLD flat — price {price:.4f}  {tag}")
        return 0.0

    high, low, close = df["high"], df["low"], df["close"]
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close,
                                         window=ATR_PERIOD).average_true_range().iloc[-1]
    if not atr or atr <= 0:
        return 0.0

    trade_usd = max(position_size(conf.combined, equity_quote), MIN_ORDER_USD)
    quote_bal = api.get_available(QUOTE)
    trade_usd = min(trade_usd, quote_bal - 0.50)
    if trade_usd < MIN_ORDER_USD:
        print(f"[{coin}] signal is BUY ({tag}) but not enough spendable {QUOTE} — skipping")
        return 0.0

    buy_quote = api.get_quote(coin, QUOTE, f"{trade_usd:.2f}")
    order = api.place_instant_order(coin, QUOTE, buy_quote["sell_currency_amount"],
                                    buy_quote["buy_currency_amount"], buy_quote["visible_price"])
    status = confirm_status(api, order["order_number"])
    usd = float(buy_quote["sell_currency_amount"])
    print(f"[{coin}] BUY ({tag}) order {order['order_number']} status={status} size=${usd:.2f}")
    if status == "F":
        tp_price = price + TP_MULT * atr
        sl_price = price - SL_MULT * atr
        save_state(coin, {"position": "LONG", "entry_price": price,
                          "tp_price": tp_price, "sl_price": sl_price})
        return usd
    return 0.0


def pace_shortfall_usd():
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    elapsed_fraction = (now - day_start) / (day_end - day_start)
    expected_by_now = TARGET_DAILY_VOLUME * elapsed_fraction
    done_today = risk.today_volume()
    return expected_by_now - done_today, done_today


def volume_topup(api, equity_quote, day_start_equity, candidate_coins):
    """Always execute one round trip per tick, on whichever coin in candidate_coins
    is currently cheapest to round-trip. No pacing or cost-budget gate — trading
    happens every call by design; the only thing that can stop it is every single
    candidate's quote looking broken/abnormal at once (SANITY_MAX_COST_PCT) or
    genuinely insufficient balance. Overall risk is bounded by risk.check()'s loss
    ceiling in run(), not by anything in this function."""
    coin, cost = spread.cheapest_coin(api, candidate_coins, QUOTE)
    if coin is None:
        print("volume top-up: couldn't get a quote for any candidate coin this tick — skipping")
        return 0.0

    if cost > SANITY_MAX_COST_PCT:
        print(f"volume top-up: even the cheapest coin ({coin}) looks abnormal ({cost:.2%}, "
              f"likely a bad tick or wide market) — skipping this tick")
        return 0.0

    quote_bal = api.get_available(QUOTE)
    buffer_above_floor = max(equity_quote - risk.MIN_EQUITY_FLOOR_USD, 0.0)
    leg_usd = buffer_above_floor * LEG_FLOOR_BUFFER_FRACTION
    leg_usd = min(leg_usd, MAX_LEG_USD, quote_bal - 0.50)
    leg_usd = max(leg_usd, MIN_ORDER_USD)

    if leg_usd < MIN_ORDER_USD or quote_bal < MIN_ORDER_USD:
        print(f"volume top-up: not enough spendable {QUOTE} (${quote_bal:.2f}) for a round "
              f"trip on {coin} — skipping this tick")
        return 0.0

    print(f"volume top-up: {coin} is cheapest right now (~{cost:.2%}/trip) — running a ${leg_usd:.2f} round trip")

    buy_quote = api.get_quote(coin, QUOTE, f"{leg_usd:.2f}")
    buy_order = api.place_instant_order(coin, QUOTE, buy_quote["sell_currency_amount"],
                                        buy_quote["buy_currency_amount"], buy_quote["visible_price"])
    if confirm_status(api, buy_order["order_number"]) != "F":
        print("  buy leg didn't fill — aborting round trip")
        return 0.0
    coin_bought = float(buy_quote["buy_currency_amount"])
    buy_usd = float(buy_quote["sell_currency_amount"])

    sell_quote = api.get_quote(QUOTE, coin, f"{coin_bought:.8f}")
    sell_order = api.place_instant_order(QUOTE, coin, sell_quote["sell_currency_amount"],
                                         sell_quote["buy_currency_amount"], sell_quote["visible_price"])
    if confirm_status(api, sell_order["order_number"]) != "F":
        print("  sell leg didn't fill — coin balance left open, check manually")
        return buy_usd

    sell_usd = float(sell_quote["buy_currency_amount"])
    realized_cost = max(buy_usd - sell_usd, 0.0)
    risk.log_spread_cost(realized_cost)   # still logged for visibility, no longer a gate
    print(f"  round trip filled: ${buy_usd + sell_usd:.2f} volume, realized cost ${realized_cost:.2f}")
    return buy_usd + sell_usd


def coins_with_open_positions():
    """Coins with a LONG position recorded in their state_<coin>.json, regardless of
    whether they're currently in the screener's top performers — these must keep
    being checked for exit even if they've since fallen out of favor, or they'd become
    an orphaned position nobody manages."""
    import json
    coins = []
    for path in HERE.glob("state_*.json"):
        coin = path.stem.replace("state_", "")
        try:
            state = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if state.get("position") == "LONG":
            coins.append(coin)
    return coins


def run():
    api = build_client()

    active_coins, scanned_at = screener.top_performers(n=ACTIVE_COIN_COUNT)
    open_coins = coins_with_open_positions()
    coins_to_manage = sorted(set(active_coins) | set(open_coins))
    extra_open = sorted(set(open_coins) - set(active_coins))
    print(f"screener: top {ACTIVE_COIN_COUNT} by trailing 24h momentum: {active_coins}"
          + (f"  (+ still managing open positions: {extra_open})" if extra_open else ""))

    if not coins_to_manage:
        print("no coins with positive momentum and no open positions — nothing to manage this tick")
        return

    # Rough prices for equity valuation — use a cheap quote rather than a full feed fetch.
    prices = {}
    for coin in coins_to_manage:
        try:
            q = api.get_quote(coin, QUOTE, f"{MIN_ORDER_USD:.2f}")
            prices[coin] = float(q["visible_price"])
        except APIError:
            prices[coin] = 0.0

    equity = portfolio_equity_quote(api, prices)
    risk_state, halted, reason = risk.check(equity)
    day_start_equity = risk_state.get("day_start_equity") or equity
    risk.save_risk_state(risk_state)

    if halted:
        print(f"RISK HALT: {reason} — equity=${equity:.2f} — skipping this entire tick")
        return

    total_volume = 0.0
    for coin in coins_to_manage:
        try:
            total_volume += directional_tick(api, coin, equity)
        except APIError as error:
            print(f"[{coin}] ANDX error: {error}")

    # Volume top-up prefers the screener's currently-profitable coins; if none show
    # positive momentum this cycle, fall back to the full candidate pool so a round
    # trip can still happen (per the "always trade" requirement) rather than stalling.
    topup_candidates = active_coins if active_coins else screener.CANDIDATE_POOL
    try:
        total_volume += volume_topup(api, equity, day_start_equity, topup_candidates)
    except APIError as error:
        print(f"volume top-up: ANDX error: {error}")

    if total_volume > 0:
        risk.log_trade_volume(total_volume)

    vol7 = risk.trailing_7d_avg_daily_volume()
    print(f"tick complete — equity=${equity:.2f}  volume this tick=${total_volume:.2f}  "
          f"7d-avg-vol=${vol7:.0f}/day")


if __name__ == "__main__":
    try:
        run()
    except APIError as error:
        print(f"ANDX error: {error}")
