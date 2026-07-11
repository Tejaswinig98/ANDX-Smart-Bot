"""Volume top-up bot: runs frequently (e.g. every 15 min) alongside main.py's hourly
ML bot. The competition ranks weekly winners by TOTAL VALID USD TRADING VOLUME first,
with return/Sharpe only as tiebreakers — so a bot that only trades on high-confidence
ML signals (main.py) will often sit idle and under-qualify, even if its calls are good.

This script closes that gap: it checks how much volume you've done today vs. a linear
pace toward TARGET_DAILY_VOLUME, and if you're behind, executes one small buy-then-sell
round trip (both legs count toward valid volume) sized to close the gap — capped by a
fraction of equity and gated by the same risk halts as main.py.

It does NOT touch state.json (main.py's position tracking) — it always round-trips back
to flat within the same run, so it never leaves an open position for main.py to trip
over. It DOES share main.py's risk halts and volume log.

Run this on its own schedule (see .github/workflows/volume-topup.yml), sharing a
concurrency group with run-bot.yml so the two never execute against your account at
the same time.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

import risk
from andx_api import ANDX, APIError

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

COIN = os.environ.get("COIN", "BTC")
QUOTE = "USDT"

# Aim above the competition's $2,500/day minimum so a few slow days don't drag the
# 7-day rolling average below the qualification bar.
TARGET_DAILY_VOLUME = 2000.0

MIN_ORDER_USD = 5.0          # ANDX competition minimum order size
MAX_EQUITY_FRACTION = 0.08   # never risk more than this fraction of equity on one round trip (was 0.20)
MAX_LEG_USD = 25.0           # hard dollar cap per leg, regardless of equity, as a sanity backstop (was 50.0)


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


def account_equity_quote(api, price):
    quote_bal = api.get_available(QUOTE)
    coin_bal = api.get_available(COIN)
    return quote_bal + coin_bal * price


def pace_shortfall_usd():
    """How far behind a linear pace toward TARGET_DAILY_VOLUME we are, right now."""
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    elapsed_fraction = (now - day_start) / (day_end - day_start)
    expected_by_now = TARGET_DAILY_VOLUME * elapsed_fraction
    done_today = risk.today_volume()
    return expected_by_now - done_today, done_today


def round_trip(api, leg_usd):
    """Buy leg_usd of COIN, then immediately sell it straight back. Returns total
    USD volume actually executed across both legs (0 if either leg didn't fill)."""
    buy_quote = api.get_quote(COIN, QUOTE, f"{leg_usd:.2f}")
    buy_order = api.place_instant_order(COIN, QUOTE, buy_quote["sell_currency_amount"],
                                        buy_quote["buy_currency_amount"], buy_quote["visible_price"])
    buy_status = confirm_status(api, buy_order["order_number"])
    if buy_status != "F":
        print(f"  round-trip buy leg not filled (status={buy_status}) — skipping sell leg")
        return 0.0
    coin_bought = float(buy_quote["buy_currency_amount"])
    buy_usd = float(buy_quote["sell_currency_amount"])

    sell_quote = api.get_quote(QUOTE, COIN, f"{coin_bought:.8f}")
    sell_order = api.place_instant_order(QUOTE, COIN, sell_quote["sell_currency_amount"],
                                         sell_quote["buy_currency_amount"], sell_quote["visible_price"])
    sell_status = confirm_status(api, sell_order["order_number"])
    if sell_status != "F":
        print(f"  round-trip sell leg not filled (status={sell_status}) — coin balance left open, "
              f"main.py's next tick will see it as untracked (manual check recommended)")
        return buy_usd

    sell_usd = float(sell_quote["buy_currency_amount"])
    return buy_usd + sell_usd


def run():
    api = build_client()

    quote = api.get_quote(COIN, QUOTE, f"{MIN_ORDER_USD:.2f}")
    price = float(quote["visible_price"])
    quote_available = api.get_available(QUOTE)
    coin_available = api.get_available(COIN)
    equity = quote_available + coin_available * price

    risk_state, halted, reason = risk.check(equity)
    risk.save_risk_state(risk_state)
    if halted:
        print(f"RISK HALT: {reason} — skipping volume top-up this tick")
        return

    shortfall, done_today = pace_shortfall_usd()
    if shortfall <= 0:
        print(f"on pace: ${done_today:.2f} done today (target ${TARGET_DAILY_VOLUME:.0f}) — no top-up needed")
        return

    leg_usd = shortfall / 2   # a round trip's buy+sell legs together cover the shortfall
    leg_usd = max(leg_usd, MIN_ORDER_USD)
    # Cap by a fraction of total equity AND by actual spendable USDT cash — total equity
    # includes coin holdings that can't be spent on the buy leg, so cash is the real limit.
    leg_usd = min(leg_usd, equity * MAX_EQUITY_FRACTION, MAX_LEG_USD, quote_available - 0.50)

    if leg_usd < MIN_ORDER_USD:
        print(f"not enough spendable {QUOTE} for a top-up round trip "
              f"(available=${quote_available:.2f}, equity=${equity:.2f}) — skipping. "
              f"If most of your balance is sitting in {COIN}, consider selling some back to {QUOTE} "
              f"so there's cash for round trips.")
        return

    print(f"behind pace by ${shortfall:.2f} (done ${done_today:.2f}/${TARGET_DAILY_VOLUME:.0f} today) "
          f"— running a ${leg_usd:.2f} round trip")
    volume = round_trip(api, leg_usd)
    if volume > 0:
        risk.log_trade_volume(volume)
        print(f"round trip complete: ${volume:.2f} valid volume logged")


if __name__ == "__main__":
    try:
        run()
    except APIError as error:
        print(f"ANDX error: {error}")
