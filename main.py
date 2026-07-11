"""Sample 3: ML strategy bot. Each run fetches hourly candles, scores the latest bar with
the trained classifier (gated by a trend filter), and buys / sells on ANDX with an
ATR take-profit / stop-loss. Confidence-scaled position sizing and self-imposed
daily/weekly drawdown halts sit on top, buffered under the competition's own
50%/day and 75%/week limits. Designed to run once per call (e.g. hourly via cron);
it remembers its open position in state_<coin>.json between runs.

Trades whichever coin is set in the COIN environment variable (default BTC) — this
lets one codebase run multiple coins in parallel (see .github/workflows/run-bot.yml's
matrix), each with its own model and position state, while sharing one account-wide
risk budget and volume log (risk halts and the $2,500/day target are about your whole
account, not any single coin).

Before the first run:  python train_model.py --coin BTC   (repeat per coin)
Run:                    COIN=BTC python main.py   (needs all five ANDX_* env vars set, in .env or the shell)
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

import feed
import model as model_lib
import risk
import strategy
from andx_api import ANDX, APIError

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")   # load .env by absolute path, so it works from any working dir (e.g. cron)

# Which coin to trade and the quote currency used for sizing / risk accounting.
# COIN is overridable via env var so the same code can run multiple coins (see the
# GitHub Actions matrix in .github/workflows/run-bot.yml).
COIN = os.environ.get("COIN", "BTC")
QUOTE = "USDT"

STATE_FILE = HERE / f"state_{COIN}.json"   # per-coin, so BTC and ETH positions don't collide


def build_client():
    """Construct the ANDX client from environment variables."""
    api = ANDX(
        os.environ.get("ANDX_USER_NAME", ""),
        os.environ.get("ANDX_TOKEN", ""),
        os.environ.get("ANDX_API_KEY", ""),
        os.environ.get("ANDX_API_SECRET", ""),
        os.environ.get("ANDX_PASSPHRASE", ""),
    )
    return api


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"position": "FLAT", "entry_price": 0, "tp_price": 0, "sl_price": 0}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def confirm_status(api, order_number, tries=5):
    """Poll an order until it settles to F/C; return the last status seen."""
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
    """Total account value in QUOTE terms: quote balance + coin balance marked at price."""
    quote_bal = api.get_available(QUOTE)
    coin_bal = api.get_available(COIN)
    return quote_bal + coin_bal * price


def execute_buy(api, trade_usd):
    """Spend trade_usd of QUOTE on COIN; return the order number."""
    quote = api.get_quote(COIN, QUOTE, f"{trade_usd:.2f}")
    order = api.place_instant_order(COIN, QUOTE, quote["sell_currency_amount"],
                                    quote["buy_currency_amount"], quote["visible_price"])
    return order["order_number"], float(quote["sell_currency_amount"])


def execute_sell(api):
    """Sell the whole COIN balance back to QUOTE; return the order number and USD notional."""
    sell_amount = api.get_available(COIN)
    quote = api.get_quote(QUOTE, COIN, f"{sell_amount:.8f}")
    order = api.place_instant_order(QUOTE, COIN, quote["sell_currency_amount"],
                                    quote["buy_currency_amount"], quote["visible_price"])
    return order["order_number"], float(quote["buy_currency_amount"])


def run():
    """One tick: load model+state -> fetch bars -> risk check -> ML decision -> execute -> save."""
    api = build_client()
    state = load_state()
    clf, meta = model_lib.load(COIN)   # raises ModelNotTrained with a clear message if missing

    days = strategy.WARMUP // 24 + 3
    df = feed.recent_bars(COIN, days=days)
    if len(df) < strategy.WARMUP:
        print(f"warmup: {len(df)}/{strategy.WARMUP} bars - skipping")
        return

    price_now = float(df["close"].iloc[-1])
    equity = account_equity_quote(api, price_now)

    risk_state, halted, reason = risk.check(equity)
    if halted:
        print(f"RISK HALT: {reason} — equity={equity:.2f} {QUOTE}, no new entries this tick")
        risk.save_risk_state(risk_state)
        return
    risk.save_risk_state(risk_state)

    decision = strategy.decide(df, state, clf, equity)
    vol7 = risk.trailing_7d_avg_daily_volume()
    print(f"{COIN}/{QUOTE} price: {decision.price:.2f}  pos: {state['position']}  "
          f"p(up)={decision.prob_up:.2f}  equity={equity:.2f}  7d-avg-vol=${vol7:.0f}/day  "
          f"-> {decision.action} ({decision.note})")

    if decision.action == "BUY" and state["position"] != "LONG":
        order_number, usd_spent = execute_buy(api, decision.trade_usd)
        status = confirm_status(api, order_number)
        print(f"BUY order {order_number} status: {status}  size=${usd_spent:.2f}")
        if status == "F":
            state = {"position": "LONG", "entry_price": decision.price,
                     "tp_price": decision.tp_price, "sl_price": decision.sl_price}
            save_state(state)
            risk.log_trade_volume(usd_spent)

    elif decision.action == "SELL" and state["position"] == "LONG":
        order_number, usd_received = execute_sell(api)
        status = confirm_status(api, order_number)
        print(f"SELL order {order_number} status: {status}  size=${usd_received:.2f}")
        if status == "F":
            state = {"position": "FLAT", "entry_price": 0, "tp_price": 0, "sl_price": 0}
            save_state(state)
            risk.log_trade_volume(usd_received)


if __name__ == "__main__":
    try:
        run()
    except model_lib.ModelNotTrained as error:
        print(error)
    except APIError as error:
        print(f"ANDX error: {error}")
