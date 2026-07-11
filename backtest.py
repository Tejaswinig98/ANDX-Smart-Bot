"""Offline backtest of the ml_trend strategy over historical Coinbase bars.

Not connected to ANDX at all — pure historical simulation, useful for sanity-checking
PARAMS in strategy.py and for writing the competition's weekly reflection (bot
performance, trading logic, risk controls, lessons learned).

Run:
    python backtest.py --coin BTC --days 180
"""

import argparse

import model as model_lib
import strategy
from feed import recent_bars


def backtest(coin, days, starting_equity=1000.0):
    clf, meta = model_lib.load(coin)
    df = recent_bars(coin=coin, days=days)
    if len(df) < strategy.WARMUP + 20:
        raise SystemExit(f"Not enough bars ({len(df)}) for a meaningful backtest — try more --days.")

    equity = starting_equity
    state = {"position": "FLAT", "entry_price": 0, "tp_price": 0, "sl_price": 0}
    coin_held = 0.0
    trades = []

    for i in range(strategy.WARMUP, len(df)):
        window = df.iloc[: i + 1]
        decision = strategy.decide(window, state, clf, equity)

        if decision.action == "BUY" and state["position"] != "LONG":
            spend = min(decision.trade_usd, equity)
            if spend >= strategy.PARAMS["min_order_usd"]:
                coin_held = spend / decision.price
                equity -= spend
                state = {"position": "LONG", "entry_price": decision.price,
                         "tp_price": decision.tp_price, "sl_price": decision.sl_price}
                trades.append({"type": "BUY", "price": decision.price, "usd": spend})

        elif decision.action == "SELL" and state["position"] == "LONG":
            proceeds = coin_held * decision.price
            equity += proceeds
            pnl = proceeds - trades[-1]["usd"] if trades else 0
            trades.append({"type": "SELL", "price": decision.price, "usd": proceeds, "pnl": pnl})
            coin_held = 0.0
            state = {"position": "FLAT", "entry_price": 0, "tp_price": 0, "sl_price": 0}

    final_price = float(df["close"].iloc[-1])
    final_equity = equity + coin_held * final_price
    n_round_trips = sum(1 for t in trades if t["type"] == "SELL")
    wins = sum(1 for t in trades if t["type"] == "SELL" and t.get("pnl", 0) > 0)
    total_volume = sum(t["usd"] for t in trades)

    print(f"bars: {len(df)}  round-trip trades: {n_round_trips}  win rate: "
          f"{(wins / n_round_trips * 100) if n_round_trips else 0:.1f}%")
    print(f"total traded volume: ${total_volume:,.2f}  (avg/day over window: "
          f"${total_volume / days:,.2f})")
    print(f"starting equity: ${starting_equity:,.2f}  ending equity: ${final_equity:,.2f}  "
          f"return: {(final_equity / starting_equity - 1) * 100:+.2f}%")

    buy_hold = starting_equity / df["close"].iloc[strategy.WARMUP] * final_price
    print(f"buy & hold over same window would be: ${buy_hold:,.2f}  "
          f"({(buy_hold / starting_equity - 1) * 100:+.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--equity", type=float, default=1000.0)
    args = parser.parse_args()
    backtest(args.coin, args.days, args.equity)
