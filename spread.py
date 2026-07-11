"""Estimates the real cost of a buy-then-sell round trip on a coin, using two live
quotes — no money is spent, ANDX's get_quote is a dry-run price check.

This exists because ANDX's instant-order spread is a real, measurable cost (observed
in practice at roughly 0.6-1% per round trip on BTC/USDT) — generating volume without
checking this first means bleeding money on every trip regardless of any signal.
"""

from andx_api import APIError


def estimate_round_trip_cost_pct(api, coin, quote, probe_usd=10.0):
    """Return the estimated round-trip cost as a fraction (e.g. 0.007 = 0.7%), or None
    if quoting failed. Costs are always >= 0 in practice (spread works against you both
    ways); a negative reading would mean something unusual and is treated as 0 cost."""
    try:
        buy_quote = api.get_quote(coin, quote, f"{probe_usd:.2f}")
        coin_amount = float(buy_quote["buy_currency_amount"])
        sell_quote = api.get_quote(quote, coin, f"{coin_amount:.8f}")
        usd_back = float(sell_quote["buy_currency_amount"])
    except (APIError, KeyError, ValueError):
        return None

    cost_pct = 1 - (usd_back / probe_usd)
    return max(cost_pct, 0.0)


def cheapest_coin(api, coins, quote, probe_usd=10.0):
    """Estimate round-trip cost for each coin in `coins`; return (coin, cost_pct) for
    the cheapest one, or (None, None) if every quote attempt failed."""
    best_coin, best_cost = None, None
    for coin in coins:
        cost = estimate_round_trip_cost_pct(api, coin, quote, probe_usd)
        if cost is None:
            continue
        if best_cost is None or cost < best_cost:
            best_coin, best_cost = coin, cost
    return best_coin, best_cost
