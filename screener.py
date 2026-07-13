"""Screens a broad candidate pool by recent price performance ("profitable" = recent
upward momentum — this is the honest, implementable meaning; nothing can predict
future profit) and returns the top performers to actually trade.

Re-scanning ~60 coins' price history every 15-minute tick would be heavy (each coin
needs a Coinbase fetch), so results are cached and only refreshed once per
REFRESH_MINUTES — intervening ticks reuse the cached ranking.
"""

import json
import time
from pathlib import Path

import feed

# A broad candidate pool. Not every ticker here will be listed on ANDX or have
# Coinbase data — both feed.recent_bars and downstream quote calls fail gracefully
# per-coin, so it's safe to list generously and let unavailable ones drop out on
# their own rather than needing an exact, curated list.
CANDIDATE_POOL = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "NEAR", "AAVE", "TRX", "SUI", "WLD",
    "LTC", "LINK", "MATIC", "AVAX", "DOT", "ATOM", "ALGO", "XLM", "ETC", "UNI", "BCH",
    "FIL", "APT", "ARB", "OP", "INJ", "RENDER", "TIA", "SEI", "STX", "IMX", "GRT",
    "SAND", "MANA", "AXS", "CRV", "MKR", "COMP", "SNX", "1INCH", "ENS", "LDO",
    "FET", "RUNE", "KAVA", "MINA", "ROSE", "ZEC", "DASH", "XTZ", "EGLD", "FLOW",
    "CHZ", "ENJ", "BAT", "ZRX", "YFI", "OMG", "SKL",
]

RANK_CACHE_PATH = Path(__file__).parent / "coin_rankings.json"
REFRESH_MINUTES = 60          # how often to re-scan the full pool
LOOKBACK_DAYS = 1             # momentum window: trailing ~24h return
MIN_BARS_REQUIRED = 20        # skip a coin if Coinbase returns too little history


def _trailing_return(coin):
    """Return the coin's trailing LOOKBACK_DAYS pct return, or None if data's
    unavailable (not listed on Coinbase, fetch failure, etc.) — callers should skip
    coins that return None rather than treating them as a 0% return."""
    try:
        df = feed.recent_bars(coin, days=LOOKBACK_DAYS)
    except Exception:
        return None
    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None
    first_close = float(df["close"].iloc[0])
    last_close = float(df["close"].iloc[-1])
    if first_close <= 0:
        return None
    return (last_close / first_close) - 1


def _rescan_pool():
    """Fetch trailing performance for every candidate; return a sorted list of
    {coin, return} dicts, best performer first, positive-return only."""
    results = []
    for coin in CANDIDATE_POOL:
        ret = _trailing_return(coin)
        if ret is not None:
            results.append({"coin": coin, "return": ret})
    results.sort(key=lambda r: r["return"], reverse=True)
    return results


def top_performers(n=5, refresh_minutes=REFRESH_MINUTES):
    """Return the top n coins by trailing return (positive only), using a cached scan
    if it's fresh enough. Returns a list of coin tickers, best performer first — may
    be shorter than n if fewer than n candidates currently have positive momentum."""
    now = time.time()
    cache = None
    if RANK_CACHE_PATH.exists():
        cache = json.loads(RANK_CACHE_PATH.read_text())

    if cache is None or (now - cache.get("scanned_at", 0)) > refresh_minutes * 60:
        results = _rescan_pool()
        cache = {"scanned_at": now, "results": results}
        RANK_CACHE_PATH.write_text(json.dumps(cache, indent=2))

    positive = [r for r in cache["results"] if r["return"] > 0]
    top = positive[:n]
    return [r["coin"] for r in top], cache["scanned_at"]
