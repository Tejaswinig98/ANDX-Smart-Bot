"""Risk management and volume tracking for the ANDX competition.

ANDX's own rules (see competition rules) already suspend an account for 24h if it
loses >50% in a day, and disqualify it from that week's rankings if it loses >75% in a
week. This module adds tighter, self-imposed limits well below those hard lines, PLUS
an absolute, permanent ceiling on total loss from your starting balance — because the
daily/weekly checks alone reset each period, so a string of "acceptable" daily losses
could otherwise still add up past your real risk tolerance without ever tripping either
one individually.

It also keeps a rolling log of executed trade notionals, since the competition's
weekly leaderboard requires an average of $2,500/day in valid USD trading volume over
7 days — trades are stamped here so main.py/smart_bot.py can report progress toward
that bar.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Self-imposed limits, buffered below ANDX's hard suspension/disqualification lines.
DAILY_LOSS_HALT = 0.08     # halt new entries once the day is down 8%
WEEKLY_LOSS_HALT = 0.15    # halt new entries once the week is down 15%

# Hard, permanent ceiling: total loss from your very first recorded balance. Once
# breached, ALL new entries stay halted indefinitely — not just for 24h — until you
# manually clear it (delete risk_state.json's "baseline_equity"/"permanent_halt" keys,
# or the whole file, once you've reviewed what happened and decided to keep going).
ABSOLUTE_LOSS_HALT = 0.25
RISK_STATE_PATH = Path(__file__).parent / "risk_state.json"
VOLUME_LOG_PATH = Path(__file__).parent / "volume_log.json"


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_risk_state(equity_now):
    """Load (or initialize) the day/week reference equity used for drawdown checks,
    plus the permanent baseline_equity used for the absolute loss ceiling."""
    if RISK_STATE_PATH.exists():
        state = json.loads(RISK_STATE_PATH.read_text())
    else:
        state = {}

    # Set once, ever — this is your starting point for the 25% absolute ceiling.
    # Never overwritten automatically; only cleared by manually editing/deleting the
    # file once you've deliberately decided to reset it.
    if "baseline_equity" not in state:
        state["baseline_equity"] = equity_now

    today = _today()
    if state.get("day") != today:
        state["day"] = today
        state["day_start_equity"] = equity_now
        state["halted_until"] = state.get("halted_until", 0)   # daily halt may still be active

    if "week_start_day" not in state or _week_elapsed(state["week_start_day"]):
        state["week_start_day"] = today
        state["week_start_equity"] = equity_now

    state.setdefault("halted_until", 0)
    state.setdefault("permanent_halt", False)
    return state


def _week_elapsed(week_start_day):
    started = datetime.strptime(week_start_day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= started + timedelta(days=7)


def save_risk_state(state):
    RISK_STATE_PATH.write_text(json.dumps(state, indent=2))


def check(equity_now):
    """Evaluate drawdown limits for the current equity; return (state, halted, reason).
    Persists any day/week rollover. Does NOT persist a halt trigger itself — call
    save_risk_state(state) after, once the caller has finished acting on the result."""
    state = load_risk_state(equity_now)

    # Absolute ceiling first: once total loss from baseline_equity hits 25%, halt
    # permanently — this does NOT reset daily/weekly and stays on until you manually
    # clear it, unlike the halts below.
    baseline = state.get("baseline_equity") or equity_now
    absolute_loss = (baseline - equity_now) / baseline if baseline else 0
    if state.get("permanent_halt") or absolute_loss >= ABSOLUTE_LOSS_HALT:
        state["permanent_halt"] = True
        return state, True, (f"PERMANENT HALT: total loss {absolute_loss:.0%} from starting "
                              f"balance ${baseline:.2f} >= your {ABSOLUTE_LOSS_HALT:.0%} ceiling. "
                              f"Trading stays off until you manually review and clear risk_state.json.")

    now_ts = time.time()
    if state["halted_until"] > now_ts:
        remaining_h = (state["halted_until"] - now_ts) / 3600
        return state, True, f"self-imposed daily halt active ({remaining_h:.1f}h remaining)"

    day_start = state["day_start_equity"] or equity_now
    daily_loss = (day_start - equity_now) / day_start if day_start else 0
    if daily_loss >= DAILY_LOSS_HALT:
        state["halted_until"] = now_ts + 24 * 3600
        return state, True, f"daily loss {daily_loss:.0%} >= {DAILY_LOSS_HALT:.0%} halt threshold"

    week_start = state["week_start_equity"] or equity_now
    weekly_loss = (week_start - equity_now) / week_start if week_start else 0
    if weekly_loss >= WEEKLY_LOSS_HALT:
        return state, True, f"weekly loss {weekly_loss:.0%} >= {WEEKLY_LOSS_HALT:.0%} halt threshold"

    return state, False, ""


def log_trade_volume(usd_amount):
    """Append an executed trade's USD notional, for tracking progress toward the
    competition's $2,500/day average-volume qualification bar."""
    log = json.loads(VOLUME_LOG_PATH.read_text()) if VOLUME_LOG_PATH.exists() else []
    log.append({"ts": int(time.time()), "usd": round(float(usd_amount), 2)})
    # Keep ~30 days of history; no need to grow this file forever.
    cutoff = time.time() - 30 * 86400
    log = [entry for entry in log if entry["ts"] >= cutoff]
    VOLUME_LOG_PATH.write_text(json.dumps(log, indent=2))


def trailing_7d_avg_daily_volume():
    """Return the average daily USD volume traded over the last 7 days."""
    if not VOLUME_LOG_PATH.exists():
        return 0.0
    log = json.loads(VOLUME_LOG_PATH.read_text())
    cutoff = time.time() - 7 * 86400
    total = sum(entry["usd"] for entry in log if entry["ts"] >= cutoff)
    return total / 7


def today_volume():
    """Return USD volume logged so far today (UTC calendar day)."""
    if not VOLUME_LOG_PATH.exists():
        return 0.0
    log = json.loads(VOLUME_LOG_PATH.read_text())
    day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = day_start.timestamp()
    return sum(entry["usd"] for entry in log if entry["ts"] >= cutoff)


# --- Spread-cost budget: separate from the drawdown halts above. Round-trip volume
# generation has a real, measurable cost (see spread.py) even when nothing is "wrong" —
# this caps how much of the day's starting equity can be deliberately spent on that
# cost, so a volume target never gets chased past what the account can safely absorb.
SPREAD_COST_LOG_PATH = Path(__file__).parent / "spread_cost_log.json"
DAILY_SPREAD_COST_BUDGET_FRACTION = 0.01   # max 1% of day-start equity spent on round-trip spread per day


def log_spread_cost(cost_usd):
    log = json.loads(SPREAD_COST_LOG_PATH.read_text()) if SPREAD_COST_LOG_PATH.exists() else []
    log.append({"ts": int(time.time()), "usd": round(float(cost_usd), 4)})
    cutoff = time.time() - 30 * 86400
    log = [entry for entry in log if entry["ts"] >= cutoff]
    SPREAD_COST_LOG_PATH.write_text(json.dumps(log, indent=2))


def today_spread_cost():
    """Return USD spent on round-trip spread cost so far today (UTC calendar day)."""
    if not SPREAD_COST_LOG_PATH.exists():
        return 0.0
    log = json.loads(SPREAD_COST_LOG_PATH.read_text())
    day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = day_start.timestamp()
    return sum(entry["usd"] for entry in log if entry["ts"] >= cutoff)


def spread_cost_budget_remaining(day_start_equity):
    """USD still available today for deliberate round-trip spread spend, per
    DAILY_SPREAD_COST_BUDGET_FRACTION. Never negative."""
    budget = day_start_equity * DAILY_SPREAD_COST_BUDGET_FRACTION
    spent = today_spread_cost()
    return max(budget - spent, 0.0)
