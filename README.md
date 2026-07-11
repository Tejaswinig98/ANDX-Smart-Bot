# sample-3 — ML strategy bot (RandomForest signal + trend filter + ATR exits)

Sample-2 traded a fixed EMA/RSI rule. This sample replaces the hand-written entry rule
with a **RandomForest classifier** trained on historical price/volume behaviour to
estimate the probability that price rises meaningfully over the next few hours. That
probability is combined with a trend filter and used to size, enter, and exit trades —
still with the same ATR take-profit/stop-loss skeleton as sample-2, plus drawdown halts
and volume tracking aimed at the competition rules.

**This is a starting point, not a guarantee of profit.** No model predicts markets
reliably; treat `PARAMS` and `train_model.py`'s settings as things to tune and validate,
not as finished answers. Read `strategy.py` before risking real funds.

## Risk limits: capped at 25% total loss, permanently

`risk.py` enforces three layers, tightest first:

1. **Absolute ceiling (25%, permanent):** tracked from the very first equity value
   ever recorded (`baseline_equity` in `risk_state.json`, set once and never reset).
   If your total loss from that starting point ever reaches 25%, **all new entries
   halt indefinitely** — not just for 24h like the daily halt below. It stays halted
   even if equity later recovers, since the flag is sticky (`permanent_halt: true`) —
   you have to deliberately review what happened and edit/delete `risk_state.json` to
   resume. This is the backstop that actually limits your worst case; the other two
   below are early warnings that fire well before you'd ever reach it.
2. **Daily halt (8%):** pauses new entries for 24h if today's loss hits 8%.
3. **Weekly halt (15%):** pauses new entries if this week's loss hits 15%.

Alongside these, `smart_bot.py`'s per-trade risk is capped at 8% of equity
(`MAX_TRADE_EQUITY_FRACTION`), stop-losses are tighter (1×ATR instead of 1.5×), and
the volume top-up's daily spread-cost budget is capped at 1% of equity
(`DAILY_SPREAD_COST_BUDGET_FRACTION` in `risk.py`) — down from the original 3%, since
live testing showed real round-trip spread costs (~0.6-1% per trip on BTC/USDT) add up
fast without a tight budget.

**None of this is a guarantee** — thresholds are checked once per tick (e.g. every 15
min), so a fast, sharp move between ticks could still cross a limit before the bot has
a chance to react. Treat 25% as the target ceiling under normal conditions, not an
absolute physical impossibility.

## Files
- `andx_api.py` — ANDX client (quote, place order, balance, order status) — unchanged from sample-2.
- `feed.py` — hourly OHLCV candles from Coinbase (public, no key needed), used for both training and live signals.
- `features.py` — causal feature engineering (returns, EMA spread, RSI, MACD, Bollinger %B, ATR%, volume z-score) shared by training and live inference.
- `train_model.py` — offline script: downloads history, builds features + a forward-return label, trains a RandomForest, saves `model_<coin>.pkl` + `model_meta_<coin>.json`.
- `model.py` — loads the trained model for a given coin for live use.
- `strategy.py` — `ml_trend` logic: BUY when in an uptrend **and** the model is confident (`p(up) >= 0.56`), size scales with confidence, exit on ATR TP/SL or if the model turns bearish.
- `risk.py` — self-imposed daily/weekly drawdown halts buffered under ANDX's 50%/day and 75%/week limits, plus a rolling trade-volume log (account-wide, shared across all coins).
- `main.py` — the runner: load state → fetch bars → risk check → ML decision → trade → save state. Trades whichever coin is set in the `COIN` env var (default `BTC`).
- `volume_topup.py` — paces small round-trip trades to hit the competition's $2,500/day volume bar; also reads `COIN` from the environment.
- `backtest.py` — offline historical simulation of the strategy, for tuning and for your weekly reflection write-up.
- `state_<coin>.json`, `risk_state.json`, `volume_log.json` — created at runtime (not gitignored — see "Run it on GitHub").

## Trading more than one coin
`main.py` and `volume_topup.py` both read which coin to trade from the `COIN`
environment variable (default `BTC`), and each coin gets its own model file
(`model_<coin>.pkl`) and position-tracking file (`state_<coin>.json`) — so BTC and ETH
(or any other coin) can run side by side without stepping on each other. Risk halts and
the volume log stay **account-wide** on purpose, since your equity and the $2,500/day
requirement are both about your whole account, not any single pair.

To run a second coin locally:
```
python train_model.py --coin ETH --days 720
COIN=ETH python main.py
```
The GitHub Actions workflows already do this for you via a matrix — see below.

## How it trades
1. **Flat:** requires EMA-fast > EMA-slow (uptrend) *and* the model's `p(up) >= buy_prob`
   (default 0.56). Trade size scales from `base_trade_usd` up to 2x as confidence rises
   above the threshold, capped at `max_trade_fraction` (15%) of account equity, and
   floored at ANDX's $5 minimum order.
2. **Long:** exits on `+3×ATR` take-profit, `-1.5×ATR` stop-loss, **or** early if the
   model's `p(up)` drops to/below `exit_prob` (0.42) — cutting losers the trend filter
   alone wouldn't catch.
3. **Risk halts:** before every entry decision, `risk.py` checks account equity against
   the day's and week's starting value. If drawdown hits 40%/day or 60%/week (buffered
   under ANDX's 50%/75% rules) new entries are blocked — existing exits still fire.
4. **Volume tracking:** every filled trade's USD notional is logged; `main.py` prints
   your trailing 7-day average daily volume each run so you can see progress toward the
   competition's $2,500/day qualification bar.

## Setup
```
pip install -r requirements.txt
cp .env.example .env        # fill in your ANDX credentials (see STEP 2 at https://andx.ai/set-up-a-bot)
```

## 1. Train the model (do this first, and periodically re-run it)
```
python train_model.py --coin BTC --days 720
python train_model.py --coin ETH --days 720
```
This downloads ~2 years of public hourly Coinbase candles, builds features, trains the
classifier on an 85/15 **chronological** split (never shuffled — that would leak future
data into training), and prints a classification report + AUC + feature importances so
you can judge whether the signal is actually useful before trading it live. It writes
`model_<coin>.pkl` and `model_meta_<coin>.json`.

Re-run this every week or so with fresh data so the model doesn't go stale — markets
regime-shift, and a model trained on old data quietly decays.

## 2. Sanity-check it offline
```
python backtest.py --coin BTC --days 180
python backtest.py --coin ETH --days 180
```
Simulates the exact same `strategy.decide()` logic bar-by-bar over historical data (no
ANDX calls at all) and prints trade count, win rate, traded volume, and return vs.
buy-and-hold. Use this to tune `PARAMS` in `strategy.py` (e.g. `buy_prob`, `tp_mult`,
`sl_mult`) before going live — this is also good raw material for the competition's
weekly reflection (bot performance, trading logic, risk controls, lessons learned).

## 3. Run live
```
python main.py                 # trades BTC (the default)
COIN=ETH python main.py        # trades ETH
```
Sample output (a tick that opens a long):
```
BTC/USDT price: 60139.52  pos: FLAT  p(up)=0.61  equity=105.32  7d-avg-vol=$14/day  -> BUY (uptrend + model p(up)=0.61)
BUY order 12345678 status: F  size=$9.10
```
Heads up: a BUY or SELL places a **real order** on ANDX every time it triggers.

## Schedule it (cron, Linux/macOS)
Same pattern as sample-2 — one tick per run, cron repeats it hourly, once per coin:
```
7 * * * * cd /path/to/sample-3 && COIN=BTC /usr/bin/python3 main.py >> bot.log 2>&1
7 * * * * cd /path/to/sample-3 && COIN=ETH /usr/bin/python3 main.py >> bot.log 2>&1
```
On Windows, use **Task Scheduler** to run `python main.py` in this folder hourly.

## Tuning notes
- `strategy.PARAMS["buy_prob"]` — raise it (e.g. 0.60+) for fewer, higher-conviction
  trades; lower it for more trade frequency (helps hit the $2,500/day volume
  requirement, at the cost of signal quality — check the backtest first).
- `train_model.HORIZON_BARS` / `LABEL_THRESHOLD` — define what "a good trade" means
  during training (how many hours ahead, and how big a move counts). Shorter horizons
  trade more often but are noisier.
- `risk.DAILY_LOSS_HALT` / `WEEKLY_LOSS_HALT` — tighten these if you want the bot to
  step aside earlier than the competition's own suspension/disqualification lines.

## Winning the competition: volume first, not just prediction accuracy

Re-read the competition rules carefully: **weekly winners are determined by total
valid trading volume**, with account return and Sharpe ratio used only to break ties.
A bot that only trades on high-confidence ML signals (like `main.py` alone) will
often sit idle for hours — good discipline for real trading, but it may not even clear
the $2,500/day average volume needed to qualify for the leaderboard, let alone win it.

`volume_topup.py` closes that gap. It runs frequently (every 15 min via
`.github/workflows/volume-topup.yml`), checks whether you're behind a linear pace
toward a $3,000/day target (buffered above the $2,500 minimum), and if so executes one
small buy-then-immediately-sell round trip sized to close the gap — both legs count as
valid volume. It shares `main.py`'s risk halts, so it stops right alongside the ML bot
if you hit a daily/weekly drawdown limit, and it never touches `state_<coin>.json`, so it
can't confuse the ML bot's position tracking.

**Be aware of the real cost here:** even with no visible fee line, ANDX's quote likely
has some spread baked in, so every round trip probably costs a little. Farming volume
aggressively without limits could quietly erode your equity. `MAX_EQUITY_FRACTION`
and `MAX_LEG_USD` in `volume_topup.py` cap how much each round trip risks — start
conservative, watch a day or two of results, and adjust `TARGET_DAILY_VOLUME` and
those caps once you can see the actual spread cost per trade in your ANDX trade
history.

**Both bots need to be running for this to work as intended:** `run-bot.yml` for
directional ML trades (helps your return/Sharpe tiebreaker), `volume-topup.yml` for
qualification volume. They share a GitHub Actions concurrency group
(`andx-bot-trading`) so they never execute against your account simultaneously.

## smart_bot.py — multi-indicator, multi-coin, spread-aware (recommended)

`main.py` + `volume_topup.py` (above) were the first version: one coin, ML-only
directional trades, and a volume top-up that chased a fixed dollar target regardless
of cost. Live testing showed that cost matters a lot — ANDX's instant-order spread on
BTC/USDT round trips measured out at roughly **0.6-1% per trip**, which is enough to
meaningfully erode a small account if you generate volume without checking it first.

`smart_bot.py` is the fixed version. Differences:

- **Multi-indicator confluence** (`signals.py`): combines SMA(20/50) trend, RSI(14)
  overbought/oversold, Bollinger Bands(20,2) mean-reversion positioning, and MACD
  histogram momentum into a single -4..+4 score, blended 50/50 with the coin's ML
  model probability if one is trained (falls back to technicals-only otherwise — so a
  coin can trade even before you've trained a model for it).
- **Multi-coin** (`COINS = ["BTC", "ETH", "SOL", "XRP"]` in `smart_bot.py`): scans all
  of them each tick, both for directional entries and to find the *cheapest* coin to
  round-trip for volume.
- **Spread-cost aware** (`spread.py` + `risk.py`'s spread-cost budget): before any
  volume top-up, it measures the actual round-trip cost via two dry-run quotes (no
  money spent) and only trades the cheapest available coin — and only up to a capped
  daily spread-cost budget (default 3% of the day's starting equity,
  `DAILY_SPREAD_COST_BUDGET_FRACTION` in `risk.py`). If every coin's spread is too
  expensive, or the budget's used up, it skips the top-up rather than force a losing
  trade to hit a number.
- **Directional entries require real agreement**: `ENTRY_SCORE = 0.55` (out of ±1)
  means multiple indicators need to point the same way, not just one.

**Be honest with yourself about what this can and can't do:** no bot can guarantee
$2,000/day in volume with zero risk of loss — every trade carries some cost. What this
design actually gives you is a bot that (a) only takes directional trades with
multi-signal agreement, (b) only spends a small, capped, pre-measured budget on
volume-generation cost, and (c) still has the daily/weekly drawdown halts underneath
as a hard backstop. Some days it will likely fall short of $2,000 rather than force a
bad trade — track your actual results with `backtest.py` per coin and adjust
`TARGET_DAILY_VOLUME`, `ENTRY_SCORE`, and `DAILY_SPREAD_COST_BUDGET_FRACTION` from
real data, not guesses.

**If you switch to smart_bot.py, disable `run-bot.yml` and `volume-topup.yml`**
(Actions tab → each workflow → "..." menu → Disable workflow) so they don't also place
trades alongside it — `smart-bot.yml` already covers both directional trades and
volume top-up in one tick.

Run it locally the same way as the others:
```
python smart_bot.py
```

## Run it on GitHub (instead of your own machine/cron)

GitHub Actions can run and schedule the bot for you, with no computer of your own left
running. The catch: Actions runners are **stateless** — each run starts from a fresh
checkout — so `state_<coin>.json`, `risk_state.json`, `volume_log.json`, and `model_<coin>.pkl` all
need to be committed back to the repo after every run, or the bot forgets its position
and risk counters between ticks. The two workflows in `.github/workflows/` do this for
you. **Use a private repository** — trade sizes and decisions show up in Action logs.

1. **Create the repo.**
   On GitHub: New repository → **Private** → don't initialize with a README (you already have one).

2. **Push this folder to it.**
   ```
   cd sample-3
   git init
   git add .
   git commit -m "initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
   `.env` is gitignored, so your credentials never get committed — good.

3. **Add your ANDX credentials as repo secrets.**
   Repo → Settings → Secrets and variables → Actions → New repository secret. Add all five:
   `ANDX_USER_NAME`, `ANDX_TOKEN`, `ANDX_API_KEY`, `ANDX_API_SECRET`, `ANDX_PASSPHRASE`.
   The `run-bot` workflow writes these into a `.env` file at the start of each run.

4. **Enable Actions** if prompted (Actions tab → "I understand my workflows, go ahead and enable them").

5. **Train the model.**
   Actions tab → **train-model** → Run workflow (leave the defaults, or set `coin`/`days`) → Run workflow.
   This downloads history, trains, and commits `model_<coin>.pkl` + `model_meta_<coin>.json` back to the repo (defaults to both BTC and ETH).
   Wait for it to finish (a couple of minutes) before the next step.

6. **(Optional) do one manual test tick.**
   Actions tab → **run-bot** → Run workflow. Check the log — you should see a price/decision
   line like in the sample output above. If it errors on a missing `model_<coin>.pkl`, step 5 hasn't
   finished yet.

7. **Let it run.**
   `run-bot` is scheduled for `7 * * * *` (7 minutes past every hour) and `train-model` for
   Mondays at 06:00 UTC — both editable in the `cron:` line of their `.yml` files. Every
   `run-bot` tick commits a `bot tick: update state` commit; every `train-model` run commits
   a `retrain model` commit. That's expected — it's how state persists across stateless runs.

8. **Monitor it.**
   Actions tab → click into a `run-bot` run → the log shows the same price/decision/order
   output as running it locally. GitHub disables scheduled workflows after 60 days with no
   repository activity — the bot's own commits count as activity, so this stays alive as
   long as it's ticking, but check in periodically regardless.

## More endpoints
This bot uses the same handful of ANDX endpoints as sample-1/sample-2. For the full
API, see the docs at [docs.andx.one](https://docs.andx.one).
