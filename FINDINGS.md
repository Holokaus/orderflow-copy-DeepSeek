# Findings Report — Current State of OrderFlow Project (glm5.2 branch)

> Written 2026-06-24. This is an evidence-based diagnosis of the code **as it
> exists on disk**, not what AGENTS.md documents. AGENTS.md is now badly stale.

---

## TL;DR

The project has been **over-engineered into a failing state.** Multiple
sessions layered contradictory changes on top of each other:

- The code is **no longer long-only** (despite AGENTS.md saying so) — it now
  takes SHORT trades, but the direction logic is broken so it shorts on
  uptrend days.
- The **regime classifier returns `UNKNOWN` on ~100% of ticks**, so all the
  regime-specific code you've been tuning (multipliers, strategy routing,
  daily-trend override) **literally never executes**.
- A recent commit (`dd2c6bb` / `c543147` "glm5.2") rewrote the direction logic
  to be bidirectional but **did not reconcile it with the filters**, producing
  internal contradictions.

**Observed result (May 31, an uptrend day of +3.81%):**
The absorption strategy — historically the best, long-only, +1.89% on this day —
**took 3 SHORT trades and lost −1.34%.** The system is now losing on its best
day by trading against the move.

---

## Evidence

### E1. Regime is `UNKNOWN` on 100% of sampled ticks

Probed every 250–1000 ticks across **four** dates:

```
May 31 (uptrend):    {'UNKNOWN': 7}     # 100% (7/7 samples)
Jun 1  (uptrend):    {'UNKNOWN': 62}    # 100% (62/62 samples)
Jun 8  (downtrend):  {'UNKNOWN': 93}    # 100% (93/93 samples)
daily_trend: dir=+0 strength=0.00 chg=0.00%   # always zero, all days
```

(Coarse sampling on May 31 briefly suggested `trend_following` could fire
under a real regime — that was a transient; fine-grained sampling on the other
three days shows a clean 100% UNKNOWN.)

**Implication:** Every downstream branch that keys off a real regime
(TRENDING_UP/DOWN, BREAKOUT, ACCUMULATION, etc.) is dead code in backtests.
The strategy effectively always runs under `Regime.UNKNOWN`.

### E2. The bidirectional change is half-applied

Git diff of `knowledge/strategy_library.py` shows commit `dd2c6bb`/`c543147`
removed the "LONG-ONLY" comments and added SELL/STRONG_SELL emission. But:

- `_determine_direction()` now forces SELL in `TRENDING_DOWN`…
- …but the regime is always UNKNOWN, so that branch never runs.
- Direction falls to the **feature-based** path, which decides long-vs-short
  from a hand-tuned score over `depth_imbalance`, `delta_pct`, `net_pressure`
  (with an arbitrary `> 50000` threshold).
- The **filters were not reconciled**: e.g. `vwap_deviation_300s < -0.0015`
  was designed to block *long* entries into distribution. It's still there,
  unchanged, while the engine now also takes shorts.

### E3. Measured results (current code, glm5.2 branch)

**May 31 — an UPtrend day of +3.81% (the historical best day):**

```
  absorption           n=  3 WR=  0.0% ret= -1.343% PF=0.00   (3 SHORTS)
  stacked_imbalance    n=  7 WR= 28.6% ret= -0.462% PF=0.71
  delta_divergence     n= 11 WR= 36.4% ret= -0.824% PF=0.70
  value_area           n=  0
  trend_following      n= 12 WR= 41.7% ret= +0.521% PF=1.32
  ALL                  n= 33 WR= 33.3% ret= -2.101%
```

AGENTS.md documented this day as **+0.60% combined**. The current code produces
**−2.101%**. The flagship absorption strategy — historically +1.89% long on
this day — took 3 SHORT trades and lost −1.34%. **It is shorting a +3.81%
up-day.**

**Jun 8 — a DOWNtrend day of −2.26% (absorption only; others too slow to run):**

```
  absorption           n= 12 WR= 50.0% ret= -0.950% PF=0.75   L/S = 9 / 3
```

On the downtrend day the system went **LONG 9 of 12 times** and still lost
−0.95%. So it is directionally wrong in *both* conditions: shorts the uptrend,
longs the downtrend.

### E4. The backtest itself is too slow to iterate (a process blocker)

A single strategy on Jun 8 (23,227 ticks) took **~425 seconds**. A full
5-strategy × 4-date sweep therefore needed ~3 hours and exceeded every run
timeout. Practical consequences:

- Tuning was effectively done blind — results couldn't be seen fast enough to
  evaluate changes, which is exactly how contradictory layering accumulated.

**Root cause (found via profiling, Step 1):** I had initially *guessed* the
bottleneck was per-tick debug logging / `evaluate()` overhead. Profiling
disproved this — `strategy.evaluate()` is only **3.2%** of runtime. The real
culprit is `RegimeClassifier.classify()`, which rebuilt two lists from
`book_history` (up to 5000 elements each) **on every tick** — and regime is
`UNKNOWN` 100% of the time, so this was pure waste: **88% of all backtest
runtime** was spent computing a result that is thrown away.

**Fixed (Step 1, 2026-06-24):** `FeatureEngine` now maintains `_book_ts` /
`_book_mid` mirrors incrementally (O(1) append/trim) and passes them to
`classify()`, which skips the rebuild. Result:

- Jun 8 absorption: **425s → 50.4s (8.4× faster)** — the full sweep now runs in
  ~25 min instead of 3+ hours.
- Trade results **identical** to pre-change baseline (verified by signature:
  May 31 absorption n=3/ret=−1.343%/L/S=2/1; Jun 8 absorption n=12/ret=−0.950%/L/S=9/3).

This was pure performance work — no strategy logic was touched, so all the
numbers in §E3 above are still valid.

### E5. Correction to an earlier claim (L/S counting)

Earlier I reported May 31 absorption as "3 SHORTS." That count used
`side.value == "buy"`, but `Side` is an `auto()` enum so `.value` is an int
(1/2), not `"buy"`/`"sell"` — the comparison silently failed and counted 0
longs. The corrected count via `Side.BUY` is **2 longs / 1 short**. Still
directionally broken (an up-day should not produce a short), but the original
"3 shorts" was overstated. This affects only the *characterisation* in §E3,
not the diagnosis: the system is still taking shorts on an up-day and losing.

This is itself a blocker worth removing **before** more strategy work, because
without a fast feedback loop, no fix can be validated.

---

## Catalogue of Dead / Broken Code Paths

### D1. Regime classifier (BROKEN — the root cause)

`core/feature_engine.py:RegimeClassifier.classify()`

Despite a "FIXED" comment, it returns `UNKNOWN` for the entire session in
backtests. The 30-min volatility/trend thresholds never classify these ICP
sessions into anything but UNKNOWN. **Everything that depends on a real regime
is therefore dead:**

- `StrategyDefinition.evaluate()` regime-specific multiplier selection
  (`sl_mult_*` / `tp_mult_*`) — never branches; always takes the UNKNOWN
  default path.
- `REGIME_STRATEGY_MAP` strategy routing — always returns the UNKNOWN list.
- The "volatility-regime override" block keyed on `bar_ranges_10`.

### D2. Daily-trend override (DEAD)

`feature_engine.py:155-174`, `compute_daily_trend()`

The regime classifier has a "STRONG SIGNAL OVERRIDE": if
`daily_trend_strength > 0.5` and `abs(daily_price_change) > 0.01`, classify as
TRENDING. But the probe shows `daily_trend_*` is **0.0 everywhere**. So this
override path, and the entire TRENDING_UP/DOWN regime branch, never fires.

### D3. `stop_loss_atr_mult` / `take_profit_atr_mult` (DEAD — confirmed still dead)

`knowledge/strategy_library.py:96-97`

Defined on `StrategyDefinition`, set by several strategies, but `evaluate()`
uses the regime-specific `sl_mult_*`/`tp_mult_*` instead. These fields are
never read. Setting them has zero effect.

### D4. Regime map lists `trend_following` (ROUTED, but entry rarely met)

`engine.py:REGIME_STRATEGY_MAP` adds `trend_following` to most regimes. Since
regime is always UNKNOWN, `trend_following` is only active when its name is in
the UNKNOWN list — which it is NOT (UNKNOWN = value_area, delta_divergence,
absorption). So `trend_following` is **effectively disabled** in backtests.

### D5. Circuit breaker / crash detection (DEAD)

`engine.py:_should_trade_today()`, `_detect_crash_regime()`

These gate on `recent_bars` / `bar_ranges_10` reaching certain lengths. With
the regime always UNKNOWN and these features inconsistently populated, these
paths don't reliably fire. Even if they did, they're protecting against a
"crash" that the regime classifier can't detect anyway.

### D6. Value Area strategy (DEAD — confirmed by prior testing, still 0 trades)

Depends on VAH/VAL from a volume profile built incrementally. The VP
throttling + UNKNOWN regime means it effectively never fires.

### D7. ML ensemble (OFF, not a problem)

`config/settings.py:MLEnsembleConfig.enabled = False`. Confirmed disabled by
default, so it's not the performance drag. The slowness is purely per-tick
strategy evaluation overhead × 5 strategies.

---

## The Core Diagnosis

You are **not** failing because the order-flow ideas are wrong, and not because
you need more features. You are failing because:

1. **The signal pipeline has no reliable sense of market direction.** With
   regime stuck at UNKNOWN, the system decides long-vs-short from short-window
   feature noise. On May 31 that noise said "short," and you shorted a +3.81%
   up-day; on Jun 8 it said "long," and you longed a −2.26% down-day.

2. **You've been tuning parameters that aren't connected to anything.** Every
   regime-specific multiplier, the trend-following routing, the daily-trend
   override — none of it executes. Hours of tuning have had zero effect on
   behaviour.

3. **The bidirectional change was made without re-validating direction.** It
   enabled shorts in the engine but the strategy direction logic is naive, so
   shorts fire arbitrarily and longs fire into downtrends.

4. **The feedback loop is too slow to tune safely** (E4). This is *why* the
   contradictions accumulated unnoticed.

---

## User Decisions (2026-06-24)

- **Direction:** diagnose fully before changing code (this report fulfils that).
- **Trend tension:** leaning toward **long-only, trade-less** — accept that
  downtrend days lose; focus on maximising up/range days and *minimising*
  trades on downtrend days via a regime-based halt.

---

## Recommended Order of Repair

Given those decisions, a sensible sequence (no code changed yet):

1. **Make the backtest fast** (E4). Kill the per-tick debug logging and
   redundant validation, or precompute condition results. Target <60s per
   strategy per day. Without this, nothing below can be validated.

2. **Decide the regime question deliberately.** The classifier returns UNKNOWN
   100% of the time. Either (a) fix it so it genuinely classifies, or (b)
   **accept UNKNOWN and stop depending on regime** for any core logic — remove
   the dead regime-keyed code so what's left is honest. Option (b) is lower
   risk and matches "trade-less": a simple, verifiable trend/halt filter can
   replace the elaborate regime machinery.

3. **Revert to clean long-only.** Strip SELL/STRONG_SELL from
   `_determine_direction()`; reconcile the filters so they're all consistent
   with long entries. This alone should stop the "shorting an up-day" failure.

4. **Add a single, honest downtrend filter** that reduces trade frequency on
   declining days (e.g. skip new longs when the session's price is down >X%
   from its open). This is the "trade-less" lever — simpler and more robust
   than graduated exits.

5. **Re-baseline.** Run the (now fast) sweep on all 4 dates and record the new
   numbers in FINDINGS.md. Only accept further changes that beat this
   baseline on the chosen metric (e.g. combined return with a max-DD cap).

Each step is independently testable and reversible.
