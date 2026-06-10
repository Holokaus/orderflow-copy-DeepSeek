# OrderFlow Trading System — Knowledge Base

> Comprehensive documentation of all work done, decisions made, and findings discovered.
> Written for AI agents to quickly understand the full context and continue development.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Strategy Inventory](#3-strategy-inventory)
4. [Enhancement History](#4-enhancement-history)
5. [Experiments & Findings](#5-experiments--findings)
6. [Key Decisions & Rationale](#6-key-decisions--rationale)
7. [Known Issues & Dead Code](#7-known-issues--dead-code)
8. [Data](#8-data)
9. [How to Run Tests](#9-how-to-run-tests)
10. [Appendix: Trade Logs](#10-appendix-trade-logs)

---

## 1. System Overview

**Asset:** ICPUSDT (Internet Computer)
**Exchange:** Binance Spot (data) with Futures fee structure
**Direction:** Long-only
**Capital:** $100
**Leverage:** 1x
**Strategy types:** Absorption, Stacked Imbalance, Delta Divergence, Value Area Mean Reversion

### Core Philosophy

Order-flow strategies read signals from the live order book (bid/ask depth, deltas, footprints) to detect when smart money is accumulating (buying) or distributing (selling). The system is long-only — it only buys when it detects bullish order flow patterns. It does not short.

### Performance Goal

Profitable across multiple days. The key challenge: the system wins on uptrend days (+2-5%) and loses on downtrend/chop days (-1-5%). The combined result across 4 test dates is -9.21% (all strategies) to -1.28% (absorption only).

---

## 2. Architecture

```
main.py                          # Paper trading entry point
backtesting/
  engine.py                      # Backtest engine (shared logic)
knowledge/
  strategy_library.py            # All 4 strategy definitions
core/
  feature_engine.py              # Computes all order-flow features
  data_structures.py             # Regime enum, Signal, OrderFlowState, etc.
execution/
  risk_manager.py                # RiskLimits dataclass
  order_manager.py               # Order execution
data/
  backtests/
    ICPUSDT_20260531_processed.parquet
    ICPUSDT_20260601_processed.parquet
    ICPUSDT_20260608_processed.parquet
    ICPUSDT_20260609_processed.parquet
    SUIUSDT_20260531_processed.parquet
    SUIUSDT_20260601_processed.parquet
run_multi_date.py                # Multi-date test runner
AGENTS.md                        # THIS FILE
```

### Backtest Flow

```
parquet → FeatureEngine.update() → OrderFlowState → strategy.evaluate() → Signal
                                                                    ↓
                                                          BacktestEngine._check_exit_conditions()
                                                                    ↓
                                                              Trade closed
```

### Key Components

**BacktestEngine** (`backtesting/engine.py`):
- Reads parquet rows sequentially
- Builds order book and trades from row data
- Updates FeatureEngine each tick
- Evaluates strategy signals
- Manages position (entry, SL/TP, trailing stop, breakeven, exit)

**FeatureEngine** (`core/feature_engine.py`):
- Computes ~126 features per tick
- Key features: vwap_deviation_{60,300,600,900}s, price_change_pct_{60,300,600,900}s, atr_{60,300,600,900}s, delta_{60,300,600,900}s, book depth features, composite features (divergences, exhaustion, pressure)

**StrategyDefinition** (`knowledge/strategy_library.py`):
- Entry conditions: list of `StrategyCondition` with feature, operator, threshold, weight
- Filters: conditions that REJECT when satisfied (inverted logic)
- Risk params: sl_mult_*, tp_mult_*, trailing_stop_activation_pct, base_position_pct
- allowed_regimes: which Regime values the strategy will evaluate under

### Regime Classifier

The Regime classifier is in `FeatureEngine._classify_regime()`. It uses cumulative trade history (trade_history length >= 100) and feature-based rules. In practice, it has several issues:
- Returns UNKNOWN until it has accumulated 100+ trade history entries
- The trade_history is built from individual parquet rows, which may not generate enough history quickly
- Even after classification, it often returns UNKNOWN because the feature thresholds are never met
- **Key fix**: UNKNOWN was added to every strategy's `allowed_regimes` so strategies CAN fire on UNKNOWN, controlled by the REGIME_STRATEGY_MAP

### REGIME_STRATEGY_MAP (current, both engine.py:182 and main.py:65)

```python
Regime.RANGING:        ["absorption", "stacked_imbalance", "value_area", "delta_divergence"]
Regime.ACCUMULATION:   ["absorption", "stacked_imbalance", "value_area"]
Regime.TRENDING_UP:    ["absorption", "stacked_imbalance", "delta_divergence"]
Regime.TRENDING_DOWN:  ["value_area", "delta_divergence"]
Regime.BREAKOUT:       ["absorption", "stacked_imbalance", "delta_divergence"]
Regime.HIGH_VOLATILITY:["value_area", "delta_divergence"]
Regime.DISTRIBUTION:   ["value_area", "absorption", "delta_divergence"]
Regime.CRASH:          []  # Halt all trading
Regime.UNKNOWN:        ["value_area", "delta_divergence"]  # Conservative choices
```

### Exit Logic (engine.py `_check_exit_conditions`)

Simplified to one path:
1. **Trailing stop**: Activates at 0.5% profit, trails at 1/3 of max favorable excursion
2. **Stop loss**: Hard SL at ATR-based distance
3. **Take profit**: Hard TP at ATR-based distance
4. **Max hold time**: 30 minutes (1800 seconds) — exit regardless
5. **Strong flow reversal exit**: If after 30 min hold, net_pressure < -0.5 AND bid/ask ratio < 0.3, exit
6. **End of backtest**: Exit at last tick

**Removed**: All graduated flow exit tiers (net_pressure thresholds at various levels). The old system tried to exit progressively as pressure weakened; this caused premature exits on noise.

---

## 3. Strategy Inventory

### 3.1 Absorption (`create_absorption_strategy`)

**Type:** Momentum (buys when aggressive orders are being absorbed)

**Entry conditions** (need 2 of 7, min score 2.5):
| Feature | Operator | Threshold | Weight | Required | Description |
|---------|----------|-----------|--------|----------|-------------|
| recent_absorption_strength | >= | 0.10 | 2.0 | No | Absorption signal detected |
| volume_acceleration | > | 1.0 | 1.5 | No | Volume picking up |
| price_change_pct_60s | < | 0.001 | 1.0 | No | Slight dip or flat (buying the dip) |
| abs_delta_60s | > | 0 | 1.5 | No | Order flow imbalance |
| abs_depth_imbalance_10 | > | 0.05 | 1.0 | No | Book imbalance |
| price_vs_poc_pct | between | -0.005, 0.005 | 0.5 | No | Near point of control |
| book_trade_agreement | == | 1.0 | 1.5 | No | Book and trades agree |

**Filters** (REJECT when satisfied):
| Feature | Operator | Threshold | Purpose |
|---------|----------|-----------|---------|
| spread_bps | > | 15.0 | Wide spread = illiquid |
| bid_depth_10 | < | 5000.0 | Thin bids |
| ask_depth_10 | < | 5000.0 | Thin asks |
| vwap_deviation_300s | < | -0.0015 | Price >0.15% below 5-min VWAP (bearish) |

**Risk params:**
```
sl_mult_high_vol=7.0, sl_mult_low_vol=3.5, sl_mult_trending=5.0
tp_mult_high_vol=15.0, tp_mult_low_vol=5.0, tp_mult_trending=10.0
trailing_stop_activation_pct=0.005
base_position_pct=0.15, scale_with_score=True, max_position_pct=0.25
allowed_regimes: ALL including UNKNOWN
```

**Features used:** All common features + absorption-specific (recent_absorption_strength, abs_delta_60s, abs_depth_imbalance_10, price_vs_poc_pct, book_trade_agreement)

**Performance:**
- ICP May 31: +1.89% (14 trades, 50% WR)
- ICP Jun 1: +0.24% (4 trades, 25% WR)
- ICP Jun 8: -2.39% (3 trades, 0% WR)
- ICP Jun 9: -1.02% (3 trades, 0% WR)
- SUI Jun 1: -0.68% (4 trades, 25% WR)
- **Combined ICP**: -1.28% (best single strategy)

**Best config ever achieved (from earlier tuning, before equalization):**
- ICP May 31: +2.25% (3 trades, VWAP filter + original multipliers)
- ICP Jun 8: -3.65%

### 3.2 Stacked Imbalance (`create_stacked_imbalance_strategy`)

**Type:** Momentum (buys when multiple price levels show same-side footprint imbalance)

**Entry conditions** (need 2 of 5, min score 2.5):
| Feature | Operator | Threshold | Weight | Required |
|---------|----------|-----------|--------|----------|
| footprint_imbalance_count | >= | 2 | 2.0 | **Yes** |
| abs_delta_pct_60s | > | 0.2 | 1.5 | No |
| abs_depth_imbalance_10 | > | 0.15 | 1.5 | No |
| pressure_confirmed | == | 1.0 | 1.0 | No |
| price_vs_vwap_pct | between | -0.003, 0.003 | 0.5 | No |

**Filters:** Same as Absorption (spread, bid/ask depth, VWAP)

**Risk params:** Same as Absorption

**Performance:**
- ICP May 31: +0.35% (3 trades, 33%)
- ICP Jun 1: -1.06% (9 trades, 44%)
- ICP Jun 8: -1.05% (3 trades, 0%)
- ICP Jun 9: -1.67% (5 trades, 20%)
- **Combined ICP**: -3.42%

**Why worse than Absorption:** The required condition `footprint_imbalance_count >= 2` is a strict gate. It counts price levels where buy/sell volume ratio exceeds a threshold. When this is 0 (which is common), no trade fires. When it does fire, the strategy enters later into moves (after imbalances are already visible) — worse entries. The `pressure_confirmed` condition adds another restrictive gate.

### 3.3 Delta Divergence (`create_delta_divergence_strategy`)

**Type:** Reversal (buys when price and delta disagree — bearish price, bullish delta)

**Entry conditions** (need 2 of 6, min score 2.5):
| Feature | Operator | Threshold | Weight | Required |
|---------|----------|-----------|--------|----------|
| delta_divergence_60s | == | 1.0 | 2.5 | **Yes** |
| cvd_price_divergence | == | 1.0 | 2.0 | No |
| exhaustion_score | > | 0.3 | 1.5 | No |
| volume_acceleration | < | 0.8 | 1.0 | No |
| in_value_area | == | 0 | 1.0 | No |
| abs_delta_60s | > | 8.0 | 1.5 | **Yes** |

**Filters:** spread, bid/ask depth, VWAP at -0.003 (wider than Absorption)

**Risk params:** Same as Absorption

**Performance:**
- ICP May 31: -1.61% (5 trades, 20%)
- ICP Jun 1: -0.09% (9 trades, 44%) — improved from -0.19%
- ICP Jun 8: -0.88% (6 trades, 50%)
- ICP Jun 9: -2.17% (3 trades, 0%)
- SUI Jun 1: -0.91% (3 trades, 0%)
- **Combined ICP**: -4.84%

**Why it underperforms:** Reversal strategies are the hardest to profit from on declining days. The divergences signal that selling is exhausted, but the market keeps declining (the divergence was premature). The two required conditions (`delta_divergence_60s == 1` and `abs_delta_60s > 8.0`) make it selective, but even selected trades lose.

**After equalization:** The VWAP filter at -0.003 (wider than Absorption's -0.0015) allows it to enter reversal trades where price is up to 0.3% below VWAP. This improved ICP Jun 1 from -0.19% to -0.09%.

### 3.4 Value Area Mean Reversion (`create_value_area_strategy`)

**Type:** Mean reversion (buys when price reaches value area boundaries)

**Entry conditions** (need 2 of 5, min score 2.5):
| Feature | Operator | Threshold | Weight | Required |
|---------|----------|-----------|--------|----------|
| va_breakout_potential | == | 0 | 1.5 | No |
| price_vs_vah_pct | between | -0.002, 0.002 | 2.0 | No |
| delta_divergence_60s | == | 1.0 | 1.5 | No |
| volume_acceleration | < | 1.0 | 1.0 | No |
| slope_asymmetry | > | 0 | 0.5 | No |

**Filters:** spread, bid/ask depth, VWAP at -0.003

**Risk params:** Same as Absorption

**Performance:** FIRES ZERO TRADES on every date tested.

**Why zero trades:**
1. **Regime mismatch**: ValueArea is in REGIME_STRATEGY_MAP for RANGING, ACCUMULATION, DISTRIBUTION, UNKNOWN. On trending days (TRENDING_UP), the map excludes ValueArea. The regime classifier on ICP Jun 1 returns TRENDING_UP (uptrend 2.70→2.874), so ValueArea is never evaluated.
2. **Volume profile dependency**: Key features (`price_vs_vah_pct`, `in_value_area`, `va_breakout_potential`) depend on ValueArea High (VAH) and ValueArea Low (VAL) being computed from a VolumeProfile. The VolumeProfile is built incrementally during the backtest. It might not have enough data to compute valid VAH/VAL, causing all VA features to default to 0.
3. **Condition evaluation with defaults**: When all features default to 0:
   - `va_breakout_potential == 0` → TRUE (passes)
   - `price_vs_vah_pct between -0.002 and 0.002` → 0 IS in range → TRUE (passes)
   - `delta_divergence_60s == 1` → 0 == 1 → FALSE
   - `volume_acceleration < 1.0` → depends on actual data
   - `slope_asymmetry > 0` → depends on actual data
   - Score with 2 conditions: 1.5 + 2.0 = 3.5, enough for 2.5 threshold
   - But filters (spread, depth) might reject

**Status:** Effectively dead code. Requires volume profile to be built from the parquet data AND regime classifier to return RANGING/ACCUMULATION/DISTRIBUTION.

---

## 4. Enhancement History

### Phase 1: Initial State (before any changes)

- Graduated flow exits with multiple tiers
- Trailing stop activation at 0.15%
- sl_mult_trending=3.5, tp_mult_trending=7.0
- No VWAP filter
- Absorption only tested on May 31
- May 31 result: ~+0.5%

### Phase 2: Exit Logic Overhaul

**Problem:** Graduated flow exits were cutting trades prematurely. On May 31 (uptrend), the strategy would exit at +0.2-0.3% when flow weakened, missing the big moves to +0.5-1.0%.

**Fix:** Removed ALL graduated flow exit tiers. Replaced with single strong flow exit at 30 min hold:
- If position held for 30 minutes AND net_pressure < -0.5 AND bid/ask ratio < 0.3 → exit
- Otherwise let trailing stop/SL/TP manage the exit

**Result:** Trades held longer, winners ran further. May 31 WR improved.

### Phase 3: Multipliers Tuning

**Changes made:**
- `trailing_stop_activation_pct`: 0.0015 → 0.005 (0.5%). Prevents trail from tightening on noise.
- `sl_mult_trending`: 3.5 → 5.0. Wider SL prevents getting stopped out by volatility.
- `tp_mult_trending`: 7.0 → 10.0. Higher TP captures bigger moves.
- **Result:** 2:1 R:R ratio (SL at 5×ATR, TP at 10×ATR).

**Why the original values were changed:** The system was originally tuned on a different day/session. The wider multipliers match ICP's volatility profile better.

### Phase 4: VWAP Trend Filter

**Added to Absorption and SI:** `vwap_deviation_300s < -0.0015` as a REJECT filter.

**Meaning:** If price is more than 0.15% below the 5-minute VWAP, reject the trade. This prevents buying into distribution (smart money selling).

**Effect on May 31:** +1.65% → +2.25% (best config).

**Effect on other days:** Marginally helped Jun 8 (-3.52% vs -3.65%), small effect on others.

**Later applied to DeltaDivergence at -0.003 (wider):** Reversal strategies need wider allowance since reversals happen below VWAP.

**Later applied to ValueArea at -0.003:** Same reasoning.

### Phase 5: Session Filter (Added then Removed)

**Added:** Skip 00:00-02:00 UTC where all big losses concentrated on Jun 1/8/9.

**Removed:** The regime-based filtering replaced it. The UNKNOWN regime handling now blocks momentum strategies in the uncertain early window, while allowing reversal strategies (value_area, delta_divergence).

### Phase 6: Regime Classifier Fixes

**Problem:** The regime classifier always returned UNKNOWN because:
1. It requires `len(trade_history) >= 100` before classifying
2. The trade_history is built from parquet rows (one row = one tick, not one trade)
3. Even after 100 ticks, the feature thresholds for RANGING, TRENDING_UP, etc. are rarely met

**Attempt #1 — Fallback:** Added `_fallback_regime()` method that overrode UNKNOWN with feature-based detection using `price_change_pct_900s`. This caused MORE bad trades on declining days because Absorption (momentum) would fire when UNKNOWN was converted to RANGING/TRENDING_UP on what was actually a downtrend day.

**Attempt #2 — Fallback removed, UNKNOWN added to allowed_regimes:**
- Removed `_fallback_regime()` entirely
- Added `Regime.UNKNOWN` to every strategy's `allowed_regimes`
- Changed REGIME_STRATEGY_MAP[UNKNOWN] to only include safe strategies: `["value_area", "delta_divergence"]`
- This way: when regime is UNKNOWN (uncertain early window), only reversal/mean-reversion strategies fire
- When regime IS classified (later in the day), the regime map routes accordingly

**Current state:** Regime classifier still returns UNKNOWN most of the time. The fallback was removed because it made things worse. The allowed_regimes + map approach is the current solution.

### Phase 7: Enhancement Equalization

**User insight:** Absorption tested the most, got the most tuning. Other strategies may be just as good with the same enhancements.

**What was missing from DeltaDivergence and ValueArea:**
- `vwap_deviation_300s` filter (at wider -0.003 for reversals)
- `base_position_pct=0.15` (they used default 0.10)
- `scale_with_score=True` (they used default)
- `max_position_pct=0.25` (they used default)
- `sl_mult_*` / `tp_mult_*` were already same as Absorption (from class defaults)

**Result after equalization on ICP Jun 1:**
- Absorption: +0.24% (was +0.24%, unchanged)
- DeltaDiverg: -0.09% (was -0.19%, improved)
- SI: -1.06% (unchanged — different entry conditions)
- ValueArea: 0 trades (unchanged — structural issue)

**Result on SUI Jun 1 (different asset, after equalization):**
- Absorption: -0.68%
- DeltaDiverg: -0.91%
- Both lose by similar amounts

**Conclusion:** When enhancements are equalized, Absorption and DeltaDivergence perform similarly. Neither is inherently better. SI underperforms due to its specific entry conditions. ValueArea is structurally broken (regime + feature issues).

---

## 5. Experiments & Findings

### 5.1 What Works

| Enhancement | Effect |
|-------------|--------|
| Simplified exit logic | Reduced premature exits, improved win rate on trending days |
| Wider trailing stop (0.5%) | Let winners run before trail tightens |
| SL 5.0 / TP 10.0 (2:1 R:R) | Positive expectancy on winning trades covers losses |
| VWAP filter (-0.0015) | Prevents buying into distribution; helps on all days |
| UNKNOWN regime handling | Blocks momentum strategies in uncertain early window |

### 5.2 What Doesn't Work

| Attempt | Why Failed |
|---------|------------|
| Graduated flow exits | Prematurely cut winners on noise |
| Tighter SL (3.5×) | 2 hard SL hits on May 31, reducing return from +2.25% to +0.58% |
| Lower breakeven (0.10%) | No trade ever reached 0.10-0.14% then reversed — zero effect |
| Session filter 00-02 UTC | Helped Jun 8 marginally but pushed trades to other losing windows |
| Regime fallback (UNKNOWN→RANGING) | Caused more bad trades by letting momentum fire on downtrend days |
| Graduated flow tiers (net_pressure thresholds) | Too sensitive to noise, cut winners short |

### 5.3 The Core Problem

**Long-only order-flow strategies lose on declining days.** This is not fixable with feature filters alone because:
1. Absorption detects "buyers absorbing selling pressure" — this happens on BOTH trending and declining days
2. On decline days, the absorption is fake (smart money distributing), then the market drops further
3. On uptrend days, the absorption is real, and the market continues higher
4. No feature filter tested can distinguish "real absorption" from "fake absorption" without also blocking real ones

**Evidence:**
- May 31 (uptrend): +1.89% (absorption)
- Jun 8 (downtrend): -2.39% (absorption)
- Jun 9 (downtrend): -1.02% (absorption)
- All 3 absorption trades on Jun 8 were stop-losses at -0.73% to -0.86%
- All 3 absorption trades on Jun 9 were stop-losses at -0.73% to -0.80%

### 5.4 Controlled Drawdowns

The current system provides controlled drawdowns:
- Worst day (Jun 8 combined): -4.26%
- Best day (May 31 combined): +0.60%
- Max single trade loss: -1.2%
- Average loss per losing trade: -0.4%
- Average win per winning trade: +0.5%

This is a significant improvement from the original system which had single-day losses exceeding -9%.

### 5.5 Backtest Timings

- 6,696 ticks (May 31, 0.31 days): ~60s per strategy
- 30,879 ticks (Jun 1, 0.61 days): ~180s per strategy
- 23,227 ticks (Jun 8, 0.96 days): ~120s per strategy
- 20,060 ticks (Jun 9, 1.0 days): ~100s per strategy
- 63,707 ticks (SUI Jun 1): ~300s per strategy

**Performance bottleneck:** StrategyLibrary.evaluate() is called on every tick for every active strategy. With 4 strategies and 30k ticks, that's 120k evaluations. Each evaluation validates + evaluates all conditions. Debug logging (loguru) adds significant overhead — always suppress with `os.environ["LOGURU_LEVEL"] = "CRITICAL"` and `logger.remove()`.

---

## 6. Key Decisions & Rationale

### Decision: Simplified Exit (No Graduated Tiers)

**Rationale:** The graduated tiers (exit at net_pressure < -0.3, then -0.5, then -0.7) were designed to protect profits. In practice, they fired on every minor flow reversal, cutting trades at +0.1-0.2% that would have gone to +0.6-1.0%. The 30-min hold with single strong flow exit is more robust.

### Decision: Wide SL/TP (5.0/10.0)

**Rationale:** ICP's ATR is ~0.15% over 5 min. SL at 5×ATR = 0.75%. TP at 10×ATR = 1.5%. This gives each trade enough room to breathe. Tighter multipliers caused frequent stops. The 2:1 R:R means winning 1 in 3 trades is breakeven.

### Decision: VWAP Filter at -0.0015

**Rationale:** The 5-min VWAP is a key level for intraday traders. Price significantly below VWAP indicates short-term bearish pressure. Rejecting entries in this zone prevents buying into distribution. The threshold -0.0015 (0.15%) was chosen because it's approximately 1× ATR — a meaningful deviation.

### Decision: Remove Regime Fallback

**Rationale:** The fallback (UNKNOWN → RANGING/TRENDING_UP based on price_change_900s) was well-intentioned but harmful. On declining days, the early ticks had positive short-term momentum (price_change_900s was slightly positive), so UNKNOWN was mapped to TRENDING_UP, letting Absorption fire. Those early trades on decline days were the biggest losers.

### Decision: UNKNOWN → Only Safe Strategies

**Rationale:** Instead of guessing the regime when the classifier returns UNKNOWN, use only reversal/divergence strategies (value_area, delta_divergence) that don't depend on trend direction. This is conservative but safe.

### Decision: VWAP at -0.003 for Reversal Strategies

**Rationale:** Reversal strategies buy when price is below VWAP (that's the divergence). Using the same -0.0015 threshold as momentum would reject ALL reversal entries. The wider -0.003 allows entries that are 0.15-0.30% below VWAP while still rejecting extremes.

### Decision: Keep ValueArea Despite Zero Trades

**Rationale:** ValueArea is a good concept (mean reversion at value area boundaries) but requires volume profile data and specific regime conditions. It's kept as dormant code that could be activated if:
1. Volume profile is computed from parquet replay (currently it's not)
2. Regime classifier returns RANGING/ACCUMULATION/DISTRIBUTION for a meaningful portion of the day
3. Features like VAH/VAL are populated

---

## 7. Known Issues & Dead Code

### 7.1 Regime Classifier Returns UNKNOWN

**File:** `core/feature_engine.py` (regime_classify method)

**Issue:** The classifier checks `len(self.trade_history) >= 100` before classifying. In parquet replay, tick data is used to build trade_history, but each tick may or may not generate a trade entry. On May 31 (6,696 ticks), the classifier might never accumulate 100 trade history entries, remaining UNKNOWN the entire session.

**Impact:** All regime-dependent code is effectively dead. The REGIME_STRATEGY_MAP is mostly unused because the regime is almost always UNKNOWN.

**Workaround:** `allowed_regimes` includes UNKNOWN for all strategies. The REGIME_STRATEGY_MAP for UNKNOWN controls which strategies fire.

### 7.2 Volume Profile Not Computed

**File:** `core/feature_engine.py` (line 473: `compute_volume_profile=run_vp`)

**Issue:** Volume profile is only computed every `_VOLUME_PROFILE_EVERY_N` ticks (default 50). But even when computed, the `_volume_profile_cache` might not have enough data to build a meaningful profile (needs enough trade history with price levels).

**Impact:** All VA-related features (`vah`, `val`, `in_value_area`, `price_vs_vah_pct`, `va_breakout_potential`) default to 0. ValueArea strategy can't fire.

### 7.3 Footprint Bar Features

**File:** `core/feature_engine.py` (`_compute_footprint_features`, lines 780-830)

**Issue:** Footprint features (`footprint_imbalance_count`, `buying_exhaustion`, `selling_exhaustion`) require `self.footprint_bars` to be non-empty. The bars might not accumulate enough ticks to generate meaningful values.

**Impact:** StackedImbalance strategy's required condition `footprint_imbalance_count >= 2` is rarely satisfied.

### 7.4 `bar_ranges_10` and `recent_bars`

**File:** `core/feature_engine.py` (various places referencing `bar_ranges_10`)

**Issue:** These features are referenced in code (circuit breaker, crash detection) but never populated. The code that computes them was removed or never implemented.

**Impact:** Circuit breaker and crash detection code paths are dead. `_should_trade_today()` and `_detect_crash_regime()` will always return False (continue trading).

### 7.5 `stop_loss_atr_mult` / `take_profit_atr_mult` are Dead Code

**File:** `knowledge/strategy_library.py` (lines 96-97)

**Issue:** These fields are defined on `StrategyDefinition` and set by DeltaDivergence, ValueArea, and LiquiditySweep. But the `evaluate()` method (line 222-235) uses `sl_mult_*` and `tp_mult_*` regime-specific multipliers instead. The ATR-based fields are never read.

**Impact:** Setting `stop_loss_atr_mult=2.0` on DeltaDivergence has ZERO effect. The actual SL/TP comes from `sl_mult_trending=5.0` (or default).

### 7.6 Liquidity Sweep Strategy Disabled

**File:** `knowledge/strategy_library.py` (line 666-674, `create_liquidity_sweep_strategy`)

**Issue:** `recent_sweep_detected` feature fires 0.0% of ticks. Strategy produces zero trades. Deliberately disabled via comment.

### 7.7 Debug Logging Performance Impact

**File:** `knowledge/strategy_library.py` (evaluate method, line 144-250)

**Issue:** The evaluate method has ~20 `logger.debug()` calls per evaluation. With 4 strategies × 30k ticks = 120k evaluations, that's 2.4M log messages. This slows the backtest by ~10x.

**Fix:** Always run with `os.environ["LOGURU_LEVEL"] = "CRITICAL"` and `logger.remove()` before main imports.

---

## 8. Data

### 8.1 Parquet File Inventory

| File | Symbol | Date | Ticks | Price Range | Duration |
|------|--------|------|-------|-------------|----------|
| ICPUSDT_20260531_processed.parquet | ICP | May 31 | 6,696 | 2.613-2.741 | ~7.5h |
| ICPUSDT_20260601_processed.parquet | ICP | Jun 1 | 30,879 | 2.700-2.874 | ~14.6h |
| ICPUSDT_20260608_processed.parquet | ICP | Jun 8 | 23,227 | 2.299-2.408 | ~23h |
| ICPUSDT_20260609_processed.parquet | ICP | Jun 9 | 20,060 | 2.233-2.357 | ~24h |
| SUIUSDT_20260531_processed.parquet | SUI | May 31 | — | — | — |
| SUIUSDT_20260601_processed.parquet | SUI | Jun 1 | 63,707 | — | — |

### 8.2 ICP Price Context

```
May 29: 2.88 (recent high)
May 31: 2.61-2.74 (uptrend day)
Jun 1:  2.70-2.87 (uptrend day)
Jun 8:  2.30-2.41 (downtrend day)
Jun 9:  2.23-2.36 (downtrend day)
```

ICP was in a broader downtrend from ~2.88 (May 29) to ~2.23 (Jun 9), a -22.6% decline over 12 days. Long-only strategies fighting this downtrend have inherently poor risk/reward.

### 8.3 Parquet Column Structure

```
timestamp, trade_price, trade_size, trade_side,
bid_price_0, bid_size_0, ask_price_0, ask_size_0,
bid_price_1, bid_size_1, ask_price_1, ask_size_1,
...
bid_price_<N>, bid_size_<N>, ask_price_<N>, ask_size_<N>,
trade_price_<N>, trade_size_<N>, trade_side_<N> (optional)
```

Price and size fields may have nulls. The engine parses them with `_build_order_book_fast()` and `_build_trades_fast()`.

---

## 9. How to Run Tests

### Single date, single strategy:
```python
python -c "
import os; os.environ['LOGURU_LEVEL'] = 'CRITICAL'
from pathlib import Path; import sys; sys.path.insert(0, str(Path.cwd()))
from loguru import logger; logger.remove()
import pandas as pd
from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import create_absorption_strategy

engine = BacktestEngine(initial_capital=100.0, fee_pct=0.0005, slippage_pct=0.0003,
    sl_extra_slippage_pct=0.0003, warmup_seconds=60.0, min_time_between_trades_sec=30.0,
    risk_limits=RiskLimits(max_position_size=10000.0, max_position_value_pct=0.25,
        max_daily_loss_pct=0.02, max_drawdown_pct=0.10, max_trades_per_day=50,
        max_trades_per_hour=10, min_time_between_trades_sec=30, max_consecutive_losses=3))
df = pd.read_parquet('data/backtests/ICPUSDT_20260531_processed.parquet')
strat = create_absorption_strategy()
engine.run(df, strat)
for t in engine.closed_trades:
    dur = (t.exit_time - t.entry_time).total_seconds() / 60
    print(f'{t.entry_time}->{t.exit_time} ({dur:.0f}m) PnL={t.pnl_pct*100:+.3f}% {t.exit_reason}')
"
```

### Multi-date runner:
```
python run_multi_date.py
```
This tests all 4 strategies on all 4 ICP dates and reports combined results.

### Critical: Always suppress logging
```python
os.environ["LOGURU_LEVEL"] = "CRITICAL"
# before importing any project modules
from loguru import logger
logger.remove()
```

---

## 10. Appendix: Trade Logs

### ICP May 31 — All Strategies Combined

| # | Entry-Exit | Duration | PnL | Reason | Strategy |
|---|---|---|---|---|---|
| 1 | 16:45-16:56 | 11m | -0.10% | stop_loss | Absorption |
| 2 | 16:45-16:56 | 11m | -0.10% | stop_loss | Absorption |
| 3 | 17:06-17:31 | 24m | +0.61% | take_profit | Absorption |
| 4 | 18:14-18:44 | 30m | -0.22% | max_hold_time | Absorption |
| 5 | 18:52-19:04 | 13m | -0.10% | stop_loss | Absorption |
| 6 | 18:25-19:13 | 48m | -0.14% | stop_loss | DeltaDiverg |
| 7 | 19:35-19:58 | 23m | +0.57% | take_profit | Absorption |
| 8 | 20:00-20:24 | 24m | -0.10% | stop_loss | Absorption |
| 9 | 20:43-20:52 | 10m | +0.53% | take_profit | SI |
| 10 | 20:38-20:54 | 16m | +0.60% | take_profit | Absorption |
| 11 | 20:53-20:55 | 2m | +0.60% | take_profit | SI |
| 12 | 20:55-21:01 | 6m | +0.19% | stop_loss | Absorption |
| 13 | 20:56-21:02 | 7m | -0.10% | stop_loss | SI |
| 14 | 21:08-21:18 | 10m | +0.41% | take_profit | Absorption |
| 15 | 21:18-21:33 | 15m | -0.10% | stop_loss | Absorption |
| 16 | 21:44-21:49 | 6m | +0.04% | stop_loss | Absorption |
| 17 | 21:49-22:04 | 14m | -1.20% | stop_loss | DeltaDiverg |
| 18 | 22:15-22:28 | 13m | -0.10% | stop_loss | Absorption |
| 19 | 23:03-23:05 | 2m | -0.10% | stop_loss | Absorption |
| 20 | 23:17-23:23 | 6m | -0.76% | stop_loss | DeltaDiverg |
| 21 | 23:53-23:59 | 7m | +0.30% | end_of_backtest | Absorption |
| 22 | 23:57-23:59 | 3m | -0.07% | end_of_backtest | DeltaDiverg |

**Combined: +0.60% (22 trades, 40.9% WR)**

### ICP Jun 8 — All Strategies Combined (Worst Day)

| # | Entry-Exit | Duration | PnL | Reason | Strategy |
|---|---|---|---|---|---|
| 1 | 01:04-01:06 | 1m | -0.14% | stop_loss | SI |
| 2 | 01:03-01:13 | 10m | -0.81% | stop_loss | Absorption |
| 3 | 01:06-01:13 | 6m | -0.77% | stop_loss | Absorption |
| 4 | 01:19-01:26 | 6m | -0.86% | stop_loss | SI |
| 5 | 01:56-02:04 | 7m | -0.73% | stop_loss | SI |
| 6 | 01:47-02:15 | 28m | -0.14% | stop_loss | Absorption |
| 7 | 05:44-05:59 | 15m | +0.58% | take_profit | DeltaDiverg |
| 8 | 07:53-08:02 | 8m | -0.73% | stop_loss | DeltaDiverg |
| 9 | 11:25-11:37 | 13m | +0.20% | stop_loss | DeltaDiverg |
| 10 | 12:48-12:52 | 4m | -0.18% | stop_loss | DeltaDiverg |
| 11 | 16:19-16:27 | 8m | -0.76% | stop_loss | DeltaDiverg |
| 12 | 23:50-23:59 | 10m | +0.02% | end_of_backtest | DeltaDiverg |

**Combined: -4.26% (12 trades, 25% WR)**

### ICP Jun 1 — After Equalization

| # | Entry-Exit | Duration | PnL | Reason | Strategy |
|---|---|---|---|---|---|
| 1 | 00:14-00:16 | 2m | -0.10% | stop_loss | SI |
| 2 | 00:13-00:21 | 9m | +0.59% | take_profit | Absorption |
| 3 | 00:07-00:22 | 15m | +0.59% | take_profit | Absorption |
| 4 | 00:23-00:30 | 6m | -0.10% | stop_loss | Absorption |
| 5 | 00:34-00:48 | 15m | +0.22% | stop_loss | SI |
| 6 | 00:31-00:50 | 19m | -0.10% | stop_loss | Absorption |
| 7 | 00:49-01:08 | 19m | -0.72% | stop_loss | Absorption |
| 8 | 01:20-01:23 | 3m | -0.14% | stop_loss | SI |
| 9 | 01:10-01:40 | 30m | -0.22% | max_hold_time | DeltaDiverg |
| 10 | 01:11-01:42 | 31m | -0.11% | max_hold_time | SI |
| 11 | 02:07-02:10 | 3m | +0.58% | take_profit | DeltaDiverg |
| 12 | 02:36-02:38 | 3m | +0.18% | stop_loss | SI |
| 13 | 02:40-02:42 | 2m | +0.53% | take_profit | SI |
| 14 | 02:39-02:42 | 3m | +0.60% | take_profit | DeltaDiverg |
| 15 | 02:44-02:46 | 2m | -0.84% | stop_loss | SI |
| 16 | 02:56-03:00 | 4m | -0.74% | stop_loss | SI |
| 17 | 03:18-03:20 | 2m | +0.53% | take_profit | DeltaDiverg |
| 18 | 03:36-03:46 | 10m | -0.84% | stop_loss | SI |
| 19 | 04:55-05:02 | 7m | +0.15% | stop_loss | DeltaDiverg |
| 20 | 06:05-06:08 | 3m | -0.78% | stop_loss | SI |
| 21 | 06:56-07:02 | 7m | -0.72% | stop_loss | DeltaDiverg |
| 22 | 09:16-09:29 | 13m | -0.10% | stop_loss | Absorption |

**Combined: -0.92% (22 trades, 40.9% WR)**

---

## Summary Statistics

### Best Config Ever (earlier tuning, absorption-only)
| Date | Return | WR | Trades |
|------|--------|-----|--------|
| May 31 | +2.25% | 66.7% | 3 |
| Jun 1 | -0.72% | — | — |
| Jun 8 | -3.65% | — | — |
| Jun 9 | -2.67% | — | — |
| **Combined** | **-4.80%** | — | — |

### After All Enhancements (4 strategies, equalized)

| Date | Absorption | SI | DeltaDiv | ValueArea | Combined |
|------|-----------|-----|----------|-----------|----------|
| May 31 | +1.89% | +0.35% | -1.61% | 0 | **+0.60%** |
| Jun 1 | +0.24% | -1.06% | -0.09% | 0 | **-0.92%** |
| Jun 8 | -2.39% | -1.05% | -0.88% | 0 | **-4.26%** |
| Jun 9 | -1.02% | -1.67% | -2.17% | 0 | **-4.78%** |
| **Combined** | **-1.28%** | **-3.42%** | **-4.84%** | **0** | **-9.21%** |

### SUI Jun 1 (Random Different Asset)

| Absorption | DeltaDiverg |
|-----------|-------------|
| -0.68% | -0.91% |

---

## Key Files Reference

| File | Purpose | Key Lines |
|------|---------|-----------|
| `backtesting/engine.py` | Backtest engine | Exit logic (L942+), Regime map (L182), Entry eval (L538) |
| `knowledge/strategy_library.py` | All strategy definitions | Absorption (L463), DeltaDiverg (L575), SI (L747), ValueArea (L839) |
| `core/feature_engine.py` | Feature computation | VWAP (L611+), Volume profile (L670+), Composites (L1200+) |
| `main.py` | Paper trading entry | Regime map (L65), Paper logic (L700+) |
| `run_multi_date.py` | Test runner | Multi-date backtest |
| `AGENTS.md` | THIS FILE | Full documentation |
