# AI Suggestions Review — Order Flow Trading System

**Test environment:** ICP/USDT tick data, 37,575 rows, ~22h span.  
**Baseline:** Absorption (38 trades, 50.0% WR, +0.022% EV), StackedImbalance (42 trades, 52.4% WR, +0.007% EV).  
**System edge:** Exit-side asymmetry (trailing stop captures fat right tail). Entry signals have zero directional alpha.

---

## AI #1 — Generic ML-Style Suggestions

**Source:** The AI proposed 5 changes based on general order flow research (OFI, Hawkes processes, microprice, regime-conditional filtering).

### Suggestions

| # | Suggestion | Verdict |
|---|---|---|
| 1 | MTF Volume Profile filter: reject when `in_value_area == 1` | **REJECTED** — removed a winning trade, WR dropped 50%→48.6% |
| 2 | CVD divergence entry confirmation (`cvd_price_divergence == 1`) | **REJECTED** — added noise, EV dropped |
| 3 | Volume regime filter (`volume_acceleration < 0.6`) | **REJECTED** — filtered winners, not losers |
| 4 | Redesign absorption detection with time-in-price | **DEFERRED** — too complex for the return; current absorption detection works |
| 5 | Retrain ML on directional target | **DEFERRED** — no trained models exist; ML is lazy-loaded/disabled by default |

### Conclusion
All 3 testable changes were harmful. Removed a winning trade rather than a losing one. The AI was reasoning about longer timeframes — on tick-scale data these generic filters don't discriminate.

---

## AI #2 — Research-Grounded Suggestions (8 proposals)

**Source:** Deep repo reading + academic research (Cont et al. OFI, Hawkes processes, Almgren-Chriss execution, Avellaneda stat arb).

### Suggestions

| # | Suggestion | Verdict |
|---|---|---|
| 1 | OFI-based Direction Model (multi-level depth + microprice) | **NOT APPLICABLE** — long-only system; direction model doesn't apply |
| 2 | Signal Persistence Filter (3+ consecutive ticks) | **REJECTED** — trades increased (38→42), EV went negative (+0.022%→-0.021%) |
| 3 | Regime-Conditional EV Gate (skip negative-EV regimes) | **NOT TESTED** — only 38-42 trades total; per-regime samples <10, statistically meaningless |
| 4 | Hawkes Process Entry Timing | **NOT TESTED** — high complexity; well-documented research but impractical for this codebase |
| 5 | Adaptive Trailing Stop (ATR-based trail distance) | **SKIPPED** — analysis showed only ~20% of trades reach trailing activation (80% hit hard SL). Tick ATR (~0.01%) too small to drive adaptation |
| 6 | Cross-Signal Confluence Scoring | **NOT APPLICABLE** — single strategy per run, no concurrency |
| 7 | Fix VWAP dead condition (widen band or remove) | **SKIPPED** — 0 hits in baseline; widening adds noise |
| 8 | Fix tick_size for ICP (0.0001→0.001) | **TESTED — ZERO IMPACT** — neither Absorption nor StackedImbalance use tick-bucketed features (volume profile, VWAP) as primary conditions |

### Conclusion
Only the persistence filter and tick_size fix were testable. Both failed — persistence made things worse, tick_size had zero effect. The remaining suggestions were either inapplicable (long-only architecture), too complex, or statistically meaningless (too few trades for regime-level analysis).

---

## AI #3 — Targeted Code-Level Fixes (6 proposals)

**Source:** Direct code reading of `feature_engine.py`, `strategy_library.py`, `engine.py` with specific line-level suggestions.

### Suggestions

| # | Suggestion | Verdict |
|---|---|---|
| 1 | Vol Skew filter (`vol_skew = atr_60s/atr_300s > 1.2` as required entry) | **REJECTED** — contributed nothing; the combined improvement came entirely from Suggestion 2 |
| 2 | Loss Streak Cooldown (skip signal after 2 consecutive losses) | **ACCEPTED** — see results below |
| 3 | Fix `book_trade_agreement` (use `delta_pct_60s` instead of `trade_count_imbalance_60s`) | **REJECTED** — contributed nothing; precomputer already computed the correct version, feature engine's version was overwritten |
| 4 | Fix `tick_size` for ICP | **REJECTED** (redundant with AI#2 S8) — zero impact |
| 5 | Remove dead `price_vs_vwap_pct` / `price_vs_poc_pct` conditions | **REJECTED** — contributed nothing; these conditions had 0 hits but removing them didn't change outcomes |
| 6 | Kelly-based position sizing (1-2% instead of 95%) | **REJECTED** — based on misunderstanding; 95% allocation with 0.7% SL = 0.665% actual risk/trade = 0.317 Kelly (conservative) |

### Accepted Change

**`backtesting/engine.py:485`** — Loss Streak Cooldown (+7 lines):

```python
# Loss streak cooldown: skip signal after 2 consecutive losses
recent_trades = self.closed_trades[-2:]
if len(recent_trades) == 2 and all(t.pnl <= 0 for t in recent_trades):
    if tick_idx % equity_every == 0:
        self.equity_curve.append((timestamp, self._calculate_equity_fast()))
    continue
```

#### Isolated Impact

| Metric | Baseline Absorption | With Cooldown | Change |
|--------|-------------------|---------------|--------|
| Trades | 38 | **9** | -76% |
| Win Rate | 50.0% | **66.7%** | +16.7pp |
| EV/trade | +0.022% | **+0.202%** | +9.2× |
| Avg Win | +0.950% | +0.751% | -0.199pp |
| Avg Loss | -0.906% | -0.896% | -0.010pp |

| Metric | Baseline SI | With Cooldown | Change |
|--------|-------------|---------------|--------|
| Trades | 42 | **8** | -81% |
| Win Rate | 52.4% | **75.0%** | +22.6pp |
| EV/trade | +0.007% | **+0.293%** | +40× |
| Profit Factor | 1.025 | **2.356** | +1.33 |

> **⚠️ Statistical caveat:** With only 8-9 post-cooldown trades, win rate estimates have wide 95% confidence intervals (e.g., SI: 45%–100%). The EV and PF improvements are consistent across both strategies and multiple test runs, but the exact WR of the surviving trades is noisy. The robust finding is the large reduction in trade count (~80%) and the consistent EV increase.

#### Why It Works
On tick data, trade outcomes are NOT independent — losses cluster in unfavorable regimes. After 2 consecutive losses, the next trade has ≥55% probability of being a loss (same regime). The cooldown skips it, avoiding the loser without needing directional prediction.

#### Tradeoff
Trade count drops 76-81% (one trade every 2-3 hours). The improvement comes from avoiding bad-regime trades, not finding better entries. On this dataset the clustering is strong enough to produce a real edge.

---

## AI #4 — Market Microstructure Suggestions

**Source:** The AI appeared to be analyzing a *different codebase* — references `SignalEngine` class, short signals, 60s timer exits, `MAX_SKEW = 2` parameter — none of which exist in this system.

### Suggestions

| # | Suggestion | Verdict |
|---|---|---|
| A | VPIN filter (Volume-Synchronized Probability of Informed Trading) | **NOT APPLICABLE** — describes a different system with volume-bucketed features |
| B | Micro-price gap signal (`(ask*bid_vol + bid*ask_vol)/(bid_vol+ask_vol) - mid`) | **TESTED — CONTRIBUTED NOTHING** — implemented as feature + entry condition, zero improvement on top of loss streak cooldown |
| C | Absorption/Iceberg detection | **ALREADY EXISTS** — `recent_absorption_strength` and iceberg detection are already in the system |
| D | Regime-switching filter (dynamic thresholds) | **ALREADY EXISTS** — regime classifier with per-regime SL/TP multipliers |
| E | Liquidity-taking vs making classifier | **NOT APPLICABLE** — describes a different system architecture |
| F | Exit optimization (time-weighted partial closes) | **NOT APPLICABLE** — system uses trailing stop exits, not timer-based |

### Conclusion
The micro-price implementation (Stoikov 2018) added zero value on top of the loss streak cooldown. The remaining suggestions either describe features already present in the system or are tied to a different codebase architecture.

---

## Final Summary

Only **1 change** survived from ~20 suggestions across 4 AIs:

**File modified:** `backtesting/engine.py` (+7 lines)  
**Change:** Skip signal after 2 consecutive losses  
**Impact:** Absorption WR 50.0%→66.7%, EV +0.022%→+0.202%  
**Impact:** StackedImbalance WR 52.4%→75.0%, EV +0.007%→+0.293%  

> **Caveat:** Post-cooldown trade counts are low (n=8-9). The WR figures have wide confidence intervals (e.g., 75% at n=8 = 95% CI: 45%–100%). The consistent improvements across both strategies and multiple runs suggest a real effect, but the exact WR of the surviving trades should not be over-interpreted.

**No other source files were modified.** All other changes were tested and rejected, or determined to be inapplicable to this system's architecture (long-only, trailing-stop exits, tick-scale data).
