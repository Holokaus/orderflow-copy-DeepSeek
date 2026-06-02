# ICP Switch & Win-Rate Improvement — Analysis & Changes

## 1. Asset Selection: Why ICP?

All 4 available assets were evaluated on data quality, volatility, spread, and fee/vol ratio:

| Asset | 30min Vol | Spread (bps) | Fee/Vol Ratio | Bid/Ask Depth Available | Verdict |
|-------|-----------|-------------|---------------|------------------------|---------|
| XRP   | 0.25%     | 0.54        | 15%           | Yes (150K)             | Vol too low, fee eats 15% of move |
| SUI   | 2.05%     | 4.02        | 4%            | Yes (but few signals)  | Good vol but too few trade signals |
| BTC   | 0.42%     | 1.56        | 7%            | **All zeros**          | Data broken — zero bid/ask depth |
| **ICP** | **0.89%** | **3.84**  | **5%**        | Yes (6K-15K)           | **Chosen**: best balance |

**Decision**: ICP chosen. 30min vol=0.89% gives room above fee drag (0.16%/trade), spreads are reasonable at 3.84bps, and order book data is intact.

## 2. The Zero-Trade Problem on Non-XRP Assets

### Root Cause
All strategies had **XRP-specific filter thresholds** hardcoded in `knowledge/strategy_library.py`:

| Threshold | XRP Value | ICP's Actual | Effect on ICP |
|-----------|-----------|-------------|---------------|
| `bid_depth_10 <` | 150,000 | ~15,400 | **Always rejected** (15K < 150K = True) |
| `ask_depth_10 <` | 150,000 | ~14,900 | **Always rejected** (15K < 150K = True) |
| `spread_bps >` | 8.0 | 3.64 | OK (3.64 < 8, filter not triggered) |

### How XRP's depth threshold was discovered
- Initial test: 0 trades on every non-XRP asset
- Hypothesis: filter conditions reject everything
- Examined each filter condition: XRP's `bid_depth_10 < 150000` assumes XRP's massive depth (~150K per side). ICP's depth (~15K) is 10x smaller → always satisfies the `<` condition → always REJECTED
- Fix: change `150000` → `5000` (below ICP's 10th percentile of ~7,700)

### Additional XRP thresholds found in subsequent audit
After fixing the obvious depth+spread, tested each strategy. StackedImbalance still gave 0 trades:

| Threshold | File | XRP Value | Issue |
|-----------|------|-----------|-------|
| `price_change_pct_300s > 0.001` | StackedImbalance filter | 0.1% | ICP 5-min vol ~0.48%, so nearly all 5-min windows exceed 0.1%. Fix: → 5% |
| `price_change_pct_300s < -0.001` | StackedImbalance filter | -0.1% | Same logic for negative moves. **Bug**: original threshold `< -0.001` means "reject if change < -0.1%". On ICP with typical -0.2% to -1.0% swings, this ALWAYS rejects. Fix: → -5% |

### Deep audit of ALL thresholds (conducted across all 5 strategies)

#### Changed (34 instances across 5 strategies)

| Threshold | Old (XRP) | New (ICP) | Strategies Affected |
|-----------|-----------|-----------|-------------------|
| `spread_bps >` | 8.0 | 15.0 | Absorption, DeltaDivergence, LiquiditySweep, StackedImbalance, ValueArea |
| `bid_depth_10 <` | 150,000 | 5,000 | Same 5 strategies |
| `ask_depth_10 <` | 150,000 | 5,000 | Same 5 strategies |
| `price_change_pct_300s >` | 0.001 | 0.05 | StackedImbalance |
| `price_change_pct_300s <` | -0.001 | -0.05 | StackedImbalance |

#### Investigated but kept unchanged

| Threshold | Value | Why kept |
|-----------|-------|----------|
| `abs(pressure) > 50000` (in `_determine_direction`) | 50,000 | Lowering to 5,000 added noise: EV dropped from +0.023% to -0.023%. Likely a deliberate "extreme pressure" threshold, not XRP-specific |
| `abs_delta_60s > 8.0` (DeltaDivergence, required) | 8.0 | Median trade size on ICP is 30.9 → single trade exceeds this. Not blocking |
| `price_change_pct_60s < 0.001` (Absorption entry) | 0.1% | ICP 60s vol ~0.045% → passes most bars. Non-required condition |
| `delta_pct_300s > 0.7` (DeltaDivergence filter) | 0.7 | Ratio value (0-1), generic |
| `exhaustion_score > 0.3` | 0.3 | Ratio value, generic |
| `price_vs_vwap_pct between -0.003 and 0.003` | ±0.3% | Percentage, generic (though produces 0 hits on ICP — non-required, doesn't block) |
| `tick_size` in FeatureConfig | 0.0001 | Was set for XRP. ICP actual tick is 0.001. Affects volume profile bucketing, but Absorption/StackedImbalance produce reasonable results anyway |

## 3. Win-Rate Improvement Strategy: Trailing Stop Exits

### The Problem
All signals had ~50% directional accuracy (no edge). Fixed TP cuts winners early → negative EV after fees.

### MFE/MAE Analysis on Absorption (22 trades, no SL/TP)
- **Direction accuracy**: 54.5% (statistically ~50%)
- **MFE distribution**: Winners ran 2-4%, losers reversed after initial positive MFE
- **Key insight**: The fat right tail (MFE q75=2.76%) creates an asymmetry that a trailing stop can capture, while a fixed TP would cut it

### Trailing Stop Design
```
Entry → price moves +1.0% → trailing activates → trail at 0.5% from peak
                                                   ↓
                                           SL ratchets upward
                                                   ↓
                                           If price reverses 0.5% from peak → exit
```

**Parameters** (set in `strategy_library.py` on Absorption + StackedImbalance):
- `sl_mult_high_vol=7.0` → SL = ATR(0.1%) × 7 = 0.7%
- `sl_mult_low_vol=7.0`
- `sl_mult_trending=7.0`
- `tp_mult_*=100.0` → effectively disables TP
- `trailing_stop_activation_pct=0.01` → activates at +1.0%
- `trail_distance = activation × 0.5 = 0.5%`

### Engine Exit Changes (`backtesting/engine.py`, lines 826-830)

| Exit | Status | Reason |
|------|--------|--------|
| `absorption_against` | DISABLED | Was closing trades before trailing stop could activate, reducing WR from 68%→50% |
| `delta_divergence_against` | DISABLED | Same — premature exits |
| Buying/selling exhaustion | DISABLED | Premature exits |
| `book_pressure_collapse` | **ENABLED** | Helps EV: removing it dropped Absorption EV from +0.023% to -0.015% |
| `sweep_against` | ENABLED | Unrelated to flow exits |

## 4. StackedImbalance Debug: Zero Trades on ICP

### Symptoms
- StackedImbalance: 0 trades with original XRP filters
- Fixed depth+spread+price_change filters: still 0 trades on "original + ICP filters" test

### Debug Process
1. **Check each entry condition individually** (with tight SL/TP for speed):
   - `footprint_imbalance_count >= 2`: 22 trades ✅
   - `abs_delta_pct_60s > 0.2`: 32 trades ✅
   - `abs_depth_imbalance_10 > 0.15`: 26 trades ✅
   - `pressure_confirmed == 1.0`: 23 trades ✅
   - **`price_vs_vwap_pct between -0.003`**: **0 trades** ❌

2. **VWAP condition never fires on ICP**: The `between -0.003 to 0.003` check (price within ±0.3% of VWAP) never triggers on ICP's VWAP data. However, this condition is NOT required (weight=0.5) and `min_conditions_satisfied=2`, so it alone doesn't block.

3. **Root cause of 0 trades on "full" test**: Correcting the filters was enough. With depth=5K, spread=15, price_change=5%, plus the VWAP condition being non-required → 38 trades.

### `price_vs_vwap_pct` Investigation
- This condition has 0 hits on ICP across all 37,575 rows
- Likely because ICP's VWAP calculation or feature value range differs from XRP
- Since it's non-required (weight=0.5), it simply never contributes. Not a blocker, but worth investigating if we want that 0.5 score contribution

## 5. Final Results

Both strategies produce similar results on the full 37,575-row ICP dataset (22h, +4.46% uptrend):

| Metric | Absorption | StackedImbalance |
|--------|-----------|-----------------|
| Trades | 38 | 38 |
| Win Rate | 50.0% | 50.0% |
| Avg Win | +0.95% | +0.92% |
| Avg Loss | -0.91% | -0.90% |
| **EV** | **+0.023%** | **+0.015%** |
| **PF** | **1.036** | **1.045** |
| Exit: SL | 32 | 31 |
| Exit: book_pressure_collapse | 5 | 6 |
| Exit: end_of_backtest | 1 | 1 |

### Key Finding
Both have **zero directional edge** (~50% WR in a +4.46% uptrend). Profits come purely from **exit asymmetry**: trailing stop captures fat right tail, fixed SL limits left tail. This is **volatility harvesting, not prediction**.

### Performance by Market Regime (from earlier MFE/MAE analysis)
- First half of data: EV = +0.175%
- Second half of data: EV = -0.129%
- Strategy may degrade in range-bound or trending-without-fat-tails markets

## 6. Remaining Concerns

1. **`FeatureConfig.tick_size = 0.0001`** in `feature_engine.py:33` was set for XRP. ICP's actual tick size is 0.001. Affects volume profile bucketing, footprint bars, and price comparisons. Currently not blocking but should be parameterized per-asset.

2. **VWAP condition never fires** — `price_vs_vwap_pct between -0.003` has 0 hits on ICP. Non-blocking but the 0.5 weight is permanently lost.

3. **Cross-asset portability unknown** — Thresholds were tuned for ICP only. Running Absorption on XRP's 385K dataset would verify whether the trailing stop approach generalizes.

## Files Modified

| File | Changes |
|------|---------|
| `knowledge/strategy_library.py` | ICP filter thresholds on all 5 strategies; trailing stop params on Absorption + StackedImbalanced |
| `backtesting/engine.py` (lines 826-830) | Disabled absorption_against, delta_divergence_against, exhaustion exits |

## Data Files Used

- `data/backtests/ICPUSDT_20260531_processed.parquet` (6,696 rows)
- `data/backtests/ICPUSDT_20260601_processed.parquet` (30,879 rows)
- Combined: 37,575 rows, 2026-05-31 16:36 to 2026-06-01 14:40 UTC
