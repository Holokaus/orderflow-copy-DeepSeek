"""
analyze_optimal_tp.py
Vectorized TP optimization for XRPUSDT with real execution constraints.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

SEP = "=" * 90
DASH = "-" * 90

print(SEP)
print("XRPUSDT TP OPTIMIZATION - REAL MARKET CONSTRAINTS (Vectorized)")
print(SEP)

df = pd.read_parquet(project_root / "data" / "backtests" / "XRPUSDT_market_data_20260328.parquet")
print(f"\nLoaded {len(df):,} rows")

trade_mask = df['trade_price'].notna() & (df['trade_price'] > 0)
trades_df = df[trade_mask].copy()
print(f"Trade rows: {len(trades_df):,}")

ROUND_TRIP_FEE = 0.001
SLIPPAGE_PER_SIDE = 0.00015
TOTAL_COST_PCT = ROUND_TRIP_FEE + 2 * SLIPPAGE_PER_SIDE

print(f"\nCost Structure:")
print(f"  Round trip fee:         {ROUND_TRIP_FEE*100:.2f}%")
print(f"  Slippage per side:      {SLIPPAGE_PER_SIDE*100:.3f}%")
print(f"  Total round trip cost:  {TOTAL_COST_PCT*100:.3f}%")

prices = trades_df['trade_price'].values.astype(np.float64)
entry_asks = trades_df['ask_price_0'].values.astype(np.float64)
exit_bids = trades_df['bid_price_0'].values.astype(np.float64)
real_entry = entry_asks * (1 + SLIPPAGE_PER_SIDE)
n = len(prices)

H = 2000

print(f"\nPrecomputing fwd max/min (H={H} ticks)...")
fwd_max = np.full(n, np.nan)
fwd_min = np.full(n, np.nan)
for i in range(n - H):
    seg = prices[i + 1:i + 1 + H]
    fwd_max[i] = np.max(seg)
    fwd_min[i] = np.min(seg)

# Forward return distributions
print(f"\nForward Return Distribution (% from current price):")
print(f"{'Metric':>10}  {'P50':>8}  {'P75':>8}  {'P90':>8}  {'P95':>8}  {'P99':>8}")
print(DASH)

vm = ~np.isnan(fwd_max)
fwd_returns_max = (fwd_max[vm] - prices[vm]) / prices[vm] * 100
fwd_returns_min = (fwd_min[vm] - prices[vm]) / prices[vm] * 100

print(f"{'Max Fwd':>10}  {np.percentile(fwd_returns_max,50):>7.3f}% {np.percentile(fwd_returns_max,75):>7.3f}% "
      f"{np.percentile(fwd_returns_max,90):>7.3f}% {np.percentile(fwd_returns_max,95):>7.3f}% "
      f"{np.percentile(fwd_returns_max,99):>7.3f}%")
print(f"{'Min Fwd':>10}  {np.percentile(fwd_returns_min,50):>7.3f}% {np.percentile(fwd_returns_min,75):>7.3f}% "
      f"{np.percentile(fwd_returns_min,90):>7.3f}% {np.percentile(fwd_returns_min,95):>7.3f}% "
      f"{np.percentile(fwd_returns_min,99):>7.3f}%")

# TP Win Rate Analysis
print(f"\n{SEP}")
print(f"TP WIN RATES (H={H} ticks forward)")
print(f"Entry=ask*1.015% slip, Fee={ROUND_TRIP_FEE*100:.1f}%, Exit slip=0.015%")
print(SEP)

tp_levels_pct = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80]

print(f"\nWin rate = % of entries where fwd_max >= entry*(1+TP%)*(1+cost)")
print(f"{'TP%':>8}  ", end="")
for tp in tp_levels_pct:
    print(f"{tp:>6.2f}%", end="")
print()
print(f"{'No SL':>8}  ", end="")
for tp_pct in tp_levels_pct:
    tp_dec = tp_pct / 100
    breakeven_mult = 1 + tp_dec + TOTAL_COST_PCT
    hits = np.sum(fwd_max[vm] >= prices[vm] * breakeven_mult)
    wr = hits / np.sum(vm) * 100
    print(f"{wr:>6.2f}%", end=" ")
print()

# Expected value
print(f"\n{SEP}")
print(f"EXPECTED VALUE (TP vs SL, H={H} ticks)")
print(f"Proportional model: closer level hits first")
print(SEP)

SL_CANDIDATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
TP_CANDIDATES = [0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

best_ev = -999.0
best_row = None
all_rows = []

for sl_pct in SL_CANDIDATES:
    sl_dec = sl_pct / 100
    print(f"\n--- SL={sl_pct:.2f}% ---")
    print(f"{'TP%':>8} {'WinRate':>9} {'ExpVal%':>9} {'PF':>8}  {'TPonly':>8} {'Both':>8} {'SLonly':>8}")
    print(DASH)
    
    for tp_pct in TP_CANDIDATES:
        tp_dec = tp_pct / 100
        tp_levels = real_entry * (1 + tp_dec)
        sl_levels = real_entry * (1 - sl_dec)
        
        tp_reached = fwd_max >= tp_levels
        sl_reached = fwd_min <= sl_levels
        
        tp_only = tp_reached & ~sl_reached
        sl_only = sl_reached & ~tp_reached
        both = tp_reached & sl_reached
        neither = ~(tp_reached | sl_reached)
        
        # Both TP and SL were contacted. Which hit first?
        # Probability proportional to distance: closer level hits first.
        # P(TP first) = distance_to_SL / (distance_to_SL + distance_to_TP)
        # = sl_dec / (sl_dec + tp_dec)
        both = tp_reached & sl_reached
        p_tp_first = sl_dec / (sl_dec + tp_dec)
        both_wins = both * p_tp_first  # fractional count
        
        wins = np.sum(tp_only) + np.sum(both_wins)
        losses = np.sum(sl_only) + np.sum(both) - np.sum(both_wins)
        total = wins + losses
        if total < 50:
            continue
        
        win_rate = wins / total * 100
        
        avg_win_ret = tp_dec - ROUND_TRIP_FEE - SLIPPAGE_PER_SIDE  # costs
        avg_loss_ret = -sl_dec - ROUND_TRIP_FEE - 2 * SLIPPAGE_PER_SIDE  # extra slip on SL
        
        exp_val = wins / total * avg_win_ret + losses / total * avg_loss_ret
        exp_val_pct = exp_val * 100
        
        gross_profit = wins * max(avg_win_ret, 0)
        gross_loss = losses * abs(min(avg_loss_ret, 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        print(f"{tp_pct:>7.2f}% {win_rate:>8.2f}% {exp_val_pct:>8.4f}% {pf:>8.3f}  "
              f"{np.sum(tp_only):>8,d} {np.sum(both):>8,d} {np.sum(sl_only):>8,d}")
        
        all_rows.append({
            'tp_pct': tp_pct, 'sl_pct': sl_pct, 'win_rate': win_rate,
            'exp_val_pct': exp_val_pct, 'trades': total, 'profit_factor': pf,
        })
        
        if exp_val_pct > best_ev:
            best_ev = exp_val_pct
            best_row = {
                'tp_pct': tp_pct, 'sl_pct': sl_pct, 'win_rate': win_rate,
                'exp_val_pct': exp_val_pct, 'trades': total, 'pf': pf,
            }

# Summary
print(f"\n{SEP}")
print(f"OPTIMAL TP/SL FOR XRPUSDT")
print(SEP)
if best_row:
    print(f"\n  Take Profit:  {best_row['tp_pct']:.2f}%")
    print(f"  Stop Loss:    {best_row['sl_pct']:.2f}%")
    print(f"  Win Rate:     {best_row['win_rate']:.2f}%")
    print(f"  Expected Val: {best_row['exp_val_pct']:+.4f}% per trade")
    print(f"  Trade Count:  {best_row['trades']:,} resolved entries")
    print(f"  Profit Fact:  {best_row['pf']:.3f}")

print(f"\n{DASH}")
print(f"KEY INSIGHTS:")
print(f"")
print(f"  1. XRP tick-level forward price moves (max, H={H} ticks):")
print(f"     - P50: {np.percentile(fwd_returns_max,50):.3f}%")
print(f"     - P90: {np.percentile(fwd_returns_max,90):.3f}%")
print(f"     - P95: {np.percentile(fwd_returns_max,95):.3f}%")
print(f"")
print(f"  2. Current TP_mult=2.5-7.0 with ATR_floor=0.1% => TP=0.25%-0.70%")
print(f"     At these levels, win rate (no SL) is:")
for tp_pct in [0.25, 0.37, 0.50, 0.70]:
    tp_dec = tp_pct / 100
    breakeven_mult = 1 + tp_dec + TOTAL_COST_PCT
    hits = np.sum(fwd_max[vm] >= prices[vm] * breakeven_mult)
    wr = hits / np.sum(vm) * 100
    print(f"       TP={tp_pct:.2f}%: {wr:.1f}% of entries")
print(f"")
print(f"  3. RECOMMENDATION: Set TP={best_row['tp_pct']:.2f}%, SL={best_row['sl_pct']:.2f}%")
print(f"     Rationale: maximizes expected value while keeping trade count high")
print(f"")
print(f"  4. Strategy edge SHOULD improve these conservative random-entry estimates")
print(DASH)

if all_rows:
    pd.DataFrame(all_rows).to_csv(project_root / "diagnostics" / "tp_optimization.csv", index=False)
    print(f"\nSaved to diagnostics/tp_optimization.csv")
