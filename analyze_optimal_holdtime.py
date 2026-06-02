"""
analyze_optimal_holdtime.py
Find the best trade hold time (horizon) for TP/SL on XRPUSDT.
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
print("XRPUSDT HOLD TIME OPTIMIZATION")
print(SEP)

df = pd.read_parquet(project_root / "data" / "backtests" / "XRPUSDT_market_data_20260328.parquet")
trade_mask = df['trade_price'].notna() & (df['trade_price'] > 0)
trades_df = df[trade_mask].copy()

prices = trades_df['trade_price'].values.astype(np.float64)
entry_asks = trades_df['ask_price_0'].values.astype(np.float64)
exit_bids = trades_df['bid_price_0'].values.astype(np.float64)
n = len(prices)

ROUND_TRIP_FEE = 0.001
SLIPPAGE_PER_SIDE = 0.00015
TOTAL_COST = ROUND_TRIP_FEE + 2 * SLIPPAGE_PER_SIDE

print(f"\nCost: {ROUND_TRIP_FEE*100:.1f}% fee + {2*SLIPPAGE_PER_SIDE*100:.2f}% slip = {TOTAL_COST*100:.2f}% total")
print(f"Trades: {n:,}\n")

# Horizons to test (in ticks)
HORIZONS = {
    '25 ticks (~3-4s)': 25,
    '50 ticks (~7-10s)': 50,
    '100 ticks (~15s)': 100,
    '250 ticks (~30-40s)': 250,
    '500 ticks (~1min)': 500,
    '1000 ticks (~2min)': 1000,
    '2000 ticks (~3-5min)': 2000,
    '5000 ticks (~10min)': 5000,
    '10000 ticks (~20min)': 10000,
}

# TP/SL candidates (in %)
TP_CANDIDATES = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
SL_CANDIDATES = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]

all_results = []
best_overall_ev = -999.0
best_overall = None

for h_label, H in HORIZONS.items():
    print(f"\n{'='*70}")
    print(f"HORIZON: {h_label} ({H} ticks)")
    print(f"{'='*70}")

    # Compute fwd max/min
    fwd_max = np.full(n, np.nan)
    fwd_min = np.full(n, np.nan)
    for i in range(n - H):
        seg = prices[i + 1:i + 1 + H]
        fwd_max[i] = np.max(seg)
        fwd_min[i] = np.min(seg)

    vm = ~np.isnan(fwd_max)
    returns_max = (fwd_max[vm] - prices[vm]) / prices[vm] * 100
    returns_min = (fwd_min[vm] - prices[vm]) / prices[vm] * 100

    print(f"Fwd return distribution: P50={np.percentile(returns_max,50):.3f}%  "
          f"P75={np.percentile(returns_max,75):.3f}%  "
          f"P90={np.percentile(returns_max,90):.3f}%  "
          f"P95={np.percentile(returns_max,95):.3f}%")
    print(f"Fwd min distribution:  P50={np.percentile(returns_min,50):.3f}%  "
          f"P25={np.percentile(returns_min,25):.3f}%  "
          f"P10={np.percentile(returns_min,10):.3f}%  "
          f"P5={np.percentile(returns_min,5):.3f}%")

    # No-SL win rates
    print(f"\nWin rate (no SL, just TP reach):")
    for tp_pct in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50]:
        tp_dec = tp_pct / 100
        mult = 1 + tp_dec + TOTAL_COST
        wr = np.sum(fwd_max[vm] >= prices[vm] * mult) / np.sum(vm) * 100
        print(f"  TP={tp_pct:.2f}%: {wr:.1f}%")

    # Best TP/SL for this horizon
    best_ev_h = -999.0
    best_h = None

    print(f"\n{'TP%':>6} {'SL%':>6} {'WinRate':>9} {'ExpVal%':>10} {'Trades':>8} {'PF':>8}")
    print(DASH)

    for tp_pct in TP_CANDIDATES:
        for sl_pct in SL_CANDIDATES:
            if sl_pct < tp_pct * 0.5:
                continue  # skip unrealistic R:R

            tp_dec = tp_pct / 100
            sl_dec = sl_pct / 100

            tp_levels = (entry_asks * (1 + SLIPPAGE_PER_SIDE)) * (1 + tp_dec)
            sl_levels = (entry_asks * (1 + SLIPPAGE_PER_SIDE)) * (1 - sl_dec)

            tp_reached = fwd_max >= tp_levels
            sl_reached = fwd_min <= sl_levels

            tp_only = tp_reached & ~sl_reached
            sl_only = sl_reached & ~tp_reached
            both = tp_reached & sl_reached

            # Proportional model: closer level hits first
            p_tp_first = sl_dec / (sl_dec + tp_dec)
            both_wins = both * p_tp_first

            wins = np.sum(tp_only) + np.sum(both_wins)
            losses = np.sum(sl_only) + np.sum(both) - np.sum(both_wins)
            total = wins + losses
            if total < 50:
                continue

            win_rate = wins / total * 100
            avg_win_ret = tp_dec - TOTAL_COST
            avg_loss_ret = -sl_dec - TOTAL_COST - SLIPPAGE_PER_SIDE
            exp_val = wins / total * avg_win_ret + losses / total * avg_loss_ret
            exp_val_pct = exp_val * 100

            gross_profit = wins * max(avg_win_ret, 0)
            gross_loss = losses * abs(min(avg_loss_ret, 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            all_results.append({
                'horizon_label': h_label,
                'horizon_ticks': H,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'win_rate': round(win_rate, 2),
                'exp_val_pct': round(exp_val_pct, 4),
                'trades': int(total),
                'profit_factor': round(pf, 3),
            })

            if exp_val_pct > best_ev_h:
                best_ev_h = exp_val_pct
                best_h = {
                    'tp': tp_pct, 'sl': sl_pct,
                    'wr': win_rate, 'ev': exp_val_pct,
                    'trades': int(total), 'pf': pf,
                }

            # Show best few rows
            if exp_val_pct > -0.05:  # show only promising ones
                print(f"{tp_pct:>5.2f}% {sl_pct:>5.2f}% {win_rate:>8.2f}% {exp_val_pct:>9.4f}% {int(total):>8,d} {pf:>8.3f}")

    if best_h:
        print(f"\n  BEST for {h_label}: TP={best_h['tp']:.2f}% SL={best_h['sl']:.2f}%  "
              f"EV={best_h['ev']:+.4f}% WR={best_h['wr']:.1f}% Trades={best_h['trades']:,}")
        if best_h['ev'] > best_overall_ev:
            best_overall_ev = best_h['ev']
            best_overall = {**best_h, 'horizon': h_label, 'horizon_ticks': H}

# Overall best
print(f"\n\n{SEP}")
print("BEST CONFIGURATION ACROSS ALL HORIZONS")
print(SEP)
if best_overall:
    print(f"\n  Horizon:      {best_overall['horizon']} ({best_overall['horizon_ticks']} ticks)")
    print(f"  Take Profit:  {best_overall['tp']:.2f}%")
    print(f"  Stop Loss:    {best_overall['sl']:.2f}%")
    print(f"  Win Rate:     {best_overall['wr']:.2f}%")
    print(f"  Expected Val: {best_overall['ev']:+.4f}% per trade")
    print(f"  Trade Count:  {best_overall['trades']:,}")
    print(f"  Profit Fact:  {best_overall['pf']:.3f}")

# Summary table: best EV at each horizon
print(f"\n{SEP}")
print("BEST EV BY HORIZON")
print(SEP)
print(f"{'Horizon':>30} {'TP%':>6} {'SL%':>6} {'WinRate':>9} {'ExpVal%':>10} {'Trades':>9} {'PF':>8}")
print(DASH)

best_by_horizon = {}
for r in all_results:
    key = r['horizon_ticks']
    if key not in best_by_horizon or r['exp_val_pct'] > best_by_horizon[key]['exp_val_pct']:
        best_by_horizon[key] = r

for h_ticks in sorted(best_by_horizon.keys()):
    r = best_by_horizon[h_ticks]
    print(f"{r['horizon_label']:>30} {r['tp_pct']:>5.2f}% {r['sl_pct']:>5.2f}% "
          f"{r['win_rate']:>8.2f}% {r['exp_val_pct']:>9.4f}% {r['trades']:>9,d} {r['profit_factor']:>7.3f}")

# Save
df_out = pd.DataFrame(all_results)
df_out.to_csv(project_root / "diagnostics" / "holdtime_optimization.csv", index=False)
print(f"\nSaved to diagnostics/holdtime_optimization.csv")
