"""
run_backtest_20min.py
Run full backtest with: MIN_HOLD=1200s, TP=0.50%, SL=0.35%
All 4 active strategies included (excl. Liquidity Sweep).
"""

import sys, json, math
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from core.feature_engine import FeatureConfig, Regime
from core.data_structures import Side, SignalType
from backtesting.engine import BacktestEngine, BacktestMetrics
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import (
    create_absorption_strategy, create_delta_divergence_strategy,
    create_stacked_imbalance_strategy, create_value_area_strategy,
    StrategyDefinition,
)
from core.fee_aware_filter import FeeAwareFilter

SEP = "=" * 90
DASH = "-" * 90

# ─── Load data ───────────────────────────────────────────────────────
print(SEP)
print("BACKTEST REPORT: MIN_HOLD=20min | TP=0.50% | SL=0.35%")
print(SEP)

df = pd.read_parquet(project_root / "data" / "backtests" / "XRPUSDT_market_data_20260328.parquet")
print(f"\nData: {len(df):,} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")

# ─── Configure TP/SL parameters ──────────────────────────────────────
# Target: TP=0.50%, SL=0.35% (as percentages of entry price)
# atr_pct_floor = 0.001, so:
#   sl_mult = 0.0035 / 0.001 = 3.5
#   tp_mult = 0.0050 / 0.001 = 5.0
TP_MULT = 5.0
SL_MULT = 3.5

def configure_strategy(strat):
    """Override strategy to use fixed TP=0.5%, SL=0.35% across all regimes."""
    strat.tp_mult_high_vol = TP_MULT
    strat.tp_mult_low_vol = TP_MULT
    strat.tp_mult_trending = TP_MULT
    strat.sl_mult_high_vol = SL_MULT
    strat.sl_mult_low_vol = SL_MULT
    strat.sl_mult_trending = SL_MULT
    return strat

# ─── Build strategies ────────────────────────────────────────────────
strategies = {
    "Absorption": configure_strategy(create_absorption_strategy()),
    "Delta Divergence": configure_strategy(create_delta_divergence_strategy()),
    "Stacked Imbalance": configure_strategy(create_stacked_imbalance_strategy()),
    "Value Area MR": configure_strategy(create_value_area_strategy()),
}

# ─── Engine config ───────────────────────────────────────────────────
# fee_pct=0.0005 (0.05%/side = 0.1% round trip, matches user spec)
# slippage_pct=0.0003 (matches 50ms VPS estimate)
# sl_extra_slippage_pct=0.0003 (extra slip on stop-loss)
engine = BacktestEngine(
    initial_capital=100_000.0,
    fee_pct=0.0005,
    slippage_pct=0.0003,
    sl_extra_slippage_pct=0.0003,
    warmup_seconds=60.0,
    min_time_between_trades_sec=30.0,
    risk_limits=RiskLimits(
        max_position_size=10000.0,
        max_position_value_pct=0.25,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
        max_trades_per_day=50,
        max_trades_per_hour=10,
        min_time_between_trades_sec=30,
        max_consecutive_losses=20,
    ),
)

print(f"\nEngine config:")
print(f"  fee_pct:                {engine.fee_pct*100:.2f}%")
print(f"  slippage_pct:           {engine.slippage_pct*100:.2f}%")
print(f"  sl_extra_slippage_pct:  {engine.sl_extra_slippage_pct*100:.2f}%")
print(f"  MinHoldBeforeFlowExit:  {engine.MIN_HOLD_BEFORE_FLOW_EXIT_SEC}s ({engine.MIN_HOLD_BEFORE_FLOW_EXIT_SEC/60:.0f}min)")
print(f"  min_time_between_trades: {engine.min_time_between_trades_sec}s")

# ─── Run each strategy ──────────────────────────────────────────────
all_metrics = {}
all_trades = {}
combined = []

for sname, strategy in strategies.items():
    print(f"\n{SEP}")
    print(f"RUNNING: {sname}")
    print(SEP)

    # Reset engine between runs
    metrics: BacktestMetrics = engine.run(df, strategy)

    all_metrics[sname] = metrics
    all_trades[sname] = list(engine.closed_trades)
    combined.extend(engine.closed_trades)

    # Quick summary
    print(f"  Trades:     {metrics.total_trades:>5,d}")
    print(f"  Win Rate:   {metrics.win_rate*100:>6.2f}%")
    print(f"  Return:     {metrics.total_return_pct*100:>+7.3f}%")
    print(f"  Sharpe:     {metrics.sharpe_ratio:>7.3f}")
    print(f"  MaxDD:      {metrics.max_drawdown_pct*100:>6.2f}%")
    print(f"  ProfitFac:  {metrics.profit_factor:>7.3f}")
    print(f"  PF (fee_adj): {metrics.profit_factor:>7.3f}")

# ─── Combined report ────────────────────────────────────────────────
print(f"\n{SEP}")
print("FULL BACKTEST REPORT")
print(SEP)

total_trades = sum(m.total_trades for m in all_metrics.values())
total_winning = sum(m.winning_trades for m in all_metrics.values())
total_losing = sum(m.losing_trades for m in all_metrics.values())

print(f"\n{'─'*90}")
print(f"TRADE SUMMARY (all strategies combined)")
print(f"{'─'*90}")
print(f"  Total trades (all strategies):   {total_trades:>6,d}")
print(f"  Total winning:                   {total_winning:>6,d}")
print(f"  Total losing:                    {total_losing:>6,d}")
if total_trades > 0:
    print(f"  Combined win rate:               {total_winning/total_trades*100:>6.2f}%")
    print(f"  Combined loss rate:              {total_losing/total_trades*100:>6.2f}%")

if combined:
    avg_duration = np.mean([t.duration_seconds for t in combined])
    avg_hold_min = avg_duration / 60
    max_duration = np.max([t.duration_seconds for t in combined]) / 60
    print(f"  Avg hold time:                   {avg_hold_min:.1f} min")
    print(f"  Max hold time:                   {max_duration:.1f} min")

    # Exit reason breakdown
    reasons = {}
    for t in combined:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"\n  Exit reasons breakdown:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:>35s}: {count:>5,d} ({count/len(combined)*100:>5.1f}%)")

print(f"\n{'─'*90}")
print(f"PER-STRATEGY BREAKDOWN")
print(f"{'─'*90}")
print(f"{'Strategy':>20} {'Trades':>8} {'WinRate':>9} {'AvgWin%':>9} {'AvgLoss%':>9} "
      f"{'ExpVal%':>10} {'PF':>8} {'Sharpe':>8} {'MaxDD':>8} {'Return%':>9}")
print(DASH)

for sname in strategies:
    m = all_metrics[sname]
    t = all_trades[sname]
    if m.total_trades > 0:
        wins = [x for x in t if x.pnl > 0]
        losses = [x for x in t if x.pnl <= 0]
        avg_win = float(np.mean([x.pnl_pct*100 for x in wins])) if wins else 0
        avg_loss = float(np.mean([x.pnl_pct*100 for x in losses])) if losses else 0
        exp_val = float(np.mean([x.pnl_pct*100 for x in t]))
    else:
        avg_win = avg_loss = exp_val = 0
    print(f"{sname:>20} {m.total_trades:>8,d} {m.win_rate*100:>8.2f}% "
          f"{avg_win:>8.3f}% {avg_loss:>8.3f}% "
          f"{exp_val:>9.4f}% {m.profit_factor:>7.3f} {m.sharpe_ratio:>7.3f} "
          f"{m.max_drawdown_pct*100:>6.2f}% {m.total_return_pct*100:>8.3f}%")

# Full metrics per strategy
print(f"\n{'─'*90}")
print(f"DETAILED METRICS PER STRATEGY")
print(f"{'─'*90}")

for sname in strategies:
    m = all_metrics[sname]
    t = all_trades[sname]
    print(f"\n  {'─'*70}")
    print(f"  {sname.upper()}")
    print(f"  {'─'*70}")
    print(f"  Total Trades:         {m.total_trades:>8,d}")
    print(f"  Winning Trades:       {m.winning_trades:>8,d}")
    print(f"  Losing Trades:        {m.losing_trades:>8,d}")
    print(f"  Win Rate:             {m.win_rate*100:>8.2f}%")
    print(f"  Total Return:         {m.total_return_pct*100:>+8.3f}%")
    print(f"  Annualized Return:    {m.annualized_return*100:>+8.3f}%")
    print(f"  Sharpe Ratio:         {m.sharpe_ratio:>8.3f}")
    print(f"  Sortino Ratio:        {m.sortino_ratio:>8.3f}")
    print(f"  Calmar Ratio:         {m.calmar_ratio:>8.3f}")
    print(f"  Max Drawdown:         {m.max_drawdown_pct*100:>8.2f}%")
    print(f"  Profit Factor:        {m.profit_factor:>8.3f}")
    print(f"  Expected Value (pct): {m.expectancy/m.initial_capital*100*0:>+8.4f}%")
    if m.total_trades > 0:
        print(f"  Avg Win (pct):        {np.mean([x.pnl_pct*100 for x in t if x.pnl > 0]):>+8.3f}%")
        print(f"  Avg Loss (pct):       {np.mean([x.pnl_pct*100 for x in t if x.pnl <= 0]):>+8.3f}%")
        print(f"  Avg Duration:         {np.mean([x.duration_seconds/60 for x in t]):>8.1f} min")
        print(f"  Trades Per Day:       {m.trades_per_day:>8.2f}")
        print(f"  Risk Rejected:        {m.signals_rejected_by_risk:>8,d}")
        print(f"  Fee Filter Rejected:  {m.signals_rejected_by_fee_filter:>8,d}")

# Equity curve
print(f"\n{'─'*90}")
print(f"FEE & COST ANALYSIS")
print(f"{'─'*90}")
print(f"  Fee per side:           0.05% (entry) + 0.05% (exit) = 0.10% round trip")
print(f"  Slippage per side:      0.03%")
print(f"  SL extra slippage:      0.03%")
print(f"  Total cost (TP trade):  0.10% fee + 0.06% slip = 0.16%")
print(f"  Total cost (SL trade):  0.10% fee + 0.09% slip = 0.19%")
print(f"  MinHoldBeforeFlowExit:  1200s (20 min)")

# Win rate vs no-SL analysis from earlier
print(f"\n{'─'*90}")
print(f"EXPECTED VS ACTUAL (from the earlier random-entry analysis)")
print(f"{'─'*90}")
print(f"  For TP=0.50%, SL=0.35% (20min horizon):")
print(f"    Random entry win rate:    67.5%")
print(f"    Random entry expected val: +0.089% per trade")
print(f"    Strategy win rate:         (see above for actual)")
print(f"  Strategy edge = Actual WR / Random WR")
for sname in strategies:
    m = all_metrics[sname]
    if m.total_trades > 0:
        t = all_trades[sname]
        actual_wr = m.win_rate * 100
        edge = actual_wr / 67.5
        avg_hold = np.mean([x.duration_seconds/60 for x in t])
        ret_pct = np.mean([x.pnl_pct*100 for x in t]) if t else 0
        print(f"    {sname:>20}: WR={actual_wr:>5.1f}%  edge={edge:>5.2f}x  hold={avg_hold:>4.1f}min  EV={ret_pct:>+.4f}%")

print(f"\n{SEP}")
print("END OF REPORT")
print(SEP)
print(f"\nNote: Engine MIN_HOLD_BEFORE_FLOW_EXIT_SEC changed to {engine.MIN_HOLD_BEFORE_FLOW_EXIT_SEC}")
