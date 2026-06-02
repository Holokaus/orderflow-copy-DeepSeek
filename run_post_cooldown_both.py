"""
Post-cooldown Both Strategies
Runs Absorption + StackedImbalance with loss streak cooldown,
combines trades chronologically, and reports combined P&L.
"""
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
logger.remove()

from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import (
    create_absorption_strategy, create_stacked_imbalance_strategy,
)

INITIAL_CAPITAL = 100.0
DATA_PATH = project_root / "data" / "backtests"

def load_data():
    df1 = pd.read_parquet(DATA_PATH / "ICPUSDT_20260531_processed.parquet")
    df2 = pd.read_parquet(DATA_PATH / "ICPUSDT_20260601_processed.parquet")
    return pd.concat([df1, df2]).reset_index(drop=True)

def make_engine():
    return BacktestEngine(
        initial_capital=INITIAL_CAPITAL, fee_pct=0.0005, slippage_pct=0.0003,
        sl_extra_slippage_pct=0.0003, warmup_seconds=60.0,
        min_time_between_trades_sec=30.0,
        risk_limits=RiskLimits(
            max_position_size=10000.0, max_position_value_pct=0.25,
            max_daily_loss_pct=0.02, max_drawdown_pct=0.10,
            max_trades_per_day=50, max_trades_per_hour=10,
            min_time_between_trades_sec=30, max_consecutive_losses=20,
        ),
    )

def run_strategy(strat):
    engine = make_engine()
    engine.run(df, strat)
    return [(t.exit_time, t.pnl_pct, t.exit_reason) for t in engine.closed_trades]

def report(label, trades, span_days):
    if not trades:
        print(f"\n  {label}: 0 trades")
        return
    pnl_pcts = [p for _, p, _ in trades]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]
    total_ret = (np.prod([1 + p for p in pnl_pcts]) - 1) * 100
    print(f"\n  {'='*65}")
    print(f"  {label}")
    print(f"  {'='*65}")
    print(f"    Trades:       {len(trades)}")
    print(f"    Win rate:     {len(wins)/len(trades)*100:.1f}%")
    print(f"    Total return: {total_ret:+.3f}% ({INITIAL_CAPITAL * (1+total_ret/100):.2f}$)")
    print(f"    Daily ret:    {total_ret/span_days:+.3f}%/day")
    print(f"    Monthly ret:  {total_ret/span_days*30:+.3f}%/mo")
    print(f"    Profit fact:  {sum(wins)/abs(sum(losses)):.3f}" if losses else "    Profit fact:  inf")
    peak = INITIAL_CAPITAL
    eq = INITIAL_CAPITAL
    max_dd = 0
    for _, p, _ in trades:
        eq *= (1 + p)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)
    print(f"    Max DD:       {max_dd:.2f}%")
    print(f"    Avg win:      {np.mean(wins)*100:+.3f}%" if wins else "")
    print(f"    Avg loss:     {np.mean(losses)*100:+.3f}%" if losses else "")
    print()
    for i, (ts, pnl, reason) in enumerate(trades):
        print(f"    #{i+1:>2}  {ts.strftime('%m/%d %H:%M')}  {pnl*100:>+7.3f}%  ({reason})")

# ── Run ──
print("Loading ICP data...")
df = load_data()
ts_min = pd.to_datetime(df['timestamp']).min()
ts_max = pd.to_datetime(df['timestamp']).max()
span_days = (ts_max - ts_min).total_seconds() / 86400
print(f"  {ts_min}  ->  {ts_max}  ({span_days:.1f} days)")

print("\nRunning Absorption (post-cooldown)...")
abs_trades = run_strategy(create_absorption_strategy())
print(f"  {len(abs_trades)} trades")

print("Running StackedImbalance (post-cooldown)...")
si_trades = run_strategy(create_stacked_imbalance_strategy())
print(f"  {len(si_trades)} trades")

both = sorted(abs_trades + si_trades, key=lambda x: x[0])

report("Absorption Only", abs_trades, span_days)
report("StackedImbalance Only", si_trades, span_days)
report("Both Strategies (merged chronologically)", both, span_days)

print(f"\n  Final equity breakdown:")
print(f"    Absorption:       ${INITIAL_CAPITAL * float(np.prod([1+p for _,p,_ in abs_trades])):.2f}")
print(f"    StackedImbalance: ${INITIAL_CAPITAL * float(np.prod([1+p for _,p,_ in si_trades])):.2f}")
print(f"    Combined:         ${INITIAL_CAPITAL * float(np.prod([1+p for _,p,_ in both])):.2f}")
