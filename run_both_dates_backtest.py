"""
Backtest both May 31 and June 8 with updated min hold (5 min) and graduated tiers.
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

def load_data(date_label):
    if date_label == "may31":
        return pd.read_parquet(DATA_PATH / "ICPUSDT_20260531_processed.parquet")
    elif date_label == "jun8":
        return pd.read_parquet(DATA_PATH / "ICPUSDT_20260608_processed.parquet")
    else:
        raise ValueError(f"Unknown date: {date_label}")

def make_engine():
    return BacktestEngine(
        initial_capital=INITIAL_CAPITAL, fee_pct=0.0005, slippage_pct=0.0003,
        sl_extra_slippage_pct=0.0003, warmup_seconds=60.0,
        min_time_between_trades_sec=30.0,
        risk_limits=RiskLimits(
            max_position_size=10000.0, max_position_value_pct=0.25,
            max_daily_loss_pct=0.02, max_drawdown_pct=0.10,
            max_trades_per_day=50, max_trades_per_hour=10,
            min_time_between_trades_sec=30, max_consecutive_losses=3,
        ),
    )

def run_strategy(df, strat):
    engine = make_engine()
    engine.run(df, strat)
    return [(t.exit_time, t.pnl_pct, t.exit_reason, t.entry_time) for t in engine.closed_trades]

def report(label, trades, span_days):
    if not trades:
        print(f"\n  {label}: 0 trades")
        return
    pnl_pcts = [p for _, p, _, _ in trades]
    durations = [(t - e).total_seconds() / 60 for t, _, _, e in trades]
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
    print(f"    Profit fact:  {sum(wins)/abs(sum(losses)):.3f}" if losses else "    Profit fact:  inf")
    print(f"    Avg dur:      {np.mean(durations):.1f} min")
    print(f"    Med dur:      {np.median(durations):.1f} min")
    peak = INITIAL_CAPITAL
    eq = INITIAL_CAPITAL
    max_dd = 0
    for _, p, _, _ in trades:
        eq *= (1 + p)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)
    print(f"    Max DD:       {max_dd:.2f}%")
    print(f"    Avg win:      {np.mean(wins)*100:+.3f}%" if wins else "")
    print(f"    Avg loss:     {np.mean(losses)*100:+.3f}%" if losses else "")
    exit_reasons = {}
    for _, _, r, _ in trades:
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    print(f"    Exit reasons: {exit_reasons}")
    print()
    for i, (ts, pnl, reason, et) in enumerate(trades):
        dur = (ts - et).total_seconds() / 60
        print(f"    #{i+1:>2}  {et.strftime('%m/%d %H:%M')}-{ts.strftime('%H:%M')} ({dur:.0f}m)  {pnl*100:>+7.3f}%  ({reason})")

# ── Run ──
for date_label in ["may31", "jun8"]:
    print(f"\n{'#'*70}")
    print(f"#  BACKTEST: {date_label.upper()}")
    print(f"{'#'*70}")
    df = load_data(date_label)
    ts_min = pd.to_datetime(df['timestamp']).min()
    ts_max = pd.to_datetime(df['timestamp']).max()
    span_days = (ts_max - ts_min).total_seconds() / 86400
    print(f"  Data: {ts_min}  ->  {ts_max}  ({span_days:.2f} days, {len(df)} ticks)")

    print(f"\n  Running Absorption...")
    abs_trades = run_strategy(df, create_absorption_strategy())

    print(f"  Running StackedImbalance...")
    si_trades = run_strategy(df, create_stacked_imbalance_strategy())

    both = sorted(abs_trades + si_trades, key=lambda x: x[0])

    report("Absorption Only", abs_trades, span_days)
    report("StackedImbalance Only", si_trades, span_days)
    report("Both Strategies (merged)", both, span_days)
