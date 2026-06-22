"""Multi-date backtest runner - all 4 strategies per date, combined results."""
import sys, os
from pathlib import Path

# Suppress ALL log output before any imports
os.environ["LOGURU_LEVEL"] = "CRITICAL"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
logger.remove()

from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import (
    create_absorption_strategy,
    create_stacked_imbalance_strategy,
    create_delta_divergence_strategy,
    create_value_area_strategy,
    create_trend_following_strategy,
)

INITIAL_CAPITAL = 100.0
DATA_PATH = project_root / "data" / "backtests"

STRATEGIES = [
    ("absorption", create_absorption_strategy),
    ("stacked_imbalance", create_stacked_imbalance_strategy),
    ("delta_divergence", create_delta_divergence_strategy),
    ("value_area", create_value_area_strategy),
    ("trend_following", create_trend_following_strategy),
]

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

def run_strategy(df, strat_fn):
    engine = make_engine()
    strat = strat_fn()
    engine.run(df, strat)
    return [(t.exit_time, t.pnl_pct, t.exit_reason, t.entry_time) for t in engine.closed_trades]

dates = [
    ("May 31", "ICPUSDT_20260531_processed.parquet"),
    ("Jun 1",  "ICPUSDT_20260601_processed.parquet"),
    ("Jun 8",  "ICPUSDT_20260608_processed.parquet"),
    ("Jun 9",  "ICPUSDT_20260609_processed.parquet"),
]

all_pnl = []
all_trades = []

for date_label, filename in dates:
    print(f"\n{'='*70}")
    print(f"  {date_label}")
    print(f"{'='*70}")
    df = pd.read_parquet(DATA_PATH / filename)
    ts_min = pd.to_datetime(df['timestamp']).min()
    ts_max = pd.to_datetime(df['timestamp']).max()
    span_days = (ts_max - ts_min).total_seconds() / 86400
    print(f"  {len(df)} ticks, {span_days:.2f} days, price: {df['trade_price'].min():.3f} - {df['trade_price'].max():.3f}")
    print()

    day_trades = []
    for name, strat_fn in STRATEGIES:
        trades = run_strategy(df, strat_fn)
        if trades:
            pnl_pcts = [p for _, p, _, _ in trades]
            wins = sum(1 for p in pnl_pcts if p > 0)
            ret = (np.prod([1 + p for p in pnl_pcts]) - 1) * 100
            losses = [p for p in pnl_pcts if p <= 0]
            pf = sum(wins for p in pnl_pcts if p > 0) / abs(sum(losses)) if losses else float('inf')
            print(f"    {name:20s} | n={len(trades):3d}  WR={wins/len(trades)*100:5.1f}%  ret={ret:>+7.3f}%  PF={pf:.3f}")
        else:
            print(f"    {name:20s} | n=  0")
        day_trades.extend(trades)

    combined = sorted(day_trades, key=lambda x: x[0])
    if combined:
        cp = [p for _, p, _, _ in combined]
        ret = (np.prod([1 + p for p in cp]) - 1) * 100
        wins = sum(1 for p in cp if p > 0)
        losses = [p for p in cp if p <= 0]
        pf = sum(wins for p in cp if p > 0) / abs(sum(losses)) if losses else float('inf')
        print(f"    {'ALL':20s} | n={len(cp):3d}  WR={wins/len(cp)*100:5.1f}%  ret={ret:>+7.3f}%  PF={pf:.3f}")
        print()
        for i, (ts, pnl, reason, et) in enumerate(combined):
            dur = (ts - et).total_seconds() / 60
            print(f"    #{i+1:>2}  {et.strftime('%m/%d %H:%M')}-{ts.strftime('%H:%M')} ({dur:.0f}m)  {pnl*100:>+7.3f}%  ({reason})")

    all_pnl.extend([p for _, p, _, _ in combined])
    all_trades.extend(combined)

print(f"\n{'='*70}")
print(f"  COMBINED")
print(f"{'='*70}")
if all_pnl:
    ret = (np.prod([1 + p for p in all_pnl]) - 1) * 100
    wins = sum(1 for p in all_pnl if p > 0)
    losses = [p for p in all_pnl if p <= 0]
    pf = sum(wins for p in all_pnl if p > 0) / abs(sum(losses)) if losses else float('inf')
    print(f"    Total trades:  {len(all_pnl)}")
    print(f"    Win rate:      {wins/len(all_pnl)*100:.1f}%")
    print(f"    Total return:  {ret:+.3f}% (${INITIAL_CAPITAL * (1+ret/100):.2f})")
    print(f"    Profit factor: {pf:.3f}")
