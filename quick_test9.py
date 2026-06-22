import os; os.environ['LOGURU_LEVEL'] = 'CRITICAL'
from loguru import logger; logger.remove()
import pandas as pd
import numpy as np
from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import (
    create_absorption_strategy, 
    create_delta_divergence_strategy, 
    create_trend_following_strategy,
    create_stacked_imbalance_strategy,
    create_value_area_strategy
)
import time

def run_strat(df, strat_fn, name):
    engine = BacktestEngine(initial_capital=100.0, fee_pct=0.0005, slippage_pct=0.0003, sl_extra_slippage_pct=0.0003, warmup_seconds=60.0, min_time_between_trades_sec=30.0, risk_limits=RiskLimits(max_position_size=10000.0, max_position_value_pct=0.25, max_daily_loss_pct=0.02, max_drawdown_pct=0.10, max_trades_per_day=50, max_trades_per_hour=10, min_time_between_trades_sec=30, max_consecutive_losses=3))
    strat = strat_fn()
    start = time.time()
    engine.run(df, strat)
    elapsed = time.time() - start
    pnls = [t.pnl_pct for t in engine.closed_trades]
    wins = sum(1 for p in pnls if p > 0)
    ret = (np.prod([1 + p for p in pnls]) - 1) * 100 if pnls else 0
    sides = [t.side.name for t in engine.closed_trades]
    print(f'  {name:20s} | n={len(pnls):3d}  WR={wins/len(pnls)*100 if pnls else 0:5.1f}%  ret={ret:>+7.3f}%  time={elapsed:.1f}s  sides={sides}')
    return pnls

# Test May 31 (Uptrend day)
print(f'=== May 31 (Uptrend) ===')
df = pd.read_parquet('data/backtests/ICPUSDT_20260531_processed.parquet')
all_pnls = []
for name, strat_fn in [
    ('absorption', create_absorption_strategy), 
    ('delta_div', create_delta_divergence_strategy), 
    ('trend_follow', create_trend_following_strategy),
    ('stacked_imb', create_stacked_imbalance_strategy),
    ('value_area', create_value_area_strategy)
]:
    pnls = run_strat(df, strat_fn, name)
    all_pnls.extend(pnls)

ret = (np.prod([1 + p for p in all_pnls]) - 1) * 100 if all_pnls else 0
wins = sum(1 for p in all_pnls if p > 0)
print(f'\n=== COMBINED (May 31, 5 strats) ===')
print(f'  Total trades: {len(all_pnls)}')
print(f'  Win rate: {wins/len(all_pnls)*100:.1f}%')
print(f'  Total return: {ret:+.3f}%')