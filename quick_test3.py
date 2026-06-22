import os; os.environ['LOGURU_LEVEL'] = 'CRITICAL'
from loguru import logger; logger.remove()
import pandas as pd
import numpy as np
from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import create_trend_following_strategy
import time

engine = BacktestEngine(initial_capital=100.0, fee_pct=0.0005, slippage_pct=0.0003, sl_extra_slippage_pct=0.0003, warmup_seconds=60.0, min_time_between_trades_sec=30.0, risk_limits=RiskLimits(max_position_size=10000.0, max_position_value_pct=0.25, max_daily_loss_pct=0.02, max_drawdown_pct=0.10, max_trades_per_day=50, max_trades_per_hour=10, min_time_between_trades_sec=30, max_consecutive_losses=3))
df = pd.read_parquet('data/backtests/ICPUSDT_20260608_processed.parquet')
strat = create_trend_following_strategy()

start = time.time()
engine.run(df, strat)
elapsed = time.time() - start

pnls = [t.pnl_pct for t in engine.closed_trades]
wins = sum(1 for p in pnls if p > 0)
ret = (np.prod([1 + p for p in pnls]) - 1) * 100 if pnls else 0
sides = [t.side.name for t in engine.closed_trades]

print(f'Time: {elapsed:.1f}s')
print(f'n={len(pnls)} WR={wins/len(pnls)*100 if pnls else 0:.1f}% ret={ret:+.3f}%')
for t in engine.closed_trades:
    dur = (t.exit_time - t.entry_time).total_seconds() / 60
    print(f'  {t.entry_time}->{t.exit_time} ({dur:.0f}m) PnL={t.pnl_pct*100:+.3f}% {t.exit_reason} side={t.side.name}')