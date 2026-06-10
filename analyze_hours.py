import sys
sys.path.insert(0, '.')
import pandas as pd
from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import create_absorption_strategy, create_stacked_imbalance_strategy
from loguru import logger
logger.remove()

DATA_PATH = 'data/backtests'
date_map = {
    'May 31': 'ICPUSDT_20260531_processed.parquet',
    'Jun 1':  'ICPUSDT_20260601_processed.parquet',
    'Jun 8':  'ICPUSDT_20260608_processed.parquet',
    'Jun 9':  'ICPUSDT_20260609_processed.parquet',
}

def make_engine():
    return BacktestEngine(
        initial_capital=100.0, fee_pct=0.0005, slippage_pct=0.0003,
        sl_extra_slippage_pct=0.0003, warmup_seconds=60.0,
        min_time_between_trades_sec=30.0,
        risk_limits=RiskLimits(
            max_position_size=10000.0, max_position_value_pct=0.25,
            max_daily_loss_pct=0.02, max_drawdown_pct=0.10,
            max_trades_per_day=50, max_trades_per_hour=10,
            min_time_between_trades_sec=30, max_consecutive_losses=3,
        ),
    )

all_rows = []
for label, filename in date_map.items():
    df = pd.read_parquet(f'{DATA_PATH}/{filename}')
    engine = make_engine()
    engine.run(df.copy(), create_absorption_strategy())
    for t in engine.closed_trades:
        all_rows.append({'date': label, 'strat': 'Abs', 'entry_hour': t.entry_time.hour, 'pnl_pct': t.pnl_pct * 100, 'exit_reason': t.exit_reason, 'entry_time': t.entry_time})
    engine2 = make_engine()
    engine2.run(df.copy(), create_stacked_imbalance_strategy())
    for t in engine2.closed_trades:
        all_rows.append({'date': label, 'strat': 'SI', 'entry_hour': t.entry_time.hour, 'pnl_pct': t.pnl_pct * 100, 'exit_reason': t.exit_reason, 'entry_time': t.entry_time})

df = pd.DataFrame(all_rows)
print(f'Total trades: {len(df)}')

print()
print('=== PnL by entry hour ===')
by_hour = df.groupby('entry_hour').agg(
    trades=('pnl_pct','count'),
    total_pnl=('pnl_pct','sum'),
    avg_pnl=('pnl_pct','mean'),
    win_rate=('pnl_pct', lambda x: (x > 0).mean() * 100),
).round(3)
print(by_hour.to_string())

print()
print('=== Total PnL by hour x date ===')
pivot = df.pivot_table(index='entry_hour', columns='date', values='pnl_pct', aggfunc='sum', fill_value=0)
print(pivot.to_string())

print()
print('=== Big losers (>0.5%) ===')
big = df[df['pnl_pct'] < -0.5].sort_values('entry_time')
for _, r in big.iterrows():
    print(f'  {r["date"]} {r["strat"]} at hour {r["entry_hour"]}: {r["pnl_pct"]:.2f}%')
