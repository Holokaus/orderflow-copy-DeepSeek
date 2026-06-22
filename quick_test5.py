import os; os.environ['LOGURU_LEVEL'] = 'CRITICAL'
from loguru import logger; logger.remove()
import pandas as pd
from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from core.feature_engine import FeatureEngine, FeatureConfig
from core.data_structures import Regime

engine = BacktestEngine(initial_capital=100.0, fee_pct=0.0005, slippage_pct=0.0003, sl_extra_slippage_pct=0.0003, warmup_seconds=60.0, min_time_between_trades_sec=30.0, risk_limits=RiskLimits(max_position_size=10000.0, max_position_value_pct=0.25, max_daily_loss_pct=0.02, max_drawdown_pct=0.10, max_trades_per_day=50, max_trades_per_hour=10, min_time_between_trades_sec=30, max_consecutive_losses=3))
df = pd.read_parquet('data/backtests/ICPUSDT_20260608_processed.parquet')
df = df.iloc[:5000]
rows = engine._preprocess(df)

fc = FeatureConfig()
fe = FeatureEngine(fc)
regimes = {}
for i, row in enumerate(rows):
    ob = engine._build_order_book_fast(row, row.timestamp)
    trades = engine._build_trades_fast(row, row.timestamp)
    state = fe.update(ob, trades, detect_patterns=(i%25==0), compute_volume_profile=(i%100==0))
    regimes[state.regime] = regimes.get(state.regime, 0) + 1
    if i % 500 == 0:
        print(f'Tick {i}: Regime={state.regime.name}, Mid={ob.mid_price:.4f}')

print('\nRegime distribution:')
for r, c in sorted(regimes.items(), key=lambda x: -x[1]):
    print(f'  {r.name}: {c}')