import sys, warnings, os
os.environ['LOGURU_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from backtesting.engine import BacktestEngine
from knowledge.strategy_library import get_strategy
from execution.risk_manager import RiskLimits

df = pd.read_parquet('data/backtests/XRPUSDT_20260329_processed.parquet').head(20000)

# Run with high debug - check each stacked imbalance signal
strat = get_strategy('stacked_imbalance')
eng = BacktestEngine(10000.0, 0.001, 0.0001,
    risk_limits=RiskLimits(max_position_size=100.0, max_position_value_pct=0.50, max_trades_per_hour=100),
    warmup_seconds=10.0)

# Monkey-patch evaluate to count reasons
eval_counts = {'total_eval': 0, 'sent_signals': 0, 'filter_reject': 0, 'regime_reject': 0,
               'required_fail': 0, 'min_cond_fail': 0, 'score_fail': 0, 'neutral_fail': 0,
               'no_depth': 0}

orig_evaluate = strat.evaluate
def debug_evaluate(state):
    eval_counts['total_eval'] += 1
    f = state.features
    if 'footprint_imbalance_count' not in f:
        eval_counts['no_depth'] += 1
        return None
    sig = orig_evaluate(state)
    if sig:
        eval_counts['sent_signals'] += 1
    return sig

strat.evaluate = debug_evaluate
m = eng.run(df, strat)

print(f'Data size: {len(df)}')
print(f'Evaluation counts:')
for k, v in eval_counts.items():
    print(f'  {k}: {v}')
print(f'Trades: {m.total_trades}')
print(f'Fee rejections: {m.signals_rejected_by_fee_filter}')
print(f'Risk rejections: {m.signals_rejected_by_risk}')
