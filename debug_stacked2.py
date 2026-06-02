import sys, warnings, os
os.environ['LOGURU_LEVEL'] = 'CRITICAL'
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path.cwd()))
from core.feature_engine import FeatureEngine, FeatureConfig
from core.data_structures import OrderBook, PriceLevel, Trade, Side, Regime
from knowledge.strategy_library import get_strategy, StrategyCondition, StrategyDefinition

df = pd.read_parquet('data/backtests/XRPUSDT_20260329_processed.parquet').head(10000)

fe = FeatureEngine(FeatureConfig())
strat = get_strategy('stacked_imbalance')

regime_counts = Counter()
has_condition_counts = Counter({c.feature: 0 for c in strat.entry_conditions})
failed_required = 0
filtered_by_filter = 0
filtered_by_regime = 0
filtered_by_neutral = 0
filtered_by_score = 0
filtered_by_min_cond = 0
sig_count = 0
total_checked = 0

entry_feats = [c.feature for c in strat.entry_conditions]

for idx in range(len(df)):
    row = df.iloc[idx]
    ts = pd.to_datetime(row['timestamp'])
    trades=[]
    tp=float(row.get('trade_price',0)); tsz=float(row.get('trade_size',0)); tsd=str(row.get('trade_side','buy')).lower().strip()
    if tp>0 and tsz>0:
        trades.append(Trade(timestamp=ts, price=tp, size=tsz, side=Side.BUY if tsd=='buy' else Side.SELL))
    bids=[PriceLevel(price=float(row[f'bid_price_{i}']), size=float(row[f'bid_size_{i}']), timestamp=ts) for i in range(20) if row.get(f'bid_price_{i}',0)>0]
    asks=[PriceLevel(price=float(row[f'ask_price_{i}']), size=float(row[f'ask_size_{i}']), timestamp=ts) for i in range(20) if row.get(f'ask_price_{i}',0)>0]
    if not bids or not asks: continue
    bids.sort(key=lambda l: l.price,reverse=True)
    asks.sort(key=lambda l: l.price)
    ob = OrderBook(timestamp=ts, bids=bids, asks=asks)
    state = fe.update(ob, trades, detect_patterns=(idx%25==0))
    f = state.features
    
    if idx < 500: continue
    total_checked += 1
    regime_counts[state.regime.name] += 1
    
    sig = strat.evaluate(state)
    if sig:
        sig_count += 1

print(f'Checked {total_checked} ticks')
print(f'\nRegime distribution:')
for r, c in regime_counts.most_common():
    print(f'  {r:20s}: {c} ({c/total_checked*100:.1f}%)')
print(f'\nAllowed regimes for stacked_imbalance: {[r.name for r in strat.allowed_regimes]}')
print(f'  RANGING not in allowed: {"RANGING" not in [r.name for r in strat.allowed_regimes]}')
print(f'\nSignals generated: {sig_count}')
