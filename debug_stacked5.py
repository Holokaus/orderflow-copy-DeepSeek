import sys
sys.path.insert(0, 'C:\\Users\\A\\Downloads\\New-folder-orderflow-copy\\orderflow-copy')
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['LOGURU_LEVEL'] = 'CRITICAL'
os.environ['PYTHONWARNINGS'] = 'ignore'

from pathlib import Path
import pandas as pd
import numpy as np

from core.feature_engine import FeatureEngine, FeatureConfig
from core.data_structures import OrderBook, PriceLevel, Trade, Side, SignalType, Regime
from knowledge.strategy_library import get_strategy

df = pd.read_parquet('data/backtests/XRPUSDT_20260329_processed.parquet').head(5000)
fe = FeatureEngine(FeatureConfig())
strat = get_strategy('stacked_imbalance')

reasons = {'low_liq': 0, 'filter': 0, 'regime': 0, 'required': 0, 'min_cond': 0, 'score': 0, 'neutral': 0, 'signal': 0}

for idx in range(len(df)):
    row = df.iloc[idx]
    ts = pd.to_datetime(row['timestamp'])
    trades = []
    tp = float(row.get('trade_price', 0)); tsz = float(row.get('trade_size', 0))
    tsd = str(row.get('trade_side', 'buy')).lower().strip()
    if tp > 0 and tsz > 0:
        trades.append(Trade(timestamp=ts, price=tp, size=tsz, side=Side.BUY if tsd == 'buy' else Side.SELL))
    bids = [PriceLevel(price=float(row[f'bid_price_{i}']), size=float(row[f'bid_size_{i}']), timestamp=ts) for i in range(20) if row.get(f'bid_price_{i}', 0) > 0]
    asks = [PriceLevel(price=float(row[f'ask_price_{i}']), size=float(row[f'ask_size_{i}']), timestamp=ts) for i in range(20) if row.get(f'ask_price_{i}', 0) > 0]
    if not bids or not asks:
        continue
    bids.sort(key=lambda l: l.price, reverse=True)
    asks.sort(key=lambda l: l.price)
    ob = OrderBook(timestamp=ts, bids=bids, asks=asks)
    state = fe.update(ob, trades, detect_patterns=(idx % 25 == 0))

    if idx < 500:
        continue

    f = state.features

    if state.regime == Regime.LOW_LIQUIDITY:
        reasons['low_liq'] += 1
        continue

    filter_hit = False
    for fcond in strat.filters:
        sat, _ = fcond.evaluate(f)
        if sat:
            filter_hit = True
            break
    if filter_hit:
        reasons['filter'] += 1
        continue

    if state.regime not in strat.allowed_regimes:
        reasons['regime'] += 1
        continue

    required_ok = True
    satisfied_count = 0
    total_score = 0.0
    for cond in strat.entry_conditions:
        sat, score = cond.evaluate(f)
        if sat:
            satisfied_count += 1
            total_score += score
        elif cond.required:
            required_ok = False
    if not required_ok:
        reasons['required'] += 1
        continue
    if satisfied_count < strat.min_conditions_satisfied:
        reasons['min_cond'] += 1
        continue
    if total_score < strat.min_score_threshold:
        reasons['score'] += 1
        continue

    direction = strat._determine_direction(state, total_score)
    if direction == SignalType.NEUTRAL:
        reasons['neutral'] += 1
        continue

    reasons['signal'] += 1

print(f'Checked ticks')
for k, v in reasons.items():
    print(f'  {k:10s}: {v}')
