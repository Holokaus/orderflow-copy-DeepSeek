import sys, warnings, os
os.environ['LOGURU_LEVEL'] = 'CRITICAL'
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from core.feature_engine import FeatureEngine, FeatureConfig
from core.data_structures import OrderBook, PriceLevel, Trade, Side, Regime

df = pd.read_parquet('data/backtests/XRPUSDT_20260329_processed.parquet').head(10000)

fe = FeatureEngine(FeatureConfig())

passed_counts = {}
total_checked = 0

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
    
    c1 = f.get('footprint_imbalance_count',0) >= 2
    c2 = f.get('abs_delta_pct_60s',0) > 0.2
    c3 = f.get('abs_depth_imbalance_10',0) > 0.15
    c4 = f.get('pressure_confirmed',0) == 1.0
    c5 = -0.003 <= f.get('price_vs_vwap_pct',0) <= 0.003
    
    filters_pass = True
    for feat, op, thresh in [('spread_bps','>',8.0), ('bid_depth_10','<',1500.0), ('ask_depth_10','<',1500.0)]:
        if op == '>' and f.get(feat,0) > thresh: filters_pass = False
        if op == '<' and f.get(feat,0) < thresh: filters_pass = False
    pcp = f.get('price_change_pct_300s',0)
    if -0.001 <= pcp <= 0.001:
        filters_pass = False
    
    cnames = ['footprint>=2','abs_delta_pct>0.2','abs_imbal>0.15','pressure_confirmed','vwap_range','filters']
    cvals = [c1,c2,c3,c4,c5,filters_pass]
    
    for n,v in zip(cnames,cvals):
        if n not in passed_counts:
            passed_counts[n] = {'pass':0,'fail':0}
        if v:
            passed_counts[n]['pass'] += 1
        else:
            passed_counts[n]['fail'] += 1
    
    if total_checked >= 5000: break

print(f'Checked {total_checked} ticks')
for n in cnames:
    pc = passed_counts.get(n,{'pass':0,'fail':0})
    total = pc['pass']+pc['fail']
    print(f'{n:30s}: pass={pc["pass"]:5d} ({pc["pass"]/total*100:5.1f}%) fail={pc["fail"]:5d}')
