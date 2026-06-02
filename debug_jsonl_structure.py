"""
Debug: Inspect raw JSONL structure to understand bid/ask format
"""

import gzip
import json
from pathlib import Path

filepath = Path("data/recorded/XRPUSDT_market_data_20260329.jsonl.gz")

print("="*80)
print("Inspecting raw JSONL structure")
print("="*80)

count_by_type = {}
sample_orderbook = None
sample_trade = None

with gzip.open(filepath, 'rt') as f:
    for line_no, line in enumerate(f, 1):
        if line_no > 1000:
            break
        
        try:
            record = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        
        data_type = record.get('data_type')
        count_by_type[data_type] = count_by_type.get(data_type, 0) + 1
        
        # Capture samples
        if data_type == 'orderbook' and sample_orderbook is None:
            sample_orderbook = record
        elif data_type == 'trade' and sample_trade is None:
            sample_trade = record

print(f"\nRecord types found:")
for dtype, count in count_by_type.items():
    print(f"  {dtype}: {count} records")

if sample_orderbook:
    print(f"\n--- ORDERBOOK SAMPLE ---")
    print(f"Keys: {list(sample_orderbook.keys())}")
    print(f"bids_json type: {type(sample_orderbook.get('bids_json'))}")
    
    bids_json = sample_orderbook.get('bids_json')
    if isinstance(bids_json, str):
        bids = json.loads(bids_json)
    else:
        bids = bids_json
    
    print(f"bids type: {type(bids)}")
    if isinstance(bids, dict):
        print(f"bids keys: {list(bids.keys())[:5]}")
        # Show first bid
        first_key = list(bids.keys())[0] if bids else None
        if first_key:
            print(f"bids['{first_key}']: {bids[first_key]}")
    elif isinstance(bids, list):
        print(f"bids is list, length: {len(bids)}")
        if bids:
            print(f"bids[0]: {bids[0]}")

if sample_trade:
    print(f"\n--- TRADE SAMPLE ---")
    print(f"Keys: {list(sample_trade.keys())}")
    print(f"price: {sample_trade.get('price')}")
    print(f"size: {sample_trade.get('size')}")
    print(f"side: {sample_trade.get('side')}")
