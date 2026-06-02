"""
Inspect the processed data to see if bid_size columns have values
"""

import pandas as pd
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(lambda msg: print(msg, end=''), colorize=False)

parquet_file = Path("data/backtests/XRPUSDT_20260329_processed.parquet")
df = pd.read_parquet(parquet_file)

print("\n" + "="*80)
print("DATA INSPECTION")
print("="*80)

print(f"\nShape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Check bid_size columns
bid_size_cols = [c for c in df.columns if c.startswith('bid_size_')]
print(f"\nbid_size columns: {bid_size_cols}")

# Check values
print(f"\nSample bid_size values (first 5 rows):")
for col in bid_size_cols[:3]:
    values = df[col].values[:5]
    print(f"  {col}: {values}")

# Check if all zeros
total_bid_size = df[[c for c in df.columns if c.startswith('bid_size_')]].sum().sum()
print(f"\nTotal sum of all bid_size columns: {total_bid_size}")

if total_bid_size == 0:
    print("⚠ WARNING: All bid_size values are 0!")
    print("The data loading from JSONL may have failed to parse depth correctly.")
else:
    print(f"✓ bid_size data present: {total_bid_size}")

# Check ask_size
total_ask_size = df[[c for c in df.columns if c.startswith('ask_size_')]].sum().sum()
print(f"\nTotal sum of all ask_size columns: {total_ask_size}")

# Show first row
print(f"\nFirst row data:")
print(df.iloc[0][['timestamp', 'trade_price', 'trade_size', 'trade_side']].to_string())
print("\nFirst row depth:")
for col in bid_size_cols[:3]:
    print(f"  {col}: {df.iloc[0][col]}")
