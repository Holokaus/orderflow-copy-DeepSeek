import pandas as pd
df = pd.read_parquet('data/backtests/ICPUSDT_20260531_processed.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Time: {df["timestamp"].min()} to {df["timestamp"].max()}')
print(f'Trade price range: {df["trade_price"].min()} - {df["trade_price"].max()}')
bid_cols = [c for c in df.columns if c.startswith('bid_size_')]
print(f'Depth levels: {len(bid_cols)}')
print(f'Sample row:')
print(df.head(1).to_dict())
