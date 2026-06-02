import pandas as pd
df1 = pd.read_parquet('data/backtests/ICPUSDT_20260531_processed.parquet')
df2 = pd.read_parquet('data/backtests/ICPUSDT_20260601_processed.parquet')
df = pd.concat([df1, df2]).reset_index(drop=True)
print(f'Combined: {len(df)} rows')
print(f'Time: {df["timestamp"].min()} to {df["timestamp"].max()}')
print(f'Duration: {pd.to_datetime(df["timestamp"].max()) - pd.to_datetime(df["timestamp"].min())}')
# Check if bid_price column exists (without suffix)
has_bid_price = 'bid_price' in df.columns
has_bid_price_0 = 'bid_price_0' in df.columns
print(f'Has bid_price: {has_bid_price}')
print(f'Has bid_price_0: {has_bid_price_0}')
