"""
Data Loader for XRPUSDT_market_data_20260329.jsonl.gz
Transforms recorded data into backtest-compatible format.
"""

import gzip
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger


def load_recorded_data(filepath: Path, max_rows: int = 2000) -> pd.DataFrame:
    """
    Load JSONL.gz market data and transform to backtest format.
    
    Input format:
        timestamp, exchange_timestamp_ms, data_type, bids_json, asks_json, 
        latency_ms, last_update_id, price, size, side, _seq
    
    bids_json/asks_json format: JSON string with [[price, size], [price, size], ...]
    
    Output format:
        timestamp, bid_price_0...9, bid_size_0...9, ask_price_0...9, ask_size_0...9,
        trade_price, trade_size, trade_side
    """
    
    logger.info(f"Loading {filepath.name}...")
    
    records = []
    current_time = None
    current_bids = {}
    current_asks = {}
    
    with gzip.open(filepath, 'rt') as f:
        for line_no, line in enumerate(f, 1):
            if line_no > max_rows:
                break
            
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            timestamp = record.get('timestamp')
            data_type = record.get('data_type')
            
            if current_time is None:
                current_time = timestamp
            
            # Parse orderbook data
            if data_type == 'orderbook':
                try:
                    bids_json = record.get('bids_json')
                    asks_json = record.get('asks_json')
                    
                    # Parse bids: JSON string with list of [price, size] pairs
                    if isinstance(bids_json, str):
                        bids_list = json.loads(bids_json)
                    else:
                        bids_list = bids_json or []
                    
                    # Parse asks: JSON string with list of [price, size] pairs
                    if isinstance(asks_json, str):
                        asks_list = json.loads(asks_json)
                    else:
                        asks_list = asks_json or []
                    
                    # Convert list format to dict: {level: {price, size}}
                    current_bids = {}
                    for level, bid in enumerate(bids_list[:20]):  # Take first 20 levels
                        if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            current_bids[level] = {
                                'price': float(bid[0]),
                                'size': float(bid[1])
                            }
                    
                    current_asks = {}
                    for level, ask in enumerate(asks_list[:20]):  # Take first 20 levels
                        if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            current_asks[level] = {
                                'price': float(ask[0]),
                                'size': float(ask[1])
                            }
                    
                    logger.debug(f"Parsed orderbook with {len(current_bids)} bid levels and {len(current_asks)} ask levels")
                    
                except Exception as e:
                    logger.debug(f"Error parsing orderbook at line {line_no}: {e}")
                    continue
            
            # Parse trade data
            elif data_type == 'trade':
                price = record.get('price')
                size = record.get('size')
                side = record.get('side')
                
                if price is not None and size is not None and len(current_bids) > 0 and len(current_asks) > 0:
                    # Build record
                    row_data = {
                        'timestamp': timestamp,
                        'trade_price': float(price),
                        'trade_size': float(size),
                        'trade_side': side.lower() if side else 'buy',
                    }
                    
                    # Add current depth levels (0-9)
                    for level in range(10):
                        if level in current_bids:
                            row_data[f'bid_price_{level}'] = current_bids[level]['price']
                            row_data[f'bid_size_{level}'] = current_bids[level]['size']
                        else:
                            row_data[f'bid_price_{level}'] = 0.0
                            row_data[f'bid_size_{level}'] = 0.0
                        
                        if level in current_asks:
                            row_data[f'ask_price_{level}'] = current_asks[level]['price']
                            row_data[f'ask_size_{level}'] = current_asks[level]['size']
                        else:
                            row_data[f'ask_price_{level}'] = 0.0
                            row_data[f'ask_size_{level}'] = 0.0
                    
                    records.append(row_data)
    
    if not records:
        logger.error("No records loaded!")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


if __name__ == "__main__":
    # Test loading
    filepath = Path("data/recorded/XRPUSDT_market_data_20260329.jsonl.gz")
    df = load_recorded_data(filepath, max_rows=2000)
    
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Check depth data
    bid_size_cols = [c for c in df.columns if c.startswith('bid_size_')]
    logger.info(f"Bid depth columns: {len(bid_size_cols)}")
    
    # Save for backtest
    output_file = Path("data/backtests/XRPUSDT_20260329_processed.parquet")
    df.to_parquet(output_file)
    logger.info(f"Saved to {output_file}")
