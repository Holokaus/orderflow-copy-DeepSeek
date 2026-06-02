"""
prepare_data_to_parquet.py
==========================
General-purpose data preparation script for converting recorded JSONL.gz to Parquet.

Usage:
    python prepare_data_to_parquet.py <input_file.jsonl.gz> [output_file.parquet] [max_rows]

Examples:
    python prepare_data_to_parquet.py data/recorded/XRPUSDT_market_data_20260329.jsonl.gz
    python prepare_data_to_parquet.py data/recorded/XRPUSDT_market_data_20260329.jsonl.gz data/backtests/processed.parquet 5000
"""

import sys
import gzip
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | {message}",
    level="INFO"
)


class DataPreparator:
    """Convert recorded JSONL.gz market data to backtest-ready Parquet format."""
    
    def __init__(self, input_file: Path, output_file: Path, max_rows: int = None):
        """
        Initialize data preparator.
        
        Args:
            input_file: Path to JSONL.gz file
            output_file: Path to output Parquet file
            max_rows: Maximum rows to load (None = all)
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.max_rows = max_rows
        self.records: List[Dict] = []
        self.stats = {
            'total_lines': 0,
            'orderbook_records': 0,
            'trade_records': 0,
            'parse_errors': 0,
            'trades_with_depth': 0,
            'final_records': 0,
        }
    
    def validate_input(self) -> bool:
        """Validate input file exists and is readable."""
        if not self.input_file.exists():
            logger.error(f"✗ Input file not found: {self.input_file}")
            return False
        
        if not self.input_file.name.endswith('.jsonl.gz'):
            logger.warning(f"⚠ File doesn't end with .jsonl.gz: {self.input_file}")
        
        logger.info(f"✓ Input file exists: {self.input_file}")
        logger.info(f"  Size: {self.input_file.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    
    def ensure_output_dir(self) -> bool:
        """Create output directory if needed."""
        output_dir = self.output_file.parent
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Output directory ready: {output_dir}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to create output directory: {e}")
            return False
    
    def load_and_parse(self) -> bool:
        """Load and parse JSONL.gz file."""
        logger.info(f"\nLoading data from {self.input_file.name}...")
        
        current_bids = {}
        current_asks = {}
        
        try:
            with gzip.open(self.input_file, 'rt', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    # Limit rows if specified
                    if self.max_rows and line_no > self.max_rows:
                        break
                    
                    # Parse JSON
                    try:
                        record = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        self.stats['parse_errors'] += 1
                        continue
                    
                    self.stats['total_lines'] += 1
                    
                    timestamp = record.get('timestamp')
                    data_type = record.get('data_type')
                    
                    # Parse orderbook: update current book state
                    if data_type == 'orderbook':
                        try:
                            # Parse bids (JSON string → list of [price, size])
                            bids_json = record.get('bids_json')
                            if isinstance(bids_json, str):
                                bids_list = json.loads(bids_json)
                            else:
                                bids_list = bids_json or []
                            
                            # Parse asks (JSON string → list of [price, size])
                            asks_json = record.get('asks_json')
                            if isinstance(asks_json, str):
                                asks_list = json.loads(asks_json)
                            else:
                                asks_list = asks_json or []
                            
                            # Convert to dict format: {level: {price, size}}
                            current_bids = {}
                            for level, bid in enumerate(bids_list[:20]):
                                if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                                    current_bids[level] = {
                                        'price': float(bid[0]),
                                        'size': float(bid[1])
                                    }
                            
                            current_asks = {}
                            for level, ask in enumerate(asks_list[:20]):
                                if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                                    current_asks[level] = {
                                        'price': float(ask[0]),
                                        'size': float(ask[1])
                                    }
                            
                            self.stats['orderbook_records'] += 1
                            
                        except Exception as e:
                            logger.debug(f"Error parsing orderbook at line {line_no}: {e}")
                            self.stats['parse_errors'] += 1
                            continue
                    
                    # Parse trade: create output record with current book state
                    elif data_type == 'trade':
                        price = record.get('price')
                        size = record.get('size')
                        side = record.get('side')
                        
                        # Only create record if we have valid trade + depth data
                        if (price is not None and size is not None and 
                            len(current_bids) > 0 and len(current_asks) > 0):
                            
                            row_data = {
                                'timestamp': timestamp,
                                'trade_price': float(price),
                                'trade_size': float(size),
                                'trade_side': side.lower() if side else 'buy',
                            }
                            
                            # Add 10 levels of bid/ask depth
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
                            
                            self.records.append(row_data)
                            self.stats['trades_with_depth'] += 1
                            self.stats['trade_records'] += 1
                        else:
                            self.stats['trade_records'] += 1
        
        except Exception as e:
            logger.error(f"✗ Fatal error reading file: {e}")
            return False
        
        logger.info(f"✓ Parsed {self.stats['total_lines']} lines")
        logger.info(f"  Orderbooks: {self.stats['orderbook_records']}")
        logger.info(f"  Trades: {self.stats['trade_records']}")
        logger.info(f"  Trades with depth: {self.stats['trades_with_depth']}")
        logger.info(f"  Parse errors: {self.stats['parse_errors']}")
        
        return True
    
    def validate_data(self) -> bool:
        """Validate parsed data quality."""
        if len(self.records) == 0:
            logger.error("✗ No records loaded!")
            return False
        
        logger.info(f"\nValidating {len(self.records)} records...")
        
        # Check for required columns
        required_cols = ['timestamp', 'trade_price', 'trade_size', 'bid_size_0', 'ask_size_0']
        if not all(col in self.records[0] for col in required_cols):
            logger.error(f"✗ Missing required columns")
            return False
        
        # Check for non-zero values
        bid_size_values = sum(r['bid_size_0'] for r in self.records)
        ask_size_values = sum(r['ask_size_0'] for r in self.records)
        
        if bid_size_values == 0:
            logger.error("✗ All bid_size_0 values are 0!")
            return False
        
        if ask_size_values == 0:
            logger.error("✗ All ask_size_0 values are 0!")
            return False
        
        logger.info(f"✓ Data validation passed")
        logger.info(f"  bid_size_0 total: {bid_size_values:,.0f}")
        logger.info(f"  ask_size_0 total: {ask_size_values:,.0f}")
        
        return True
    
    def create_dataframe(self) -> bool:
        """Convert records to DataFrame."""
        logger.info(f"\nCreating DataFrame...")
        
        try:
            self.df = pd.DataFrame(self.records)
            self.stats['final_records'] = len(self.df)
            
            # Convert timestamp to datetime
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            logger.info(f"✓ DataFrame created")
            logger.info(f"  Shape: {self.df.shape}")
            logger.info(f"  Columns: {len(self.df.columns)}")
            logger.info(f"  Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Failed to create DataFrame: {e}")
            return False
    
    def save_parquet(self) -> bool:
        """Save DataFrame to Parquet."""
        logger.info(f"\nSaving to Parquet...")
        
        try:
            self.df.to_parquet(self.output_file, engine='pyarrow', compression='snappy')
            
            file_size = self.output_file.stat().st_size / 1024 / 1024
            logger.info(f"✓ Saved to {self.output_file}")
            logger.info(f"  Size: {file_size:.1f} MB")
            logger.info(f"  Records: {len(self.df)}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Failed to save Parquet: {e}")
            return False
    
    def print_summary(self):
        """Print final summary."""
        print("\n" + "="*80)
        print("DATA PREPARATION SUMMARY")
        print("="*80)
        
        logger.info(f"\nInput:  {self.input_file}")
        logger.info(f"Output: {self.output_file}")
        logger.info(f"\nStatistics:")
        for key, value in self.stats.items():
            logger.info(f"  {key}: {value}")
        
        if len(self.records) > 0:
            bid_depth_10 = self.df[[f'bid_size_{i}' for i in range(10)]].sum(axis=1)
            logger.info(f"\nbid_depth_10 (SUM of 10 levels):")
            logger.info(f"  Mean: {bid_depth_10.mean():,.0f}")
            logger.info(f"  Min: {bid_depth_10.min():,.0f}")
            logger.info(f"  Max: {bid_depth_10.max():,.0f}")
            logger.info(f"  Ticks > 1000: {(bid_depth_10 > 1000).sum()}/{len(bid_depth_10)}")
        
        print("="*80)
    
    def run(self) -> bool:
        """Execute full data preparation pipeline."""
        logger.info("="*80)
        logger.info("DATA PREPARATION PIPELINE")
        logger.info("="*80 + "\n")
        
        # Step 1: Validate input
        if not self.validate_input():
            return False
        
        # Step 2: Ensure output directory
        if not self.ensure_output_dir():
            return False
        
        # Step 3: Load and parse
        if not self.load_and_parse():
            return False
        
        # Step 4: Validate data
        if not self.validate_data():
            return False
        
        # Step 5: Create DataFrame
        if not self.create_dataframe():
            return False
        
        # Step 6: Save Parquet
        if not self.save_parquet():
            return False
        
        # Step 7: Summary
        self.print_summary()
        
        logger.info("\n✓✓✓ DATA PREPARATION COMPLETE ✓✓✓")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert recorded JSONL.gz market data to Parquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python prepare_data_to_parquet.py data/recorded/XRPUSDT_market_data_20260329.jsonl.gz
  python prepare_data_to_parquet.py data/recorded/XRPUSDT_market_data_20260329.jsonl.gz data/backtests/processed.parquet
  python prepare_data_to_parquet.py data/recorded/XRPUSDT_market_data_20260329.jsonl.gz data/backtests/processed.parquet 5000
        '''
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Input JSONL.gz file path'
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        help='Output Parquet file path (default: auto-generate from input name)'
    )
    
    parser.add_argument(
        'max_rows',
        type=int,
        nargs='?',
        help='Maximum rows to load (default: all)'
    )
    
    args = parser.parse_args()
    
    # Determine output file
    input_path = Path(args.input_file)
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Auto-generate output name
        stem = input_path.stem.replace('.jsonl', '')
        output_path = input_path.parent.parent / 'backtests' / f'{stem}_processed.parquet'
    
    # Create preparator and run
    preparator = DataPreparator(input_path, output_path, args.max_rows)
    success = preparator.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
