"""
parquet_data_information.py
============================
Examine parquet files - inspect time ranges, columns, data types, and sample data.

Usage:
    python parquet_data_information.py <parquet_file>
    python parquet_data_information.py data/backtests/XRPUSDT_20260329_merged.parquet
    python parquet_data_information.py data/backtests/XRPUSDT_market_data_20260328.parquet

Shows:
    - File size
    - Number of records
    - Column names and data types
    - Time range (min/max timestamp)
    - Sample rows
    - Data statistics
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | {message}",
    level="INFO"
)


def examine_parquet(parquet_file: Path) -> None:
    """Examine and display parquet file information."""
    
    if not parquet_file.exists():
        logger.error(f"✗ File not found: {parquet_file}")
        return
    
    try:
        # Load parquet
        logger.info(f"Loading {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)
        
        # File size
        file_size_mb = parquet_file.stat().st_size / 1024 / 1024
        
        print("\n" + "="*80)
        print(f"PARQUET FILE INFORMATION: {parquet_file.name}")
        print("="*80)
        
        # File info
        print(f"\nFile Info:")
        print(f"  Path: {parquet_file}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Records: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        # Time range (if timestamp column exists)
        if 'timestamp' in df.columns:
            ts_min = df['timestamp'].min()
            ts_max = df['timestamp'].max()
            duration = ts_max - ts_min
            
            print(f"\nTime Range:")
            print(f"  Min: {ts_min}")
            print(f"  Max: {ts_max}")
            print(f"  Duration: {duration}")
        
        # Column info
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            null_pct = (1 - non_null / len(df)) * 100
            print(f"  {col:<25} {dtype:<10} non-null: {non_null:,} ({100-null_pct:.1f}%)")
        
        # Data statistics
        print(f"\nData Statistics:")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"  Numeric columns: {', '.join(numeric_cols)}")
            for col in numeric_cols[:5]:  # Show first 5
                print(f"    {col}:")
                print(f"      Min: {df[col].min():.4f}")
                print(f"      Max: {df[col].max():.4f}")
                print(f"      Mean: {df[col].mean():.4f}")
        
        # Sample data
        print(f"\nFirst 5 Rows:")
        print(df.head().to_string())
        
        print("\n" + "="*80)
        logger.info(f"✓ Examined {len(df):,} records from {parquet_file.name}")
        
    except Exception as e:
        logger.error(f"✗ Error reading parquet file: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Examine parquet files - inspect time ranges, columns, and data"
    )
    
    parser.add_argument(
        'parquet_file',
        type=str,
        help='Path to parquet file to examine'
    )
    
    args = parser.parse_args()
    parquet_file = Path(args.parquet_file)
    
    examine_parquet(parquet_file)


if __name__ == '__main__':
    main()
