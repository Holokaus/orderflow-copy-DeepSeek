"""
Batch-convert all recorded ICPUSDT_market_data_*.jsonl.gz files to backtest-ready parquet.

Idempotent: skips any output parquet that already exists. Pass --force to re-process all.

Usage:
    python scripts/process_all_recorded.py
    python scripts/process_all_recorded.py --force
    python scripts/process_all_recorded.py --symbol ICPUSDT --skip-existing
"""
import sys
import os
import re
import argparse
from pathlib import Path

# Suppress logging BEFORE any project imports (loguru adds ~10x overhead otherwise).
os.environ["LOGURU_LEVEL"] = "CRITICAL"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
logger.remove()
logger.add(sys.stderr, format="{message}", level="WARNING")

import pandas as pd

# Import the existing preparator class (single source of truth for the conversion).
from prepare_data_to_parquet import DataPreparator


RECORDED_DIR = PROJECT_ROOT / "data" / "recorded"
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtests"


def discover_files(symbol: str):
    """
    Find recorded market_data files and map them to parquet output paths.

    A raw file like  ICPUSDT_market_data_20260608.jsonl.gz
    maps to output   ICPUSDT_20260608_processed.parquet
    (matching the existing convention used by the backtest engine).
    """
    pattern = re.compile(rf"^{symbol}_market_data_(\d{{8}})\.jsonl\.gz$")
    pairs = []
    for raw in sorted(RECORDED_DIR.glob(f"{symbol}_market_data_*.jsonl.gz")):
        m = pattern.match(raw.name)
        if not m:
            continue
        date_tag = m.group(1)
        # Canonical short name: ICPUSDT_YYYYMMDD_processed.parquet
        out = BACKTEST_DIR / f"{symbol}_{date_tag}_processed.parquet"
        pairs.append((raw, out, date_tag))
    return pairs


def process_one(raw_path: Path, out_path: Path):
    """Run DataPreparator on one file. Returns (rows, duration_days, price_lo, price_hi) or None on failure."""
    prep = DataPreparator(raw_path, out_path, max_rows=None)
    ok = prep.run()
    if not ok:
        return None
    # Read back stats for the summary table.
    df = pd.read_parquet(out_path)
    ts = pd.to_datetime(df["timestamp"])
    span = (ts.max() - ts.min()).total_seconds() / 86400.0
    return (
        len(df),
        span,
        float(df["trade_price"].min()),
        float(df["trade_price"].max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="ICPUSDT", help="Symbol prefix (default: ICPUSDT)")
    parser.add_argument("--force", action="store_true", help="Re-process even if output parquet exists")
    args = parser.parse_args()

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    pairs = discover_files(args.symbol)
    if not pairs:
        print(f"No raw files found matching {args.symbol}_market_data_*.jsonl.gz in {RECORDED_DIR}")
        return 1

    print(f"Discovered {len(pairs)} raw {args.symbol} files.\n")
    header = f"{'date':<10} {'status':<12} {'rows':>8}  {'days':>6}  {'price_low':>9}  {'price_high':>9}  output"
    print(header)
    print("-" * len(header))

    n_done = n_skipped = n_failed = 0
    for raw_path, out_path, date_tag in pairs:
        if out_path.exists() and not args.force:
            # Read stats for already-processed files so the table is complete.
            try:
                df = pd.read_parquet(out_path)
                ts = pd.to_datetime(df["timestamp"])
                span = (ts.max() - ts.min()).total_seconds() / 86400.0
                print(f"{date_tag:<10} {'exists':<12} {len(df):>8}  {span:>6.2f}  "
                      f"{df['trade_price'].min():>9.3f}  {df['trade_price'].max():>9.3f}  {out_path.name}")
                n_skipped += 1
                continue
            except Exception:
                # Corrupt/unreadable → fall through and reprocess.
                pass

        res = process_one(raw_path, out_path)
        if res is None:
            print(f"{date_tag:<10} {'FAILED':<12}  {raw_path.name}")
            n_failed += 1
            continue
        rows, span, lo, hi = res
        print(f"{date_tag:<10} {'processed':<12} {rows:>8}  {span:>6.2f}  {lo:>9.3f}  {hi:>9.3f}  {out_path.name}")
        n_done += 1

    print("-" * len(header))
    print(f"Done: {n_done} processed, {n_skipped} already existed, {n_failed} failed.")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
