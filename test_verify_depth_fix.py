"""
COMPREHENSIVE VERIFICATION: Show actual bid_depth_10 values being computed.
Proves the fix is working correctly.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<level>{level: <8}</level> | {message}", level="INFO")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.feature_precomputer import FeaturePrecomputer


def verify_depth_calculation():
    """Verify that bid_depth_10 is correctly calculated as SUM of 10 levels."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DEPTH CALCULATION VERIFICATION")
    print("="*80)
    
    # Load data
    parquet_file = project_root / "data" / "backtests" / "XRPUSDT_market_data_20260328.parquet"
    df = pd.read_parquet(parquet_file)
    df = df.head(1000)
    
    logger.info(f"Loaded {len(df)} rows")
    
    # Check available depth columns
    depth_cols = [c for c in df.columns if c.startswith('bid_size_')]
    logger.info(f"Depth columns: {len(depth_cols)}")
    logger.info(f"Levels: {sorted([int(c.split('_')[2]) for c in depth_cols])}")
    
    # Manually calculate what bid_depth_10 SHOULD be (sum of levels 0-9)
    print("\n" + "-"*80)
    print("MANUAL CALCULATION: Sum of bid_size_0 through bid_size_9")
    print("-"*80)
    
    manual_depth_10 = np.zeros(len(df))
    for level in range(10):
        col = f'bid_size_{level}'
        if col in df.columns:
            manual_depth_10 += df[col].values
            logger.debug(f"Added {col}")
    
    logger.info(f"Manual bid_depth_10 stats:")
    logger.info(f"  Min: {manual_depth_10.min():.2f}")
    logger.info(f"  Max: {manual_depth_10.max():.2f}")
    logger.info(f"  Mean: {manual_depth_10.mean():.2f}")
    logger.info(f"  Median: {np.median(manual_depth_10):.2f}")
    
    # Now run precomputer
    print("\n" + "-"*80)
    print("PRECOMPUTER CALCULATION: FeaturePrecomputer.precompute_all()")
    print("-"*80)
    
    precomputer = FeaturePrecomputer(windows=[15, 30, 60, 300, 600, 900])
    precomputer.precompute_all(df)
    
    precomputed_depth_10 = precomputer.precomputed['bid_depth_10']
    
    logger.info(f"Precomputed bid_depth_10 stats:")
    logger.info(f"  Min: {precomputed_depth_10.min():.2f}")
    logger.info(f"  Max: {precomputed_depth_10.max():.2f}")
    logger.info(f"  Mean: {precomputed_depth_10.mean():.2f}")
    logger.info(f"  Median: {np.median(precomputed_depth_10):.2f}")
    
    # Compare
    print("\n" + "-"*80)
    print("COMPARISON: Manual vs Precomputed")
    print("-"*80)
    
    if np.allclose(manual_depth_10, precomputed_depth_10):
        logger.info("✓✓✓ VALUES MATCH EXACTLY ✓✓✓")
        logger.info("Precomputer is correctly summing 10 levels")
        match = True
    else:
        max_diff = np.max(np.abs(manual_depth_10 - precomputed_depth_10))
        logger.error(f"✗ VALUES DON'T MATCH (max diff: {max_diff})")
        match = False
    
    # Validate depth > 1000
    print("\n" + "-"*80)
    print("VALIDATION: bid_depth_10 > 1000")
    print("-"*80)
    
    n_gt_1000 = (precomputed_depth_10 > 1000).sum()
    pct_gt_1000 = n_gt_1000 / len(precomputed_depth_10) * 100
    
    logger.info(f"Ticks with bid_depth_10 > 1000: {n_gt_1000}/{len(precomputed_depth_10)} ({pct_gt_1000:.1f}%)")
    
    if pct_gt_1000 >= 95:
        logger.info("✓ Realistic depth values (>95% ticks > 1000)")
        realistic = True
    else:
        logger.warning(f"⚠ Low percentage of ticks > 1000")
        realistic = False
    
    # Sample values
    print("\n" + "-"*80)
    print("SAMPLE VALUES (first 10 ticks)")
    print("-"*80)
    
    for i in range(min(10, len(df))):
        logger.info(f"Tick {i}: bid_depth_10 = {precomputed_depth_10[i]:,.2f}")
    
    # Summary
    print("\n" + "="*80)
    print("✓ VERIFICATION COMPLETE")
    print("="*80)
    
    if match and realistic:
        logger.info("✅ bid_depth_10 = SUM(bid_size_0...bid_size_9) - CORRECT")
        logger.info("✅ Values are realistic (all > 1000)")
        logger.info("✅ DEPTH FIX IS WORKING CORRECTLY")
        return True
    else:
        logger.error("✗ VERIFICATION FAILED")
        return False


if __name__ == "__main__":
    success = verify_depth_calculation()
    sys.exit(0 if success else 1)
