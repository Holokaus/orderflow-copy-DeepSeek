"""
Inspect absorption_strength values to see why they don't meet 0.30 threshold
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

from backtesting.engine import BacktestEngine, _Row
from knowledge.strategy_library import get_strategy
from execution.risk_manager import RiskLimits
from core.feature_engine import FeatureEngine, FeatureConfig
from core.feature_precomputer import FeaturePrecomputer


def inspect_absorption_strength():
    """Inspect absorption_strength values at each tick."""
    
    print("\n" + "="*80)
    print("INSPECTION: absorption_strength values")
    print("="*80)
    
    # Load data
    parquet_file = project_root / "data" / "backtests" / "XRPUSDT_20260329_processed.parquet"
    df = pd.read_parquet(parquet_file)
    
    logger.info(f"Loaded {len(df)} records")
    
    # Precompute features
    precomputer = FeaturePrecomputer(windows=[15, 30, 60, 300, 600, 900])
    precomputer.precompute_all(df)
    
    # Get feature engine
    feature_config = FeatureConfig()
    feature_engine = FeatureEngine(config=feature_config)
    
    # Preprocess data
    timestamps = pd.to_datetime(df['timestamp']).dt.to_pydatetime()
    
    bid_price = df['bid_price_0'].values if 'bid_price_0' in df.columns else np.zeros(len(df))
    ask_price = df['ask_price_0'].values if 'ask_price_0' in df.columns else np.zeros(len(df))
    bid_size = df['bid_size_0'].values if 'bid_size_0' in df.columns else np.zeros(len(df))
    ask_size = df['ask_size_0'].values if 'ask_size_0' in df.columns else np.zeros(len(df))
    
    trade_price = df['trade_price'].values
    trade_size = df['trade_size'].values
    trade_side = df['trade_side'].values
    
    has_depth = 'bid_price_0' in df.columns
    depth_bids_p = depth_bids_s = depth_asks_p = depth_asks_s = None
    
    if has_depth:
        n_depth = 0
        while f'bid_price_{n_depth}' in df.columns:
            n_depth += 1
        
        zeros = np.zeros(len(df))
        depth_bids_p = np.column_stack([
            df.get(f'bid_price_{i}', zeros) for i in range(n_depth)
        ])
        depth_bids_s = np.column_stack([
            df.get(f'bid_size_{i}', zeros) for i in range(n_depth)
        ])
        depth_asks_p = np.column_stack([
            df.get(f'ask_price_{i}', zeros) for i in range(n_depth)
        ])
        depth_asks_s = np.column_stack([
            df.get(f'ask_size_{i}', zeros) for i in range(n_depth)
        ])
    
    # Build rows
    from backtesting.engine import _Row
    rows = []
    for i in range(len(df)):
        rows.append(_Row(
            timestamp=timestamps[i],
            bid_price=float(bid_price[i]),
            ask_price=float(ask_price[i]),
            bid_size=float(bid_size[i]),
            ask_size=float(ask_size[i]),
            trade_price=float(trade_price[i]),
            trade_size=float(trade_size[i]),
            trade_side=str(trade_side[i]),
            depth_bids_p=depth_bids_p[i] if has_depth else None,
            depth_bids_s=depth_bids_s[i] if has_depth else None,
            depth_asks_p=depth_asks_p[i] if has_depth else None,
            depth_asks_s=depth_asks_s[i] if has_depth else None,
        ))
    
    # Process each tick and collect absorption_strength
    absorption_strengths = []
    bid_depths = []
    
    logger.info("Processing ticks to collect absorption_strength values...")
    
    for tick_idx in range(len(rows)):
        row = rows[tick_idx]
        
        # Get precomputed features
        precomputed_features = precomputer.get_features_for_tick(tick_idx)
        
        # Build order book
        order_book = feature_engine._create_order_book_from_row(row, tick_idx)
        
        # Build trades
        trades = []
        if not np.isnan(row.trade_price) and row.trade_size > 0:
            from core.data_structures import Trade, Side
            side = Side.BUY if row.trade_side.lower() == 'buy' else Side.SELL
            trades.append(Trade(
                timestamp=row.timestamp,
                price=row.trade_price,
                size=row.trade_size,
                side=side,
            ))
        
        # Update feature engine
        state = feature_engine.update(
            order_book, trades, run_patterns=False, run_vp=False,
            precomputed_features=precomputed_features
        )
        
        # Collect values
        if state:
            absorption_strength = state.features.get('recent_absorption_strength', 0)
            absorption_strengths.append(absorption_strength)
            bid_depths.append(precomputed_features.get('bid_depth_10', 0))
    
    # Analysis
    absorption_strengths = np.array(absorption_strengths)
    bid_depths = np.array(bid_depths)
    
    logger.info(f"\nabsorption_strength statistics:")
    logger.info(f"  Min: {absorption_strengths.min():.4f}")
    logger.info(f"  Max: {absorption_strengths.max():.4f}")
    logger.info(f"  Mean: {absorption_strengths.mean():.4f}")
    logger.info(f"  Median: {np.median(absorption_strengths):.4f}")
    logger.info(f"  Std: {absorption_strengths.std():.4f}")
    
    logger.info(f"\nbd_depth_10 statistics:")
    logger.info(f"  Min: {bid_depths.min():.0f}")
    logger.info(f"  Max: {bid_depths.max():.0f}")
    logger.info(f"  Mean: {bid_depths.mean():.0f}")
    
    # Count ticks meeting threshold
    threshold = 0.30
    meeting_threshold = (absorption_strengths >= threshold).sum()
    pct_meeting = meeting_threshold / len(absorption_strengths) * 100 if len(absorption_strengths) > 0 else 0
    
    logger.info(f"\nThreshold analysis (threshold={threshold}):")
    logger.info(f"  Ticks meeting threshold: {meeting_threshold}/{len(absorption_strengths)} ({pct_meeting:.1f}%)")
    
    # Distribution
    logger.info(f"\nDistribution of absorption_strength values:")
    for threshold_level in [0.10, 0.20, 0.30, 0.40, 0.50]:
        count = (absorption_strengths >= threshold_level).sum()
        pct = count / len(absorption_strengths) * 100 if len(absorption_strengths) > 0 else 0
        logger.info(f"  >= {threshold_level}: {count} ticks ({pct:.1f}%)")
    
    print("\n" + "="*80)
    if pct_meeting > 0:
        logger.info(f"✓ {meeting_threshold} ticks would generate entry signals")
        logger.info(f"✓ Depth fix is working, absorption_strength IS being calculated")
        logger.info(f"  Max absorption_strength observed: {absorption_strengths.max():.4f}")
    else:
        logger.warning(f"✗ 0 ticks meet threshold")
        logger.warning(f"  Max absorption_strength observed: {absorption_strengths.max():.4f}")
        logger.warning(f"  This dataset may not have strong enough order absorption patterns")
    
    return 0


if __name__ == "__main__":
    sys.exit(inspect_absorption_strength())
