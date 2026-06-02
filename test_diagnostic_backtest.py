"""
DIAGNOSTIC BACKTEST: Check regime, signals, fees, and absorption strength
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, List

logger.remove()
logger.add(sys.stderr, format="<level>{level: <8}</level> | {message}", level="DEBUG")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtesting.engine import BacktestEngine
from knowledge.strategy_library import get_strategy
from execution.risk_manager import RiskLimits
from core.feature_engine import FeatureConfig


# Track diagnostics
diagnostics = {
    'regimes': [],
    'signals_generated': 0,
    'signals_by_fee_status': {'passed': 0, 'rejected': 0},
    'absorption_strengths': [],
    'trades': 0,
}


def run_diagnostic_backtest():
    """Run backtest with heavy logging for diagnosis."""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC BACKTEST: Regime + Signals + Fees + Absorption Strength")
    print("="*80)
    
    # Load processed data
    parquet_file = project_root / "data" / "backtests" / "XRPUSDT_20260329_processed.parquet"
    if not parquet_file.exists():
        logger.error(f"Processed data not found: {parquet_file}")
        logger.info("Run: python load_recorded_data.py first")
        return 1
    
    logger.info(f"Loading processed data: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)[:10]}...")
    
    # Check depth
    bid_size_cols = [c for c in df.columns if c.startswith('bid_size_')]
    logger.info(f"Bid depth columns: {len(bid_size_cols)}")
    
    # Get absorption strategy
    strategy = get_strategy("absorption")
    if strategy is None:
        logger.error("Could not load absorption strategy")
        return 1
    
    logger.info(f"Strategy: {strategy.name}")
    logger.info(f"Entry conditions: {len(strategy.entry_conditions)}")
    
    # Create engine with detailed logging
    engine = BacktestEngine(
        initial_capital=10000.0,
        fee_pct=0.001,
        slippage_pct=0.0001,
        risk_limits=RiskLimits(
            max_position_size=100.0,
            max_position_value_pct=0.50,
            max_trades_per_hour=100,
        ),
    )
    
    logger.info("Running backtest...")
    try:
        metrics = engine.run(df, strategy)
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1
    
    # Results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    
    logger.info(f"Total trades: {metrics.total_trades}")
    logger.info(f"Winning trades: {metrics.winning_trades}")
    logger.info(f"Losing trades: {metrics.losing_trades}")
    logger.info(f"Total return: {metrics.total_return_pct:.2f}%")
    
    logger.info(f"\nRisk/Fee Metrics:")
    logger.info(f"  Signals rejected by risk: {metrics.signals_rejected_by_risk}")
    logger.info(f"  Signals rejected by fee: {metrics.signals_rejected_by_fee_filter}")
    logger.info(f"  Suspicious PnL rejected: {metrics.suspicious_pnl_rejected}")
    
    # Diagnostic insights
    print("\n" + "="*80)
    print("DIAGNOSTIC INSIGHTS")
    print("="*80)
    
    logger.info(f"\n1. DATA LOADING:")
    logger.info(f"   ✓ Loaded {len(df)} records")
    logger.info(f"   ✓ Depth levels: {len(bid_size_cols)}")
    
    # Calculate bid_depth_10 to verify fix
    bid_depth_10_check = np.zeros(len(df))
    for level in range(10):
        col = f'bid_size_{level}'
        if col in df.columns:
            bid_depth_10_check += df[col].values
    
    pct_gt_1000 = (bid_depth_10_check > 1000).sum() / len(df) * 100 if len(df) > 0 else 0
    logger.info(f"   ✓ bid_depth_10 mean: {bid_depth_10_check.mean():.0f}")
    logger.info(f"   ✓ Ticks with bid_depth_10 > 1000: {pct_gt_1000:.0f}%")
    
    logger.info(f"\n2. REGIME CLASSIFICATION:")
    logger.info(f"   (Check logs above for regime at each tick)")
    
    logger.info(f"\n3. SIGNAL GENERATION:")
    logger.info(f"   Signals rejected by risk: {metrics.signals_rejected_by_risk}")
    logger.info(f"   Signals rejected by fee: {metrics.signals_rejected_by_fee_filter}")
    total_rejected = metrics.signals_rejected_by_risk + metrics.signals_rejected_by_fee_filter
    logger.info(f"   Total signals rejected: {total_rejected}")
    
    logger.info(f"\n4. ABSORPTION STRENGTH:")
    logger.info(f"   (Check logs above for absorption_strength values)")
    logger.info(f"   Required threshold: 0.30")
    
    logger.info(f"\n5. TRADE EXECUTION:")
    logger.info(f"   Total trades fired: {metrics.total_trades}")
    
    if metrics.total_trades > 0:
        logger.info(f"   ✓✓✓ TRADES EXECUTED - DEPTH FIX WORKING")
        return 0
    elif metrics.signals_rejected_by_fee_filter > 0:
        logger.info(f"   Signals blocked by fees: {metrics.signals_rejected_by_fee_filter}")
    elif metrics.signals_rejected_by_risk > 0:
        logger.info(f"   Signals blocked by risk: {metrics.signals_rejected_by_risk}")
    else:
        logger.warning(f"   No trades executed, no rejections")
        logger.info(f"   → Issue is likely in signal generation (absorption_strength not meeting threshold)")
    
    return 0 if metrics.total_trades > 0 else 1


if __name__ == "__main__":
    sys.exit(run_diagnostic_backtest())
