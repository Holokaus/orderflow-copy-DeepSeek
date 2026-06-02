"""
DEBUG TEST: Check what's happening with signal generation
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<level>{level: <8}</level> | {message}", level="DEBUG")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtesting.engine import BacktestEngine
from knowledge.strategy_library import StrategyDefinition, StrategyCategory, StrategyCondition
from execution.risk_manager import RiskLimits


def main():
    print("\n" + "="*80)
    print("DEBUG: Check Feature Values and Signal Generation")
    print("="*80)
    
    # Load small data sample
    parquet_file = project_root / "data" / "backtests" / "XRPUSDT_market_data_20260328.parquet"
    df = pd.read_parquet(parquet_file)
    df = df.head(500)
    
    logger.info(f"Loaded {len(df)} rows")
    
    # Create minimal strategy - just one condition
    entry_conditions = [
        StrategyCondition(
            feature="bid_depth_10",
            operator=">",
            threshold=100.0,
            weight=1.0,
        ),
    ]
    
    strategy = StrategyDefinition(
        name="DebugStrategy",
        category=StrategyCategory.MOMENTUM,
        description="Debug: Just check bid_depth_10",
        entry_conditions=entry_conditions,
        min_conditions_satisfied=1,
        min_score_threshold=0.1,
        stop_loss_atr_mult=1.0,
        take_profit_atr_mult=1.0,
    )
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=10000.0,
        fee_pct=0.002,
        slippage_pct=0.0002,
        risk_limits=RiskLimits(),
    )
    
    logger.info("Running backtest...")
    metrics = engine.run(df, strategy)
    
    # Report
    logger.info(f"Total trades: {metrics.total_trades}")
    logger.info(f"Signals rejected (risk): {metrics.signals_rejected_by_risk}")
    logger.info(f"Signals rejected (fee): {metrics.signals_rejected_by_fee_filter}")
    logger.info(f"Suspicious PnL rejected: {metrics.suspicious_pnl_rejected}")
    
    return 0 if metrics.total_trades > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
