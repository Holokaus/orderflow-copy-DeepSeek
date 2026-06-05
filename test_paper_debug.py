#!/usr/bin/env python3
"""
Debug Paper Trading - Trace every step of orderbook sync and strategy evaluation.
Run this to see detailed logs showing where trades are being blocked.
"""
import sys
import os
from pathlib import Path
import asyncio
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import OrderFlowSystem
from config.settings import Settings, TradingConfig, Exchange
from pydantic import ConfigDict
from loguru import logger

# Configure logging for maximum verbosity
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level="DEBUG",  # MAXIMUM VERBOSITY
    format="{time:HH:mm:ss} | {level: <8} | {message}",
    colorize=True
)

class DebugPaperSettings(Settings):
    """ICP/USDT settings for debugging paper trading."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

settings = DebugPaperSettings(
    trading=TradingConfig(
        symbol="ICPUSDT",
        exchange=Exchange.BINANCE,
        max_position_size=1000.0,
        max_position_value_pct=0.25,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
        max_consecutive_losses=20,
        min_time_between_trades_sec=30,
        slippage_estimate_pct=0.0005,
        fee_pct=0.0005,
        tick_size=0.001,
        feature_windows=[15, 30, 60, 300, 600, 900],
    ),
)

STRATEGIES = ["absorption", "stacked_imbalance"]


async def main():
    """Run paper trading with debug logging."""
    print("\n" + "="*70)
    print("🔍 DEBUG PAPER TRADING - TRACE ALL STEPS")
    print("="*70)
    print("\nThis will show:")
    print("✓ Every orderbook update received")
    print("✓ Every strategy evaluation")
    print("✓ Every filter/rejection reason")
    print("✓ Gap detection and re-sync triggers")
    print("\n" + "-"*70 + "\n")
    
    system = OrderFlowSystem(settings)
    
    try:
        # Run paper trading for 120 seconds, then exit
        paper_task = asyncio.create_task(
            system.run_paper(STRATEGIES, testnet=False)
        )
        
        # Wait 120 seconds
        await asyncio.sleep(120)
        
        print("\n" + "="*70)
        print("⏹️  Stopping debug session...")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Error in debug session: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Cleanup
        system.running = False
        if system.exchange:
            await system.exchange.disconnect()
        
        # Print summary
        if system.exchange:
            health = system.exchange.get_health_report()
            print("\n" + "="*70)
            print("📊 HEALTH REPORT")
            print("="*70)
            print(f"Events received: {health['events_received']}")
            print(f"Events applied: {health['events_applied']}")
            print(f"Gaps detected: {health['gaps_detected']}")
            print(f"Re-syncs: {health['resyncs']}")
            print(f"Book initialized: {health['book_initialized']}")
            print(f"Buffer size: {health['buffer_size']}")
            print(f"Completed trades: {len(system.paper_closed_trades)}")
            print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
