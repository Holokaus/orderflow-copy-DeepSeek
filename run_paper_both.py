#!/usr/bin/env python3
"""
Paper Trading Runner: Post-cooldown Both Strategies (ICP/USDT)
Runs Absorption + StackedImbalance with loss streak cooldown on real-time ICP data.
"""
import sys
from pathlib import Path
import asyncio

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import OrderFlowSystem
from config.settings import Settings, TradingConfig, Exchange

class PaperSettings(Settings):
    """ICP/USDT settings for paper trading."""
    class Config:
        arbitrary_types_allowed = True

settings = PaperSettings(
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
    system = OrderFlowSystem(settings)
    print("=" * 60)
    print("  PAPER TRADING: Post-cooldown Both Strategies")
    print("  Symbol: ICP/USDT (1x, no leverage)")
    print(f"  Strategies: {', '.join(STRATEGIES)}")
    print("  Initial capital: $100")
    print("  Loss streak cooldown: active (skip after 2 losses)")
    print("=" * 60)
    await system.run_paper(STRATEGIES)

if __name__ == "__main__":
    asyncio.run(main())
