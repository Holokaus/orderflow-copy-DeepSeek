#!/usr/bin/env python3
"""
Paper Trading Runner: Post-cooldown Both Strategies (ICP/USDT)
Runs Absorption + StackedImbalance with loss streak cooldown on real-time ICP data.
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

class PaperSettings(Settings):
    """ICP/USDT settings for paper trading."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

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


def print_status(system, elapsed_seconds):
    """Print periodic status of paper trading"""
    print("\n" + "-" * 60)
    print(f"📊 STATUS UPDATE [{datetime.now().strftime('%H:%M:%S')}]")
    print("-" * 60)
    
    # Timing
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60
    print(f"⏱️  Running time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    # Capital & P&L
    initial = system.paper_initial_capital
    current = system.paper_capital
    equity = current
    pnl_amount = current - initial
    pnl_pct = (pnl_amount / initial * 100) if initial > 0 else 0
    
    pnl_emoji = "📈" if pnl_amount > 0 else "📉" if pnl_amount < 0 else "➡️"
    print(f"💰 Equity: ${equity:.2f}")
    print(f"{pnl_emoji} P&L: ${pnl_amount:+.2f} ({pnl_pct:+.2f}%)")
    
    # Position
    if system.paper_position:
        pos = system.paper_position
        print(f"📍 OPEN POSITION:")
        print(f"   Strategy: {pos['strategy']}")
        print(f"   Entry: ${pos['entry_price']:.4f} | Size: {pos['size']:.4f}")
        print(f"   SL: ${pos['stop_loss']:.4f} | TP: ${pos['take_profit']:.4f}")
        unrealized = pos.get('unrealized_pnl', 0)
        unrealized_pct = (unrealized / pos.get('allocated', 1)) * 100 if pos.get('allocated') > 0 else 0
        print(f"   Unrealized: ${unrealized:+.2f} ({unrealized_pct:+.2f}%)")
    else:
        print(f"📍 Position: FLAT (no open trade)")
    
    # Trades
    num_trades = len(system.paper_closed_trades)
    print(f"📋 Completed Trades: {num_trades}")
    if num_trades > 0:
        wins = sum(1 for t in system.paper_closed_trades if t['pnl'] > 0)
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
        total_pnl = sum(t['pnl'] for t in system.paper_closed_trades)
        print(f"   ├─ Win Rate: {win_rate:.1f}% ({wins}/{num_trades})")
        print(f"   ├─ Total P&L: ${total_pnl:+.2f}")
        avg_pnl = total_pnl / num_trades if num_trades > 0 else 0
        print(f"   └─ Avg Trade: ${avg_pnl:+.2f}")
    
    # Cooldown status
    if system.paper_consecutive_losses >= 2:
        print(f"⏸️  COOLDOWN ACTIVE: {system.paper_consecutive_losses} consecutive losses")
    
    print("-" * 60)


async def main():
    system = OrderFlowSystem(settings)
    
    # Load API credentials from environment
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    
    if not api_key or not api_secret:
        print("\n⚠️  WARNING: BINANCE_API_KEY or BINANCE_API_SECRET not found in .env file")
        print("   Paper trading will run in DATA-ONLY mode (no trading execution)")
        print("   To enable execution, add credentials to .env:\n")
        print("   BINANCE_API_KEY=your_key")
        print("   BINANCE_API_SECRET=your_secret\n")
    
    print("=" * 60)
    print("  PAPER TRADING: Post-cooldown Both Strategies")
    print("  Symbol: ICP/USDT (1x, no leverage)")
    print(f"  Strategies: {', '.join(STRATEGIES)}")
    print("  Initial capital: $100")
    print("  Loss streak cooldown: active (skip after 2 losses)")
    print("  Exchange: Binance PRODUCTION (real market data)")
    print(f"  API Credentials: {'✓ Configured' if api_key else '✗ Missing'}")
    print("=" * 60)
    print("\n⏳ Connecting to Binance...\n")
    
    # Initialize exchange with API credentials (not async) — FUTURES mode for backtest-consistent data
    system._init_components('paper', testnet=False, use_futures=True)
    if system.exchange:
        system.exchange.config.api_key = api_key
        system.exchange.config.api_secret = api_secret
    
    start_time = asyncio.get_event_loop().time()
    status_interval = 180  # Print status every 180 seconds
    last_status_time = start_time
    
    try:
        # Create task for paper trading
        trading_task = asyncio.create_task(system.run_paper(STRATEGIES, testnet=False, use_futures=True))
        
        # Status monitor task
        async def status_monitor():
            nonlocal last_status_time
            while not trading_task.done():
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                
                if current_time - last_status_time >= status_interval:
                    print_status(system, elapsed)
                    last_status_time = current_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
        
        monitor_task = asyncio.create_task(status_monitor())
        
        # Wait for trading to complete
        await trading_task
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Paper trading stopped by user")
        elapsed = asyncio.get_event_loop().time() - start_time
        print_status(system, elapsed)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if "restricted location" in str(e).lower():
            print("\n⚠️  GEOGRAPHIC RESTRICTION DETECTED:")
            print("   Binance is blocking access from your location.")
            print("   Solutions:")
            print("   1. Use a VPN to connect from an allowed region")
            print("   2. Contact Binance support to whitelist your IP")
        raise
    finally:
        # Cleanup resources
        if system.exchange:
            await system.exchange.disconnect()
        elapsed = asyncio.get_event_loop().time() - start_time
        print("\n\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print_status(system, elapsed)

if __name__ == "__main__":
    asyncio.run(main())
