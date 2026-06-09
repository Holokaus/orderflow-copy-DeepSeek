#!/usr/bin/env python3
"""
Paper Trading Runner: Multi-trade version.
Same as run_paper_both.py but with:
  - Relaxed fee filter spread (0.05% instead of 0.01%)
  - Up to 2 concurrent positions instead of single-position-only
Uses spot data stream (for reliable trade data) + futures-equivalent fees.
No source files modified — all changes are local to this script.
"""
import sys, os, types
from pathlib import Path
import asyncio
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import OrderFlowSystem
from config.settings import Settings, TradingConfig, Exchange
from pydantic import ConfigDict

from core.data_structures import Side, SignalType, Trade, OrderFlowState, OrderBook, PriceLevel
from core.feature_engine import FeatureEngine, FeatureConfig
from execution.risk_manager import RiskAction
from loguru import logger

class PaperSettings(Settings):
    model_config = ConfigDict(arbitrary_types_allowed=True)

settings = PaperSettings(
    trading=TradingConfig(
        symbol="ICPUSDT", exchange=Exchange.BINANCE,
        max_position_size=1000.0, max_position_value_pct=0.25,
        max_daily_loss_pct=0.02, max_drawdown_pct=0.10,
        max_consecutive_losses=20, min_time_between_trades_sec=30,
        slippage_estimate_pct=0.0005, fee_pct=0.0005,
        tick_size=0.001,
        feature_windows=[15, 30, 60, 300, 600, 900],
    ),
)

STRATEGIES = ["absorption", "stacked_imbalance"]
MAX_CONCURRENT = 2


def print_status(system, elapsed_seconds):
    """Print periodic status — adapted for multi-position."""
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60
    initial = system.paper_initial_capital
    current = system.paper_capital
    unrealized = sum(p.get('unrealized_pnl', 0) for p in getattr(system, 'paper_positions', []))
    equity = current + unrealized
    pnl_amount = equity - initial
    pnl_pct = (pnl_amount / initial * 100) if initial > 0 else 0
    num_pos = len(getattr(system, 'paper_positions', []))
    num_trades = len(system.paper_closed_trades)
    wins = sum(1 for t in system.paper_closed_trades if t['pnl'] > 0)
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
    total_pnl = sum(t['pnl'] for t in system.paper_closed_trades)
    print("\n" + "-" * 60)
    print(f"STATUS [{datetime.now().strftime('%H:%M:%S')}]  running {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("-" * 60)
    print(f"  Equity: ${equity:.2f}  P&L: ${pnl_amount:+.2f} ({pnl_pct:+.2f}%)")
    print(f"  Open positions: {num_pos}/{MAX_CONCURRENT}")
    for i, p in enumerate(getattr(system, 'paper_positions', [])):
        print(f"    [{i+1}] {p['strategy']} entry=${p['entry_price']:.4f} sz={p['size']:.4f} "
              f"SL=${p['stop_loss']:.4f} TP=${p['take_profit']:.4f} "
              f"uPNL=${p.get('unrealized_pnl',0):+.4f}")
    print(f"  Closed trades: {num_trades}  ({win_rate:.1f}% win)  total P&L: ${total_pnl:+.2f}")
    if system.paper_consecutive_losses >= 2:
        print(f"  COOLDOWN: {system.paper_consecutive_losses} consecutive losses")
    print("-" * 60)


# ── Multi-position replacement for _on_paper_order_book ──
async def _multi_on_paper_order_book(self, order_book: dict):
    """Replacement for _on_paper_order_book supporting up to MAX_CONCURRENT positions."""
    self._paper_tick_count += 1
    ntrades = len(self._pending_trades)
    self._paper_total_trades_received += ntrades

    if not self.running or not self.paper_strategies:
        return

    try:
        ts = datetime.now()

        # 1) Build OrderBook
        bids_raw = order_book.get('bids', [])
        asks_raw = order_book.get('asks', [])
        bids = [PriceLevel(price=float(p), size=float(s), timestamp=ts)
                for p, s in bids_raw[:20] if float(p) > 0 and float(s) > 0]
        asks = [PriceLevel(price=float(p), size=float(s), timestamp=ts)
                for p, s in asks_raw[:20] if float(p) > 0 and float(s) > 0]
        bids.sort(key=lambda l: l.price, reverse=True)
        asks.sort(key=lambda l: l.price)
        ob = OrderBook(timestamp=ts, bids=bids, asks=asks)

        # 2) Convert trades
        trades = []
        for t in self._pending_trades:
            price = float(t.get('price', 0))
            size_val = float(t.get('size', 0))
            if size_val == 0:
                size_val = float(t.get('q', 0))
            side_str = str(t.get('side', 'buy')).lower().strip()
            side = Side.BUY if side_str == 'buy' else Side.SELL
            if price > 0 and size_val > 0:
                trades.append(Trade(timestamp=ts, price=price, size=size_val, side=side))
        self._pending_trades.clear()

        # 3) Skip if no book
        if not ob.best_bid or not ob.best_ask or ob.mid_price <= 0:
            self._paper_skipped_no_book += 1
            return

        # 4) Update FeatureEngine
        state = self.paper_feature_engine.update(ob, trades)

        # 5) Mark-to-market ALL open positions & check exits
        for pos in list(getattr(self, 'paper_positions', [])):
            # Update P&L
            book = state.order_book
            mid = book.mid_price
            mark = book.best_bid.price if book.best_bid else mid
            pos['unrealized_pnl'] = (mark - pos['entry_price']) * pos['size']
            pos['highest_price'] = max(pos.get('highest_price', 0), mid)

            # Trailing stop
            activation = pos.get('trailing_activation', 0)
            if activation > 0:
                mark = book.best_bid.price if book.best_bid else book.mid_price
                move_pct = (mark - pos['entry_price']) / pos['entry_price']
                if move_pct >= activation:
                    pos['trailing_active'] = True
                    trail_distance = activation * 0.5
                    new_stop = mark * (1 - trail_distance)
                    if new_stop > pos.get('trailing_stop_price', 0):
                        pos['trailing_stop_price'] = new_stop
                        pos['stop_loss'] = max(pos['stop_loss'], new_stop)

            # Exit check
            exit_price = book.best_bid.price if book.best_bid else book.mid_price
            if exit_price <= pos['stop_loss']:
                self._close_multi_paper_trade(state, ts, 'stop_loss', pos)
            elif exit_price >= pos['take_profit']:
                self._close_multi_paper_trade(state, ts, 'take_profit', pos)
            else:
                hold_sec = (ts - pos['entry_time']).total_seconds()
                if hold_sec >= 1200:
                    net_pressure = state.features.get('net_pressure', 0)
                    if net_pressure < -0.5:
                        bid_depth = state.features.get('bid_depth_10', 0)
                        ask_depth = state.features.get('ask_depth_10', 0)
                        if ask_depth > 0 and bid_depth / (ask_depth + 1e-9) < 0.3:
                            self._close_multi_paper_trade(state, ts, 'book_pressure_collapse', pos)

        # 6) Evaluate strategies (allow if < MAX_CONCURRENT positions)
        positions = getattr(self, 'paper_positions', [])
        if len(positions) >= MAX_CONCURRENT:
            self._paper_skipped_has_pos += 1
            return

        # Cooldown
        if self.paper_last_trade_time is not None:
            elapsed = (ts - self.paper_last_trade_time).total_seconds()
            if elapsed < self.settings.trading.min_time_between_trades_sec:
                return

        # Loss streak cooldown
        if self.paper_consecutive_losses >= 2:
            self._paper_skipped_loss_streak += 1
            return

        # Evaluate strategies
        mid = ob.mid_price
        for strat_name, strat in self.paper_strategies:
            self._paper_signals_evaluated += 1
            signal = strat.evaluate(state)
            if not signal or not signal.is_actionable:
                continue
            if signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL):
                continue

            entry_price = ob.best_ask.price * (1 + self.settings.trading.slippage_estimate_pct)
            risk_action, adjusted_signal, reason = self.risk_manager.check_signal(signal, mid, ts)
            if risk_action in (RiskAction.HALT_TRADING, RiskAction.REJECT):
                continue

            # Fee-aware filter
            if signal.entry_price > 0 and signal.take_profit != signal.entry_price:
                predicted_move = abs(signal.take_profit - signal.entry_price) / signal.entry_price
            else:
                predicted_move = 0.002

            fee_check = self.order_manager.validate_signal_with_fee_filter(
                signal, predicted_move, signal.confidence,
                ob.best_bid.price, ob.best_ask.price, ob.mid_price
            )
            if fee_check['status'] == 'REJECTED':
                continue

            sig = adjusted_signal or signal
            pos_value = self.paper_capital * sig.position_size
            size = pos_value / entry_price

            logger.info(f"[{strat_name}] PAPER ENTRY @ {entry_price:.4f} | "
                       f"SL: {sig.stop_loss:.4f} | TP: {sig.take_profit:.4f} | "
                       f"Size: {size:.4f} | Open: {len(positions)}/{MAX_CONCURRENT}")

            new_pos = {
                'strategy': strat_name, 'entry_time': ts,
                'entry_price': entry_price, 'size': size,
                'allocated': pos_value, 'stop_loss': sig.stop_loss,
                'take_profit': sig.take_profit,
                'trailing_activation': strat.trailing_stop_activation_pct,
                'trailing_active': False, 'trailing_stop_price': 0.0,
                'highest_price': entry_price,
                'entry_fee': pos_value * self.settings.trading.fee_pct,
            }
            self.paper_positions.append(new_pos)
            self.paper_entry_timestamp = ts
            self.paper_capital -= pos_value + (pos_value * self.settings.trading.fee_pct)
            break

        # Health report
        if self._paper_tick_count % 1000 == 0:
            fcount = len(state.features) if state and state.features else 0
            regime_name = state.regime.name if state and state.regime else 'UNKNOWN'
            logger.info(
                f"[HEALTH] Ticks:{self._paper_tick_count} "
                f"TradesRcv:{self._paper_total_trades_received} "
                f"Features:{fcount} Regime:{regime_name} "
                f"Mid:{mid:.4f} "
                f"Pos:{len(positions)}/{MAX_CONCURRENT} "
                f"ClTrades:{len(self.paper_closed_trades)} "
                f"ConsecLoss:{self.paper_consecutive_losses}"
            )
    except Exception as e:
        logger.error(f"Paper trading error: {e}")
        import traceback
        logger.error(traceback.format_exc())


def _close_multi_paper_trade(self, state, ts, reason, pos):
    """Close a specific paper position (removes from paper_positions list)."""
    if pos not in getattr(self, 'paper_positions', []):
        return
    book = state.order_book
    best_bid = book.best_bid.price if book.best_bid else book.mid_price
    best_ask = book.best_ask.price if book.best_ask else book.mid_price
    adverse_slip = (self.settings.trading.slippage_estimate_pct + 0.0003
                   if reason == 'stop_loss' else self.settings.trading.slippage_estimate_pct)
    exit_price = best_bid * (1 - adverse_slip)
    gross_pnl = (exit_price - pos['entry_price']) * pos['size']
    exit_value = exit_price * pos['size']
    exit_fee = exit_value * self.settings.trading.fee_pct
    net_pnl = gross_pnl - exit_fee
    notional = pos['entry_price'] * pos['size']
    pnl_pct = net_pnl / notional if notional > 0 else 0.0
    duration = (ts - pos['entry_time']).total_seconds()
    self.paper_capital += (pos['allocated'] - pos.get('entry_fee', 0)) + net_pnl
    self.paper_closed_trades.append({
        'exit_time': ts, 'exit_price': exit_price,
        'entry_price': pos['entry_price'], 'exit_reason': reason,
        'pnl': net_pnl, 'pnl_pct': pnl_pct,
        'strategy': pos['strategy'], 'duration_seconds': duration,
        'entry_time': pos['entry_time'],
    })
    self.risk_manager.record_trade_closed(net_pnl)
    self.paper_last_trade_time = ts
    if net_pnl <= 0:
        self.paper_consecutive_losses += 1
    else:
        self.paper_consecutive_losses = 0
    self.paper_positions.remove(pos)


async def main():
    system = OrderFlowSystem(settings)

    # Init state for multi-position
    system.paper_positions = []
    system.paper_entry_timestamp = None

    # Replace position methods with multi-position versions
    system._on_paper_order_book = types.MethodType(_multi_on_paper_order_book, system)
    system._close_multi_paper_trade = types.MethodType(_close_multi_paper_trade, system)

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        print("WARNING: BINANCE_API_KEY or BINANCE_API_SECRET not found")
        print("  Paper trading will run in DATA-ONLY mode (no execution)")

    print("=" * 60)
    print("  PAPER TRADING: Multi-trade version")
    print(f"  Strategies: {', '.join(STRATEGIES)}")
    print(f"  Max concurrent positions: {MAX_CONCURRENT}")
    print(f"  Spread threshold: 0.05% (relaxed for spot)")
    print(f"  Initial capital: $100")
    print("=" * 60)
    print("\nConnecting to Binance (spot data stream)...\n")

    system._init_components('paper', testnet=False, use_futures=False)
    if system.exchange:
        system.exchange.config.api_key = api_key
        system.exchange.config.api_secret = api_secret

    # Relax fee filter spread
    if system.order_manager and hasattr(system.order_manager, 'fee_filter'):
        ff = system.order_manager.fee_filter
        ff.expected_spread = 0.0005
        ff.total_cost = ff.entry_fee + ff.exit_fee + ff.expected_spread + ff.min_profit

    start_time = asyncio.get_event_loop().time()
    status_interval = 180

    try:
        trading_task = asyncio.create_task(
            system.run_paper(STRATEGIES, testnet=False, use_futures=False)
        )

        async def status_monitor():
            last = start_time
            while not trading_task.done():
                now = asyncio.get_event_loop().time()
                if now - last >= status_interval:
                    print_status(system, now - start_time)
                    last = now
                await asyncio.sleep(5)

        monitor_task = asyncio.create_task(status_monitor())
        await trading_task
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        elapsed = asyncio.get_event_loop().time() - start_time
        print_status(system, elapsed)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
    finally:
        if system.exchange:
            await system.exchange.disconnect()
        elapsed = asyncio.get_event_loop().time() - start_time
        print("\n\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print_status(system, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
