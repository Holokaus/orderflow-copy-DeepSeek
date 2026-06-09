"""
VIRTUAL ONE-TIME TEST: Relaxed spread (0.05%) + max 2 concurrent trades.
No changes to any source files.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import (
    create_absorption_strategy, create_stacked_imbalance_strategy,
    create_delta_divergence_strategy, create_liquidity_sweep_strategy,
    create_value_area_strategy,
)
from core.fee_aware_filter import FeeAwareFilter
from core.feature_engine import FeatureConfig
from core.feature_precomputer import FeaturePrecomputer
from core.data_structures import Side, SignalType
from backtesting.engine import Position, ClosedTrade
from execution.risk_manager import RiskAction

INITIAL_CAPITAL = 100.0
DATA_PATH = project_root / "data" / "backtests"

# ── Patch FeeAwareFilter default BEFORE any engine is created ──
FeeAwareFilter.__init__.__defaults__ = (0.0002, 0.0005, 0.0005, 0.0002)


class MultiPositionEngine(BacktestEngine):
    """Same as BacktestEngine but allows up to 2 concurrent positions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positions: list = []
        self.max_concurrent = 2
        # Apply relaxed spread
        self.fee_filter.expected_spread = 0.0005
        self.fee_filter.total_cost = (
            self.fee_filter.entry_fee + self.fee_filter.exit_fee +
            self.fee_filter.expected_spread + self.fee_filter.min_profit
        )

    def reset(self) -> None:
        super().reset()
        self.positions = []

    def _open_position(self, signal, state, timestamp, strategy, risk_action):
        book = state.order_book
        best_ask = book.best_ask.price if book.best_ask else book.mid_price
        entry_price = best_ask * (1 + self.slippage_pct)
        side = Side.BUY
        position_value = self.capital * signal.position_size
        size = position_value / entry_price
        entry_fee = position_value * self.fee_pct
        allocated = position_value + entry_fee
        if allocated > self.capital:
            allocated = self.capital * 0.95
            position_value = allocated / (1 + self.fee_pct)
            size = position_value / entry_price
            entry_fee = position_value * self.fee_pct
        self.capital -= allocated
        pos = Position(
            entry_time=timestamp,
            entry_price=entry_price,
            allocated_capital=allocated,
            size=size,
            side=side,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal=signal,
            strategy_name=strategy.name,
            trailing_stop_activation_pct=strategy.trailing_stop_activation_pct,
            risk_action=risk_action,
            highest_price=entry_price,
            lowest_price=entry_price,
            entry_fee=entry_fee,
        )
        self.positions.append(pos)
        self.risk_manager.record_trade_opened(entry_price, size, side, timestamp)
        self._last_signal_time = timestamp
        self._last_signal_strategy = strategy.name

    def _close_position(self, state, timestamp, reason, strategy, pos):
        if pos not in self.positions:
            return
        book = state.order_book
        best_bid = book.best_bid.price if book.best_bid else book.mid_price
        best_ask = book.best_ask.price if book.best_ask else book.mid_price
        adverse_slip = (self.slippage_pct + self.sl_extra_slippage_pct
                       if reason == "stop_loss" else self.slippage_pct)
        exit_price = best_bid * (1 - adverse_slip)
        gross_pnl = (exit_price - pos.entry_price) * pos.size
        exit_value = exit_price * pos.size
        exit_fee = exit_value * self.fee_pct
        pnl = gross_pnl - exit_fee
        notional = pos.entry_price * pos.size
        pnl_pct = pnl / notional if notional > 0 else 0.0
        if abs(pnl_pct) > self.suspicious_pnl_pct and reason != "end_of_backtest":
            self._suspicious_rejected += 1
            self.capital += pos.allocated_capital
            self.positions.remove(pos)
            return
        self.capital += (pos.allocated_capital - pos.entry_fee) + pnl
        duration = (timestamp - pos.entry_time).total_seconds()
        self.closed_trades.append(ClosedTrade(
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            side=pos.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            duration_seconds=duration,
            signal_confidence=pos.signal.confidence,
            risk_action=pos.risk_action,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
        ))
        self.risk_manager.record_trade_closed(pnl)
        self._last_trade_close_time = timestamp
        self.positions.remove(pos)

    def run(self, data, strategy, params=None):
        """Modified run() supporting up to 2 concurrent positions."""
        self.reset()
        if params:
            strategy = self._apply_params(strategy, params)
        rows = self._preprocess(data)
        n_rows = len(rows)
        self.precomputer = FeaturePrecomputer(windows=self.windows)
        self.precomputer.precompute_all(data)
        if len(rows) > 0:
            sample = rows[0]
            if not sample._has_depth:
                logger.error("WARNING: No depth data detected!")
        first_timestamp = None
        state = None
        timestamp = None
        pattern_every = self.PATTERN_DETECTION_EVERY_N
        equity_every = self.EQUITY_SAMPLE_EVERY_N
        vp_every = self.VOLUME_PROFILE_EVERY_N
        for tick_idx in range(n_rows):
            row = rows[tick_idx]
            timestamp = row.timestamp
            if first_timestamp is None:
                first_timestamp = timestamp
                self.equity_curve.append((first_timestamp, self.initial_capital))
            self._handle_daily_reset(timestamp)
            current_equity = self._calculate_equity_fast()
            if current_equity < self.equity_floor:
                logger.warning(f"Equity floor breached. Halting.")
                break
            order_book = self._build_order_book_fast(row, timestamp)
            trades = self._build_trades_fast(row, timestamp)
            run_patterns = (tick_idx % pattern_every == 0)
            run_vp = (tick_idx % vp_every == 0)
            precomputed = self.precomputer.get_features_for_tick(tick_idx) if self.precomputer else None
            state = self.feature_engine.update(
                order_book, trades,
                detect_patterns=run_patterns,
                compute_volume_profile=run_vp,
                precomputed_features=precomputed,
            )
            # Update open positions
            for pos in list(self.positions):
                self._update_position_generic(state, pos)
                self._update_trailing_stop_generic(state, pos)
                exit_reason = self._check_exit_conditions_generic(state, timestamp, pos)
                if exit_reason:
                    self._close_position(state, timestamp, exit_reason, strategy, pos)
            # Warmup gate
            elapsed = (timestamp - first_timestamp).total_seconds()
            if elapsed < self.warmup_seconds:
                if tick_idx % equity_every == 0:
                    self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                continue
            # Signal evaluation (allow if < max_concurrent positions)
            if len(self.positions) < self.max_concurrent:
                if not self._cooldown_passed(timestamp):
                    if tick_idx % equity_every == 0:
                        self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                    continue
                recent_trades = self.closed_trades[-2:]
                if len(recent_trades) == 2 and all(t.pnl <= 0 for t in recent_trades):
                    if tick_idx % equity_every == 0:
                        self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                    continue
                signal = strategy.evaluate(state)
                if signal and signal.is_actionable:
                    if signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL):
                        if tick_idx % equity_every == 0:
                            self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                        continue
                    if self._is_duplicate_signal(signal, strategy, timestamp):
                        if tick_idx % equity_every == 0:
                            self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                        continue
                    risk_action, adjusted_signal, reason = self.risk_manager.check_signal(
                        signal, state.order_book.mid_price, timestamp
                    )
                    if risk_action.__class__.__name__ == 'HALT_TRADING':
                        self._halts += 1
                        continue
                    elif risk_action.__class__.__name__ == 'REJECT':
                        self._signals_rejected += 1
                        continue
                    signal_to_check = adjusted_signal or signal
                    if all([
                        signal_to_check.entry_price > 0,
                        signal_to_check.take_profit != signal_to_check.entry_price,
                    ]):
                        predicted_move_pct = abs(signal_to_check.take_profit - signal_to_check.entry_price) / signal_to_check.entry_price
                    else:
                        predicted_move_pct = self._estimate_predicted_move(state, strategy, signal_to_check)
                    if predicted_move_pct > 0:
                        should_ignore, fee_reason = self.fee_filter.should_ignore_signal(
                            signal_to_check, predicted_move_pct, signal_to_check.confidence
                        )
                        if should_ignore:
                            self._signals_rejected_by_fee_filter += 1
                            if tick_idx % equity_every == 0:
                                self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                            continue
                    if risk_action == RiskAction.REDUCE_SIZE:
                        self._signals_reduced += 1
                        self._open_position(adjusted_signal or signal, state, timestamp, strategy, risk_action.name)
                    else:
                        self._open_position(adjusted_signal or signal, state, timestamp, strategy, risk_action.name)
            if tick_idx % equity_every == 0:
                self.equity_curve.append((timestamp, self._calculate_equity_fast()))
        # Close remaining positions
        for pos in list(self.positions):
            self._close_position(state, timestamp, "end_of_backtest", strategy, pos)
        return self._calculate_metrics()

    def _update_position_generic(self, state, pos):
        book = state.order_book
        mid = book.mid_price
        mark_price = book.best_bid.price if book.best_bid else mid
        pos.unrealized_pnl = (mark_price - pos.entry_price) * pos.size
        pos.highest_price = max(pos.highest_price, mid)
        if pos.lowest_price <= 0:
            pos.lowest_price = mid
        else:
            pos.lowest_price = min(pos.lowest_price, mid)

    def _update_trailing_stop_generic(self, state, pos):
        if pos.trailing_stop_activation_pct <= 0:
            return
        book = state.order_book
        mark = book.best_bid.price if book.best_bid else book.mid_price
        move_pct = (mark - pos.entry_price) / pos.entry_price
        if move_pct >= pos.trailing_stop_activation_pct:
            pos.trailing_stop_active = True
            trail_distance = pos.trailing_stop_activation_pct * 0.5
            new_stop = mark * (1 - trail_distance)
            pos.trailing_stop_price = max(pos.trailing_stop_price, new_stop)
            if pos.trailing_stop_price > pos.stop_loss:
                pos.stop_loss = pos.trailing_stop_price

    def _check_exit_conditions_generic(self, state, timestamp, pos):
        if not pos:
            return None
        book = state.order_book
        features = state.features
        hold_duration = (timestamp - pos.entry_time).total_seconds()
        flow_exits_allowed = hold_duration >= self.MIN_HOLD_BEFORE_FLOW_EXIT_SEC
        exit_check_price = book.best_bid.price if book.best_bid else book.mid_price
        if exit_check_price <= pos.stop_loss:
            return "stop_loss"
        if exit_check_price >= pos.take_profit:
            return "take_profit"
        if flow_exits_allowed and state.sweeps:
            latest_sweep = state.sweeps[-1]
            if (latest_sweep.direction == Side.SELL and
                    latest_sweep.reversal_strength < 0.4):
                return "sweep_against_long"
        if flow_exits_allowed:
            net_pressure = features.get("net_pressure", 0)
            if net_pressure < -0.5:
                bid_depth = features.get("bid_depth_10", 0)
                ask_depth = features.get("ask_depth_10", 0)
                if ask_depth > 0 and bid_depth / (ask_depth + 1e-9) < 0.3:
                    return "book_pressure_collapse"
        return None

    def _calculate_equity_fast(self) -> float:
        equity = self.capital
        for pos in self.positions:
            equity += pos.unrealized_pnl
            equity += (pos.allocated_capital - pos.entry_fee)
        return equity


def load_data():
    df1 = pd.read_parquet(DATA_PATH / "ICPUSDT_20260531_processed.parquet")
    df2 = pd.read_parquet(DATA_PATH / "ICPUSDT_20260601_processed.parquet")
    return pd.concat([df1, df2]).reset_index(drop=True)


def make_engine():
    return MultiPositionEngine(
        initial_capital=INITIAL_CAPITAL, fee_pct=0.0005, slippage_pct=0.0003,
        sl_extra_slippage_pct=0.0003, warmup_seconds=60.0,
        min_time_between_trades_sec=30.0,
        risk_limits=RiskLimits(
            max_position_size=10000.0, max_position_value_pct=0.25,
            max_daily_loss_pct=0.02, max_drawdown_pct=0.10,
            max_trades_per_day=50, max_trades_per_hour=10,
            min_time_between_trades_sec=30, max_consecutive_losses=20,
        ),
    )


def run_strategy(strat):
    engine = make_engine()
    engine.run(df, strat)
    return [(t.exit_time, t.pnl_pct, t.exit_reason) for t in engine.closed_trades]


def report(label, trades, span_days, initial_cap):
    if not trades:
        print(f"\n  {label}: 0 trades")
        return
    pnl_pcts = [p for _, p, _ in trades]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]
    total_ret = (np.prod([1 + p for p in pnl_pcts]) - 1) * 100
    print(f"\n  {'='*65}")
    print(f"  {label}  [spread=0.05% | max {2} concurrent trades]")
    print(f"  {'='*65}")
    print(f"    Trades:       {len(trades)}")
    print(f"    Win rate:     {len(wins)/len(trades)*100:.1f}%")
    print(f"    Total return: {total_ret:+.3f}% ({initial_cap * (1+total_ret/100):.2f}$)")
    print(f"    Daily ret:    {total_ret/span_days:+.3f}%/day")
    print(f"    Profit fact:  {sum(wins)/abs(sum(losses)):.3f}" if losses else "    Profit fact:  inf")
    peak = initial_cap
    eq = initial_cap
    max_dd = 0
    for _, p, _ in trades:
        eq *= (1 + p)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)
    print(f"    Max DD:       {max_dd:.2f}%")
    print(f"    Avg win:      {np.mean(wins)*100:+.3f}%" if wins else "")
    print(f"    Avg loss:     {np.mean(losses)*100:+.3f}%" if losses else "")
    print()
    for i, (ts, pnl, reason) in enumerate(trades):
        print(f"    #{i+1:>2}  {ts.strftime('%m/%d %H:%M')}  {pnl*100:>+7.3f}%  ({reason})")


# ── Run ──
print("Loading ICP data...")
df = load_data()
ts_min = pd.to_datetime(df['timestamp']).min()
ts_max = pd.to_datetime(df['timestamp']).max()
span_days = (ts_max - ts_min).total_seconds() / 86400
print(f"  {ts_min}  ->  {ts_max}  ({span_days:.1f} days)")
print(f"\n{'='*70}")
print(f"  ONE-TIME VIRTUAL TEST: spread=0.05% + max 2 concurrent trades")
print(f"{'='*70}")

print("\nRunning Absorption (multi-position)...")
abs_trades = run_strategy(create_absorption_strategy())
print(f"  {len(abs_trades)} trades")

print("Running Delta Divergence (multi-position)...")
dd_trades = run_strategy(create_delta_divergence_strategy())
print(f"  {len(dd_trades)} trades")

print("Running Liquidity Sweep (multi-position)...")
ls_trades = run_strategy(create_liquidity_sweep_strategy())
print(f"  {len(ls_trades)} trades")

print("Running StackedImbalance (multi-position)...")
si_trades = run_strategy(create_stacked_imbalance_strategy())
print(f"  {len(si_trades)} trades")

print("Running Value Area (multi-position)...")
va_trades = run_strategy(create_value_area_strategy())
print(f"  {len(va_trades)} trades")

all_trades = sorted(abs_trades + dd_trades + ls_trades + si_trades + va_trades, key=lambda x: x[0])

report("Absorption", abs_trades, span_days, INITIAL_CAPITAL)
report("Delta Divergence", dd_trades, span_days, INITIAL_CAPITAL)
report("Liquidity Sweep", ls_trades, span_days, INITIAL_CAPITAL)
report("StackedImbalance", si_trades, span_days, INITIAL_CAPITAL)
report("Value Area", va_trades, span_days, INITIAL_CAPITAL)
report("All 5 Strategies (merged chronologically)", all_trades, span_days, INITIAL_CAPITAL)

abs_ret = float(np.prod([1+p for _,p,_ in abs_trades]))
dd_ret = float(np.prod([1+p for _,p,_ in dd_trades]))
ls_ret = float(np.prod([1+p for _,p,_ in ls_trades]))
si_ret = float(np.prod([1+p for _,p,_ in si_trades]))
va_ret = float(np.prod([1+p for _,p,_ in va_trades]))
all_ret = float(np.prod([1+p for _,p,_ in all_trades]))
print(f"\n  Final equity:")
print(f"    Absorption:       ${INITIAL_CAPITAL * abs_ret:.2f}")
print(f"    Delta Divergence: ${INITIAL_CAPITAL * dd_ret:.2f}")
print(f"    Liquidity Sweep:  ${INITIAL_CAPITAL * ls_ret:.2f}")
print(f"    StackedImbalance: ${INITIAL_CAPITAL * si_ret:.2f}")
print(f"    Value Area:       ${INITIAL_CAPITAL * va_ret:.2f}")
print(f"    Combined (all):   ${INITIAL_CAPITAL * all_ret:.2f}")
