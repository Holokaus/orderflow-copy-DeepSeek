##==========================================
# Section 1 Backtesting/engine.py
##==========================================
"""
Backtesting Engine v5 — Integrated with Enhanced FeatureEngine
=================================================================
Fixes from v4:
1. Accepts FeatureConfig from caller (tick_size=0.5, regime params, etc.)
2. Removes duplicate footprint bar system (FeatureEngine is single source)
3. Removes duplicate regime detection (RegimeClassifier is single source)
4. WalkForwardValidator properly isolates trials for parallel execution
5. Min-trades filtering in OOS results

Performance optimisations from v4 preserved:
- Pre-extracted NumPy arrays
- Lightweight _Row container
- Pattern/volume profile throttling
- Equity curve downsampling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import copy
import math

from loguru import logger

from core.data_structures import (
    OrderFlowState, Signal, SignalType, Side, Trade,
    OrderBook, PriceLevel, FootprintBar, Regime
)
from core.feature_engine import FeatureEngine, FeatureConfig
from knowledge.strategy_library import StrategyDefinition
from execution.risk_manager import RiskManager, RiskLimits, RiskAction


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Track an open position"""
    entry_time: datetime
    entry_price: float
    allocated_capital: float
    size: float
    side: Side
    stop_loss: float
    take_profit: float
    signal: Signal
    strategy_name: str
    trailing_stop_activation_pct: float
    risk_action: str = ""

    entry_fee: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0


@dataclass
class ClosedTrade:
    """Record of a completed trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    side: Side
    pnl: float
    pnl_pct: float
    exit_reason: str
    duration_seconds: float
    signal_confidence: float
    risk_action: str = ""


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    avg_trade_duration_seconds: float = 0.0
    max_consecutive_losses: int = 0

    trades_per_day: float = 0.0
    calmar_ratio: float = 0.0

    signals_rejected_by_risk: int = 0
    signals_reduced_by_risk: int = 0
    halts_triggered: int = 0
    suspicious_pnl_rejected: int = 0


# ---------------------------------------------------------------------------
# Lightweight row container
# ---------------------------------------------------------------------------

class _Row:
    """Ultra-lightweight row accessor backed by raw numpy arrays."""
    __slots__ = (
        'timestamp', 'bid_price', 'ask_price', 'bid_size', 'ask_size',
        'trade_price', 'trade_size', 'trade_side',
        '_depth_bids_p', '_depth_bids_s', '_depth_asks_p', '_depth_asks_s',
        '_has_depth', '_n_depth',
    )

    def __init__(
        self,
        timestamp, bid_price, ask_price, bid_size, ask_size,
        trade_price, trade_size, trade_side,
        depth_bids_p=None, depth_bids_s=None,
        depth_asks_p=None, depth_asks_s=None,
    ):
        self.timestamp = timestamp
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.trade_price = trade_price
        self.trade_size = trade_size
        self.trade_side = trade_side
        self._depth_bids_p = depth_bids_p
        self._depth_bids_s = depth_bids_s
        self._depth_asks_p = depth_asks_p
        self._depth_asks_s = depth_asks_s
        self._has_depth = depth_bids_p is not None
        self._n_depth = len(depth_bids_p) if depth_bids_p is not None else 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven backtesting engine — integrated with enhanced FeatureEngine.
    """

    # Throttling knobs
    PATTERN_DETECTION_EVERY_N = 25
    EQUITY_SAMPLE_EVERY_N = 50
    VOLUME_PROFILE_EVERY_N = 100

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_pct: float = 0.0004,
        slippage_pct: float = 0.0005,
        sl_extra_slippage_pct: float = 0.0003,
        warmup_seconds: float = 60.0,
        min_time_between_trades_sec: float = 30.0,
        equity_floor_pct: float = 0.50,
        suspicious_pnl_pct: float = 0.10,
        risk_limits: Optional[RiskLimits] = None,
        feature_config: Optional[FeatureConfig] = None,  # NEW: accept config
    ):
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.sl_extra_slippage_pct = sl_extra_slippage_pct
        self.warmup_seconds = warmup_seconds
        self.min_time_between_trades_sec = min_time_between_trades_sec
        self.equity_floor_pct = equity_floor_pct
        self.equity_floor = initial_capital * equity_floor_pct
        self.suspicious_pnl_pct = suspicious_pnl_pct

        # FIX: Store feature config for creating FeatureEngine
        self._feature_config = feature_config or FeatureConfig()

        self.risk_limits = risk_limits or RiskLimits(
            max_position_size=10000.0, #1.0 for BTC change to 10000 for XRP
            max_position_value_pct=0.25,
            max_daily_loss_pct=0.02,
            max_weekly_loss_pct=0.05,
            max_drawdown_pct=0.10,
            max_trades_per_day=50,
            max_trades_per_hour=10,
            min_time_between_trades_sec=int(min_time_between_trades_sec),
            max_consecutive_losses=5,
        )

        # State (initialized in reset())
        self.capital: float = 0.0
        self.position: Optional[Position] = None
        self.closed_trades: List[ClosedTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.feature_engine: Optional[FeatureEngine] = None
        self.risk_manager: Optional[RiskManager] = None

        self._last_trade_close_time: Optional[datetime] = None
        self._last_signal_strategy: Optional[str] = None
        self._last_signal_time: Optional[datetime] = None
        self._current_day = None

        # Tracking
        self._signals_rejected = 0
        self._signals_reduced = 0
        self._halts = 0
        self._suspicious_rejected = 0

    def reset(self) -> None:
        """Reset state for new backtest."""
        self.capital = self.initial_capital
        self.position = None
        self.closed_trades = []
        self.equity_curve = []
        
        # FIX: Create FeatureEngine with proper config
        self.feature_engine = FeatureEngine(self._feature_config)
        
        self.risk_manager = RiskManager(
            limits=self.risk_limits,
            initial_equity=self.initial_capital,
        )
        self._last_trade_close_time = None
        self._last_signal_strategy = None
        self._last_signal_time = None
        self._current_day = None
        self._signals_rejected = 0
        self._signals_reduced = 0
        self._halts = 0
        self._suspicious_rejected = 0
        self._entry_tick_idx: int = -1 # ADD Initialize entry tick tracker

    # ------------------------------------------------------------------
    # DataFrame → numpy pre-extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(data: pd.DataFrame) -> List[_Row]:
        """Convert DataFrame to lightweight _Row objects."""
        
        # === FIX: Map _0 columns to base columns ===
        # (Because we dropped the redundant ones in the Parquet conversion)
        if 'bid_price' not in data.columns and 'bid_price_0' in data.columns:
            data = data.copy()
            data['bid_price'] = data['bid_price_0']
            data['ask_price'] = data['ask_price_0']
            data['bid_size'] = data['bid_size_0']
            data['ask_size'] = data['ask_size_0']
        # ============================================
        
        timestamps = pd.to_datetime(data['timestamp']).dt.to_pydatetime()

        bid_price = data['bid_price'].values if 'bid_price' in data.columns else np.zeros(len(data))
        ask_price = data['ask_price'].values if 'ask_price' in data.columns else np.zeros(len(data))
        bid_size = data['bid_size'].values if 'bid_size' in data.columns else np.zeros(len(data))
        ask_size = data['ask_size'].values if 'ask_size' in data.columns else np.zeros(len(data))

        trade_price = data['trade_price'].values if 'trade_price' in data.columns else np.full(len(data), np.nan)
        trade_size = data['trade_size'].values if 'trade_size' in data.columns else np.zeros(len(data))

        if 'trade_side' in data.columns:
            trade_side = data['trade_side'].astype(str).str.lower().values
        else:
            trade_side = np.full(len(data), 'buy')

        has_depth = 'bid_price_0' in data.columns
        depth_bids_p = depth_bids_s = depth_asks_p = depth_asks_s = None

        if has_depth:
            n_depth = 0
            while f'bid_price_{n_depth}' in data.columns:
                n_depth += 1

            zeros = np.zeros(len(data))
            depth_bids_p = np.column_stack([
                data.get(f'bid_price_{i}', zeros) for i in range(n_depth)
            ])
            depth_bids_s = np.column_stack([
                data.get(f'bid_size_{i}', zeros) for i in range(n_depth)
            ])
            depth_asks_p = np.column_stack([
                data.get(f'ask_price_{i}', zeros) for i in range(n_depth)
            ])
            depth_asks_s = np.column_stack([
                data.get(f'ask_size_{i}', zeros) for i in range(n_depth)
            ])

        n = len(data)
        rows: List[_Row] = []
        for i in range(n):
            ts = timestamps[i]
            rows.append(_Row(
                timestamp=ts,
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

        return rows

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        strategy: StrategyDefinition,
        params: Optional[Dict[str, Any]] = None,
    ) -> BacktestMetrics:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with columns: timestamp, bid_price, ask_price,
                  bid_size, ask_size, trade_price, trade_size, trade_side
            strategy: Strategy to test
            params: Override strategy parameters

        Returns:
            BacktestMetrics
        """
        self.reset()

        if params:
            strategy = self._apply_params(strategy, params)

        rows = self._preprocess(data)
        n_rows = len(rows)
        logger.debug(f"Running backtest with {n_rows} data points")
        
        # DEPTH CHECK - Verify data loading
        if len(rows) > 0:
            sample = rows[0]
            logger.info(f"DEPTH CHECK: Has depth={sample._has_depth}, Levels={sample._n_depth}")
            if not sample._has_depth:
                logger.error("WARNING: No depth data detected in backtest data! Order book will have only best bid/ask.")
            else:
                logger.info(f"✓ Depth data loaded successfully ({sample._n_depth} levels per row)")

        first_timestamp: Optional[datetime] = None
        state: Optional[OrderFlowState] = None
        timestamp = None

        pattern_every = self.PATTERN_DETECTION_EVERY_N
        equity_every = self.EQUITY_SAMPLE_EVERY_N
        vp_every = self.VOLUME_PROFILE_EVERY_N

        for tick_idx in range(n_rows):
            row = rows[tick_idx]
            timestamp = row.timestamp
            
            # DEBUG: First 5 iterations - verify order book construction
            if tick_idx < 5:
                logger.debug(f"DEBUG Row {tick_idx}: Has depth={row._has_depth}, Levels={row._n_depth}, "
                           f"Best bid={row.bid_price}, Best ask={row.ask_price}")

            if first_timestamp is None:
                first_timestamp = timestamp
                self.equity_curve.append((first_timestamp, self.initial_capital))

            # Daily reset
            self._handle_daily_reset(timestamp)

            # Equity floor check
            current_equity = self._calculate_equity_fast()
            if current_equity < self.equity_floor:
                logger.warning(f"Equity floor breached. Halting.")
                break

            # Build order book
            order_book = self._build_order_book_fast(row, timestamp)

            # Build trades
            trades = self._build_trades_fast(row, timestamp)

            # FIX: REMOVED _accumulate_footprint() — FeatureEngine handles this

            # Update feature engine (single source of truth)
            run_patterns = (tick_idx % pattern_every == 0)
            run_vp = (tick_idx % vp_every == 0)

            state = self.feature_engine.update(
                order_book, trades,
                detect_patterns=run_patterns,
                compute_volume_profile=run_vp,
            )

            # FIX: REMOVED state.regime override — FeatureEngine's classifier is used

            # Update open position
            if self.position:
                self._update_position(state)
                self._update_trailing_stop(state)

                if tick_idx > self._entry_tick_idx:  # Only gate the EXIT check
                    exit_reason = self._check_exit_conditions(state, timestamp)
                    if exit_reason:
                        self._close_position(state, timestamp, exit_reason, strategy)

            # Warmup gate
            elapsed = (timestamp - first_timestamp).total_seconds()
            if elapsed < self.warmup_seconds:
                if tick_idx % equity_every == 0:
                    self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                continue

            # Signal evaluation (only when flat)
            if not self.position:
                if not self._cooldown_passed(timestamp):
                    if tick_idx % equity_every == 0:
                        self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                    continue

                signal = strategy.evaluate(state)

                if signal and signal.is_actionable:
                    if self._is_duplicate_signal(signal, strategy, timestamp):
                        if tick_idx % equity_every == 0:
                            self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                        continue

                    risk_action, adjusted_signal, reason = self.risk_manager.check_signal(
                        signal, state.order_book.mid_price
                    )

                    if risk_action == RiskAction.HALT_TRADING:
                        self._halts += 1
                        continue
                    elif risk_action == RiskAction.REJECT:
                        self._signals_rejected += 1
                        continue
                    elif risk_action == RiskAction.REDUCE_SIZE:
                        self._signals_reduced += 1
                        self._open_position(adjusted_signal, state, timestamp, strategy, risk_action.name)
                        self._entry_tick_idx = tick_idx #ADD THIS: Initialize entry tick tracker
                    elif risk_action == RiskAction.ALLOW:
                        self._open_position(adjusted_signal or signal, state, timestamp, strategy, risk_action.name)
                        self._entry_tick_idx = tick_idx #ADD THIS: Initialize entry tick tracker

            # Equity curve (down-sampled)
            if tick_idx % equity_every == 0:
                self.equity_curve.append((timestamp, self._calculate_equity_fast()))

        # Close remaining position
        if self.position and state is not None and timestamp is not None:
            self._close_position(state, timestamp, "end_of_backtest", strategy)

        if timestamp is not None:
            self.equity_curve.append((timestamp, self._calculate_equity_fast()))

        return self._calculate_metrics()

    # ------------------------------------------------------------------
    # Parameter application
    # ------------------------------------------------------------------

    def _apply_params(
        self,
        strategy: StrategyDefinition,
        params: Dict[str, Any],
    ) -> StrategyDefinition:
        """
        Apply optimized parameters to strategy definition.
        CRITICAL: Handles param_key mapping and symmetric ranges correctly.
        DEBUG: Logs every parameter application for verification.
        """
        strategy = copy.deepcopy(strategy)
        applied_keys = set()
        
        logger.debug(f"[_apply_params] Applying {len(params)} parameters to {strategy.name}")
        
        for param_name, value in params.items():
            if value is None:
                logger.debug(f"[_apply_params] Skipping {param_name}: None value")
                continue
            
            # DEBUG: Log parameter being processed
            logger.debug(f"[_apply_params] Processing: {param_name} = {value}")
            
            # 1. Direct strategy attributes (regime multipliers, risk params)
            # CRITICAL: Check if this is a direct attribute of the strategy
            if hasattr(strategy, param_name):
                current_val = getattr(strategy, param_name)
                setattr(strategy, param_name, value)
                applied_keys.add(param_name)
                logger.debug(f"[_apply_params] Applied to attribute: {param_name} ({current_val} -> {value})")
                continue
            
            # 2. Entry conditions and filters via param_key
            all_conditions = list(strategy.entry_conditions) + list(strategy.filters)
            matched = False
            
            for cond in all_conditions:
                if getattr(cond, "param_key", None) != param_name:
                    continue
                
                # CRITICAL: Handle symmetric "between" ranges
                if param_name.endswith("_range") and cond.operator == "between":
                    abs_val = abs(value)
                    old_low, old_high = cond.threshold, cond.threshold_high
                    cond.threshold = -abs_val
                    cond.threshold_high = abs_val
                    logger.debug(f"[_apply_params] Applied symmetric range: {param_name} = ±{abs_val} (was {old_low} to {old_high})")
                # Handle asymmetric high threshold
                elif param_name.endswith("_high") and hasattr(cond, "threshold_high"):
                    cond.threshold_high = value
                    logger.debug(f"[_apply_params] Applied threshold_high: {param_name} = {value}")
                # Standard threshold
                else:
                    old_val = cond.threshold
                    cond.threshold = value
                    logger.debug(f"[_apply_params] Applied threshold: {param_name} = {value} (was {old_val})")
                
                applied_keys.add(param_name)
                matched = True
                break
            
            if not matched:
                logger.warning(f"[_apply_params] Parameter {param_name} not matched to any condition or attribute")
        
        # CRITICAL: Validate all conditions after modification
        logger.debug(f"[_apply_params] Validating {len(strategy.entry_conditions)} entry conditions and {len(strategy.filters)} filters")
        for cond in strategy.entry_conditions + strategy.filters:
            try:
                cond.validate()
            except ValueError as e:
                logger.error(f"[_apply_params] Validation failed for {cond.feature}: {e}")
                raise
        
        # Warn about unmapped parameters
        unmapped = set(params.keys()) - applied_keys
        if unmapped:
            logger.warning(f"[_apply_params] UNMAPPED PARAMETERS (will have no effect): {unmapped}")
        
        # Guard: min_conditions_satisfied must never exceed the number of actual
        # entry conditions \u2014 otherwise the strategy can never fire.  # [FIXED]
        entry_count = len(strategy.entry_conditions)
        if strategy.min_conditions_satisfied > entry_count:
            logger.warning(
                f"[_apply_params] min_conditions_satisfied={strategy.min_conditions_satisfied} "
                f"exceeds entry condition count={entry_count}. "
                f"Clamping to {entry_count}."
            )
            strategy.min_conditions_satisfied = entry_count
        
        logger.info(f"[_apply_params] Successfully applied {len(applied_keys)}/{len(params)} parameters to {strategy.name}")
        return strategy

    # ------------------------------------------------------------------
    # Fast order book builder
    # ------------------------------------------------------------------

    def _build_order_book_fast(self, row: _Row, timestamp: datetime) -> OrderBook:
        """Build OrderBook from pre-extracted _Row."""
        best_bid = row.bid_price
        best_ask = row.ask_price
        best_bid_size = row.bid_size
        best_ask_size = row.ask_size

        if best_bid == 0 and best_ask == 0:
            tp = row.trade_price
            if not math.isnan(tp) and tp > 0:
                spread_est = tp * 0.0001
                best_bid = tp - spread_est / 2
                best_ask = tp + spread_est / 2
                best_bid_size = row.trade_size if row.trade_size > 0 else 1.0
                best_ask_size = best_bid_size

        n_levels = 20
        bids: List[PriceLevel] = []
        asks: List[PriceLevel] = []

        if row._has_depth:
            for i in range(min(n_levels, row._n_depth)):
                bp = float(row._depth_bids_p[i])
                bs = float(row._depth_bids_s[i])
                ap = float(row._depth_asks_p[i])
                a_s = float(row._depth_asks_s[i])
                if bp > 0 and bs > 0:
                    bids.append(PriceLevel(price=bp, size=bs, timestamp=timestamp))
                if ap > 0 and a_s > 0:
                    asks.append(PriceLevel(price=ap, size=a_s, timestamp=timestamp))
        else:
            # NO SYNTHETIC DEPTH - Data missing real depth levels
            logger.warning(f"No depth data available at {timestamp}. Book contains only best bid/ask.")
            if best_bid > 0:
                bids.append(PriceLevel(price=best_bid, size=best_bid_size, timestamp=timestamp))
            if best_ask > 0:
                asks.append(PriceLevel(price=best_ask, size=best_ask_size, timestamp=timestamp))

        bids.sort(key=lambda l: l.price, reverse=True)
        asks.sort(key=lambda l: l.price)

        return OrderBook(timestamp=timestamp, bids=bids, asks=asks)

    # ------------------------------------------------------------------
    # Fast trade builder
    # ------------------------------------------------------------------

    def _build_trades_fast(self, row: _Row, timestamp: datetime) -> List[Trade]:
        """Build Trade list from pre-extracted _Row."""
        if math.isnan(row.trade_price) or row.trade_price <= 0:
            return []
        if math.isnan(row.trade_size) or row.trade_size <= 0:
            return []

        side = Side.BUY if str(row.trade_side).lower().strip() == 'buy' else Side.SELL
        return [Trade(
            timestamp=timestamp,
            price=row.trade_price,
            size=row.trade_size,
            side=side,
        )]

    # NOTE: _accumulate_footprint() REMOVED — FeatureEngine handles this

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _update_position(self, state: OrderFlowState) -> None:
        """Update position P&L using best bid/ask."""
        if not self.position:
            return

        book = state.order_book
        mid = book.mid_price

        if self.position.side == Side.BUY:
            mark_price = book.best_bid.price if book.best_bid else mid
            self.position.unrealized_pnl = (mark_price - self.position.entry_price) * self.position.size
        else:
            mark_price = book.best_ask.price if book.best_ask else mid
            self.position.unrealized_pnl = (self.position.entry_price - mark_price) * self.position.size

        self.position.highest_price = max(self.position.highest_price, mid)
        if self.position.lowest_price <= 0:
            self.position.lowest_price = mid
        else:
            self.position.lowest_price = min(self.position.lowest_price, mid)

    def _update_trailing_stop(self, state: OrderFlowState) -> None:
        """Activate and ratchet trailing stop."""
        if not self.position:
            return

        book = state.order_book
        activation = self.position.trailing_stop_activation_pct
        if activation <= 0:
            return

        if self.position.side == Side.BUY:
            mark = book.best_bid.price if book.best_bid else book.mid_price
            move_pct = (mark - self.position.entry_price) / self.position.entry_price
            if move_pct >= activation:
                self.position.trailing_stop_active = True
                trail_distance = activation * 0.5
                new_stop = mark * (1 - trail_distance)
                self.position.trailing_stop_price = max(
                    self.position.trailing_stop_price, new_stop
                )
                if self.position.trailing_stop_price > self.position.stop_loss:
                    self.position.stop_loss = self.position.trailing_stop_price
        else:
            mark = book.best_ask.price if book.best_ask else book.mid_price
            move_pct = (self.position.entry_price - mark) / self.position.entry_price
            if move_pct >= activation:
                self.position.trailing_stop_active = True
                trail_distance = activation * 0.5
                new_stop = mark * (1 + trail_distance)
                if self.position.trailing_stop_price <= 0:
                    self.position.trailing_stop_price = new_stop
                else:
                    self.position.trailing_stop_price = min(
                        self.position.trailing_stop_price, new_stop
                    )
                if self.position.trailing_stop_price < self.position.stop_loss:
                    self.position.stop_loss = self.position.trailing_stop_price

    # ------------------------------------------------------------------
    # Exit conditions
    # ------------------------------------------------------------------

    def _check_exit_conditions(
        self,
        state: OrderFlowState,
        timestamp: datetime,
    ) -> Optional[str]:
        """Check if position should be closed."""
        # [APPLIED] No time-based exit logic exists here - exits are flow-based (SL/TP/absorption/divergence/etc)
        # max_holding_seconds=86400 in StrategyDefinition makes time exits effectively unlimited
        if not self.position:
            return None

        book = state.order_book
        features = state.features

        if self.position.side == Side.BUY:
            exit_check_price = book.best_bid.price if book.best_bid else book.mid_price
        else:
            exit_check_price = book.best_ask.price if book.best_ask else book.mid_price

        # 1) Hard stop loss
        if self.position.side == Side.BUY:
            if exit_check_price <= self.position.stop_loss:
                return "stop_loss"
        else:
            if exit_check_price >= self.position.stop_loss:
                return "stop_loss"

        # 2) Take profit
        if self.position.side == Side.BUY:
            if exit_check_price >= self.position.take_profit:
                return "take_profit"
        else:
            if exit_check_price <= self.position.take_profit:
                return "take_profit"

        # 3a) Absorption against
        if state.absorptions:
            latest_abs = state.absorptions[-1]
            if (self.position.side == Side.BUY and
                    latest_abs.absorbing_side == Side.SELL and
                    latest_abs.strength >= 0.6):
                return "absorption_against"
            if (self.position.side == Side.SELL and
                    latest_abs.absorbing_side == Side.BUY and
                    latest_abs.strength >= 0.6):
                return "absorption_against"

        # 3b) Delta divergence against
        delta_div = features.get("delta_divergence_60s", 0)
        if delta_div == 1.0:
            delta_60 = features.get("delta_60s", 0)
            if self.position.side == Side.BUY and delta_60 < 0:
                return "delta_divergence_against"
            if self.position.side == Side.SELL and delta_60 > 0:
                return "delta_divergence_against"

        # 3c) Exhaustion
        if self.position.side == Side.BUY:
            if features.get("buying_exhaustion", 0) >= 1.0:
                return "buying_exhaustion"
        else:
            if features.get("selling_exhaustion", 0) >= 1.0:
                return "selling_exhaustion"

        # 3d) Sweep against
        if state.sweeps:
            latest_sweep = state.sweeps[-1]
            if (self.position.side == Side.BUY and
                    latest_sweep.direction == Side.SELL and
                    latest_sweep.reversal_strength < 0.4):
                return "sweep_against_long"
            if (self.position.side == Side.SELL and
                    latest_sweep.direction == Side.BUY and
                    latest_sweep.reversal_strength < 0.4):
                return "sweep_against_short"

        # 3e) Book pressure collapse
        net_pressure = features.get("net_pressure", 0)
        if self.position.side == Side.BUY and net_pressure < -0.5:
            bid_depth = features.get("bid_depth_10", 0)
            ask_depth = features.get("ask_depth_10", 0)
            if ask_depth > 0 and bid_depth / (ask_depth + 1e-9) < 0.3:
                return "book_pressure_collapse"
        if self.position.side == Side.SELL and net_pressure > 0.5:
            bid_depth = features.get("bid_depth_10", 0)
            ask_depth = features.get("ask_depth_10", 0)
            if bid_depth > 0 and ask_depth / (bid_depth + 1e-9) < 0.3:
                return "book_pressure_collapse"

        return None

    # ------------------------------------------------------------------
    # Open/Close position
    # ------------------------------------------------------------------

    def _open_position(
        self,
        signal: Signal,
        state: OrderFlowState,
        timestamp: datetime,
        strategy: StrategyDefinition,
        risk_action: str = "ALLOW",
    ) -> None:
        """Open a new position using best bid/ask fill model."""
        book = state.order_book
        best_bid = book.best_bid.price if book.best_bid else book.mid_price
        best_ask = book.best_ask.price if book.best_ask else book.mid_price

        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            entry_price = best_ask * (1 + self.slippage_pct)
            side = Side.BUY
        else:
            entry_price = best_bid * (1 - self.slippage_pct)
            side = Side.SELL

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

        self.position = Position(
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

        self.risk_manager.record_trade_opened(entry_price, size, side)
        self._last_signal_time = timestamp
        self._last_signal_strategy = strategy.name

    def _close_position(
        self,
        state: OrderFlowState,
        timestamp: datetime,
        reason: str,
        strategy: StrategyDefinition,
    ) -> None:
        """Close current position using best bid/ask fill model."""
        if not self.position:
            return

        book = state.order_book
        best_bid = book.best_bid.price if book.best_bid else book.mid_price
        best_ask = book.best_ask.price if book.best_ask else book.mid_price

        adverse_slip = (self.slippage_pct + self.sl_extra_slippage_pct 
                       if reason == "stop_loss" else self.slippage_pct)

        if self.position.side == Side.BUY:
            exit_price = best_bid * (1 - adverse_slip)
            gross_pnl = (exit_price - self.position.entry_price) * self.position.size
        else:
            exit_price = best_ask * (1 + adverse_slip)
            gross_pnl = (self.position.entry_price - exit_price) * self.position.size

        exit_value = exit_price * self.position.size
        exit_fee = exit_value * self.fee_pct
        pnl = gross_pnl - exit_fee

        logger.info(
            f"EXIT: {reason} | Entry: {self.position.entry_price} | Exit: {exit_price} | PnL: {pnl}"
        )

        notional = self.position.entry_price * self.position.size
        pnl_pct = pnl / notional if notional > 0 else 0.0

        # Suspicious PnL rejection
        if abs(pnl_pct) > self.suspicious_pnl_pct and reason != "end_of_backtest":
            self._suspicious_rejected += 1
            self.capital += self.position.allocated_capital
            self.position = None
            return

        self.capital += (self.position.allocated_capital - self.position.entry_fee) + pnl

        duration = (timestamp - self.position.entry_time).total_seconds()

        self.closed_trades.append(ClosedTrade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            size=self.position.size,
            side=self.position.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            duration_seconds=duration,
            signal_confidence=self.position.signal.confidence,
            risk_action=self.position.risk_action,
        ))

        self.risk_manager.record_trade_closed(pnl)
        self._last_trade_close_time = timestamp
        self.position = None

    # ------------------------------------------------------------------
    # Signal de-duplication / cooldown
    # ------------------------------------------------------------------

    def _is_duplicate_signal(
        self,
        signal: Signal,
        strategy: StrategyDefinition,
        timestamp: datetime,
    ) -> bool:
        if self._last_signal_time is None:
            return False
        if self._last_signal_strategy != strategy.name:
            return False
        elapsed = (timestamp - self._last_signal_time).total_seconds()
        return elapsed < 5.0

    def _cooldown_passed(self, timestamp: datetime) -> bool:
        if self._last_trade_close_time is None:
            return True
        elapsed = (timestamp - self._last_trade_close_time).total_seconds()
        return elapsed >= self.min_time_between_trades_sec

    # ------------------------------------------------------------------
    # Daily resets
    # ------------------------------------------------------------------

    def _handle_daily_reset(self, timestamp: datetime) -> None:
        day = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        if self._current_day is None:
            self._current_day = day
            return

        if day != self._current_day:
            self._current_day = day
            self.risk_manager.state.daily_pnl = 0.0
            self.risk_manager.state.trades_today = 0
            self.risk_manager.state.trades_this_hour = 0

            if (self.risk_manager.state.trading_halted and
                    "Daily" in self.risk_manager.state.halt_reason):
                self.risk_manager.resume_trading()
                logger.info("New trading day — daily halt lifted.")

    # ------------------------------------------------------------------
    # Equity helpers
    # ------------------------------------------------------------------

    def _calculate_equity_fast(self) -> float:
        equity = self.capital
        if self.position:
            equity += self.position.unrealized_pnl
            equity += (self.position.allocated_capital - self.position.entry_fee)
        return equity

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()

        if not self.closed_trades:
            return metrics

        metrics.total_trades = len(self.closed_trades)

        final_equity = self._calculate_equity_fast()
        metrics.total_return = final_equity - self.initial_capital
        metrics.total_return_pct = metrics.total_return / self.initial_capital

        winning = [t for t in self.closed_trades if t.pnl > 0]
        losing = [t for t in self.closed_trades if t.pnl <= 0]

        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = len(winning) / len(self.closed_trades)

        metrics.avg_win = float(np.mean([t.pnl for t in winning])) if winning else 0
        metrics.avg_loss = float(np.mean([t.pnl for t in losing])) if losing else 0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        pnls = [t.pnl for t in self.closed_trades]
        metrics.expectancy = float(np.mean(pnls))

        durations = [t.duration_seconds for t in self.closed_trades]
        metrics.avg_trade_duration_seconds = float(np.mean(durations))

        max_consec = 0
        current_consec = 0
        for trade in self.closed_trades:
            if trade.pnl <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        metrics.max_consecutive_losses = max_consec

        if len(self.equity_curve) > 1:
            timestamps_eq = [e[0] for e in self.equity_curve]
            equities = [e[1] for e in self.equity_curve]

            peak = equities[0]
            max_dd = 0.0
            for eq in equities:
                peak = max(peak, eq)
                dd = (peak - eq) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            metrics.max_drawdown_pct = max_dd
            metrics.max_drawdown = max_dd * self.initial_capital

            equity_series = pd.Series(
                equities,
                index=pd.DatetimeIndex(timestamps_eq),
            )
            daily_equity = equity_series.resample('D').last().dropna()

            if len(daily_equity) > 2:
                daily_returns = daily_equity.pct_change().dropna().values

                if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                    metrics.sharpe_ratio = float(
                        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                    )

                    downside = daily_returns[daily_returns < 0]
                    if len(downside) > 0:
                        downside_std = np.std(downside)
                        if downside_std > 0:
                            metrics.sortino_ratio = float(
                                np.mean(daily_returns) / downside_std * np.sqrt(252)
                            )

            if metrics.max_drawdown_pct > 0:
                metrics.calmar_ratio = metrics.total_return_pct / metrics.max_drawdown_pct

            total_seconds = (timestamps_eq[-1] - timestamps_eq[0]).total_seconds()
            total_days = total_seconds / 86400.0
            if total_days > 0:
                metrics.trades_per_day = metrics.total_trades / total_days
                metrics.annualized_return = metrics.total_return_pct * (365.0 / total_days)

        metrics.signals_rejected_by_risk = self._signals_rejected
        metrics.signals_reduced_by_risk = self._signals_reduced
        metrics.halts_triggered = self._halts
        metrics.suspicious_pnl_rejected = self._suspicious_rejected

        return metrics


# ======================================================================
# Walk-Forward Validator — Fixed for Parallel Safety
# ======================================================================

class WalkForwardValidator:
    """
    Walk-forward validation with proper isolation for parallel optimization.
    """

    def __init__(
        self,
        train_days: int = 60,
        test_days: int = 14,
        step_days: int = 7,
        min_oos_trades: int = 10,  # NEW: filter out useless folds
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.min_oos_trades = min_oos_trades

    def generate_splits(
        self,
        data: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits"""
        data = data.sort_values('timestamp').reset_index(drop=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        start_date = data['timestamp'].min()
        end_date = data['timestamp'].max()

        splits = []
        current_start = start_date

        while True:
            train_end = current_start + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)

            if test_end > end_date:
                break

            train_data = data[
                (data['timestamp'] >= current_start) &
                (data['timestamp'] < train_end)
            ]

            test_data = data[
                (data['timestamp'] >= train_end) &
                (data['timestamp'] < test_end)
            ]

            if len(train_data) > 100 and len(test_data) > 10:
                splits.append((train_data, test_data))

            current_start += timedelta(days=self.step_days)

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def validate(
        self,
        data: pd.DataFrame,
        strategy: StrategyDefinition,
        optimizer,
        n_trials_per_fold: int = 50,
        feature_config: Optional[FeatureConfig] = None,  # NEW: pass config
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation with proper trial isolation.

        FIX: Each trial creates its own BacktestEngine to avoid race conditions
        when n_jobs > 1 in optimization.
        """
        splits = self.generate_splits(data)

        results: Dict[str, Any] = {
            "folds": [],
            "oos_metrics": [],
            "params_stability": {},
            "skipped_folds": 0,  # NEW: track filtered folds
        }

        all_params: List[Dict] = []
        valid_oos_sharpes: List[float] = []  # NEW: for proper averaging

        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Processing fold {i+1}/{len(splits)}")

            # FIX: Create engine INSIDE backtest_fn for thread safety
            # When n_jobs > 1, multiple trials run in parallel
            def train_backtest(params, _train_data=train_data, _fc=feature_config):
                # Each trial gets its own engine instance
                engine = BacktestEngine(feature_config=_fc)
                metrics = engine.run(_train_data, strategy, params)
                return {
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "profit_factor": metrics.profit_factor,
                    "win_rate": metrics.win_rate,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "total_trades": metrics.total_trades,
                }

            # Optimize with n_jobs=1 for walk-forward (safer, clearer logs)
            # The inner optimization can still be fast with TPE
            opt_result = optimizer.optimize(
                train_backtest, 
                n_trials=n_trials_per_fold,
                n_jobs=1  # FIX: Force sequential for fold stability
            )
            best_params = opt_result.best_params
            all_params.append(best_params)

            # OOS test with fresh engine
            oos_engine = BacktestEngine(feature_config=feature_config)
            oos_metrics = oos_engine.run(test_data, strategy, best_params)

            # FIX: Filter out folds with insufficient trades
            if oos_metrics.total_trades < self.min_oos_trades:
                logger.warning(
                    f"Fold {i+1} skipped: OOS trades ({oos_metrics.total_trades}) "
                    f"< min ({self.min_oos_trades})"
                )
                results["skipped_folds"] += 1
                continue

            fold_result = {
                "fold": i,
                "train_score": opt_result.best_score,
                "oos_sharpe": oos_metrics.sharpe_ratio,
                "oos_return": oos_metrics.total_return_pct,
                "oos_win_rate": oos_metrics.win_rate,
                "oos_trades": oos_metrics.total_trades,
                "oos_max_dd": oos_metrics.max_drawdown_pct,
                "best_params": best_params,
            }

            results["folds"].append(fold_result)
            results["oos_metrics"].append(oos_metrics)
            valid_oos_sharpes.append(oos_metrics.sharpe_ratio)

            logger.info(
                f"Fold {i+1}: Train={opt_result.best_score:.2f}, "
                f"OOS Sharpe={oos_metrics.sharpe_ratio:.2f}, "
                f"OOS Trades={oos_metrics.total_trades}"
            )

        # Parameter stability
        if all_params:
            param_names = list(all_params[0].keys())
            for param in param_names:
                values = [p.get(param, 0) for p in all_params]
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                results["params_stability"][param] = {
                    "mean": mean_val,
                    "std": std_val,
                    "cv": std_val / (abs(mean_val) + 1e-9),
                }

        # FIX: Use only valid folds for summary
        if valid_oos_sharpes:
            results["summary"] = {
                "avg_oos_sharpe": float(np.mean(valid_oos_sharpes)),
                "std_oos_sharpe": float(np.std(valid_oos_sharpes)),
                "min_oos_sharpe": float(np.min(valid_oos_sharpes)),
                "max_oos_sharpe": float(np.max(valid_oos_sharpes)),
                "pct_profitable_folds": sum(1 for s in valid_oos_sharpes if s > 0) / len(valid_oos_sharpes),
                "total_folds": len(splits),
                "valid_folds": len(valid_oos_sharpes),
                "skipped_folds": results["skipped_folds"],
            }
        else:
            results["summary"] = {
                "avg_oos_sharpe": 0.0,
                "std_oos_sharpe": 0.0,
                "pct_profitable_folds": 0.0,
                "total_folds": len(splits),
                "valid_folds": 0,
                "skipped_folds": results["skipped_folds"],
                "error": "No valid folds (all had insufficient OOS trades)",
            }

        logger.info(f"Walk-Forward Summary:")
        logger.info(f"  Valid folds: {results['summary']['valid_folds']}/{results['summary']['total_folds']}")
        logger.info(f"  Avg OOS Sharpe: {results['summary']['avg_oos_sharpe']:.2f}")
        logger.info(f"  Profitable folds: {results['summary']['pct_profitable_folds']*100:.1f}%")

        return results
    
    
# ======================================================================
# Section 2 core/feature_engine.py
# ======================================================================
"""
Feature Engine (v6.12 Expert Optimized)
Computes ALL order flow metrics from raw data.

Key Optimizations:
- O(log N) window lookups via bisect on Lists
- O(1) incremental CVD tracking
- Complete pattern-to-feature bridge (critical fix)
- Full regime classification
- Temporal feature snapshots
- Cloud-ready memory management
"""

import bisect
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .data_structures import (
    OrderBook, Trade, Side, FootprintBar, VolumeProfile,
    Imbalance, Absorption, LiquiditySweep, IcebergOrder,
    PriceLevel, OrderFlowState, Regime
)


@dataclass
class FeatureConfig:
    """v6.12 Expert Configuration for BTCUSD"""
    windows: List[int] = field(default_factory=lambda: [15, 30, 60, 300, 600, 900])
    book_depth_levels: int = 20
    volume_profile_lookback: int = 3600
    tick_size: float = 0.0001  # 0.5 for BTCUSD specific (was 0.01 - too granular) Now 0.0001 for XRP
    
    # Absorption detection
    absorption_volume_multiplier: float = 2.0
    absorption_price_threshold_pct: float = 0.0001
    absorption_min_duration_sec: float = 1.0
    
    # Imbalance detection
    imbalance_ratio_threshold: float = 3.0
    stacked_imbalance_min_levels: int = 3
    
    # Iceberg detection
    iceberg_refill_threshold: int = 3
    iceberg_size_threshold: float = 1.0
    
    # Sweep detection
    sweep_speed_threshold_sec: float = 5.0
    sweep_reversal_threshold_pct: float = 0.002
    
    # Footprint bar generation
    footprint_bar_duration_sec: int = 5
    
    # Regime classifier parameters
    regime_lookback_seconds: int = 1800  # 30 minutes
    regime_vol_low_threshold: float = 0.003
    regime_vol_high_threshold: float = 0.008
    regime_trend_strength_threshold: float = 0.5
    low_liquidity_spread_bps: float = 20.0
    low_liquidity_min_depth_10: float = 1.0
    high_volatility_multiplier: float = 1.5
    breakout_volume_accel_threshold: float = 1.0
    breakout_va_breakout_threshold: float = 0.5
    breakout_trend_strength_threshold: float = 0.4
    accum_dist_pressure_threshold: float = 0.5


class RegimeClassifier:
    """
    Expert Market Regime Classifier
    
    Evaluates volatility, trend strength, and liquidity to determine market state.
    Priority: LOW_LIQUIDITY > HIGH_VOL > BREAKOUT > TRENDING > ACCUM/DIST > RANGING
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def classify(
        self,
        trade_history: List[Trade],
        book_features: Dict[str, float],
        current_ts: datetime
    ) -> Regime:
        """Classify market regime from price action + order flow features."""
        if len(trade_history) < 100:
            return Regime.UNKNOWN
        
        # Get lookback window via binary search
        cutoff = current_ts - timedelta(seconds=self.config.regime_lookback_seconds)
        timestamps = [t.timestamp for t in trade_history]
        idx = bisect.bisect_left(timestamps, cutoff)
        window_trades = trade_history[idx:]
        
        if len(window_trades) < 50:
            return Regime.UNKNOWN
        
        # Calculate volatility and trend from log returns
        prices = [t.price for t in window_trades[:500]]  # Cap at 500 for speed
        if len(prices) < 10:
            return Regime.UNKNOWN
        
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                log_returns.append(np.log(prices[i] / prices[i - 1]))
        
        if len(log_returns) < 10:
            return Regime.UNKNOWN
        
        volatility = float(np.std(log_returns))
        
        # Trend strength: net move / total range (0 to 1)
        start_px, end_px = prices[0], prices[-1]
        price_range = max(prices) - min(prices)
        net_move = abs(end_px - start_px)
        trend_strength = net_move / (price_range + 1e-9)
        price_dir = float(np.sign(end_px - start_px))
        
        # Get book features for liquidity checks
        spread_bps = book_features.get("spread_bps", 0)
        depth_10 = book_features.get("bid_depth_10", 0) + book_features.get("ask_depth_10", 0)
        net_pressure = book_features.get("net_pressure", 0)
        vol_accel = book_features.get("volume_acceleration", 1.0)
        
        cfg = self.config
        
        # 1. LOW_LIQUIDITY (highest priority - avoid trading)
        if spread_bps > cfg.low_liquidity_spread_bps:
            return Regime.LOW_LIQUIDITY
        if depth_10 < cfg.low_liquidity_min_depth_10 * 2:
            return Regime.LOW_LIQUIDITY
        
        # 2. HIGH_VOLATILITY (widen stops)
        if volatility > cfg.regime_vol_high_threshold * cfg.high_volatility_multiplier:
            return Regime.HIGH_VOLATILITY
        
        # 3. BREAKOUT (momentum opportunity)
        if (volatility >= cfg.regime_vol_low_threshold
                and trend_strength >= cfg.breakout_trend_strength_threshold
                and vol_accel > cfg.breakout_volume_accel_threshold):
            return Regime.BREAKOUT
        
        # 4. TRENDING (follow the trend)
        if (volatility >= cfg.regime_vol_low_threshold
                and trend_strength >= cfg.regime_trend_strength_threshold):
            return Regime.TRENDING_UP if price_dir > 0 else Regime.TRENDING_DOWN
        
        # 5. ACCUMULATION / DISTRIBUTION (low vol, directional pressure)
        if volatility < cfg.regime_vol_low_threshold:
            if abs(net_pressure) > cfg.accum_dist_pressure_threshold:
                return Regime.ACCUMULATION if net_pressure > 0 else Regime.DISTRIBUTION
        
        # 6. RANGING (default for low vol, weak trend)
        if (volatility < cfg.regime_vol_high_threshold
                and trend_strength < cfg.regime_trend_strength_threshold):
            return Regime.RANGING
        
        return Regime.RANGING


class FeatureEngine:
    """
    Optimized Feature Engine for Cloud Deployment
    
    Computes comprehensive order flow features:
    1. Order Book Features (depth, imbalance, slope, pressure)
    2. Trade Flow Features (delta, aggression, intensity) - windowed
    3. Volume Profile Features (POC, Value Area, distribution)
    4. Footprint Features (delta at price, imbalances)
    5. Pattern Features (absorption, sweeps, icebergs) - THE BRIDGE
    6. Composite Features (momentum, divergence, relative strength)
    7. Regime Classification (market state detection)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Using Lists instead of deque for native bisect compatibility
        self.trade_history: List[Trade] = []
        self.book_history: List[OrderBook] = []
        self.footprint_bars: List[FootprintBar] = []
        
        # Tracking for pattern detection
        self.price_level_activity: Dict[float, Dict] = {}
        self.potential_icebergs: Dict[float, Dict] = {}
        
        # Cached computations
        self._volume_profile_cache: Optional[VolumeProfile] = None
        self._cache_timestamp: Optional[datetime] = None
        self._vp_features_cache: Optional[Dict[str, float]] = None
        
        # O(1) incremental CVD
        self._cvd: float = 0.0
        
        # Cached pattern results
        self._cached_absorptions: List[Absorption] = []
        self._cached_sweeps: List[LiquiditySweep] = []
        self._cached_icebergs: List[IcebergOrder] = []
        
        # Data-driven timestamp (not wall-clock)
        self._current_timestamp: Optional[datetime] = None
        
        # Footprint bar state
        self._current_footprint_bar: Optional[FootprintBar] = None
        self._footprint_bar_start: Optional[datetime] = None
        
        # Temporal feature snapshots
        self._feature_snapshots: List[Tuple[datetime, Dict[str, float]]] = []
        
        # Expert regime classifier
        self._regime_classifier = RegimeClassifier(self.config)
    
    def reset(self) -> None:
        """Reset all state - call between backtest runs for clean isolation."""
        self.trade_history.clear()
        self.book_history.clear()
        self.footprint_bars.clear()
        self.price_level_activity.clear()
        self.potential_icebergs.clear()
        self._volume_profile_cache = None
        self._cache_timestamp = None
        self._vp_features_cache = None
        self._cvd = 0.0
        self._cached_absorptions = []
        self._cached_sweeps = []
        self._cached_icebergs = []
        self._current_timestamp = None
        self._current_footprint_bar = None
        self._footprint_bar_start = None
        self._feature_snapshots = []
    
    def update(
        self,
        order_book: OrderBook,
        trades: List[Trade],
        detect_patterns: bool = True,
        compute_volume_profile: bool = True,
    ) -> OrderFlowState:
        """
        Main entry point - update with new data and compute all features.
        
        Args:
            order_book: Current order book snapshot
            trades: Trades since last update
            detect_patterns: If False, reuse cached pattern results (backtest throttle)
            compute_volume_profile: If False, reuse cached volume profile (backtest throttle)
        """
        # Data-driven timestamp
        self._current_timestamp = order_book.timestamp
        
        # Store history with incremental CVD
        self.book_history.append(order_book)
        for trade in trades:
            self.trade_history.append(trade)
            self._cvd += trade.size if trade.side == Side.BUY else -trade.size
            self._update_footprint_bars(trade)
        
        # Cloud memory management - trim when large
        if len(self.trade_history) > 20000:
            self.trade_history = self.trade_history[-10000:]
        if len(self.book_history) > 2000:
            self.book_history = self.book_history[-1000:]
        
        # Create state
        state = OrderFlowState(
            timestamp=order_book.timestamp,
            order_book=order_book,
            recent_trades=trades
        )
        
        # Compute base features
        state.features = self._compute_all_features(
            order_book, 
            trades,
            compute_volume_profile=compute_volume_profile
        )
        
        # Detect patterns (or reuse cache)
        if detect_patterns:
            self._cached_absorptions = self._detect_absorptions()
            state.imbalances = self._detect_imbalances(order_book)
            self._cached_sweeps = self._detect_liquidity_sweeps()
            self._cached_icebergs = self._detect_icebergs()
        else:
            state.imbalances = self._detect_imbalances(order_book)
        
        state.absorptions = self._cached_absorptions
        state.sweeps = self._cached_sweeps
        state.icebergs = self._cached_icebergs
        
        # Volume profile
        if compute_volume_profile:
            state.volume_profile = self._compute_volume_profile()
        else:
            state.volume_profile = self._volume_profile_cache
        
        # CRITICAL: Bridge detected patterns → strategy-consumable features
        state.features.update(self._compute_pattern_features(state))
        
        # Classify market regime
        state.regime = self._regime_classifier.classify(
            self.trade_history,
            state.features,
            self._current_timestamp
        )
        
        # Store temporal snapshot
        self._store_feature_snapshot(order_book.timestamp, state.features)
        
        return state
    
    # ==================== CORE FEATURE COMPUTATION ====================
    
    def _compute_all_features(
        self,
        book: OrderBook,
        trades: List[Trade],
        compute_volume_profile: bool = True
    ) -> Dict[str, float]:
        """Compute all feature categories"""
        features: Dict[str, float] = {}
        
        # 1. Order Book Features
        features.update(self._compute_book_features(book))
        
        # 2. Trade Flow Features (windowed) - O(log N) per window
        trade_times = [t.timestamp for t in self.trade_history]
        
        for window in self.config.windows:
            cutoff = self._current_timestamp - timedelta(seconds=window)
            idx = bisect.bisect_left(trade_times, cutoff)
            window_trades = self.trade_history[idx:]
            
            window_features = self._compute_trade_features(window_trades, window)
            for key, value in window_features.items():
                features[f"{key}_{window}s"] = value
        
        # 2b. Non-windowed CVD (composites reference "cvd" directly)
        features["cvd"] = self._cvd
        
        # 3. Volume Profile Features
        features.update(
            self._compute_volume_profile_features(use_cached=not compute_volume_profile)
        )
        
        # 4. Footprint Features
        features.update(self._compute_footprint_features())
        
        # 5. Composite/Derived Features (ALL 8 that strategies need)
        features.update(self._compute_composite_features(features))
        
        return features
    
    # ==================== ORDER BOOK FEATURES ====================
    
    def _compute_book_features(self, book: OrderBook) -> Dict[str, float]:
        """Order book structure features"""
        features: Dict[str, float] = {}
        
        if not book.bids or not book.asks:
            return self._empty_book_features()
        
        # Basic spread and pricing
        features["spread_bps"] = book.spread_bps
        features["mid_price"] = book.mid_price
        features["microprice"] = book.microprice
        features["microprice_vs_mid"] = (
            (book.microprice - book.mid_price) / book.mid_price 
            if book.mid_price else 0
        )
        
        # Best level features
        features["best_bid_size"] = book.best_bid.size
        features["best_ask_size"] = book.best_ask.size
        features["best_level_imbalance"] = (
            (book.best_bid.size - book.best_ask.size) / 
            (book.best_bid.size + book.best_ask.size + 1e-9)
        )
        
        # Depth at multiple levels
        bid_sizes = [level.size for level in book.bids[:self.config.book_depth_levels]]
        ask_sizes = [level.size for level in book.asks[:self.config.book_depth_levels]]
        
        for depth in [5, 10, 20]:
            bid_depth = sum(bid_sizes[:depth])
            ask_depth = sum(ask_sizes[:depth])
            total_depth = bid_depth + ask_depth
            
            features[f"bid_depth_{depth}"] = bid_depth
            features[f"ask_depth_{depth}"] = ask_depth
            features[f"depth_imbalance_{depth}"] = (bid_depth - ask_depth) / (total_depth + 1e-9)
        
        # Book slope (closed-form regression, not np.polyfit)
        features["bid_slope"] = self._compute_book_slope(book.bids)
        features["ask_slope"] = self._compute_book_slope(book.asks)
        features["slope_asymmetry"] = features["bid_slope"] - features["ask_slope"]
        
        # Pressure indicators
        features["bid_pressure"] = self._compute_pressure(book.bids, book.mid_price)
        features["ask_pressure"] = self._compute_pressure(book.asks, book.mid_price)
        features["net_pressure"] = features["bid_pressure"] - features["ask_pressure"]
        
        # Order count (if available)
        if book.bids[0].order_count > 0:
            features["bid_order_count"] = sum(l.order_count for l in book.bids[:10])
            features["ask_order_count"] = sum(l.order_count for l in book.asks[:10])
            features["avg_bid_order_size"] = features["bid_depth_10"] / (features["bid_order_count"] + 1e-9)
            features["avg_ask_order_size"] = features["ask_depth_10"] / (features["ask_order_count"] + 1e-9)
        
        # Liquidity concentration
        features["bid_liquidity_concentration"] = bid_sizes[0] / (sum(bid_sizes) + 1e-9)
        features["ask_liquidity_concentration"] = ask_sizes[0] / (sum(ask_sizes) + 1e-9)
        
        # Gap detection
        features["bid_gap_exists"] = float(self._detect_gap(book.bids))
        features["ask_gap_exists"] = float(self._detect_gap(book.asks))
        
        return features
    
    @staticmethod
    def _compute_book_slope(levels: List[PriceLevel]) -> float:
        """
        Compute book slope using closed-form simple linear regression.
        ~100x faster than np.polyfit which uses full SVD.
        """
        n = min(len(levels), 10)
        if n < 3:
            return 0.0
        
        # Closed-form sums
        sum_x = n * (n - 1) / 2.0
        sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0
        
        sum_y = 0.0
        sum_xy = 0.0
        for i in range(n):
            s = levels[i].size
            sum_y += s
            sum_xy += i * s
        
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom
    
    def _compute_pressure(self, levels: List[PriceLevel], mid_price: float) -> float:
        """Compute pressure as size-weighted proximity to mid."""
        if not levels or mid_price == 0:
            return 0.0
        
        pressure = 0.0
        for level in levels[:self.config.book_depth_levels]:
            distance = abs(level.price - mid_price) / mid_price
            weight = 1.0 / (1.0 + distance * 100)
            pressure += level.size * weight
        
        return pressure
    
    @staticmethod
    def _detect_gap(levels: List[PriceLevel], gap_multiplier: float = 3.0) -> bool:
        """Detect significant gap in order book (avoids np.median sorting)."""
        n = min(10, len(levels) - 1)
        if n < 1:
            return False
        
        total_gap = sum(abs(levels[i].price - levels[i + 1].price) for i in range(n))
        avg_gap = total_gap / n
        threshold = avg_gap * gap_multiplier
        
        for i in range(min(20, len(levels) - 1)):
            if abs(levels[i].price - levels[i + 1].price) > threshold:
                return True
        return False
    
    def _empty_book_features(self) -> Dict[str, float]:
        """Return empty features when book is invalid"""
        return {key: 0.0 for key in [
            "spread_bps", "mid_price", "microprice", "microprice_vs_mid",
            "best_bid_size", "best_ask_size", "best_level_imbalance",
            "bid_depth_5", "ask_depth_5", "depth_imbalance_5",
            "bid_depth_10", "ask_depth_10", "depth_imbalance_10",
            "bid_depth_20", "ask_depth_20", "depth_imbalance_20",
            "bid_slope", "ask_slope", "slope_asymmetry",
            "bid_pressure", "ask_pressure", "net_pressure",
            "bid_liquidity_concentration", "ask_liquidity_concentration",
            "bid_gap_exists", "ask_gap_exists"
        ]}
    
    # ==================== TRADE FLOW FEATURES ====================
    
    def _compute_trade_features(
        self, 
        window_trades: List[Trade], 
        window_seconds: int
    ) -> Dict[str, float]:
        """Trade flow features for a specific time window."""
        features: Dict[str, float] = {}
        
        if not window_trades:
            return self._empty_trade_features()
        
        # Separate buy/sell trades
        buy_trades = [t for t in window_trades if t.side == Side.BUY]
        sell_trades = [t for t in window_trades if t.side == Side.SELL]
        
        buy_volume = sum(t.size for t in buy_trades)
        sell_volume = sum(t.size for t in sell_trades)
        total_volume = buy_volume + sell_volume
        
        # Volume metrics
        features["buy_volume"] = buy_volume
        features["sell_volume"] = sell_volume
        features["total_volume"] = total_volume
        
        # Delta
        features["delta"] = buy_volume - sell_volume
        features["delta_pct"] = features["delta"] / (total_volume + 1e-9)
        features["abs_delta"] = abs(features["delta"])
        
        # Trade count
        features["trade_count"] = len(window_trades)
        features["buy_trade_count"] = len(buy_trades)
        features["sell_trade_count"] = len(sell_trades)
        features["trade_count_imbalance"] = (
            (len(buy_trades) - len(sell_trades)) / (len(window_trades) + 1e-9)
        )
        
        # Trade intensity
        features["trade_intensity"] = len(window_trades) / window_seconds
        
        # Average trade size
        features["avg_trade_size"] = total_volume / (len(window_trades) + 1e-9)
        features["avg_buy_size"] = buy_volume / (len(buy_trades) + 1e-9) if buy_trades else 0
        features["avg_sell_size"] = sell_volume / (len(sell_trades) + 1e-9) if sell_trades else 0
        
        # Large trade detection (sorted partition, not np.percentile)
        if len(window_trades) >= 10:
            sizes = sorted((t.size for t in window_trades), reverse=True)
            p90_idx = max(1, len(sizes) // 10)
            size_threshold = sizes[p90_idx - 1]
            
            large_count = 0
            large_vol = 0.0
            large_buy_vol = 0.0
            for t in window_trades:
                if t.size >= size_threshold:
                    large_count += 1
                    large_vol += t.size
                    if t.side == Side.BUY:
                        large_buy_vol += t.size
            
            features["large_trade_count"] = large_count
            features["large_trade_volume_pct"] = large_vol / (total_volume + 1e-9)
            features["large_trade_buy_pct"] = large_buy_vol / (large_vol + 1e-9)
        else:
            features["large_trade_count"] = 0
            features["large_trade_volume_pct"] = 0
            features["large_trade_buy_pct"] = 0
        
        # Price movement
        if len(window_trades) >= 2:
            price_change = window_trades[-1].price - window_trades[0].price
            features["price_change"] = price_change
            features["price_change_pct"] = price_change / (window_trades[0].price + 1e-9)
        else:
            features["price_change"] = 0
            features["price_change_pct"] = 0
        
        # VWAP
        total_value = sum(t.value for t in window_trades)
        features["vwap"] = total_value / (total_volume + 1e-9)
        
        # VWAP deviation
        if len(window_trades) >= 2 and features["vwap"] > 0:
            features["vwap_deviation"] = (
                (window_trades[-1].price - features["vwap"]) / features["vwap"]
            )
        else:
            features["vwap_deviation"] = 0
        
        return features
    
    def _empty_trade_features(self) -> Dict[str, float]:
        """Return empty features when no trades"""
        features = {key: 0.0 for key in [
            "buy_volume", "sell_volume", "total_volume",
            "delta", "delta_pct", "abs_delta",
            "trade_count", "buy_trade_count", "sell_trade_count",
            "trade_count_imbalance", "trade_intensity",
            "avg_trade_size", "avg_buy_size", "avg_sell_size",
            "large_trade_count", "large_trade_volume_pct", "large_trade_buy_pct",
            "price_change", "price_change_pct", "vwap", "vwap_deviation"
        ]}
        # Always include live CVD even when no window trades
        features["cvd"] = self._cvd
        return features
    
    # ==================== VOLUME PROFILE FEATURES ====================
    
    def _compute_volume_profile(self) -> VolumeProfile:
        """Compute full volume profile"""
        cutoff = self._current_timestamp - timedelta(seconds=self.config.volume_profile_lookback)
        
        # Binary search for start index
        timestamps = [t.timestamp for t in self.trade_history]
        idx = bisect.bisect_left(timestamps, cutoff)
        recent_trades = self.trade_history[idx:]
        
        profile = VolumeProfile(
            timestamp=self._current_timestamp,
            lookback_seconds=self.config.volume_profile_lookback
        )
        
        if not recent_trades:
            return profile
        
        # Bucket trades by price
        tick = self.config.tick_size
        for trade in recent_trades:
            bucket_price = round(trade.price / tick) * tick
            profile.volume_at_price[bucket_price] = (
                profile.volume_at_price.get(bucket_price, 0) + trade.size
            )
        
        profile.compute_value_area()
        self._volume_profile_cache = profile
        self._cache_timestamp = self._current_timestamp
        
        return profile
    
    def _compute_volume_profile_features(self, use_cached: bool = True) -> Dict[str, float]:
        """Extract features from volume profile"""
        profile = self._volume_profile_cache
        
        if not profile or not profile.volume_at_price:
            return {
                "poc": 0, "vah": 0, "val": 0,
                "value_area_width_pct": 0, "price_vs_poc_pct": 0,
                "price_vs_vah_pct": 0, "price_vs_val_pct": 0,
                "in_value_area": 0, "volume_profile_skew": 0
            }
        
        # Use cached features if available and recent
        if use_cached and self._vp_features_cache and self._cache_timestamp:
            cache_age = (self._current_timestamp - self._cache_timestamp).total_seconds()
            if cache_age < 100:
                return self._vp_features_cache
        
        # Calculate features
        features = self._calculate_vp_features(profile)
        self._vp_features_cache = features
        return features
    
    def _calculate_vp_features(self, profile: VolumeProfile) -> Dict[str, float]:
        """Extract actual features from volume profile"""
        features: Dict[str, float] = {}
        current_price = self.book_history[-1].mid_price if self.book_history else 0
        
        features["poc"] = profile.poc
        features["vah"] = profile.vah
        features["val"] = profile.val
        features["value_area_width_pct"] = (
            (profile.vah - profile.val) / (profile.poc + 1e-9)
        )
        
        if current_price > 0:
            features["price_vs_poc_pct"] = (current_price - profile.poc) / profile.poc
            features["price_vs_vah_pct"] = (
                (current_price - profile.vah) / profile.vah if profile.vah else 0
            )
            features["price_vs_val_pct"] = (
                (current_price - profile.val) / profile.val if profile.val else 0
            )
            features["in_value_area"] = float(profile.val <= current_price <= profile.vah)
        else:
            features["price_vs_poc_pct"] = 0
            features["price_vs_vah_pct"] = 0
            features["price_vs_val_pct"] = 0
            features["in_value_area"] = 0
        
        # Profile skew
        prices = sorted(profile.volume_at_price.keys())
        volumes = [profile.volume_at_price[p] for p in prices]
        if len(volumes) > 2:
            total_vol = sum(volumes)
            weighted_price = sum(p * v for p, v in zip(prices, volumes)) / (total_vol + 1e-9)
            mid_price_range = (prices[0] + prices[-1]) / 2
            features["volume_profile_skew"] = (
                (weighted_price - mid_price_range) / (mid_price_range + 1e-9)
            )
        else:
            features["volume_profile_skew"] = 0
        
        return features
    
    # ==================== FOOTPRINT FEATURES ====================
    
    def _update_footprint_bars(self, trade: Trade) -> None:
        """Build footprint bars incrementally from incoming trades."""
        bucket_price = round(trade.price / self.config.tick_size) * self.config.tick_size
        
        # Start new bar if needed
        if (self._current_footprint_bar is None or 
            (trade.timestamp - self._footprint_bar_start).total_seconds() 
            >= self.config.footprint_bar_duration_sec):
            
            # Archive old bar
            if self._current_footprint_bar:
                self.footprint_bars.append(self._current_footprint_bar)
                if len(self.footprint_bars) > 1000:
                    self.footprint_bars = self.footprint_bars[-500:]
            
            self._start_new_footprint_bar(trade)
        
        bar = self._current_footprint_bar
        
        # Update OHLC
        bar.high_price = max(bar.high_price, trade.price)
        bar.low_price = min(bar.low_price, trade.price)
        bar.close_price = trade.price
        
        # Volume at price (bucketed)
        b_vol, s_vol = bar.volume_at_price.get(bucket_price, (0.0, 0.0))
        if trade.side == Side.BUY:
            bar.volume_at_price[bucket_price] = (b_vol + trade.size, s_vol)
        else:
            bar.volume_at_price[bucket_price] = (b_vol, s_vol + trade.size)
    
    def _start_new_footprint_bar(self, trade: Trade) -> None:
        """Initialize a fresh footprint bar."""
        self._footprint_bar_start = trade.timestamp
        self._current_footprint_bar = FootprintBar(
            timestamp=trade.timestamp,
            duration_seconds=self.config.footprint_bar_duration_sec,
            open_price=trade.price,
            high_price=trade.price,
            low_price=trade.price,
            close_price=trade.price,
            volume_at_price={}
        )
    
    def _compute_footprint_features(self) -> Dict[str, float]:
        """Footprint-specific features"""
        features: Dict[str, float] = {
            "footprint_delta": 0, "highest_delta_level": 0,
            "lowest_delta_level": 0, "delta_concentration": 0,
            "footprint_imbalance_count": 0, "buying_exhaustion": 0,
            "selling_exhaustion": 0
        }
        
        if not self.footprint_bars:
            return features
        
        latest_bar = self.footprint_bars[-1]
        features["footprint_delta"] = latest_bar.delta
        
        if latest_bar.volume_at_price:
            delta_by_price = {
                price: buy - sell 
                for price, (buy, sell) in latest_bar.volume_at_price.items()
            }
            
            features["highest_delta_level"] = max(delta_by_price.keys(), 
                                                   key=lambda p: delta_by_price[p])
            features["lowest_delta_level"] = min(delta_by_price.keys(), 
                                                  key=lambda p: delta_by_price[p])
            
            max_delta = max(delta_by_price.values())
            min_delta = min(delta_by_price.values())
            total_abs_delta = sum(abs(d) for d in delta_by_price.values())
            features["delta_concentration"] = (
                (abs(max_delta) + abs(min_delta)) / (total_abs_delta + 1e-9)
            )
            
            # Count imbalances
            features["footprint_imbalance_count"] = len([
                price for price, (buy, sell) in latest_bar.volume_at_price.items()
                if (buy / (sell + 1e-9) >= self.config.imbalance_ratio_threshold or
                    sell / (buy + 1e-9) >= self.config.imbalance_ratio_threshold)
            ])
        
        # Exhaustion detection
        if len(self.footprint_bars) >= 2:
            prev_bar = self.footprint_bars[-2]
            features["buying_exhaustion"] = float(
                latest_bar.high_price > prev_bar.high_price and
                latest_bar.delta < prev_bar.delta
            )
            features["selling_exhaustion"] = float(
                latest_bar.low_price < prev_bar.low_price and
                latest_bar.delta > prev_bar.delta
            )
        
        return features
    
    # ==================== PATTERN DETECTION ====================
    
    def _detect_absorptions(self) -> List[Absorption]:
        """Detect absorption events (time-based window)."""
        absorptions = []
        
        if len(self.trade_history) < 10:
            return absorptions
        
        recent_trades = self.trade_history[-200:]
        duration = (recent_trades[-1].timestamp - recent_trades[0].timestamp).total_seconds()
        avg_volume_per_second = sum(t.size for t in recent_trades) / (duration + 1)
        
        min_duration = self.config.absorption_min_duration_sec
        
        for i in range(len(recent_trades)):
            start_trade = recent_trades[i]
            target_time = start_trade.timestamp + timedelta(seconds=min_duration)
            
            end_idx = -1
            for j in range(i + 1, len(recent_trades)):
                if recent_trades[j].timestamp >= target_time:
                    end_idx = j
                    break
            
            if end_idx == -1:
                break
            
            window_trades = recent_trades[i:end_idx + 1]
            window_volume = sum(t.size for t in window_trades)
            window_duration = (window_trades[-1].timestamp - window_trades[0].timestamp).total_seconds()
            
            if window_duration < min_duration:
                continue
            
            price_change_pct = abs(
                window_trades[-1].price - window_trades[0].price
            ) / (window_trades[0].price + 1e-9)
            
            volume_rate = window_volume / (window_duration + 1e-9)
            
            if (volume_rate > avg_volume_per_second * self.config.absorption_volume_multiplier and
                price_change_pct < self.config.absorption_price_threshold_pct):
                
                buy_vol = sum(t.size for t in window_trades if t.side == Side.BUY)
                sell_vol = sum(t.size for t in window_trades if t.side == Side.SELL)
                
                if buy_vol > sell_vol:
                    absorbing_side = Side.BUY
                    strength = sell_vol / (buy_vol + 1e-9)
                else:
                    absorbing_side = Side.SELL
                    strength = buy_vol / (sell_vol + 1e-9)
                
                absorptions.append(Absorption(
                    timestamp=window_trades[-1].timestamp,
                    price=window_trades[-1].price,
                    absorbed_volume=window_volume,
                    price_change_pct=price_change_pct,
                    duration_seconds=window_duration,
                    absorbing_side=absorbing_side,
                    strength=min(strength, 1.0)
                ))
        
        return absorptions[-10:]
    
    def _detect_imbalances(self, book: OrderBook) -> List[Imbalance]:
        """Detect order book imbalances using true mirror level matching."""
        imbalances = []
        
        if not book.bids or not book.asks:
            return imbalances
        
        mid = book.mid_price
        bid_imbalances = self._find_side_imbalances(book.bids, book.asks, Side.BUY, mid)
        ask_imbalances = self._find_side_imbalances(book.asks, book.bids, Side.SELL, mid)
        
        imbalances.extend(bid_imbalances)
        imbalances.extend(ask_imbalances)
        
        self._mark_stacked_imbalances(imbalances)
        return imbalances
    
    def _find_side_imbalances(
        self,
        levels: List[PriceLevel],
        opposing_levels: List[PriceLevel],
        side: Side,
        mid_price: float
    ) -> List[Imbalance]:
        """Find imbalances by matching price distance from mid (True Mirror Levels)."""
        imbalances = []
        tolerance = self.config.tick_size * 1.5
        
        for level in levels[:self.config.book_depth_levels]:
            distance = abs(mid_price - level.price)
            mirror_price = mid_price + distance if side == Side.BUY else mid_price - distance
            
            matching_opp_size = 0
            for opp in opposing_levels:
                if abs(opp.price - mirror_price) <= tolerance:
                    matching_opp_size = opp.size
                    break
            
            ratio = level.size / matching_opp_size if matching_opp_size > 0 else float('inf')
            
            if ratio >= self.config.imbalance_ratio_threshold:
                imbalances.append(Imbalance(
                    timestamp=self._current_timestamp or datetime.now(),
                    price=level.price,
                    buy_volume=level.size if side == Side.BUY else matching_opp_size,
                    sell_volume=matching_opp_size if side == Side.BUY else level.size,
                    imbalance_ratio=ratio,
                    direction=side
                ))
        
        return imbalances
    
    def _mark_stacked_imbalances(self, imbalances: List[Imbalance]) -> None:
        """Mark consecutive imbalances as stacked"""
        if len(imbalances) < self.config.stacked_imbalance_min_levels:
            return
        
        sorted_imbalances = sorted(imbalances, key=lambda x: x.price)
        current_stack = [sorted_imbalances[0]]
        
        for i in range(1, len(sorted_imbalances)):
            if sorted_imbalances[i].direction == current_stack[-1].direction:
                current_stack.append(sorted_imbalances[i])
            else:
                if len(current_stack) >= self.config.stacked_imbalance_min_levels:
                    for imb in current_stack:
                        imb.is_stacked = True
                        imb.stack_count = len(current_stack)
                current_stack = [sorted_imbalances[i]]
        
        if len(current_stack) >= self.config.stacked_imbalance_min_levels:
            for imb in current_stack:
                imb.is_stacked = True
                imb.stack_count = len(current_stack)
    
    def _detect_liquidity_sweeps(self) -> List[LiquiditySweep]:
        """Detect liquidity sweeps / stop hunts."""
        sweeps = []
        
        if len(self.trade_history) < 50:
            return sweeps
        
        recent_trades = self.trade_history[-200:]
        window_size = 20
        
        for i in range(window_size * 2, len(recent_trades)):
            sweep_window = recent_trades[i-window_size*2:i-window_size]
            reversal_window = recent_trades[i-window_size:i]
            
            sweep_duration = (
                sweep_window[-1].timestamp - sweep_window[0].timestamp
            ).total_seconds()
            
            if sweep_duration > self.config.sweep_speed_threshold_sec:
                continue
            
            sweep_move = sweep_window[-1].price - sweep_window[0].price
            sweep_move_pct = abs(sweep_move) / (sweep_window[0].price + 1e-9)
            reversal_move = reversal_window[-1].price - reversal_window[0].price
            
            if (sweep_move_pct > self.config.sweep_reversal_threshold_pct and
                sweep_move * reversal_move < 0):
                
                reversal_pct = abs(reversal_move) / abs(sweep_move) if sweep_move != 0 else 0
                
                if reversal_pct > 0.5:
                    sweeps.append(LiquiditySweep(
                        timestamp=reversal_window[-1].timestamp,
                        sweep_price=sweep_window[-1].price,
                        reversal_price=reversal_window[-1].price,
                        direction=Side.BUY if sweep_move > 0 else Side.SELL,
                        volume_swept=sum(t.size for t in sweep_window),
                        speed_seconds=sweep_duration,
                        reversal_strength=reversal_pct
                    ))
        
        return sweeps[-5:]
    
    def _detect_icebergs(self) -> List[IcebergOrder]:
        """Detect iceberg orders with spoofing filter."""
        icebergs = []
        
        if len(self.book_history) < 10:
            return icebergs
        
        recent_books = self.book_history[-30:]
        refill_counts: Dict[float, Dict] = {}
        
        for i in range(1, len(recent_books)):
            self._track_refills(
                recent_books[i-1].bids, recent_books[i].bids,
                refill_counts, Side.BUY
            )
            self._track_refills(
                recent_books[i-1].asks, recent_books[i].asks,
                refill_counts, Side.SELL
            )
        
        # Build traded prices set (spoofing filter)
        traded_prices: set = set()
        for trade in self.trade_history[-100:]:
            bucket = round(trade.price / self.config.tick_size) * self.config.tick_size
            traded_prices.add(bucket)
        
        for price, data in refill_counts.items():
            if data["refill_count"] >= self.config.iceberg_refill_threshold:
                price_bucket = round(price / self.config.tick_size) * self.config.tick_size
                if price_bucket not in traded_prices:
                    continue  # Spoofing filter
                
                icebergs.append(IcebergOrder(
                    timestamp=self._current_timestamp or datetime.now(),
                    price=price,
                    visible_size=data["visible_size"],
                    estimated_hidden_size=data["total_refilled"],
                    refill_count=data["refill_count"],
                    side=data["side"],
                    confidence=min(data["refill_count"] / 10, 1.0)
                ))
        
        return icebergs
    
    def _track_refills(
        self,
        prev_levels: List[PriceLevel],
        curr_levels: List[PriceLevel],
        refill_counts: Dict[float, Dict],
        side: Side
    ) -> None:
        """Track order refills at price levels"""
        prev_by_price = {l.price: l.size for l in prev_levels[:20]}
        curr_by_price = {l.price: l.size for l in curr_levels[:20]}
        
        for price in prev_by_price:
            if price in curr_by_price:
                prev_size = prev_by_price[price]
                curr_size = curr_by_price[price]
                
                if prev_size < self.config.iceberg_size_threshold and curr_size > prev_size * 2:
                    if price not in refill_counts:
                        refill_counts[price] = {
                            "refill_count": 0,
                            "total_refilled": 0,
                            "visible_size": curr_size,
                            "side": side
                        }
                    
                    refill_counts[price]["refill_count"] += 1
                    refill_counts[price]["total_refilled"] += curr_size - prev_size
                    refill_counts[price]["visible_size"] = curr_size
    
    # ==================== PATTERN-TO-FEATURE BRIDGE (CRITICAL FIX) ====================
    
    def _compute_pattern_features(self, state: OrderFlowState) -> Dict[str, float]:
        """
        Bridge detected patterns into named features that strategies consume.
        
        WITHOUT this method, strategies reference features like
        'recent_absorption_strength' that never exist → no signals fire.
        """
        features: Dict[str, float] = {}
        now = self._current_timestamp or datetime.now()
        
        # ── Absorption features ──
        recent_abs = [
            a for a in state.absorptions
            if (now - a.timestamp).total_seconds() < 60
        ]
        if recent_abs:
            latest = recent_abs[-1]
            features["recent_absorption_strength"] = latest.strength
            features["recent_absorption_detected"] = 1.0
            features["recent_absorption_direction"] = (
                1.0 if latest.absorbing_side == Side.BUY else -1.0
            )
            features["max_absorption_strength_60s"] = max(
                a.strength for a in recent_abs
            )
            features["absorption_count_60s"] = float(len(recent_abs))
            features["avg_absorption_volume"] = sum(
                a.absorbed_volume for a in recent_abs
            ) / len(recent_abs)
        else:
            features["recent_absorption_strength"] = 0.0
            features["recent_absorption_detected"] = 0.0
            features["recent_absorption_direction"] = 0.0
            features["max_absorption_strength_60s"] = 0.0
            features["absorption_count_60s"] = 0.0
            features["avg_absorption_volume"] = 0.0
        
        # ── Sweep features ──
        recent_sw = [
            s for s in state.sweeps
            if (now - s.timestamp).total_seconds() < 120
        ]
        if recent_sw:
            latest = recent_sw[-1]
            features["recent_sweep_detected"] = 1.0
            features["recent_sweep_reversal_strength"] = latest.reversal_strength
            features["recent_sweep_direction"] = (
                1.0 if latest.direction == Side.BUY else -1.0
            )
            features["recent_sweep_volume"] = latest.volume_swept
            features["sweep_count_120s"] = float(len(recent_sw))
        else:
            features["recent_sweep_detected"] = 0.0
            features["recent_sweep_reversal_strength"] = 0.0
            features["recent_sweep_direction"] = 0.0
            features["recent_sweep_volume"] = 0.0
            features["sweep_count_120s"] = 0.0
        
        # ── Stacked imbalance features ──
        stacked = [i for i in state.imbalances if i.is_stacked]
        if stacked:
            features["stacked_imbalance_count"] = float(len(stacked))
            features["max_stack_size"] = float(max(i.stack_count for i in stacked))
            buy_imb = sum(1 for i in stacked if i.direction == Side.BUY)
            sell_imb = sum(1 for i in stacked if i.direction == Side.SELL)
            features["imbalance_direction_ratio"] = (buy_imb - sell_imb) / (
                buy_imb + sell_imb + 1e-9
            )
        else:
            features["stacked_imbalance_count"] = 0.0
            features["max_stack_size"] = 0.0
            features["imbalance_direction_ratio"] = 0.0
        
        if state.imbalances:
            features["avg_imbalance_ratio"] = sum(
                i.imbalance_ratio for i in state.imbalances
            ) / len(state.imbalances)
        else:
            features["avg_imbalance_ratio"] = 0.0
        
        # ── Iceberg features ──
        if state.icebergs:
            best = max(state.icebergs, key=lambda x: x.confidence)
            features["iceberg_detected"] = 1.0
            features["iceberg_confidence"] = best.confidence
            features["iceberg_direction"] = (
                1.0 if best.side == Side.BUY else -1.0
            )
            features["iceberg_estimated_size"] = best.estimated_hidden_size
            features["iceberg_count"] = float(len(state.icebergs))
        else:
            features["iceberg_detected"] = 0.0
            features["iceberg_confidence"] = 0.0
            features["iceberg_direction"] = 0.0
            features["iceberg_estimated_size"] = 0.0
            features["iceberg_count"] = 0.0
        
        return features
    
    # ==================== COMPOSITE FEATURES (ALL 8 STRATEGIES NEED) ====================
    
    def _compute_composite_features(self, base_features: Dict[str, float]) -> Dict[str, float]:
        """
        Derived features combining multiple base signals.
        Every feature referenced by any strategy is computed here.
        """
        features: Dict[str, float] = {}
        
        # 1. Volume acceleration: 60s vs 300s normalized
        short_vol = base_features.get("total_volume_60s", 0)
        long_vol = base_features.get("total_volume_300s", 0)
        features["volume_acceleration"] = (short_vol * 5) / (long_vol + 1e-9)
        
        # 2. Delta divergence: price and delta disagree
        price_dir_60 = np.sign(base_features.get("price_change_pct_60s", 0))
        delta_dir_60 = np.sign(base_features.get("delta_pct_60s", 0))
        features["delta_divergence_60s"] = float(
            price_dir_60 != delta_dir_60 and price_dir_60 != 0
        )
        
        # 3. CVD divergence: cumulative delta vs price
        cvd_sign = np.sign(base_features.get("cvd", 0))
        price_dir_300 = np.sign(base_features.get("price_change_pct_300s", 0))
        features["cvd_price_divergence"] = float(
            cvd_sign != price_dir_300 and price_dir_300 != 0
        )
        
        # 4. Exhaustion score (from footprint)
        buying_ex = base_features.get("buying_exhaustion", 0)
        selling_ex = base_features.get("selling_exhaustion", 0)
        features["exhaustion_score"] = max(buying_ex, selling_ex)
        
        # 5. Pressure confirmed: book pressure aligns with delta
        net_pressure = base_features.get("net_pressure", 0)
        delta_60 = base_features.get("delta_60s", 0)
        features["pressure_confirmed"] = float(
            np.sign(net_pressure) == np.sign(delta_60) and net_pressure != 0
        )
        
        # 6. Book-trade agreement: depth imbalance aligns with trade flow
        book_bias = base_features.get("depth_imbalance_10", 0)
        trade_bias = base_features.get("delta_pct_60s", 0)
        features["book_trade_agreement"] = float(
            np.sign(book_bias) == np.sign(trade_bias) and book_bias != 0
        )
        
        # 7. VA breakout potential: outside VA with rising volume
        in_va = base_features.get("in_value_area", 0)
        vol_accel = features.get("volume_acceleration", 1.0)
        features["va_breakout_potential"] = float(in_va == 0 and vol_accel > 1.5)
        
        # 8. Price vs VWAP (alias for vwap_deviation)
        features["price_vs_vwap_pct"] = base_features.get("vwap_deviation_60s", 0)
        
        # Composite: Book-Trade Direction Agreement  # [ADDED]
        # Requires BOTH order book imbalance AND trade flow imbalance to agree
        # on direction above their respective noise thresholds.
        # This prevents the strategy from entering on book imbalance alone,
        # which flips multiple times per second on XRP due to market-maker refresh.
        # Value is 1.0 (agree) or 0.0 (disagree/ambiguous).
        book_imbal  = features.get("depth_imbalance_10", 0)
        trade_imbal = features.get("trade_count_imbalance_60s", 0)

        if (book_imbal > 0.08 and trade_imbal > 0.1) or \
           (book_imbal < -0.08 and trade_imbal < -0.1):
            features["book_trade_agreement"] = 1.0
        else:
            features["book_trade_agreement"] = 0.0   # No agreement or below threshold
        
        return features
    
    # ==================== TEMPORAL FEATURE TRACKING ====================
    
    def _store_feature_snapshot(
        self, 
        timestamp: datetime, 
        features: Dict[str, float]
    ) -> None:
        """Store snapshot of key features for temporal look-back."""
        _TRACKED = [
            "recent_absorption_strength", "recent_absorption_direction",
            "depth_imbalance_10", "delta_60s", "delta_pct_60s",
            "footprint_imbalance_count", "volume_acceleration",
            "recent_sweep_detected", "recent_sweep_reversal_strength",
            "net_pressure", "spread_bps", "mid_price",
        ]
        snapshot = {k: features.get(k, 0.0) for k in _TRACKED}
        self._feature_snapshots.append((timestamp, snapshot))
        
        # Keep only last 60 seconds
        cutoff = timestamp - timedelta(seconds=60)
        while self._feature_snapshots and self._feature_snapshots[0][0] < cutoff:
            self._feature_snapshots.pop(0)
    
    def get_feature_at_lag(self, feature_name: str, seconds_ago: float) -> float:
        """Retrieve a feature value from N seconds ago."""
        if not self._feature_snapshots or not self._current_timestamp:
            return 0.0
        
        target = self._current_timestamp - timedelta(seconds=seconds_ago)
        best_snap = None
        best_diff = float("inf")
        
        for ts, snap in self._feature_snapshots:
            diff = abs((ts - target).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_snap = snap
        
        if best_snap and best_diff < seconds_ago * 0.5:
            return best_snap.get(feature_name, 0.0)
        return 0.0
    
#=========================================
# Section 3 knowledge/strategy_library.py
#=========================================

"""
Strategy Library
Complete definitions of order flow trading strategies.
Each strategy encapsulates domain knowledge in executable form.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
import numpy as np
from loguru import logger

from core.data_structures import (
    OrderFlowState, Signal, SignalType, Side, Regime
)


class StrategyCategory(Enum):
    ABSORPTION = auto()
    MOMENTUM = auto()
    REVERSAL = auto()
    BREAKOUT = auto()
    MEAN_REVERSION = auto()


@dataclass
class StrategyCondition:
    """Single condition for strategy entry/exit with optimizer parameter mapping"""
    feature: str
    operator: str  # ">", "<", ">=", "<=", "==", "between"
    threshold: float
    threshold_high: Optional[float] = None  # For "between" operator
    weight: float = 1.0
    required: bool = False
    param_key: Optional[str] = None  # CRITICAL: Links to Optuna parameter name
    
    def validate(self) -> bool:
        """Validate condition configuration before runtime"""
        if self.operator == "between":
            if self.threshold_high is None:
                raise ValueError(f"[{self.feature}] 'between' operator requires threshold_high")
            if self.threshold >= self.threshold_high:
                raise ValueError(f"[{self.feature}] threshold ({self.threshold}) must be < threshold_high ({self.threshold_high})")
        # DEBUG: Log validation success
        logger.debug(f"[StrategyCondition] Validated: {self.feature} (operator={self.operator})")
        return True
    
    def evaluate(self, features: Dict[str, float]) -> tuple:
        """Returns (satisfied: bool, score: float)"""
        if self.feature not in features:
            logger.debug(f"[StrategyCondition] Feature {self.feature} not found in features")
            return (False, 0.0)
        
        value = features[self.feature]
        
        if self.operator == ">":
            satisfied = value > self.threshold
        elif self.operator == "<":
            satisfied = value < self.threshold
        elif self.operator == ">=":
            satisfied = value >= self.threshold
        elif self.operator == "<=":
            satisfied = value <= self.threshold
        elif self.operator == "==":
            satisfied = abs(value - self.threshold) < 1e-9
        elif self.operator == "between":
            satisfied = self.threshold <= value <= self.threshold_high
        else:
            satisfied = False
        
        # Calculate score (how far beyond threshold)
        if satisfied and self.operator in [">", ">="]:
            score = min((value - self.threshold) / (abs(self.threshold) + 1e-9), 2.0)
        elif satisfied and self.operator in ["<", "<="]:
            score = min((self.threshold - value) / (abs(self.threshold) + 1e-9), 2.0)
        elif satisfied:
            score = 1.0
        else:
            score = 0.0
        
        logger.debug(f"[StrategyCondition] {self.feature}: value={value:.4f}, threshold={self.threshold}, satisfied={satisfied}")
        return (satisfied, score * self.weight)


@dataclass
class StrategyDefinition:
    """Complete strategy definition with optimizer-controlled parameters"""
    name: str
    category: StrategyCategory
    description: str
    
    entry_conditions: List[StrategyCondition] = field(default_factory=list)
    min_conditions_satisfied: int = 3
    min_score_threshold: float = 2.0
    
    # Risk parameters (base values - may be overridden by optimizer)
    stop_loss_atr_mult: float = 2.5
    take_profit_atr_mult: float = 6.0
    max_holding_seconds: int = 86400  # [APPLIED] 24h = unlimited hold time, optimizer controls exits via SL/TP only
    trailing_stop_activation_pct: float = 0.005
    
    # CRITICAL: Regime-specific multipliers (optimizer-controlled)
    # These allow different risk parameters for different market conditions
    sl_mult_high_vol: float = 3.5
    sl_mult_low_vol: float = 1.8
    sl_mult_trending: float = 2.5
    tp_mult_high_vol: float = 7.0
    tp_mult_low_vol: float = 2.5
    tp_mult_trending: float = 5.0
    
    filters: List[StrategyCondition] = field(default_factory=list)
    allowed_regimes: List[Regime] = field(default_factory=lambda: list(Regime))
    
    # Position sizing
    base_position_pct: float = 0.1
    max_position_pct: float = 0.25
    scale_with_score: bool = True
    
    def evaluate(self, state: OrderFlowState) -> Optional[Signal]:
        """
        Evaluate strategy conditions and return signal if triggered.
        CRITICAL: Uses regime-specific multipliers controlled by optimizer.
        """
        features = state.features
        
        # DEBUG: Log regime check
        logger.debug(f"[{self.name}] Evaluating - Regime: {state.regime}, Features: {len(features)}")
        
        # Check for LOW_LIQUIDITY regime (hard reject)
        if state.regime == Regime.LOW_LIQUIDITY:
            logger.debug(f"[{self.name}] REJECTED: LOW_LIQUIDITY regime")
            return None
        
        # Validate all conditions before evaluation
        try:
            for cond in self.entry_conditions + self.filters:
                cond.validate()
        except ValueError as e:
            logger.error(f"[{self.name}] Validation error: {e}")
            return None
        
        # Check filters first (if satisfied, REJECT the trade)
        for i, filter_cond in enumerate(self.filters):
            satisfied, _ = filter_cond.evaluate(features)
            if satisfied:
                logger.debug(f"[{self.name}] FILTER REJECTED by condition {i}: {filter_cond.feature}")
                return None
        
        # Check allowed regimes
        if state.regime not in self.allowed_regimes:
            logger.debug(f"[{self.name}] REJECTED: Regime {state.regime} not in allowed list")
            return None
        
        # Evaluate entry conditions
        satisfied_count = 0
        total_score = 0.0
        required_satisfied = True
        reasons = []
        failed_conditions = []
        
        for condition in self.entry_conditions:
            satisfied, score = condition.evaluate(features)
            
            if satisfied:
                satisfied_count += 1
                total_score += score
                reasons.append(f"{condition.feature} {condition.operator} {condition.threshold:.4f}")
                logger.debug(f"[{self.name}] Condition SATISFIED: {condition.feature}")
            else:
                failed_conditions.append(f"{condition.feature} ({condition.operator} {condition.threshold})")
                if condition.required:
                    required_satisfied = False
        
        # DEBUG: Log evaluation summary
        logger.debug(f"[{self.name}] Stats: satisfied={satisfied_count}/{len(self.entry_conditions)}, "
                    f"score={total_score:.2f}, required_ok={required_satisfied}")
        
        # Check entry criteria
        if not required_satisfied:
            logger.debug(f"[{self.name}] REJECTED: Required condition failed")
            return None
        
        if satisfied_count < self.min_conditions_satisfied:
            logger.debug(f"[{self.name}] REJECTED: Only {satisfied_count} conditions, need {self.min_conditions_satisfied}")
            return None
        
        if total_score < self.min_score_threshold:
            logger.debug(f"[{self.name}] REJECTED: Score {total_score:.2f} < threshold {self.min_score_threshold}")
            return None
        
        # Determine direction
        direction = self._determine_direction(state, total_score)
        if direction == SignalType.NEUTRAL:
            logger.debug(f"[{self.name}] REJECTED: No clear direction (NEUTRAL)")
            return None
        
        # CRITICAL: Select regime-specific multipliers
        regime = state.regime
        if regime == Regime.HIGH_VOLATILITY:
            sl_mult, tp_mult = self.sl_mult_high_vol, self.tp_mult_high_vol
            logger.debug(f"[{self.name}] Using HIGH_VOL multipliers: SL={sl_mult}, TP={tp_mult}")
        elif regime in (Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION):
            sl_mult, tp_mult = self.sl_mult_low_vol, self.tp_mult_low_vol
            logger.debug(f"[{self.name}] Using LOW_VOL multipliers: SL={sl_mult}, TP={tp_mult}")
        else:  # TRENDING_UP, TRENDING_DOWN, BREAKOUT, etc.
            sl_mult, tp_mult = self.sl_mult_trending, self.tp_mult_trending
            logger.debug(f"[{self.name}] Using TRENDING multipliers: SL={sl_mult}, TP={tp_mult}")
        
        # Calculate stops using selected multipliers
        atr = self._estimate_atr(state)
        mid_price = state.order_book.mid_price
        
        stop_dist = atr * sl_mult
        tp_dist = atr * tp_mult
        
        # Enforce minimum TP distance (0.37% of price)
        min_tp_dist = mid_price * 0.0037
        if tp_dist < min_tp_dist:
            logger.debug(f"[{self.name}] TP distance adjusted from {tp_dist:.2f} to {min_tp_dist:.2f} (min 0.37%)")
            tp_dist = min_tp_dist
        
        if direction in (SignalType.BUY, SignalType.STRONG_BUY):
            stop_loss = mid_price - stop_dist
            take_profit = mid_price + tp_dist
        else:
            stop_loss = mid_price + stop_dist
            take_profit = mid_price - tp_dist
        
        # Calculate position size
        position_pct = self.base_position_pct
        if self.scale_with_score:
            position_pct = min(self.base_position_pct * min(total_score / self.min_score_threshold, 2.0), 
                             self.max_position_pct)
        
        confidence = min(total_score / (self.min_score_threshold * 2), 1.0)
        
        # DEBUG: Log signal generation
        logger.info(f"[{self.name}] SIGNAL: {direction.name} @ {mid_price:.4f} | "
                   f"SL={stop_loss:.4f} TP={take_profit:.4f} | "
                   f"Regime={regime.name} | Score={total_score:.2f}")
        
        return Signal(
            timestamp=state.timestamp,
            signal_type=direction,
            confidence=confidence,
            entry_price=mid_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_pct,
            primary_reason=reasons[0] if reasons else self.name,
            supporting_factors=reasons[1:5],
            risk_reward_ratio=abs(tp_dist) / abs(stop_dist)
        )
    
    def _determine_direction(self, state: OrderFlowState, score: float) -> SignalType:
        """Determine signal direction from state"""
        features = state.features
        
        # Use delta and imbalance to determine direction
        delta = features.get("delta_60s", 0)
        imbalance = features.get("depth_imbalance_10", 0)
        pressure = features.get("net_pressure", 0)
        
        bullish_score = sum([
            delta > 0,
            imbalance > 0.1,  # Preserve the existing noise filter threshold
            pressure > 0
        ])
        
        if bullish_score == 3:
            return SignalType.STRONG_BUY if score > self.min_score_threshold * 1.5 else SignalType.BUY
        elif bullish_score == 2:
            return SignalType.BUY
        elif bullish_score == 1:
            return SignalType.NEUTRAL # Ambiguous — 1 of 3 indicators bullish, no clear edge
        elif bullish_score == 0:
            return SignalType.STRONG_SELL if score > self.min_score_threshold * 1.5 else SignalType.SELL
        
    
    def _estimate_atr(self, state: OrderFlowState, default: float = 100.0) -> float:
        """Estimate ATR for stop/TP placement.
        
        Priority order:
          1. Use atr_60s from FeatureEngine if available (most accurate)
          2. Use price_range_60s as a proxy if atr_60s missing
          3. Fall back to 0.5% of mid price (last resort)
        
        Always enforce a minimum of 0.5% of price so stops are never
        tighter than the spread.
        """
        mid_price = state.order_book.mid_price
        if mid_price <= 0:
            return default

        features = state.features

        # Priority 1: Real ATR from feature engine  # [FIXED]
        if features.get("atr_60s", 0) > 0:
            return max(features["atr_60s"], mid_price * 0.005)

        # Priority 2: Price range as ATR proxy  # [FIXED]
        if features.get("price_range_60s", 0) > 0:
            return max(features["price_range_60s"], mid_price * 0.005)

        # Priority 3: Fallback — 0.5% of price
        return mid_price * 0.005


# ==================== PRE-DEFINED STRATEGIES ====================

def create_absorption_strategy() -> StrategyDefinition:
    """
    Absorption Strategy with full optimizer control.
    CRITICAL: All thresholds mapped via param_key for optimization.
    """
    return StrategyDefinition(
        name="Absorption",
        category=StrategyCategory.ABSORPTION,
        description="Trade after detecting absorption of aggressive orders",
        
        entry_conditions=[
            # Core absorption detection
            StrategyCondition(
                feature="recent_absorption_strength",
                operator=">=",
                threshold=0.30,
                weight=2.0,
                required=True,
                param_key="abs__entry_str_min"  # Optimizer controls this threshold
            ),
            StrategyCondition(
                feature="volume_acceleration",
                operator=">",
                threshold=1.0,
                weight=1.5,
                param_key="abs__entry_vol_min"
            ),
            StrategyCondition(
                feature="price_change_pct_60s",
                operator="<",
                threshold=0.001,
                weight=1.0,
                param_key="abs__entry_chg60_max"
            ),
            StrategyCondition(
                feature="abs_delta_60s",
                operator=">",
                threshold=0,
                weight=1.5,
                param_key="abs__entry_delta_min"
            ),
            StrategyCondition(
                feature="depth_imbalance_10",
                operator=">",
                threshold=0.05,
                weight=1.0,
                param_key="abs__entry_imbal_min"
            ),
            # POC proximity - symmetric range (will use abs(value) for both bounds)
            StrategyCondition(
                feature="price_vs_poc_pct",
                operator="between",
                threshold=-0.005,  # Will be overridden by optimizer via param_key
                threshold_high=0.005,
                weight=0.5,
                param_key="abs__entry_poc_range"
            ),
            StrategyCondition(
                feature="book_trade_agreement",
                operator="==",
                threshold=1.0,
                weight=1.5,       # High weight — strongly preferred but not required
                required=False,   # NOT required: early absorption entries lag trade flow
                                  # Making it required would filter out the best entries
                param_key="abs__entry_agreement"   # Exposed to optimizer  # [FIXED]
            ),
        ],
        
        filters=[
            # Spread filter: REJECT if spread > threshold
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=15.0,
                param_key="abs__filter_spread_max"
            ),
            # Depth filters: REJECT if depth < threshold (illiquidity protection)
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=1500.0,
                param_key="abs__filter_bid_min"
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=1500.0,
                param_key="abs__filter_ask_min"
            ),
            # Price change filter: REJECT if outside symmetric range (dead/choppy market protection)
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.005,
                threshold_high=0.005,
                param_key="abs__filter_chg300_range"
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        
        # Base risk parameters (optimizer can override via param_keys below)
        stop_loss_atr_mult=2.5,
        take_profit_atr_mult=6.0,
        trailing_stop_activation_pct=0.005,
        
        # CRITICAL: Regime-specific multipliers exposed to optimizer
        sl_mult_high_vol=3.5,
        sl_mult_low_vol=1.8,
        sl_mult_trending=2.5,
        tp_mult_high_vol=7.0,
        tp_mult_low_vol=2.5,
        tp_mult_trending=5.0,
        
        allowed_regimes=[
            Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION, 
            Regime.TRENDING_UP, Regime.TRENDING_DOWN
        ]
    )


def create_delta_divergence_strategy() -> StrategyDefinition:
    """
    Delta Divergence Strategy
    
    Enters when price and delta disagree, indicating potential reversal.
    """
    return StrategyDefinition(
        name="Delta Divergence",
        category=StrategyCategory.REVERSAL,
        description="Trade reversals when price diverges from cumulative delta",
        
        entry_conditions=[
            # Divergence detected
            StrategyCondition(
                feature="delta_divergence_60s",
                operator="==",
                threshold=1.0,
                weight=2.5,
                required=True
            ),
            # CVD divergence confirms
            StrategyCondition(
                feature="cvd_price_divergence",
                operator="==",
                threshold=1.0,
                weight=2.0
            ),
            # Exhaustion signals
            StrategyCondition(
                feature="exhaustion_score",
                operator=">",
                threshold=0.3,
                weight=1.5
            ),
            # Volume declining
            StrategyCondition(
                feature="volume_acceleration",
                operator="<",
                threshold=0.8,
                weight=1.0
            ),
            # At value area extreme
            StrategyCondition(
                feature="in_value_area",
                operator="==",
                threshold=0,  # Outside value area
                weight=1.0
            ),
            # FIX 5: Strengthen delta threshold - ensure divergence has volume behind it
            StrategyCondition(
                feature="abs_delta_60s",
                operator=">",
                threshold=8.0,
                weight=1.5,
                required=True  # Significant order flow imbalance required
            ),
        ],
        
        filters=[
            # Don't fight strong momentum
            StrategyCondition(
                feature="delta_pct_300s",
                operator=">",
                threshold=0.7
            ),
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.RANGING]
    )


def create_liquidity_sweep_strategy() -> StrategyDefinition:
    """
    Liquidity Sweep Strategy
    
    Enters after a stop hunt / liquidity sweep reverses.
    """
    return StrategyDefinition(
        name="Liquidity Sweep",
        category=StrategyCategory.REVERSAL,
        description="Fade liquidity sweeps after reversal confirmation",
        
        entry_conditions=[
            # Sweep detected
            StrategyCondition(
                feature="recent_sweep_detected",
                operator="==",
                threshold=1.0,
                weight=2.5,
                required=True
            ),
            # Strong reversal
            StrategyCondition(
                feature="recent_sweep_reversal_strength",
                operator=">",
                threshold=0.35,
                weight=2.0
            ),
            # Volume spike during sweep
            StrategyCondition(
                feature="volume_acceleration",
                operator=">",
                threshold=1.5,
                weight=1.5
            ),
            # Price back inside range
            StrategyCondition(
                feature="in_value_area",
                operator="==",
                threshold=1.0,
                weight=1.0
            ),
            # Book shifted in reversal direction
            StrategyCondition(
                feature="book_trade_agreement",
                operator="==",
                threshold=1.0,
                weight=2.5,      # High weight but NOT required
                required=False   # Let it influence the score, not veto the trade
            ),
        ],
        
        filters=[
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        # Note: LOW_LIQUIDITY regime handled by FIX 3 in evaluate()
        allowed_regimes=[Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION]
    )


def create_stacked_imbalance_strategy() -> StrategyDefinition:
    """
    Stacked Imbalance Strategy
    
    Enters in direction of stacked footprint imbalances.
    """
    return StrategyDefinition(
        name="Stacked Imbalance",
        category=StrategyCategory.MOMENTUM,
        description="Trade momentum when multiple price levels show same-side imbalance",
        
        entry_conditions=[
            # Stacked imbalance detected
            StrategyCondition(
                feature="footprint_imbalance_count",
                operator=">=",
                threshold=2,
                weight=2.0,
                required=True
            ),
            # Delta confirms direction
            StrategyCondition(
                feature="delta_pct_60s",
                operator=">",
                threshold=0.2,
                weight=1.5
            ),
            # Book supports
            StrategyCondition(
                feature="depth_imbalance_10",
                operator=">",
                threshold=0.15,
                weight=1.5
            ),
            # Pressure confirms
            StrategyCondition(
                feature="pressure_confirmed",
                operator="==",
                threshold=1.0,
                weight=1.0
            ),
            # Not overextended
            StrategyCondition(
                feature="price_vs_vwap_pct",
                operator="between",
                threshold=-0.003,
                threshold_high=0.003,
                weight=0.5
            ),
        ],
        
        filters=[
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.BREAKOUT, Regime.ACCUMULATION, Regime.DISTRIBUTION] #Add Regime.ACCUMULATION, Regime.DISTRIBUTION
    )


def create_value_area_strategy() -> StrategyDefinition:
    """
    Value Area Strategy
    
    Mean reversion trades at value area boundaries.
    """
    return StrategyDefinition(
        name="Value Area Mean Reversion",
        category=StrategyCategory.MEAN_REVERSION,
        description="Fade moves to value area boundaries",
        
        entry_conditions=[
            # At value area boundary
            StrategyCondition(
                feature="va_breakout_potential",
                operator="==",
                threshold=0,  # NOT breaking out = mean reversion setup
                weight=1.5
            ),
            # Price at VAH or VAL
            StrategyCondition(
                feature="price_vs_vah_pct",
                operator="between",
                threshold=-0.002,
                threshold_high=0.002,
                weight=2.0
            ),
            # Delta showing rejection
            StrategyCondition(
                feature="delta_divergence_60s",
                operator="==",
                threshold=1.0,
                weight=1.5
            ),
            # Volume declining
            StrategyCondition(
                feature="volume_acceleration",
                operator="<",
                threshold=1.0,
                weight=1.0
            ),
            # Book shifting
            StrategyCondition(
                feature="slope_asymmetry",
                operator=">",
                threshold=0,
                weight=0.5
            ),
        ],
        
        filters=[
            # Don't fade strong momentum
            StrategyCondition(
                feature="trade_intensity_60s",
                operator=">",
                threshold=80  # Too active
            ),
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.RANGING]
    )


# Strategy registry
STRATEGY_LIBRARY = {
    "absorption": create_absorption_strategy,
    "delta_divergence": create_delta_divergence_strategy,
    "liquidity_sweep": create_liquidity_sweep_strategy,
    "stacked_imbalance": create_stacked_imbalance_strategy,
    "value_area": create_value_area_strategy,
}


def get_all_strategies() -> Dict[str, StrategyDefinition]:
    """Get all strategy definitions"""
    return {name: factory() for name, factory in STRATEGY_LIBRARY.items()}


def get_strategy(name: str) -> Optional[StrategyDefinition]:
    """Get a specific strategy by name"""
    if name in STRATEGY_LIBRARY:
        return STRATEGY_LIBRARY[name]()
    return None

# ==============================================================================
# section 5 Optimization/optuna_optimizer.py
# ==============================================================================

"""
Optuna Optimizer - Enhanced for Perfect Optimization
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from loguru import logger

from knowledge.llm_advisor import LLMAdvisor, ABSORPTION_KNOWLEDGE, DELTA_DIVERGENCE_KNOWLEDGE
from knowledge.strategy_library import StrategyDefinition, get_strategy

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    """Results from optimization run"""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    study_name: str
    timestamp: datetime
    all_trials: List[Dict] = None
    objective_type: str = "robust"


class StrategyOptimizer:
    """
    Enhanced strategy parameter optimizer.
    
    Features:
    1. Warm-start from LLM domain knowledge
    2. Multiple objective functions (sharpe, profit, robust)
    3. Minimum trade count enforcement
    4. Correlation-aware parameter constraints
    5. Advanced pruning
    """
    
    def __init__(
        self,
        strategy_name: str,
        llm_advisor: Optional[LLMAdvisor] = None,
        storage: str = "sqlite:///optuna_studies.db"
    ):
        self.strategy_name = strategy_name
        self.llm_advisor = llm_advisor
        self.storage = storage
        
        self.base_strategy = get_strategy(strategy_name)
        if not self.base_strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        self.param_ranges = self._get_parameter_ranges()
        self.warm_start_params = self._get_warm_start_params()
    
    def _get_parameter_ranges(self) -> Dict[str, Dict]:
        """
        Full optimizer-controlled parameter space.
        CRITICAL: 
        - NO dead parameters (everything used in evaluation)
        - All regime multipliers included
        - Proper bounds for XRP/USDT microstructure
        
        Naming convention: {strategy}__{type}_{feature}_{bound}
        """
        
        # DEBUG: Log strategy being configured
        logger.debug(f"[_get_parameter_ranges] Building ranges for {self.strategy_name}")
        
        # CRITICAL FIX: Removed base stop_loss_atr_mult and take_profit_atr_mult 
        # because evaluate() uses regime-specific multipliers exclusively.
        # Including them would create "dead parameters" that Optuna wastes time optimizing.
        
        common = {
            "stop_loss_atr_mult":           {"min": 0.5,  "max": 20.0,  "step": 0.25,  "default": 2.5},
            "take_profit_atr_mult":         {"min": 1.0,  "max": 50.0,  "step": 0.5,   "default": 6.0},
            "trailing_stop_activation_pct": {"min": 0.001,"max": 0.05,  "step": 0.001, "default": 0.005},
            "min_conditions_satisfied":     {"min": 1,    "max": 6,     "step": 1,     "default": 2,   "type": "int"},
            "min_score_threshold":          {"min": 0.5,  "max": 10.0,  "step": 0.25,  "default": 2.5},
            "base_position_pct":            {"min": 0.01, "max": 0.50,  "step": 0.01,  "default": 0.10},
        }  # [APPLIED] Wide bounds to free optimizer, max 6 conditions to prevent dead strategies
        
        strategy_params = {
            "absorption": {
                "abs__entry_str_min":      {"min": 0.05, "max": 0.90, "step": 0.05,  "default": 0.45},
                "abs__entry_vol_min":      {"min": 0.3,  "max": 5.0,  "step": 0.1,   "default": 1.0},
                "abs__entry_chg60_max":    {"min": 0.0,  "max": 0.05, "step": 0.001, "default": 0.001},
                "abs__entry_delta_min":    {"min": 0,    "max": 50000,"step": 100,   "default": 0,    "type": "int"},
                "abs__entry_imbal_min":    {"min": 0.0,  "max": 0.50, "step": 0.01,  "default": 0.05},
                "abs__entry_poc_range":    {"min": 0.0,  "max": 0.05, "step": 0.001, "default": 0.005},
                "abs__filter_spread_max":  {"min": 1.0,  "max": 100.0,"step": 1.0,   "default": 15.0},
                "abs__filter_bid_min":     {"min": 10.0, "max": 100000.0, "step": 100.0, "default": 1500.0},
                "abs__filter_ask_min":     {"min": 10.0, "max": 100000.0, "step": 100.0, "default": 1500.0},
                "abs__filter_chg300_range":{"min": 0.003,"max": 0.10, "step": 0.001, "default": 0.005},
            },
        }  # [APPLIED]
        
        # Merge parameters
        result = common.copy()
        strat_key = self.strategy_name.lower()
        
        if strat_key in strategy_params:
            result.update(strategy_params[strat_key])
            logger.debug(f"[_get_parameter_ranges] Added {len(strategy_params[strat_key])} strategy-specific params")
        else:
            logger.warning(f"[_get_parameter_ranges] No specific params defined for {strat_key}")
        
        logger.info(f"[_get_parameter_ranges] Total parameter space: {len(result)} dimensions")
        return result
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial with proper type handling.
        DEBUG: Logs every suggestion for traceability.
        """
        ranges = self._get_parameter_ranges()
        params = {}
        
        logger.debug(f"[_suggest_params] Suggesting {len(ranges)} parameters for trial {trial.number}")
        
        for param_name, config in ranges.items():
            param_type = config.get("type", "float")
            step = config.get("step")
            
            try:
                if param_type == "int":
                    value = trial.suggest_int(
                        param_name,
                        config["min"],
                        config["max"],
                        step=step or 1
                    )
                elif param_type == "categorical":
                    value = trial.suggest_categorical(param_name, config["choices"])
                else:  # float
                    value = trial.suggest_float(
                        param_name,
                        config["min"],
                        config["max"],
                        step=step
                    )
                
                params[param_name] = value
                logger.debug(f"[_suggest_params] {param_name} = {value}")
                
            except Exception as e:
                logger.error(f"[_suggest_params] Failed to suggest {param_name}: {e}")
                raise
        
        return params
    
    def _get_warm_start_params(self) -> Dict[str, float]:
        """Get initial parameter values for warm start"""
        warm_start = {}
        
        for param, range_dict in self.param_ranges.items():
            if "default" in range_dict:
                warm_start[param] = range_dict["default"]
        
        # Enforce logical constraints in warm start (regime-specific multipliers)
        regimes = ["high_vol", "low_vol", "trending"]
        for regime in regimes:
            sl_key = f"sl_mult_{regime}"
            tp_key = f"tp_mult_{regime}"
            if sl_key in warm_start and tp_key in warm_start:
                # Ensure 1.5:1 minimum risk/reward for each regime
                warm_start[tp_key] = max(
                    warm_start[tp_key],
                    warm_start[sl_key] * 1.5
                )
        
        if self.llm_advisor:
            try:
                llm_params = self.llm_advisor.get_initial_parameters(self.strategy_name)
                for param, value in llm_params.items():
                    if param in self.param_ranges:
                        range_dict = self.param_ranges[param]
                        if range_dict["min"] <= value <= range_dict["max"]:
                            warm_start[param] = value
            except Exception as e:
                logger.warning(f"Failed to get LLM initial params: {e}")
        
        return warm_start
    
    def create_objective(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        objective_type: str = "robust"
    ) -> Callable[[optuna.Trial], float]:
        """
        Create Optuna objective function.
        CRITICAL: Uses _suggest_params and _enforce_constraints (no re-suggestion).
        """
        
        def objective(trial: optuna.Trial) -> float:
            # DEBUG: Log trial start
            logger.debug(f"[objective] Starting trial {trial.number}")
            
            # Step 1: Suggest parameters (single source of truth)
            params = self._suggest_params(trial)
            
            # Step 2: Enforce constraints (clamping only, no re-suggestion)
            params = self._enforce_constraints(params)
            
            # DEBUG: Log final params being used
            logger.debug(f"[objective] Trial {trial.number} final params: {params}")
            
            # Step 3: Run backtest
            try:
                metrics = backtest_fn(params)
                logger.debug(f"[objective] Trial {trial.number} metrics: {metrics}")
            except Exception as e:
                logger.error(f"[objective] Trial {trial.number} backtest failed: {e}")
                return float("-inf")
            
            # Check if penalized (too few trades)
            if metrics.get("_penalized", False):
                logger.warning(f"[objective] Trial {trial.number} penalized (insufficient trades)")
                trial.set_user_attr("penalized", True)
                return float("-inf")
            
            # Store all metrics as user attributes for analysis
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            
            # Calculate objective score
            if objective_type == "sharpe":
                score = metrics.get("sharpe_ratio", 0)
            elif objective_type == "profit":
                score = self._profit_objective(metrics)
            elif objective_type == "profit_dd_trades":
                score = self._profit_dd_trades_objective(metrics)
            else:  # robust
                score = self._robust_objective(metrics)
            
            # DEBUG: Log score
            logger.debug(f"[objective] Trial {trial.number} score ({objective_type}): {score:.4f}")
            
            # Report for pruning
            trial.report(score, step=1)
            if trial.should_prune():
                logger.debug(f"[objective] Trial {trial.number} pruned")
                raise optuna.TrialPruned()
            
            return score
        
        return objective
    
    def _enforce_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clamp parameters to prevent numerical crashes ONLY.
        CRITICAL: Never call trial.suggest_*() here - that creates duplicate parameters.
        """
        # Prevent ATR math crashes (keep within float-safe bounds)
        params["stop_loss_atr_mult"] = max(0.5, min(20.0, params.get("stop_loss_atr_mult", 2.5)))
        params["take_profit_atr_mult"] = max(1.0, min(50.0, params.get("take_profit_atr_mult", 6.0)))
        
        # Prevent filter degeneracy (rejecting all signals or accepting all)
        params["abs__filter_spread_max"] = max(1.0, min(100.0, params.get("abs__filter_spread_max", 15.0)))
        params["abs__filter_bid_min"] = max(10.0, min(100000.0, params.get("abs__filter_bid_min", 1500.0)))
        params["abs__filter_ask_min"] = max(10.0, min(100000.0, params.get("abs__filter_ask_min", 1500.0)))
        params["abs__filter_chg300_range"] = max(0.003, min(0.10, params.get("abs__filter_chg300_range", 0.005)))
        
        return params  # [APPLIED]

    
    def _profit_objective(self, metrics: Dict[str, float]) -> float:
        """
        Profit-focused objective with drawdown penalty.
        
        Score = Return - 3 * MaxDrawdown
        """
        total_return = metrics.get("total_return_pct", 0) * 100  # Convert to %
        max_drawdown = metrics.get("max_drawdown_pct", 0) * 100
        
        return total_return - 3 * max_drawdown
    
    def _robust_objective(self, metrics: Dict[str, float]) -> float:
        """
        Robust multi-factor objective.
        
        Combines:
        - Sharpe ratio (risk-adjusted returns)
        - Profit factor (win/loss ratio)
        - Win rate (consistency)
        - Drawdown penalty (risk control)
        
        This objective prefers strategies that are:
        - Profitable (positive return)
        - Consistent (high win rate)
        - Risk-controlled (low drawdown)
        - Efficient (good profit factor)
        """
        sharpe = metrics.get("sharpe_ratio", 0)
        profit_factor = metrics.get("profit_factor", 0)
        win_rate = metrics.get("win_rate", 0)
        max_drawdown = metrics.get("max_drawdown_pct", 0)
        total_trades = metrics.get("total_trades", 0)
        
        # Component 1: Sharpe (normalized to 0-1 range, cap at 3)
        sharpe_component = min(sharpe / 3.0, 1.0)
        
        # Component 2: Profit factor (normalized, cap at 3)
        pf_component = min(profit_factor / 3.0, 1.0) if profit_factor > 0 else 0
        
        # Component 3: Win rate (direct, but weighted lower)
        win_rate_component = win_rate * 0.8  # 80% win rate = 0.64
        
        # Component 4: Drawdown penalty (exponential decay)
        # 0% DD = 1.0, 10% DD = 0.37, 20% DD = 0.14
        drawdown_penalty = np.exp(-max_drawdown * 10)
        
        # Component 5: Trade count sufficiency
        # Sigmoid function: approaches 1 around 100 trades
        trade_sufficiency = 1 / (1 + np.exp(-0.05 * (total_trades - 100)))
        
        # Weighted combination
        weights = {
            "sharpe": 0.30,
            "profit_factor": 0.25,
            "win_rate": 0.15,
            "drawdown": 0.20,
            "trade_count": 0.10
        }
        
        score = (
            weights["sharpe"] * sharpe_component +
            weights["profit_factor"] * pf_component +
            weights["win_rate"] * win_rate_component +
            weights["drawdown"] * drawdown_penalty +
            weights["trade_count"] * trade_sufficiency
        )
        
        # Hard penalty: negative return = bad
        if metrics.get("total_return_pct", 0) < 0:
            score *= 0.5
        
        return score
    
    def _profit_dd_trades_objective(self, metrics: Dict[str, float]) -> float:
        """
        Custom objective balancing winning trades, drawdown minimization, and average win size.
        
        Weights:
        - Winning trades: 35% (sigmoid-scaled trade count)
        - Drawdown penalty: 40% (exponential decay)
        - Average win size: 25% (normalized avg_win)
        
        This objective prioritizes strategies that:
        - Generate consistent winning trades
        - Maintain low drawdown risk
        - Achieve meaningful per-trade profitability
        """
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0)
        winning_trades = total_trades * win_rate
        
        max_drawdown = metrics.get("max_drawdown_pct", 0)
        avg_win = metrics.get("avg_win", 0)  # Use avg_win not avg_win_pct
        
        # Component 1: Winning trades (sigmoid scaling, peaks at ~50 trades)
        # 10 winning trades = ~0.5, 50 winning trades = ~0.9
        winning_trades_component = 1 / (1 + np.exp(-0.15 * (winning_trades - 25)))
        
        # Component 2: Drawdown penalty (exponential decay, stronger than robust)
        # 0% DD = 1.0, 5% DD = 0.61, 10% DD = 0.37, 15% DD = 0.22
        drawdown_penalty = np.exp(-max_drawdown * 20)
        
        # Component 3: Average win size (normalized, assume $10-100 range is good)
        # Scale so $50 avg win = 1.0, $10 = 0.2, $100 = 2.0 (capped at 1.0)
        avg_win_component = min(avg_win / 50.0, 1.0) if avg_win > 0 else 0
        
        # Weighted combination
        weights = {
            "winning_trades": 0.35,
            "drawdown": 0.40,
            "avg_win": 0.25
        }
        
        score = (
            weights["winning_trades"] * winning_trades_component +
            weights["drawdown"] * drawdown_penalty +
            weights["avg_win"] * avg_win_component
        )
        
        # Hard penalty: no winning trades = very bad
        if winning_trades < 1:
            score *= 0.1
        
        return score
    
    def optimize(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: int = 200,
        n_jobs: int = -1,
        timeout: Optional[int] = None,
        objective_type: str = "robust"
    ) -> OptimizationResult:
        """
        Run optimization with warm start and enhanced objective.
        """
        study_name = f"{self.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Enhanced sampler with warm start
        sampler = TPESampler(
            n_startup_trials=20,
            multivariate=True,  # Consider parameter correlations
            warn_independent_sampling=False
        )
        
        # Hyperband pruner for aggressive early stopping
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=1,
            reduction_factor=3
        )
        
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        # Enqueue warm-start trial
        if self.warm_start_params:
            valid_warm_start = {
                k: v for k, v in self.warm_start_params.items()
                if k in self.param_ranges
            }
            if valid_warm_start:
                study.enqueue_trial(valid_warm_start)
                logger.info(f"Enqueued warm-start trial with {len(valid_warm_start)} params")
        
        objective = self.create_objective(backtest_fn, objective_type)
        
        logger.info(f"Starting optimization: {n_trials} trials, objective={objective_type}, n_jobs={n_jobs}")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=False  # Cleaner logs
        )
        
        # Filter out penalized trials for analysis
        valid_trials = [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "user_attrs": t.user_attrs
            }
            for t in study.trials
            if t.value is not None and not t.user_attrs.get("_penalized", False)
        ]
        
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study_name=study_name,
            timestamp=datetime.now(),
            all_trials=valid_trials,
            objective_type=objective_type
        )
        
        logger.info(f"Optimization complete.")
        logger.info(f"  Best score ({objective_type}): {result.best_score:.4f}")
        logger.info(f"  Valid trials: {len(valid_trials)} / {len(study.trials)}")
        logger.info(f"  Best params: {result.best_params}")
        
        return result
    
    def save_results(self, result: OptimizationResult, path: str) -> None:
        """Save optimization results to file"""
        output = {
            "strategy_name": self.strategy_name,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "valid_trials": len(result.all_trials) if result.all_trials else 0,
            "timestamp": result.timestamp.isoformat(),
            "objective_type": result.objective_type,
            "param_ranges": self.param_ranges,
            "warm_start_params": self.warm_start_params
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved results to {path}")
    
    def analyze_parameter_sensitivity(self, result: OptimizationResult) -> Dict[str, float]:
        """
        Analyze which parameters most affect the objective.
        Returns importance scores for each parameter.
        """
        if not result.all_trials or len(result.all_trials) < 10:
            return {}
        
        # Get parameter names
        param_names = list(result.all_trials[0]["params"].keys())
        
        importance = {}
        
        for param in param_names:
            # Get param values and corresponding scores
            pairs = [
                (t["params"].get(param), t["value"])
                for t in result.all_trials
                if param in t["params"] and t["value"] is not None
            ]
            
            if len(pairs) < 5:
                continue
            
            values = np.array([p[0] for p in pairs])
            scores = np.array([p[1] for p in pairs])
            
            # Calculate correlation as importance
            if np.std(values) > 0 and np.std(scores) > 0:
                correlation = np.corrcoef(values, scores)[0, 1]
                importance[param] = abs(correlation)
            else:
                importance[param] = 0.0
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance