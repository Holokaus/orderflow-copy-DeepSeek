# ==============================================================================
#  SECTION 2: backtesting/engine.py
# ==============================================================================

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
    
    
    
# ==============================================================================
# SECTION 4: knowledge/strategy_library.py
# ==============================================================================
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
    max_holding_seconds: int = 3600
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
        
        bullish_score = (
            (1 if delta > 0 else 0) +
            (1 if imbalance > 0.1 else 0) + # Use 0.1 not 0.0 — filters noise Filters micro-fluctuation
            (1 if pressure > 0 else 0)
        )
        
        if bullish_score >= 2:
            return SignalType.STRONG_BUY if score > self.min_score_threshold * 1.5 else SignalType.BUY
        elif bullish_score <= 1:
            return SignalType.STRONG_SELL if score > self.min_score_threshold * 1.5 else SignalType.SELL
        else:
            return SignalType.NEUTRAL
    
    def _estimate_atr(self, state: OrderFlowState, default: float = 100.0) -> float:
        """Estimate ATR using True Range from recent trades.
        
        FIX 1: Calculate True Range (High-Low) from last 500-1000 trades
        with a 0.3% minimum floor.
        """
        mid_price = state.order_book.mid_price
        
        # Fallback if no mid price
        if mid_price <= 0:
            return default
        
        # Get recent trades - prefer state.recent_trades, fallback to empty list
        trades = state.recent_trades if state.recent_trades else []
        
        # Need at least some trades to calculate range
        if len(trades) < 2:
            return mid_price * 0.003  # 0.3% fallback
        
        # Use last 500-1000 trades
        sample_trades = trades[-1000:] if len(trades) > 1000 else trades[-500:] if len(trades) >= 500 else trades
        
        # Calculate True Range: max(price) - min(price)
        prices = [t.price for t in sample_trades]
        if not prices:
            return mid_price * 0.003
        
        price_range = max(prices) - min(prices)
        
        # Apply 0.3% minimum floor
        min_atr = mid_price * 0.005 #0.003 for BTC it's too tight for XRP make the stop inside the spread Change it to 0.5%
        atr = max(price_range, min_atr)
        
        return atr


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
# SECTION 6: optimization/optuna_optimizer.py
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
        
        common_params = {
            # Regime-specific multipliers (CRITICAL: These are actually used in evaluate())
            "sl_mult_high_vol":            {"min": 2.0, "max": 6.0, "step": 0.25, "default": 3.5},
            "sl_mult_low_vol":             {"min": 1.0, "max": 3.0, "step": 0.25, "default": 1.8},
            "sl_mult_trending":            {"min": 1.5, "max": 4.0, "step": 0.25, "default": 2.5},
            "tp_mult_high_vol":            {"min": 3.0, "max": 10.0, "step": 0.5, "default": 7.0},
            "tp_mult_low_vol":             {"min": 1.5, "max": 4.0, "step": 0.25, "default": 2.5},
            "tp_mult_trending":            {"min": 3.0, "max": 8.0, "step": 0.5, "default": 5.0},
            
            # Strategy logic parameters
            "min_conditions_satisfied":    {"min": 1, "max": 5, "step": 1, "default": 2, "type": "int"},
            "min_score_threshold":         {"min": 1.5, "max": 4.0, "step": 0.25, "default": 2.5},
            "base_position_pct":           {"min": 0.02, "max": 0.15, "step": 0.01, "default": 0.08},
            "trailing_stop_activation_pct": {"min": 0.002, "max": 0.015, "step": 0.001, "default": 0.005},
        }
        
        # Strategy-specific parameters (mapped via param_key in strategy definition)
        strategy_params = {
            "absorption": {
                # Entry conditions
                "abs__entry_str_min":      {"min": 0.15, "max": 0.65, "step": 0.05, "default": 0.30},
                "abs__entry_vol_min":      {"min": 0.7, "max": 2.5, "step": 0.1, "default": 1.0},
                "abs__entry_chg60_max":    {"min": 0.0003, "max": 0.004, "step": 0.0002, "default": 0.001},
                "abs__entry_delta_min":    {"min": 0, "max": 5000, "step": 100, "default": 0, "type": "int"},
                "abs__entry_imbal_min":    {"min": 0.02, "max": 0.20, "step": 0.02, "default": 0.05},
                "abs__entry_poc_range":    {"min": 0.001, "max": 0.015, "step": 0.001, "default": 0.005},
                
                # Filters (CRITICAL: These control market quality rejection thresholds)
                "abs__filter_spread_max":  {"min": 5.0, "max": 25.0, "step": 2.0, "default": 15.0},  # Max acceptable spread (bps)
                "abs__filter_bid_min":     {"min": 500.0, "max": 5000.0, "step": 250.0, "default": 1500.0},  # Min bid depth
                "abs__filter_ask_min":     {"min": 500.0, "max": 5000.0, "step": 250.0, "default": 1500.0},  # Min ask depth
                "abs__filter_chg300_range": {"min": 0.002, "max": 0.020, "step": 0.001, "default": 0.005},  # Dead market filter (symmetric)
            }
            # TODO: Add other strategies (delta_divergence, liquidity_sweep, etc.) here
        }
        
        # Merge parameters
        result = common_params.copy()
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
        Enforce mathematical constraints by CLAMPING ONLY.
        CRITICAL: NEVER call trial.suggest_* here - that creates duplicate parameters!
        
        Constraints enforced:
        1. TP >= 1.5x SL for each regime (risk/reward minimum)
        2. Price change range filters not too tight (prevent dead market rejection)
        3. Depth filters within reasonable bounds for XRP
        """
        logger.debug(f"[_enforce_constraints] Enforcing constraints on {len(params)} parameters")
        
        # Constraint 1: Regime-specific TP/SL ratios (CRITICAL FIX: AI3 Gap 2)
        regimes = ["high_vol", "low_vol", "trending"]
        
        for regime in regimes:
            sl_key = f"sl_mult_{regime}"
            tp_key = f"tp_mult_{regime}"
            
            if sl_key in params and tp_key in params:
                sl_val = params[sl_key]
                tp_val = params[tp_key]
                min_tp = sl_val * 1.5
                
                if tp_val < min_tp:
                    old_tp = tp_val
                    params[tp_key] = min_tp
                    logger.debug(f"[_enforce_constraints] CLAMPED {tp_key}: {old_tp} -> {min_tp} (1.5x {sl_key})")
        
        # Constraint 2: Prevent dead-market filters (min activity threshold)
        for key in list(params.keys()):
            if "chg300_range" in key or "chg60_max" in key:
                old_val = params[key]
                # Ensure minimum 0.2% activity allowed
                params[key] = max(0.002, min(0.02, params[key]))
                if params[key] != old_val:
                    logger.debug(f"[_enforce_constraints] CLAMPED {key}: {old_val} -> {params[key]}")
        
        # Constraint 3: XRP depth scale bounds (prevent unrealistic values)
        for key in list(params.keys()):
            if "filter_bid_min" in key or "filter_ask_min" in key:
                old_val = params[key]
                params[key] = max(100.0, min(10000.0, params[key]))
                if params[key] != old_val:
                    logger.debug(f"[_enforce_constraints] CLAMPED {key}: {old_val} -> {params[key]}")
        
        # Constraint 4: Min conditions can't exceed available conditions
        # (This is validated at runtime, but we can warn here)
        
        logger.info(f"[_enforce_constraints] Constraints applied. Final param count: {len(params)}")
        return params
    
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