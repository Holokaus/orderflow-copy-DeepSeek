        for price, data in refill_counts.items():
            if data['refill_count'] >= self.config.iceberg_refill_threshold:
                visible_size = data['size']
                estimated_hidden = visible_size * (data['refill_count'] - 1)
                
                if estimated_hidden >= self.config.iceberg_size_threshold:
                    icebergs.append(IcebergOrder(
                        timestamp=self._current_timestamp or datetime.now(),
                        price=price,
                        visible_size=visible_size,
                        estimated_hidden_size=estimated_hidden,
                        refill_count=data['refill_count'],
                        side=data['side'],
                        confidence=min(data['refill_count'] / 10.0, 1.0)
                    ))
        
        return icebergs[-5:]
    
    def _track_refills(
        self,
        prev_levels: List[PriceLevel],
        curr_levels: List[PriceLevel],
        refill_counts: Dict[float, Dict],
        side: Side
    ) -> None:
        """Track potential iceberg refills."""
        prev_sizes = {level.price: level.size for level in prev_levels[:10]}
        curr_sizes = {level.price: level.size for level in curr_levels[:10]}
        
        for price, curr_size in curr_sizes.items():
            if price in prev_sizes:
                prev_size = prev_sizes[price]
                if curr_size > prev_size * 1.1:  # Size increased significantly
                    if price not in refill_counts:
                        refill_counts[price] = {
                            'size': curr_size,
                            'refill_count': 1,
                            'side': side
                        }
                    else:
                        refill_counts[price]['refill_count'] += 1
                        refill_counts[price]['size'] = curr_size
    
    # ==================== PATTERN FEATURES (CRITICAL BRIDGE) ====================
    
    def _compute_pattern_features(self, state: OrderFlowState) -> Dict[str, float]:
        """
        CRITICAL: Bridge detected patterns → strategy-consumable features.
        
        This is the missing link that makes patterns actionable for strategies.
        """
        features: Dict[str, float] = {}
        
        # Absorption features
        if state.absorptions:
            latest_abs = state.absorptions[-1]
            features["absorption_strength"] = latest_abs.strength
            features["absorption_volume"] = latest_abs.absorbed_volume
            features["absorption_duration"] = latest_abs.duration_seconds
            features["absorption_side"] = 1 if latest_abs.absorbing_side == Side.BUY else -1
        else:
            features["absorption_strength"] = 0
            features["absorption_volume"] = 0
            features["absorption_duration"] = 0
            features["absorption_side"] = 0
        
        # Imbalance features
        if state.imbalances:
            strongest_imb = max(state.imbalances, key=lambda x: x.imbalance_ratio)
            features["imbalance_ratio"] = strongest_imb.imbalance_ratio
            features["imbalance_side"] = 1 if strongest_imb.direction == Side.BUY else -1
            features["imbalance_stacked"] = float(strongest_imb.is_stacked)
            features["imbalance_stack_count"] = strongest_imb.stack_count
        else:
            features["imbalance_ratio"] = 0
            features["imbalance_side"] = 0
            features["imbalance_stacked"] = 0
            features["imbalance_stack_count"] = 0
        
        # Sweep features
        if state.sweeps:
            latest_sweep = state.sweeps[-1]
            features["sweep_strength"] = latest_sweep.reversal_strength
            features["sweep_volume"] = latest_sweep.volume_swept
            features["sweep_speed"] = latest_sweep.speed_seconds
            features["sweep_direction"] = 1 if latest_sweep.direction == Side.BUY else -1
        else:
            features["sweep_strength"] = 0
            features["sweep_volume"] = 0
            features["sweep_speed"] = 0
            features["sweep_direction"] = 0
        
        # Iceberg features
        if state.icebergs:
            strongest_iceberg = max(state.icebergs, key=lambda x: x.confidence)
            features["iceberg_confidence"] = strongest_iceberg.confidence
            features["iceberg_hidden_ratio"] = (
                strongest_iceberg.estimated_hidden_size / 
                (strongest_iceberg.visible_size + 1e-9)
            )
            features["iceberg_refill_count"] = strongest_iceberg.refill_count
            features["iceberg_side"] = 1 if strongest_iceberg.side == Side.BUY else -1
        else:
            features["iceberg_confidence"] = 0
            features["iceberg_hidden_ratio"] = 0
            features["iceberg_refill_count"] = 0
            features["iceberg_side"] = 0
        
        # Pattern counts
        features["absorption_count"] = len(state.absorptions)
        features["imbalance_count"] = len(state.imbalances)
        features["sweep_count"] = len(state.sweeps)
        features["iceberg_count"] = len(state.icebergs)
        
        return features
    
    # ==================== COMPOSITE FEATURES ====================
    
    def _compute_composite_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Compute derived features that strategies need."""
        composites: Dict[str, float] = {}
        
        # Momentum indicators
        composites["volume_momentum"] = (
            features.get("total_volume_15s", 0) / (features.get("total_volume_300s", 0) + 1e-9)
        )
        composites["delta_momentum"] = (
            features.get("delta_15s", 0) / (features.get("abs_delta_300s", 0) + 1e-9)
        )
        
        # Divergence signals
        composites["price_volume_divergence"] = (
            features.get("price_change_pct_60s", 0) * 
            (features.get("delta_pct_60s", 0) * -1)
        )
        
        # Relative strength
        composites["bid_ask_relative_strength"] = (
            features.get("bid_pressure", 0) / (features.get("ask_pressure", 0) + 1e-9)
        )
        
        # Volume acceleration
        composites["volume_acceleration"] = (
            features.get("total_volume_15s", 0) / (features.get("total_volume_60s", 0) + 1e-9)
        )
        
        # Delta acceleration
        composites["delta_acceleration"] = (
            features.get("delta_15s", 0) / (features.get("delta_60s", 0) + 1e-9)
        )
        
        # Liquidity health
        composites["liquidity_health"] = (
            (features.get("bid_depth_10", 0) + features.get("ask_depth_10", 0)) /
            (features.get("spread_bps", 0) + 1e-9)
        )
        
        # Pressure divergence
        composites["pressure_divergence"] = (
            features.get("bid_pressure", 0) - features.get("ask_pressure", 0)
        )
        
        return composites
    
    # ==================== TEMPORAL FEATURES ====================
    
    def _store_feature_snapshot(
        self, 
        timestamp: datetime, 
        features: Dict[str, float]
    ) -> None:
        """Store temporal feature snapshots for time-series analysis."""
        self._feature_snapshots.append((timestamp, features.copy()))
        
        # Keep only recent snapshots (memory management)
        if len(self._feature_snapshots) > 1000:
            self._feature_snapshots = self._feature_snapshots[-500:]
    
    def get_temporal_features(
        self, 
        lookback_seconds: int = 300
    ) -> Dict[str, float]:
        """Extract temporal features from stored snapshots."""
        if not self._feature_snapshots:
            return {}
        
        cutoff = self._current_timestamp - timedelta(seconds=lookback_seconds)
        
        # Binary search for start index
        timestamps = [ts for ts, _ in self._feature_snapshots]
        idx = bisect.bisect_left(timestamps, cutoff)
        recent_snapshots = self._feature_snapshots[idx:]
        
        if len(recent_snapshots) < 2:
            return {}
        
        features: Dict[str, float] = {}
        
        # Trend features
        delta_trend = []
        volume_trend = []
        
        for i in range(1, len(recent_snapshots)):
            prev_features = recent_snapshots[i-1][1]
            curr_features = recent_snapshots[i][1]
            
            delta_trend.append(curr_features.get("delta", 0) - prev_features.get("delta", 0))
            volume_trend.append(
                curr_features.get("total_volume", 0) - prev_features.get("total_volume", 0)
            )
        
        if delta_trend:
            features["delta_trend_slope"] = sum(delta_trend) / len(delta_trend)
            features["delta_trend_volatility"] = np.std(delta_trend) if len(delta_trend) > 1 else 0
        
        if volume_trend:
            features["volume_trend_slope"] = sum(volume_trend) / len(volume_trend)
            features["volume_trend_volatility"] = np.std(volume_trend) if len(volume_trend) > 1 else 0
        
        return features

# ==============================================================================
# SECTION 6: backtesting/engine.py
# ==============================================================================

"""
Backtesting Engine (v6.12 Expert Optimized)
Event-driven backtesting with lightweight data processing.

Key Optimizations:
- O(1) position tracking with incremental updates
- Minimal memory footprint for cloud deployment
- Risk management integration
- Feature caching for repeated evaluations
- Walk-forward validation support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from core.data_structures import Side, SignalType, Signal
from core.feature_engine import FeatureEngine, FeatureConfig


class BacktestResult(Enum):
    SUCCESS = "success"
    INSUFFICIENT_DATA = "insufficient_data"
    NO_TRADES = "no_trades"
    ERROR = "error"


@dataclass
class TradeRecord:
    """Individual trade record for backtesting"""
    timestamp: datetime
    side: Side
    price: float
    size: float
    pnl: float = 0.0
    fees: float = 0.0
    reason: str = ""
    
    @property
    def value(self) -> float:
        return self.price * self.size


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    # Basic metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    volatility_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    value_at_risk_pct: float = 0.0  # 95% VaR
    expected_shortfall_pct: float = 0.0  # 95% CVaR
    calmar_ratio: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_hours: float = 0.0
    
    # Performance metrics
    kelly_criterion: float = 0.0
    recovery_factor: float = 0.0
    payoff_ratio: float = 0.0
    
    # Risk-adjusted metrics
    information_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # Additional stats
    total_fees: float = 0.0
    final_equity: float = 0.0
    total_volume: float = 0.0
    
    def compute_from_trades(
        self, 
        trades: List[TradeRecord], 
        initial_capital: float,
        total_days: float
    ) -> None:
        """Compute all metrics from trade records"""
        if not trades:
            return
        
        # Extract equity curve
        equity_curve = self._build_equity_curve(trades, initial_capital)
        
        # Basic returns
        self.final_equity = equity_curve[-1]
        self.total_return_pct = (self.final_equity - initial_capital) / initial_capital
        
        if total_days > 0:
            self.annualized_return_pct = (
                (1 + self.total_return_pct) ** (365 / total_days) - 1
            )
        
        # Volatility and Sharpe
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            self.volatility_pct = float(np.std(returns))
            
            if self.volatility_pct > 0:
                self.sharpe_ratio = self.annualized_return_pct / self.volatility_pct
        
        # Drawdown
        self.max_drawdown_pct = self._calculate_max_drawdown(equity_curve, initial_capital)
        
        # Trade metrics
        self.total_trades = len(trades)
        self.total_fees = sum(t.fees for t in trades)
        self.total_volume = sum(t.value for t in trades)
        
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            self.win_rate = len(winning_trades) / len(trades)
            
            if winning_trades:
                self.avg_win_pct = np.mean([t.pnl / initial_capital for t in winning_trades])
            if losing_trades:
                self.avg_loss_pct = np.mean([t.pnl / initial_capital for t in losing_trades])
            
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            
            if total_losses > 0:
                self.profit_factor = total_wins / total_losses
            
            # Payoff ratio
            if losing_trades:
                avg_win = total_wins / len(winning_trades) if winning_trades else 0
                avg_loss = total_losses / len(losing_trades)
                self.payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Risk metrics
        if len(equity_curve) > 30:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            self.value_at_risk_pct = float(np.percentile(returns, 5))
            self.expected_shortfall_pct = float(np.mean(returns[returns <= self.value_at_risk_pct]))
        
        # Advanced ratios
        if self.max_drawdown_pct > 0:
            self.calmar_ratio = self.annualized_return_pct / self.max_drawdown_pct
        
        if self.sharpe_ratio > 0:
            self.sortino_ratio = self.annualized_return_pct / self._downside_deviation(returns)
        
        # Kelly criterion
        if self.win_rate > 0 and self.payoff_ratio > 0:
            win_rate = self.win_rate
            loss_rate = 1 - win_rate
            avg_win = self.avg_win_pct if self.avg_win_pct > 0 else 0
            avg_loss = abs(self.avg_loss_pct) if self.avg_loss_pct < 0 else 0
            
            if avg_loss > 0:
                kelly = win_rate / avg_loss - loss_rate / avg_win
                self.kelly_criterion = max(0, kelly)
        
        # Recovery factor
        if self.max_drawdown_pct > 0:
            self.recovery_factor = self.total_return_pct / self.max_drawdown_pct
    
    def _build_equity_curve(
        self, 
        trades: List[TradeRecord], 
        initial_capital: float
    ) -> List[float]:
        """Build equity curve from trades"""
        equity = initial_capital
        curve = [equity]
        
        for trade in trades:
            equity += trade.pnl
            curve.append(equity)
        
        return curve
    
    def _calculate_max_drawdown(
        self, 
        equity_curve: List[float], 
        initial_capital: float
    ) -> float:
        """Calculate maximum drawdown percentage"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = initial_capital
        max_dd = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _downside_deviation(self, returns: np.ndarray) -> float:
        """Calculate downside deviation for Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        
        return float(np.std(negative_returns))


@dataclass
class BacktestEngine:
    """
    Lightweight event-driven backtesting engine.
    
    Optimized for:
    - Cloud deployment (minimal memory)
    - Fast parameter sweeps
    - Risk management integration
    - Feature caching
    """
    
    initial_capital: float
    fee_pct: float = 0.0004
    slippage_pct: float = 0.0005
    feature_config: Optional[FeatureConfig] = None
    
    # Internal state
    _current_position: float = field(default=0.0, init=False)
    _entry_price: float = field(default=0.0, init=False)
    _equity: float = field(default=0.0, init=False)
    _trades: List[TradeRecord] = field(default_factory=list, init=False)
    _feature_engine: Optional[FeatureEngine] = field(default=None, init=False)
    
    def __post_init__(self):
        self._equity = self.initial_capital
        if self.feature_config:
            self._feature_engine = FeatureEngine(self.feature_config)
    
    def run(
        self, 
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict[str, Any] = None
    ) -> BacktestMetrics:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with columns [timestamp, price, size, side, bid_price, ask_price, bid_size, ask_size]
            strategy_func: Function that takes features dict and returns Signal
            params: Strategy parameters
        """
        params = params or {}
        
        # Reset state
        self._reset()
        
        # Prepare data
        if not self._validate_data(data):
            return BacktestMetrics()
        
        processed_data = self._preprocess_data(data)
        
        # Run simulation
        for idx, row in processed_data.iterrows():
            self._process_bar(row, strategy_func, params)
        
        # Close any open position
        self._close_position(processed_data.iloc[-1], "end_of_data")
        
        # Calculate metrics
        total_days = (processed_data.iloc[-1]['timestamp'] - processed_data.iloc[0]['timestamp']).total_seconds() / 86400
        
        metrics = BacktestMetrics()
        metrics.compute_from_trades(self._trades, self.initial_capital, total_days)
        
        return metrics
    
    def _reset(self) -> None:
        """Reset engine state between runs"""
        self._current_position = 0.0
        self._entry_price = 0.0
        self._equity = self.initial_capital
        self._trades.clear()
        
        if self._feature_engine:
            self._feature_engine.reset()
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns"""
        required_cols = ['timestamp', 'price', 'size', 'side']
        return all(col in data.columns for col in required_cols)
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and sort data"""
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add order book columns if missing
        if 'bid_price' not in df.columns:
            df['bid_price'] = df['price'] * 0.9999
        if 'ask_price' not in df.columns:
            df['ask_price'] = df['price'] * 1.0001
        if 'bid_size' not in df.columns:
            df['bid_size'] = df['size'] * 0.5
        if 'ask_size' not in df.columns:
            df['ask_size'] = df['size'] * 0.5
        
        return df
    
    def _process_bar(
        self, 
        row: pd.Series, 
        strategy_func: Callable,
        params: Dict[str, Any]
    ) -> None:
        """Process a single bar/tick"""
        # Create order book snapshot
        order_book = self._create_order_book_from_row(row)
        
        # Create trade object
        trade = self._create_trade_from_row(row)
        
        # Update feature engine
        if self._feature_engine:
            state = self._feature_engine.update(order_book, [trade])
            features = state.features
        else:
            features = {}
        
        # Get strategy signal
        signal = strategy_func(features, params)
        
        # Execute signal
        self._execute_signal(signal, row, features)
    
    def _create_order_book_from_row(self, row: pd.Series):
        """Create order book from DataFrame row"""
        from core.data_structures import OrderBook, PriceLevel
        
        return OrderBook(
            timestamp=row['timestamp'],
            bids=[PriceLevel(
                price=row['bid_price'],
                size=row['bid_size'],
                timestamp=row['timestamp']
            )],
            asks=[PriceLevel(
                price=row['ask_price'], 
                size=row['ask_size'],
                timestamp=row['timestamp']
            )]
        )
    
    def _create_trade_from_row(self, row: pd.Series):
        """Create trade from DataFrame row"""
        from core.data_structures import Trade, Side
        
        return Trade(
            timestamp=row['timestamp'],
            price=row['price'],
            size=row['size'],
            side=Side.BUY if row['side'] == 'buy' else Side.SELL
        )
    
    def _execute_signal(
        self, 
        signal: Signal, 
        row: pd.Series,
        features: Dict[str, float]
    ) -> None:
        """Execute trading signal"""
        if not signal.is_actionable:
            return
        
        # Determine trade side and size
        if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            trade_side = Side.BUY
        elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            trade_side = Side.SELL
        else:
            return
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, row['price'])
        
        # Check if we need to close existing position
        if self._current_position != 0:
            current_side = Side.BUY if self._current_position > 0 else Side.SELL
            if current_side != trade_side:
                self._close_position(row, "signal_reversal")
        
        # Open new position
        if self._current_position == 0:
            self._open_position(trade_side, position_size, row, signal.primary_reason)
    
    def _calculate_position_size(self, signal: Signal, price: float) -> float:
        """Calculate position size based on signal and risk management"""
        # Simple fixed percentage for now
        max_position_value = self._equity * 0.1  # 10% of equity
        position_size = max_position_value / price
        
        # Apply signal position size modifier
        position_size *= signal.position_size
        
        return min(position_size, max_position_value / price)
    
    def _open_position(
        self, 
        side: Side, 
        size: float, 
        row: pd.Series,
        reason: str
    ) -> None:
        """Open a new position"""
        price = row['ask_price'] if side == Side.BUY else row['bid_price']
        price += price * self.slippage_pct  # Apply slippage
        
        self._current_position = size if side == Side.BUY else -size
        self._entry_price = price
        
        # Record trade
        trade = TradeRecord(
            timestamp=row['timestamp'],
            side=side,
            price=price,
            size=size,
            fees=price * size * self.fee_pct,
            reason=reason
        )
        
        self._trades.append(trade)
        self._equity -= trade.fees
    
    def _close_position(self, row: pd.Series, reason: str) -> None:
        """Close current position"""
        if self._current_position == 0:
            return
        
        side = Side.SELL if self._current_position > 0 else Side.BUY
        size = abs(self._current_position)
        
        price = row['bid_price'] if side == Side.SELL else row['ask_price']
        price -= price * self.slippage_pct  # Apply slippage (opposite for closing)
        
        # Calculate P&L
        if self._current_position > 0:  # Long position
            pnl = (price - self._entry_price) * size
        else:  # Short position
            pnl = (self._entry_price - price) * size
        
        fees = price * size * self.fee_pct
        
        # Record trade
        trade = TradeRecord(
            timestamp=row['timestamp'],
            side=side,
            price=price,
            size=size,
            pnl=pnl,
            fees=fees,
            reason=reason
        )
        
        self._trades.append(trade)
        self._equity += pnl - fees
        
        # Reset position
        self._current_position = 0
        self._entry_price = 0
    
    # ==================== WALK-FORWARD VALIDATION ====================
    
@dataclass
class WalkForwardFold:
    """Single walk-forward validation fold"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict[str, Any]
    in_sample_metrics: BacktestMetrics
    out_of_sample_metrics: BacktestMetrics
    fold_number: int


@dataclass
class WalkForwardValidator:
    """
    Walk-forward validation with expanding/rolling windows.
    
    Prevents overfitting by testing on future unseen data.
    """
    
    train_days: int = 60
    test_days: int = 14
    step_days: int = 7
    min_oos_trades: int = 10
    
    def validate(
        self,
        data: pd.DataFrame,
        strategy: Callable,
        optimizer,
        n_trials_per_fold: int = 50,
        feature_config: FeatureConfig = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.
        
        Returns:
            {
                'summary': {...},
                'folds': [WalkForwardFold, ...],
                'params_stability': {...},
                'error': str (if any)
            }
        """
        folds = []
        skipped_folds = 0
        
        # Sort data
        data = data.sort_values('timestamp').reset_index(drop=True)
        start_date = data['timestamp'].min()
        end_date = data['timestamp'].max()
        
        current_train_end = start_date + timedelta(days=self.train_days)
        
        fold_num = 1
        
        while current_train_end + timedelta(days=self.test_days) <= end_date:
            train_start = current_train_end - timedelta(days=self.train_days)
            test_end = current_train_end + timedelta(days=self.test_days)
            
            # Split data
            train_data = data[
                (data['timestamp'] >= train_start) & 
                (data['timestamp'] < current_train_end)
            ]
            test_data = data[
                (data['timestamp'] >= current_train_end) & 
                (data['timestamp'] < test_end)
            ]
            
            if len(train_data) < 1000 or len(test_data) < 100:
                skipped_folds += 1
                current_train_end += timedelta(days=self.step_days)
                continue
            
            # Optimize on training data
            def backtest_fn(params):
                engine = BacktestEngine(
                    initial_capital=100000,
                    fee_pct=0.0004,
                    slippage_pct=0.0005,
                    feature_config=feature_config
                )
                metrics = engine.run(train_data, strategy, params)
                return {
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'profit_factor': metrics.profit_factor,
                    'win_rate': metrics.win_rate,
                    'max_drawdown_pct': metrics.max_drawdown_pct,
                    'total_trades': metrics.total_trades,
                    'total_return_pct': metrics.total_return_pct
                }
            
            try:
                result = optimizer.optimize(
                    backtest_fn, 
                    n_trials=n_trials_per_fold,
                    n_jobs=1,  # Sequential for stability
                    objective_type='robust'
                )
                
                best_params = result.best_params
                
                # Test on out-of-sample data
                engine = BacktestEngine(
                    initial_capital=100000,
                    fee_pct=0.0004,
                    slippage_pct=0.0005,
                    feature_config=feature_config
                )
                
                is_metrics = engine.run(train_data, strategy, best_params)
                oos_metrics = engine.run(test_data, strategy, best_params)
                
                # Skip fold if insufficient OOS trades
                if oos_metrics.total_trades < self.min_oos_trades:
                    skipped_folds += 1
                    current_train_end += timedelta(days=self.step_days)
                    continue
                
                fold = WalkForwardFold(
                    train_start=train_start,
                    train_end=current_train_end,
                    test_start=current_train_end,
                    test_end=test_end,
                    best_params=best_params,
                    in_sample_metrics=is_metrics,
                    out_of_sample_metrics=oos_metrics,
                    fold_number=fold_num
                )
                
                folds.append(fold)
                fold_num += 1
                
            except Exception as e:
                print(f"Error in fold {fold_num}: {e}")
                skipped_folds += 1
            
            current_train_end += timedelta(days=self.step_days)
        
        # Calculate summary statistics
        summary = self._calculate_wf_summary(folds, skipped_folds)
        params_stability = self._analyze_parameter_stability(folds)
        
        return {
            'summary': summary,
            'folds': folds,
            'params_stability': params_stability,
            'skipped_folds': skipped_folds
        }
    
    def _calculate_wf_summary(self, folds: List[WalkForwardFold], skipped: int) -> Dict[str, float]:
        """Calculate walk-forward summary statistics"""
        if not folds:
            return {'error': 'No valid folds completed'}
        
        oos_sharpes = [f.out_of_sample_metrics.sharpe_ratio for f in folds]
        oos_returns = [f.out_of_sample_metrics.total_return_pct for f in folds]
        oos_win_rates = [f.out_of_sample_metrics.win_rate for f in folds]
        oos_max_dds = [f.out_of_sample_metrics.max_drawdown_pct for f in folds]
        
        profitable_folds = sum(1 for f in folds if f.out_of_sample_metrics.total_return_pct > 0)
        
        return {
            'total_folds': len(folds) + skipped,
            'valid_folds': len(folds),
            'skipped_folds': skipped,
            'avg_oos_sharpe': np.mean(oos_sharpes),
            'avg_oos_return_pct': np.mean(oos_returns),
            'avg_oos_win_rate': np.mean(oos_win_rates),
            'avg_oos_max_dd_pct': np.mean(oos_max_dds),
            'pct_profitable_folds': profitable_folds / len(folds),
            'oos_sharpe_std': np.std(oos_sharpes),
            'oos_return_std': np.std(oos_returns)
        }
    
    def _analyze_parameter_stability(self, folds: List[WalkForwardFold]) -> Dict[str, Dict[str, float]]:
        """Analyze parameter stability across folds"""
        if not folds:
            return {}
        
        stability = {}
        param_names = list(folds[0].best_params.keys())
        
        for param in param_names:
            values = [f.best_params[param] for f in folds if param in f.best_params]
            
            if len(values) < 2:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
            
            stability[param] = {
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'min': min(values),
                'max': max(values)
            }
        
        return stability

# ==============================================================================
# SECTION 7: config/settings.py (continued)
# ==============================================================================

# Already included in SECTION 3

# ==============================================================================
# SECTION 8: core/data_structures.py (continued)
# ==============================================================================

# Already included in SECTION 4

# ==============================================================================
# SECTION 9: core/feature_engine.py (continued)
# ==============================================================================

# Already included in SECTION 5

# ==============================================================================
# SECTION 10: backtesting/engine.py (continued)
# ==============================================================================

# Already included in SECTION 6

# ==============================================================================
# SECTION 11: data/data_recorder.py
# ==============================================================================

"""
Data Recorder
Records market data for backtesting with async buffering and gap detection.
"""

import asyncio
import json
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import pandas as pd

from loguru import logger


class DataRecorder:
    """
    Async data recorder with gap detection and compression.
    
    Features:
    - Async buffering for high-frequency data
    - Automatic gap detection and filling
    - Compressed JSONL output
    - Memory-efficient batching
    """
    
    def __init__(
        self, 
        symbol: str,
        output_dir: str = "./data/recorded/",
        buffer_size: int = 1000,
        flush_interval_sec: int = 30
    ):
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffers
        self.order_book_buffer: List[Dict] = []
        self.trade_buffer: List[Dict] = []
        self.buffer_size = buffer_size
        
        # Async control
        self.flush_interval = flush_interval_sec
        self.running = False
        self.flush_task: Optional[asyncio.Task] = None
        
        # Gap detection
        self.last_order_book_time: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None
        self.expected_intervals: Dict[str, float] = {
            'order_book': 0.1,  # 100ms
            'trade': 1.0        # 1 second max gap
        }
        
        # Stats
        self.stats = {
            'order_books_recorded': 0,
            'trades_recorded': 0,
            'gaps_detected': 0,
            'files_written': 0
        }
    
    async def start_recording(self) -> None:
        """Start the recording process"""
        self.running = True
        self.flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(f"Started recording for {self.symbol}")
    
    async def stop_recording(self) -> None:
        """Stop recording and flush remaining data"""
        self.running = False
        
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush_buffers()
        logger.info(f"Stopped recording. Stats: {self.stats}")
    
    def record_order_book(self, order_book: Dict) -> None:
        """Record order book snapshot"""
        if not self.running:
            return
        
        # Add timestamp if missing
        if 'timestamp' not in order_book:
            order_book['timestamp'] = datetime.now().isoformat()
        
        # Detect gaps
        current_time = datetime.fromisoformat(order_book['timestamp'])
        if self.last_order_book_time:
            gap_duration = (current_time - self.last_order_book_time).total_seconds()
            if gap_duration > self.expected_intervals['order_book'] * 10:  # 10x expected
                logger.warning(f"Order book gap detected: {gap_duration:.1f}s")
                self.stats['gaps_detected'] += 1
        
        self.last_order_book_time = current_time
        self.order_book_buffer.append(order_book)
        self.stats['order_books_recorded'] += 1
        
        # Flush if buffer full
        if len(self.order_book_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_buffers())
    
    def record_trade(self, trade: Dict) -> None:
        """Record trade execution"""
        if not self.running:
            return
        
        # Add timestamp if missing
        if 'timestamp' not in trade:
            trade['timestamp'] = datetime.now().isoformat()
        
        # Detect gaps
        current_time = datetime.fromisoformat(trade['timestamp'])
        if self.last_trade_time:
            gap_duration = (current_time - self.last_trade_time).total_seconds()
            if gap_duration > self.expected_intervals['trade']:
                logger.warning(f"Trade gap detected: {gap_duration:.1f}s")
                self.stats['gaps_detected'] += 1
        
        self.last_trade_time = current_time
        self.trade_buffer.append(trade)
        self.stats['trades_recorded'] += 1
        
        # Flush if buffer full
        if len(self.trade_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_buffers())
    
    async def _periodic_flush(self) -> None:
        """Periodically flush buffers to disk"""
        while self.running:
            await asyncio.sleep(self.flush_interval)
            await self._flush_buffers()
    
    async def _flush_buffers(self) -> None:
        """Flush all buffers to compressed files"""
        if not self.order_book_buffer and not self.trade_buffer:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Flush order books
        if self.order_book_buffer:
            filename = f"{self.symbol}_order_books_{timestamp}.jsonl.gz"
            await self._write_compressed_jsonl(
                self.order_book_buffer, 
                self.output_dir / filename
            )
            self.order_book_buffer.clear()
        
        # Flush trades
        if self.trade_buffer:
            filename = f"{self.symbol}_trades_{timestamp}.jsonl.gz"
            await self._write_compressed_jsonl(
                self.trade_buffer, 
                self.output_dir / filename
            )
            self.trade_buffer.clear()
        
        self.stats['files_written'] += 1
        logger.debug(f"Flushed buffers to disk ({self.stats['files_written']} files)")
    
    async def _write_compressed_jsonl(
        self, 
        data: List[Dict], 
        filepath: Path
    ) -> None:
        """Write data as compressed JSONL"""
        try:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f)
                    f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write {filepath}: {e}")
    
    def load_recorded_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        data_type: str = 'trades'
    ) -> pd.DataFrame:
        """
        Load recorded data from files.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            data_type: 'trades' or 'order_books'
        """
        pattern = f"{self.symbol}_{data_type}_*.jsonl.gz"
        files = list(self.output_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No {data_type} files found for {self.symbol}")
            return pd.DataFrame()
        
        # Filter files by date range
        relevant_files = []
        for file in files:
            # Extract date from filename
            try:
                date_str = file.stem.split('_')[-1]  # YYYYMMDD_HHMMSS
                file_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                
                # Include file if it overlaps with our date range
                if file_date <= end_date and file_date >= start_date - timedelta(days=1):
                    relevant_files.append(file)
            except ValueError:
                continue
        
        if not relevant_files:
            return pd.DataFrame()
        
        # Load and combine data
        dfs = []
        for file in sorted(relevant_files):
            try:
                df = self._load_single_file(file)
                if not df.empty:
                    # Filter by exact date range
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[
                        (df['timestamp'] >= start_date) & 
                        (df['timestamp'] < end_date)
                    ]
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined)} {data_type} records")
        return combined
    
    def _load_single_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single compressed JSONL file"""
        data = []
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return pd.DataFrame()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Standardize column names
        if 'data_type' not in df.columns:
            df['data_type'] = 'trades' if 'side' in df.columns else 'order_books'
        
        return df
    
    def merge_order_books_and_trades(
        self,
        start_date: datetime,
        end_date: datetime,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Merge order book and trade data into unified format for backtesting.
        
        Output format:
        - timestamp: datetime
        - price: float (last trade price)
        - size: float (last trade size)
        - side: str ('buy' or 'sell')
        - bid_price: float
        - ask_price: float
        - bid_size: float
        - ask_size: float
        """
        # Load data
        trades_df = self.load_recorded_data(start_date, end_date, 'trades')
        books_df = self.load_recorded_data(start_date, end_date, 'order_books')
        
        if trades_df.empty:
            logger.error("No trade data available")
            return pd.DataFrame()
        
        # Merge on timestamp (forward fill order books)
        books_df = books_df.set_index('timestamp')
        trades_df = trades_df.set_index('timestamp')
        
        # Resample books to trade timestamps
        merged = pd.merge_asof(
            trades_df.sort_index(), 
            books_df.sort_index(),
            left_index=True, 
            right_index=True,
            direction='backward'  # Use previous book for each trade
        ).reset_index()
        
        # Fill missing book data with last known values
        book_cols = ['bids', 'asks']
        if all(col in merged.columns for col in book_cols):
            merged[book_cols] = merged[book_cols].fillna(method='ffill')
        
        # Extract best bid/ask from nested structure
        merged['bid_price'] = merged.get('bids', [{}]).apply(
            lambda x: x[0].get('price', 0) if isinstance(x, list) and x else 0
        )
        merged['bid_size'] = merged.get('bids', [{}]).apply(
            lambda x: x[0].get('size', 0) if isinstance(x, list) and x else 0
        )
        merged['ask_price'] = merged.get('asks', [{}]).apply(
            lambda x: x[0].get('price', 0) if isinstance(x, list) and x else 0
        )
        merged['ask_size'] = merged.get('asks', [{}]).apply(
            lambda x: x[0].get('size', 0) if isinstance(x, list) and x else 0
        )
        
        # Select final columns
        result = merged[[
            'timestamp', 'price', 'size', 'side',
            'bid_price', 'ask_price', 'bid_size', 'ask_size'
        ]].copy()
        
        # Save if requested
        if output_file:
            result.to_parquet(output_file, index=False)
            logger.info(f"Saved merged data to {output_file}")
        
        return result

# ==============================================================================
# SECTION 12: data/exchange_connector.py
# ==============================================================================

"""
Exchange Connector
WebSocket and REST connectivity with failover and local order book maintenance.
"""

import asyncio
import json
import websockets
import aiohttp
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import numpy as np

from loguru import logger


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    exchange_id: str
    testnet: bool = False
    api_key: Optional[str] = None
    secret: Optional[str] = None
    sandbox: bool = False
    
    # Connection settings
    ws_timeout: int = 30
    rest_timeout: int = 10
    max_reconnects: int = 5
    reconnect_delay: float = 1.0
    
    # Rate limiting
    rate_limit: int = 1000  # requests per second
    enable_rate_limiter: bool = True
    
    # Data settings
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []


class ExchangeConnector:
    """
    Multi-exchange connector with WebSocket and REST support.
    
    Features:
    - Automatic failover between exchanges
    - Local order book maintenance
    - Connection health monitoring
    - Rate limiting
    """
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        
        # CCXT exchange instance
        self.exchange = getattr(ccxt, config.exchange_id)({
            'apiKey': config.api_key,
            'secret': config.secret,
            'testnet': config.testnet,
            'sandbox': config.sandbox,
            'enableRateLimit': config.enable_rate_limiter,
            'rateLimit': config.rate_limit,
        })
        
        # Connection state
        self.connected = False
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self.rest_session: Optional[aiohttp.ClientSession] = None
        
        # Local order book
        self.order_book: Dict[str, Dict] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # Callbacks
        self.on_order_book_update: Optional[Callable[[Dict], None]] = None
        self.on_trade: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Control
        self.running = False
        self.reconnect_count = 0
        
        # Stats
        self.stats = {
            'messages_received': 0,
            'order_books_updated': 0,
            'trades_received': 0,
            'errors': 0,
            'reconnects': 0
        }
    
    async def connect(self) -> None:
        """Establish connections"""
        try:
            # REST connection
            self.rest_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.rest_timeout)
            )
            
            # Test REST connectivity
            await self._test_rest_connection()
            
            self.connected = True
            logger.info(f"Connected to {self.config.exchange_id}")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            await self.disconnect()
            raise
    
    async def disconnect(self) -> None:
        """Clean up connections"""
        self.running = False
        
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        
        if self.rest_session:
            await self.rest_session.close()
            self.rest_session = None
        
        self.connected = False
        logger.info("Disconnected")
    
    async def _test_rest_connection(self) -> None:
        """Test REST API connectivity"""
        try:
            # Try to fetch ticker
            test_symbol = self.config.symbols[0] if self.config.symbols else "BTC/USDT"
            ticker = await self.exchange.fetch_ticker(test_symbol)
            
            if not ticker:
                raise Exception("No ticker data received")
                
        except Exception as e:
            logger.error(f"REST connectivity test failed: {e}")
            raise
    
    async def start_websocket(self, symbol: str) -> None:
        """
        Start WebSocket streaming for symbol.
        
        Handles:
        - Order book updates
        - Trade executions
        - Automatic reconnection
        """
        if not self.connected:
            raise Exception("Not connected to exchange")
        
        self.running = True
        
        while self.running and self.reconnect_count < self.config.max_reconnects:
            try:
                # Get WebSocket URL
                ws_url = self._get_websocket_url(symbol)
                
                async with websockets.connect(
                    ws_url,
                    extra_headers=self._get_ws_headers(),
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as websocket:
                    
                    self.ws_connection = websocket
                    self.reconnect_count = 0  # Reset on successful connection
                    
                    logger.info(f"WebSocket connected to {ws_url}")
                    
                    # Subscribe to streams
                    await self._subscribe_to_streams(websocket, symbol)
                    
                    # Message loop
                    async for message in websocket:
                        try:
                            await self._process_message(message, symbol)
                            self.stats['messages_received'] += 1
                            
                        except Exception as e:
                            logger.error(f"Message processing error: {e}")
                            self.stats['errors'] += 1
                            
                            if self.on_error:
                                self.on_error(e)
                
            except (websockets.exceptions.ConnectionClosed, 
                    asyncio.TimeoutError) as e:
                
                self.reconnect_count += 1
                self.stats['reconnects'] += 1
                
                if self.running:
                    delay = self.config.reconnect_delay * (2 ** (self.reconnect_count - 1))
                    logger.warning(f"WebSocket disconnected, reconnecting in {delay:.1f}s (attempt {self.reconnect_count})")
                    await asyncio.sleep(delay)
            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.stats['errors'] += 1
                
                if self.on_error:
                    self.on_error(e)
                
                break
        
        logger.info("WebSocket streaming stopped")
    
    def _get_websocket_url(self, symbol: str) -> str:
        """Get WebSocket URL for exchange and symbol"""
        exchange_id = self.config.exchange_id
        
        if exchange_id == 'binance':
            base = "wss://stream.binance.com:9443/ws/"
            stream = f"{symbol.lower().replace('/', '')}@depth@100ms/{symbol.lower().replace('/', '')}@trade"
            return base + stream
        
        elif exchange_id == 'coinbasepro':
            return f"wss://ws-feed.pro.coinbase.com"
        
        elif exchange_id == 'bybit':
            return f"wss://stream.bybit.com/v5/public/spot"
        
        elif exchange_id == 'okx':
            return f"wss://wsaws.okx.com:8443/ws/v5/public"
        
        else:
            raise NotImplementedError(f"WebSocket not implemented for {exchange_id}")
    
    def _get_ws_headers(self) -> Dict[str, str]:
        """Get headers for WebSocket authentication"""
        # Most exchanges don't require auth for public streams
        return {}
    
    async def _subscribe_to_streams(self, websocket, symbol: str) -> None:
        """Subscribe to order book and trade streams"""
        exchange_id = self.config.exchange_id
        
        if exchange_id == 'binance':
            # Binance uses URL-based subscription
            pass
            
        elif exchange_id == 'coinbasepro':
            subscription = {
                "type": "subscribe",
                "channels": [
                    {"name": "level2", "product_ids": [symbol]},
                    {"name": "matches", "product_ids": [symbol]}
                ]
            }
            await websocket.send(json.dumps(subscription))
            
        elif exchange_id == 'bybit':
            subscription = {
                "op": "subscribe",
                "args": [
                    f"orderbook.50.{symbol}",
                    f"publicTrade.{symbol}"
                ]
            }
            await websocket.send(json.dumps(subscription))
        
        elif exchange_id == 'okx':
            subscription = {
                "op": "subscribe",
                "args": [
                    {"channel": "books", "instId": symbol},
                    {"channel": "trades", "instId": symbol}
                ]
            }
            await websocket.send(json.dumps(subscription))
    
    async def _process_message(self, message: str, symbol: str) -> None:
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Skip subscription confirmations
            if isinstance(data, dict) and data.get('type') == 'subscriptions':
                return
            
            # Route by exchange
            exchange_id = self.config.exchange_id
            
            if exchange_id == 'binance':
                await self._process_binance_message(data, symbol)
            elif exchange_id == 'coinbasepro':
                await self._process_coinbase_message(data, symbol)
            elif exchange_id == 'bybit':
                await self._process_bybit_message(data, symbol)
            elif exchange_id == 'okx':
                await self._process_okx_message(data, symbol)
                
        except json.JSONDecodeError:
            logger.debug("Invalid JSON message")
    
    async def _process_binance_message(self, data: Dict, symbol: str) -> None:
        """Process Binance WebSocket message"""
        stream = data.get('stream', '')
        
        if 'depth' in stream:
            # Order book update
            order_book = self._parse_binance_order_book(data['data'])
            await self._handle_order_book_update(order_book)
            
        elif 'trade' in stream:
            # Trade update
            trade = self._parse_binance_trade(data['data'])
            await self._handle_trade_update(trade)
    
    def _parse_binance_order_book(self, data: Dict) -> Dict:
        """Parse Binance order book data"""
        return {
            'timestamp': datetime.fromtimestamp(data['E'] / 1000),
            'symbol': data['s'],
            'bids': [[float(price), float(qty)] for price, qty in data['b']],
            'asks': [[float(price), float(qty)] for price, qty in data['a']],
            'last_update_id': data['u']
        }
    
    def _parse_binance_trade(self, data: Dict) -> Dict:
        """Parse Binance trade data"""
        return {
            'timestamp': datetime.fromtimestamp(data['T'] / 1000),
            'symbol': data['s'],
            'price': float(data['p']),
            'size': float(data['q']),
            'side': 'buy' if data['m'] else 'sell',
            'trade_id': data['t']
        }
    
    async def _process_coinbase_message(self, data: Dict, symbol: str) -> None:
        """Process Coinbase Pro WebSocket message"""
        msg_type = data.get('type')
        
        if msg_type == 'l2update':
            # Order book update
            order_book = self._parse_coinbase_order_book(data)
            await self._handle_order_book_update(order_book)
            
        elif msg_type == 'match':
            # Trade
            trade = self._parse_coinbase_trade(data)
            await self._handle_trade_update(trade)
    
    def _parse_coinbase_order_book(self, data: Dict) -> Dict:
        """Parse Coinbase order book update"""
        # Coinbase sends incremental updates, we'd need full book maintenance
        # This is simplified - real implementation needs full order book
        return {
            'timestamp': datetime.fromisoformat(data['time'].replace('Z', '+00:00')),
            'symbol': data['product_id'],
            'bids': [],  # Would need to maintain full book
            'asks': [],
            'changes': data['changes']
        }
    
    def _parse_coinbase_trade(self, data: Dict) -> Dict:
        """Parse Coinbase trade"""
        return {
            'timestamp': datetime.fromisoformat(data['time'].replace('Z', '+00:00')),
            'symbol': data['product_id'],
            'price': float(data['price']),
            'size': float(data['size']),
            'side': data['side'],
            'trade_id': data['trade_id']
        }
    
    async def _process_bybit_message(self, data: Dict, symbol: str) -> None:
        """Process Bybit WebSocket message"""
        topic = data.get('topic', '')
        
        if 'orderbook' in topic:
            order_book = self._parse_bybit_order_book(data)
            await self._handle_order_book_update(order_book)
            
        elif 'publicTrade' in topic:
            for trade_data in data.get('data', []):
                trade = self._parse_bybit_trade(trade_data)
                await self._handle_trade_update(trade)
    
    def _parse_bybit_order_book(self, data: Dict) -> Dict:
        """Parse Bybit order book"""
        return {
            'timestamp': datetime.fromtimestamp(data['ts'] / 1000),
            'symbol': data['symbol'],
            'bids': [[float(level[0]), float(level[1])] for level in data['b']],
            'asks': [[float(level[0]), float(level[1])] for level in data['a']]
        }
    
    def _parse_bybit_trade(self, data: Dict) -> Dict:
        """Parse Bybit trade"""
        return {
            'timestamp': datetime.fromtimestamp(data['T'] / 1000),
            'symbol': data['S'],
            'price': float(data['p']),
            'size': float(data['v']),
            'side': 'buy' if data['S'][0] == 'B' else 'sell',
            'trade_id': data['i']
        }
    
    async def _process_okx_message(self, data: Dict, symbol: str) -> None:
        """Process OKX WebSocket message"""
        arg = data.get('arg', {})
        channel = arg.get('channel')
        
        if channel == 'books':
            order_book = self._parse_okx_order_book(data)
            await self._handle_order_book_update(order_book)
            
        elif channel == 'trades':
            for trade_data in data.get('data', []):
                trade = self._parse_okx_trade(trade_data)
                await self._handle_trade_update(trade)
    
    def _parse_okx_order_book(self, data: Dict) -> Dict:
        """Parse OKX order book"""
        book_data = data['data'][0]
        return {
            'timestamp': datetime.fromtimestamp(int(book_data['ts']) / 1000),
            'symbol': data['arg']['instId'],
            'bids': [[float(level[0]), float(level[1])] for level in book_data['bids']],
            'asks': [[float(level[0]), float(level[1])] for level in book_data['asks']]
        }
    
    def _parse_okx_trade(self, data: Dict) -> Dict:
        """Parse OKX trade"""
        return {
            'timestamp': datetime.fromtimestamp(int(data['ts']) / 1000),
            'symbol': data['instId'],
            'price': float(data['px']),
            'size': float(data['sz']),
            'side': 'buy' if data['side'] == 'buy' else 'sell',
            'trade_id': data['tradeId']
        }
    
    async def _handle_order_book_update(self, order_book: Dict) -> None:
        """Handle order book update"""
        # Update local order book
        symbol = order_book['symbol']
        self.order_book[symbol] = order_book
        self.last_update[symbol] = datetime.now()
        
        self.stats['order_books_updated'] += 1
        
        # Call callback
        if self.on_order_book_update:
            await self.on_order_book_update(order_book)
    
    async def _handle_trade_update(self, trade: Dict) -> None:
        """Handle trade update"""
        self.stats['trades_received'] += 1
        
        # Call callback
        if self.on_trade:
            await self.on_trade(trade)
    
    async def test_connectivity(self, symbol: str) -> Dict[str, Any]:
        """
        Test connectivity to REST and WebSocket endpoints.
        
        Returns detailed connectivity report.
        """
        results = {
            'rest': {'ok': False, 'latency_ms': 0, 'error': None},
            'ws_urls': []
        }
        
        # Test REST
        try:
            start_time = datetime.now()
            ticker = await self.exchange.fetch_ticker(symbol)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            results['rest'] = {
                'ok': True,
                'latency_ms': round(latency, 1),
                'error': None
            }
            
        except Exception as e:
            results['rest'] = {
                'ok': False,
                'latency_ms': 0,
                'error': str(e)
            }
        
        # Test WebSocket URLs
        ws_urls = self._get_websocket_test_urls(symbol)
        
        for url_info in ws_urls:
            try:
                start_time = datetime.now()
                
                async with websockets.connect(
                    url_info['url'],
                    extra_headers=self._get_ws_headers(),
                    open_timeout=5,
                    ping_timeout=5
                ) as ws:
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    
                    results['ws_urls'].append({
                        'url': url_info['url'],
                        'ok': True,
                        'latency_ms': round(latency, 1),
                        'error': None
                    })
                    
            except Exception as e:
                results['ws_urls'].append({
                    'url': url_info['url'],
                    'ok': False,
                    'latency_ms': 0,
                    'error': str(e)
                })
        
        return results
    
    def _get_websocket_test_urls(self, symbol: str) -> List[Dict[str, str]]:
        """Get WebSocket URLs to test"""
        exchange_id = self.config.exchange_id
        
        if exchange_id == 'binance':
            return [{
                'url': f"wss://stream.binance.com:9443/ws/{symbol.lower().replace('/', '')}@ticker",
                'description': 'Binance Ticker Stream'
            }]
        
        elif exchange_id == 'coinbasepro':
            return [{
                'url': "wss://ws-feed.pro.coinbase.com",
                'description': 'Coinbase Pro WebSocket'
            }]
        
        elif exchange_id == 'bybit':
            return [{
                'url': "wss://stream.bybit.com/v5/public/spot",
                'description': 'Bybit Public Stream'
            }]
        
        elif exchange_id == 'okx':
            return [{
                'url': "wss://wsaws.okx.com:8443/ws/v5/public",
                'description': 'OKX Public WebSocket'
            }]
        
        return []

# ==============================================================================
# SECTION 13: execution/order_manager.py
# ==============================================================================

"""
Order Manager
Handles order lifecycle, submission, tracking, fills, and bracket orders.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Trading order"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)
    average_fill_price: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Metadata
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    fees: float = 0.0
    
    # Callbacks
    on_fill: Optional[Callable] = None
    on_cancel: Optional[Callable] = None
    on_reject: Optional[Callable] = None
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN]
    
    @property
    def is_closed(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                              OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    def update_fill(self, fill_price: float, fill_quantity: float, fees: float = 0.0) -> None:
        """Update order with partial or full fill"""
        if fill_quantity > self.remaining_quantity:
            raise ValueError("Fill quantity exceeds remaining quantity")
        
        # Update quantities
        self.filled_quantity += fill_quantity
        self.remaining_quantity -= fill_quantity
        
        # Update average price
        if self.filled_quantity > 0:
            total_value = self.average_fill_price * (self.filled_quantity - fill_quantity) + fill_price * fill_quantity
            self.average_fill_price = total_value / self.filled_quantity
        
        # Update fees
        self.fees += fees
        
        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.OPEN
        
        self.updated_at = datetime.now()
        
        # Call callback
        if self.on_fill:
            self.on_fill(self, fill_price, fill_quantity)
    
    def cancel(self) -> None:
        """Cancel the order"""
        if not self.is_active:
            return
        
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now()
        
        if self.on_cancel:
            self.on_cancel(self)
    
    def reject(self, reason: str) -> None:
        """Reject the order"""
        self.status = OrderStatus.REJECTED
        self.updated_at = datetime.now()
        
        if self.on_reject:
            self.on_reject(self, reason)


@dataclass
class BracketOrder:
    """Bracket order with entry, stop loss, and take profit"""
    bracket_id: str
    
    # Entry order
    entry_order: Order
    
    # Exit orders
    stop_loss_order: Optional[Order] = None
    take_profit_order: Optional[Order] = None
    
    # Risk management
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    
    # Status
    status: str = "pending"  # pending, active, closed
    
    def activate(self) -> None:
        """Activate the bracket order"""
        self.status = "active"
    
    def close(self) -> None:
        """Close the bracket order"""
        self.status = "closed"
        
        # Cancel any remaining orders
        if self.stop_loss_order and self.stop_loss_order.is_active:
            self.stop_loss_order.cancel()
        
        if self.take_profit_order and self.take_profit_order.is_active:
            self.take_profit_order.cancel()


class OrderManager:
    """
    Order lifecycle manager with bracket order support.
    
    Features:
    - Order submission and tracking
    - Fill handling and callbacks
    - Bracket orders (entry + SL + TP)
    - Risk management integration
    - Order persistence
    """
    
    def __init__(self):
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.bracket_orders: Dict[str, BracketOrder] = {}
        
        # Callbacks
        self.on_order_update: Optional[Callable] = None
        self.on_bracket_update: Optional[Callable] = None
        
        # Stats
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'total_fees': 0.0
        }
    
    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Order:
        """Submit a single order"""
        order_id = str(uuid.uuid4())
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            client_order_id=client_order_id or order_id
        )
        
        # Store order
        self.orders[order_id] = order
        
        # Set callbacks
        order.on_fill = lambda o, p, q: self._on_order_fill(o, p, q)
        order.on_cancel = lambda o: self._on_order_cancel(o)
        order.on_reject = lambda o, r: self._on_order_reject(o, r)
        
        self.stats['orders_submitted'] += 1
        
        logger.info(f"Submitted order {order_id}: {side} {quantity} {symbol} @ {price or 'market'}")
        
        # Notify
        if self.on_order_update:
            self.on_order_update(order)
        
        return order
    
    def submit_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> BracketOrder:
        """
        Submit a bracket order with entry, stop loss, and take profit.
        
        Either specify percentages or absolute prices for SL/TP.
        """
        bracket_id = str(uuid.uuid4())
        
        # Create entry order
        entry_order = self.submit_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT if entry_price else OrderType.MARKET,
            quantity=quantity,
            price=entry_price
        )
        
        bracket = BracketOrder(
            bracket_id=bracket_id,
            entry_order=entry_order,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        
        # Create exit orders (will be submitted after entry fills)
        entry_order.on_fill = lambda o, p, q: self._on_bracket_entry_fill(bracket, p)
        
        # Store bracket
        self.bracket_orders[bracket_id] = bracket
        
        logger.info(f"Submitted bracket order {bracket_id}: {side} {quantity} {symbol}")
        
        return bracket
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        if order.is_active:
            order.cancel()
            self.stats['orders_cancelled'] += 1
            logger.info(f"Cancelled order {order_id}")
            return True
        
        return False
    
    def cancel_bracket_order(self, bracket_id: str) -> bool:
        """Cancel a bracket order"""
        bracket = self.bracket_orders.get(bracket_id)
        if not bracket:
            return False
        
        bracket.close()
        logger.info(f"Cancelled bracket order {bracket_id}")
        return True
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_bracket_order(self, bracket_id: str) -> Optional[BracketOrder]:
        """Get bracket order by ID"""
        return self.bracket_orders.get(bracket_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol"""
        orders = [o for o in self.orders.values() if o.is_active]
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders
    
    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get order history"""
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return sorted(orders, key=lambda o: o.created_at, reverse=True)
    
    # ==================== EVENT HANDLERS ====================
    
    def _on_order_fill(self, order: Order, fill_price: float, fill_quantity: float) -> None:
        """Handle order fill"""
        logger.info(f"Order {order.order_id} filled: {fill_quantity} @ {fill_price}")
        
        self.stats['orders_filled'] += 1
        self.stats['total_fees'] += order.fees
        
        if self.on_order_update:
            self.on_order_update(order)
    
    def _on_order_cancel(self, order: Order) -> None:
        """Handle order cancellation"""
        logger.info(f"Order {order.order_id} cancelled")
        
        if self.on_order_update:
            self.on_order_update(order)
    
    def _on_order_reject(self, order: Order, reason: str) -> None:
        """Handle order rejection"""
        logger.error(f"Order {order.order_id} rejected: {reason}")
        
        self.stats['orders_rejected'] += 1
        
        if self.on_order_update:
            self.on_order_update(order)
    
    def _on_bracket_entry_fill(self, bracket: BracketOrder, fill_price: float) -> None:
        """Handle bracket entry order fill - submit exit orders"""
        logger.info(f"Bracket {bracket.bracket_id} entry filled @ {fill_price}")
        
        bracket.activate()
        
        # Calculate exit prices
        if bracket.stop_loss_pct:
            if bracket.entry_order.side == 'buy':
                sl_price = fill_price * (1 - bracket.stop_loss_pct)
            else:
                sl_price = fill_price * (1 + bracket.stop_loss_pct)
        else:
            sl_price = None
        
        if bracket.take_profit_pct:
            if bracket.entry_order.side == 'buy':
                tp_price = fill_price * (1 + bracket.take_profit_pct)
            else:
                tp_price = fill_price * (1 - bracket.take_profit_pct)
        else:
            tp_price = None
        
        # Submit stop loss
        if sl_price:
            bracket.stop_loss_order = self.submit_order(
                symbol=bracket.entry_order.symbol,
                side='sell' if bracket.entry_order.side == 'buy' else 'buy',
                order_type=OrderType.STOP,
                quantity=bracket.entry_order.filled_quantity,
                stop_price=sl_price
            )
        
        # Submit take profit
        if tp_price:
            bracket.take_profit_order = self.submit_order(
                symbol=bracket.entry_order.symbol,
                side='sell' if bracket.entry_order.side == 'buy' else 'buy',
                order_type=OrderType.LIMIT,
                quantity=bracket.entry_order.filled_quantity,
                price=tp_price
            )
        
        # Set up bracket closure on exit fills
        for exit_order in [bracket.stop_loss_order, bracket.take_profit_order]:
            if exit_order:
                exit_order.on_fill = lambda o, p, q: self._on_bracket_exit_fill(bracket)
        
        if self.on_bracket_update:
            self.on_bracket_update(bracket)
    
    def _on_bracket_exit_fill(self, bracket: BracketOrder) -> None:
        """Handle bracket exit order fill - close bracket"""
        logger.info(f"Bracket {bracket.bracket_id} exit filled")
        
        bracket.close()
        
        if self.on_bracket_update:
            self.on_bracket_update(bracket)
    
    # ==================== UTILITY METHODS ====================
    
    def get_position_size(self, symbol: str) -> float:
        """Get current position size for symbol"""
        position = 0.0
        
        for order in self.get_active_orders(symbol):
            if order.side == 'buy':
                position += order.filled_quantity
            else:
                position -= order.filled_quantity
        
        return position
    
    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized P&L for symbol"""
        position = self.get_position_size(symbol)
        
        if position == 0:
            return 0.0
        
        # Find average entry price
        total_cost = 0.0
        total_quantity = 0.0
        
        for order in self.get_order_history(symbol):
            if order.status == OrderStatus.FILLED and order.average_fill_price > 0:
                if (position > 0 and order.side == 'buy') or (position < 0 and order.side == 'sell'):
                    total_cost += order.average_fill_price * order.filled_quantity
                    total_quantity += order.filled_quantity
        
        if total_quantity == 0:
            return 0.0
        
        avg_entry_price = total_cost / total_quantity
        
        if position > 0:
            return (current_price - avg_entry_price) * position
        else:
            return (avg_entry_price - current_price) * abs(position)
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics"""
        today = datetime.now().date()
        
        today_orders = [
            o for o in self.orders.values() 
            if o.created_at.date() == today and o.status == OrderStatus.FILLED
        ]
        
        if not today_orders:
            return {
                'date': today.isoformat(),
                'trades': 0,
                'volume': 0.0,
                'fees': 0.0,
                'pnl': 0.0
            }
        
        volume = sum(o.filled_quantity * o.average_fill_price for o in today_orders)
        fees = sum(o.fees for o in today_orders)
        
        # Simplified P&L calculation (would need position tracking)
        pnl = 0.0  # Placeholder
        
        return {
            'date': today.isoformat(),
            'trades': len(today_orders),
            'volume': volume,
            'fees': fees,
            'pnl': pnl
        }

# ==============================================================================
# SECTION 14: execution/risk_manager.py
# ==============================================================================

"""
Risk Manager
Position limits, drawdown checks, and trading halts.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class RiskEvent(Enum):
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    DRAWDOWN_LIMIT_EXCEEDED = "drawdown_limit_exceeded"
    DAILY_LOSS_LIMIT_EXCEEDED = "daily_loss_limit_exceeded"
    CONSECUTIVE_LOSSES_EXCEEDED = "consecutive_losses_exceeded"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_DRYUP = "liquidity_dryup"


@dataclass
class RiskLimits:
    """Risk management limits"""
    # Position limits
    max_position_size: float = 10000.0
    max_position_value_pct: float = 0.25  # 25% of account
    
    # Loss limits
    max_daily_loss_pct: float = 0.02  # 2%
    max_drawdown_pct: float = 0.05  # 5%
    max_consecutive_losses: int = 5
    
    # Volatility limits
    max_volatility_pct: float = 0.10  # 10%
    volatility_lookback_hours: int = 24
    
    # Liquidity limits
    min_liquidity_depth: float = 1000.0
    max_spread_bps: float = 50.0  # 50 basis points
    
    # Time limits
    max_trades_per_hour: int = 10
    max_trades_per_day: int = 50
    
    # Circuit breaker
    circuit_breaker_trigger_pct: float = 0.15  # 15% move triggers halt


@dataclass
class RiskPosition:
    """Current position for risk tracking"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        return self.size * self.current_price
    
    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


@dataclass
class RiskState:
    """Current risk state"""
    # Account state
    initial_equity: float
    current_equity: float = field(init=False)
    
    # Daily tracking
    daily_start_equity: float = field(init=False)
    daily_pnl: float = 0.0
    daily_trades: int = 0
    
    # Loss tracking
    consecutive_losses: int = 0
    last_trade_pnl: float = 0.0
    
    # Positions
    positions: Dict[str, RiskPosition] = field(default_factory=dict)
    
    # Circuit breaker
    trading_halted: bool = False
    halt_reason: Optional[str] = None
    halt_timestamp: Optional[datetime] = None
    
    # Stats
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    
    def __post_init__(self):
        self.current_equity = self.initial_equity
        self.daily_start_equity = self.initial_equity


class RiskManager:
    """
    Comprehensive risk management system.
    
    Monitors:
    - Position sizes and limits
    - Drawdown and daily losses
    - Trading frequency
    - Market volatility and liquidity
    - Circuit breaker for extreme events
    """
    
    def __init__(self, limits: RiskLimits, initial_equity: float):
        self.limits = limits
        self.state = RiskState(initial_equity=initial_equity)
        
        # Callbacks
        self.on_risk_event: Optional[Callable[[RiskEvent, str], None]] = None
        self.on_trading_halt: Optional[Callable[[str], None]] = None
        self.on_trading_resume: Optional[Callable[[], None]] = None
        
        # Market data tracking
        self.volatility_history: List[float] = []
        self.price_history: Dict[str, List[float]] = {}
        
        logger.info(f"Risk Manager initialized with equity: ${initial_equity:,.2f}")
    
    def check_trade_allowed(
        self, 
        symbol: str, 
        side: str, 
        size: float, 
        price: float
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a trade is allowed based on risk limits.
        
        Returns: (allowed, reason_if_not)
        """
        # Check trading halt
        if self.state.trading_halted:
            return False, f"Trading halted: {self.state.halt_reason}"
        
        # Check position limits
        current_position = self.state.positions.get(symbol, RiskPosition(symbol, 0, 0, price, 0))
        new_position_size = current_position.size + (size if side == 'buy' else -size)
        
        if abs(new_position_size) > self.limits.max_position_size:
            self._trigger_risk_event(
                RiskEvent.POSITION_LIMIT_EXCEEDED,
                f"Position limit exceeded: {abs(new_position_size)} > {self.limits.max_position_size}"
            )
            return False, "Position size limit exceeded"
        
        # Check position value percentage
        new_market_value = abs(new_position_size) * price
        max_value = self.state.current_equity * self.limits.max_position_value_pct
        
        if new_market_value > max_value:
            self._trigger_risk_event(
                RiskEvent.POSITION_LIMIT_EXCEEDED,
                f"Position value limit exceeded: ${new_market_value:,.2f} > ${max_value:,.2f}"
            )
            return False, "Position value limit exceeded"
        
        # Check daily loss limit
        if self.state.daily_pnl < -self.state.daily_start_equity * self.limits.max_daily_loss_pct:
            self._trigger_risk_event(
                RiskEvent.DAILY_LOSS_LIMIT_EXCEEDED,
                f"Daily loss limit exceeded: {self.state.daily_pnl/self.state.daily_start_equity:.2%}"
            )
            return False, "Daily loss limit exceeded"
        
        # Check consecutive losses
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            self._trigger_risk_event(
                RiskEvent.CONSECUTIVE_LOSSES_EXCEEDED,
                f"Consecutive losses limit exceeded: {self.state.consecutive_losses}"
            )
            return False, "Consecutive losses limit exceeded"
        
        # Check trading frequency
        if self.state.daily_trades >= self.limits.max_trades_per_day:
            return False, "Daily trade limit exceeded"
        
        # All checks passed
        return True, None
    
    def update_position(
        self, 
        symbol: str, 
        size_change: float, 
        price: float,
        realized_pnl: float = 0.0
    ) -> None:
        """Update position after trade execution"""
        # Update position
        if symbol not in self.state.positions:
            self.state.positions[symbol] = RiskPosition(symbol, 0, price, price, 0)
        
        position = self.state.positions[symbol]
        old_size = position.size
        
        # Update position size and entry price
        if old_size == 0:
            # Opening new position
            position.size = size_change
            position.entry_price = price
        elif (old_size > 0 and size_change > 0) or (old_size < 0 and size_change < 0):
            # Adding to position
            total_value = old_size * position.entry_price + size_change * price
            position.size += size_change
            position.entry_price = total_value / position.size
        else:
            # Reducing or closing position
            position.size += size_change
        
        position.current_price = price
        position.unrealized_pnl = position.pnl_pct * abs(position.size) * position.entry_price
        position.timestamp = datetime.now()
        
        # Update equity and P&L
        self.state.current_equity += realized_pnl
        self.state.total_pnl += realized_pnl
        self.state.daily_pnl += realized_pnl
        
        # Update trade stats
        if realized_pnl != 0:
            self.state.total_trades += 1
            self.state.daily_trades += 1
            
            if realized_pnl > 0:
                self.state.winning_trades += 1
                self.state.consecutive_losses = 0
            else:
                self.state.consecutive_losses += 1
            
            self.state.last_trade_pnl = realized_pnl
        
        # Check drawdown
        self._check_drawdown()
        
        # Clean up closed positions
        if abs(position.size) < 1e-8:
            del self.state.positions[symbol]
    
    def update_market_data(
        self, 
        symbol: str, 
        price: float, 
        order_book_depth: float = 0.0,
        spread_bps: float = 0.0
    ) -> None:
        """Update with market data for risk monitoring"""
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep only recent history
        max_history = self.limits.volatility_lookback_hours * 3600  # Assuming 1 price per second
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        
        # Update position prices
        if symbol in self.state.positions:
            self.state.positions[symbol].current_price = price
            position = self.state.positions[symbol]
            position.unrealized_pnl = position.pnl_pct * abs(position.size) * position.entry_price
        
        # Check liquidity
        if order_book_depth < self.limits.min_liquidity_depth:
            self._trigger_risk_event(
                RiskEvent.LIQUIDITY_DRYUP,
                f"Low liquidity: {order_book_depth} < {self.limits.min_liquidity_depth}"
            )
        
        if spread_bps > self.limits.max_spread_bps:
            self._trigger_risk_event(
                RiskEvent.LIQUIDITY_DRYUP,
                f"High spread: {spread_bps} > {self.limits.max_spread_bps} bps"
            )
        
        # Check volatility
        if len(self.price_history[symbol]) >= 100:
            prices = self.price_history[symbol][-100:]
            returns = [abs(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            volatility = sum(returns) / len(returns)
            
            if volatility > self.limits.max_volatility_pct:
                self._trigger_risk_event(
                    RiskEvent.VOLATILITY_SPIKE,
                    f"High volatility: {volatility:.2%} > {self.limits.max_volatility_pct:.2%}"
                )
        
        # Check circuit breaker
        self._check_circuit_breaker(symbol, price)
    
    def _check_drawdown(self) -> None:
        """Check if drawdown limits are exceeded"""
        if self.state.current_equity <= 0:
            self._halt_trading("Account equity depleted")
            return
        
        drawdown_pct = (self.state.current_equity - self.state.initial_equity) / self.state.initial_equity
        
        if drawdown_pct < -self.limits.max_drawdown_pct:
            self._trigger_risk_event(
                RiskEvent.DRAWDOWN_LIMIT_EXCEEDED,
                f"Drawdown limit exceeded: {drawdown_pct:.2%}"
            )
            self._halt_trading("Maximum drawdown exceeded")
    
    def _check_circuit_breaker(self, symbol: str, current_price: float) -> None:
        """Check for extreme price movements that should trigger circuit breaker"""
        if len(self.price_history[symbol]) < 10:
            return
        
        recent_prices = self.price_history[symbol][-10:]
        avg_price = sum(recent_prices) / len(recent_prices)
        move_pct = abs(current_price / avg_price - 1)
        
        if move_pct > self.limits.circuit_breaker_trigger_pct:
            self._halt_trading(f"Circuit breaker triggered: {move_pct:.2%} price move")
    
    def _trigger_risk_event(self, event: RiskEvent, message: str) -> None:
        """Trigger a risk event"""
        logger.warning(f"Risk Event: {event.value} - {message}")
        
        if self.on_risk_event:
            self.on_risk_event(event, message)
    
    def _halt_trading(self, reason: str) -> None:
        """Halt all trading"""
        if self.state.trading_halted:
            return
        
        self.state.trading_halted = True
        self.state.halt_reason = reason
        self.state.halt_timestamp = datetime.now()
        
        logger.error(f"TRADING HALTED: {reason}")
        
        if self.on_trading_halt:
            self.on_trading_halt(reason)
    
    def resume_trading(self) -> None:
        """Resume trading after halt"""
        if not self.state.trading_halted:
            return
        
        self.state.trading_halted = False
        self.state.halt_reason = None
        self.state.halt_timestamp = None
        
        # Reset daily stats at midnight
        now = datetime.now()
        if now.hour == 0 and now.minute < 5:  # Reset around midnight
            self.state.daily_start_equity = self.state.current_equity
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
        
        logger.info("Trading resumed")
        
        if self.on_trading_resume:
            self.on_trading_resume()
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        return {
            'equity': {
                'initial': self.state.initial_equity,
                'current': self.state.current_equity,
                'total_pnl': self.state.total_pnl,
                'daily_pnl': self.state.daily_pnl
            },
            'positions': {
                symbol: {
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'pnl_pct': pos.pnl_pct
                }
                for symbol, pos in self.state.positions.items()
            },
            'trading_stats': {
                'total_trades': self.state.total_trades,
                'winning_trades': self.state.winning_trades,
                'win_rate': self.state.winning_trades / max(self.state.total_trades, 1),
                'consecutive_losses': self.state.consecutive_losses
            },
            'risk_limits': {
                'max_position_size': self.limits.max_position_size,
                'max_daily_loss_pct': self.limits.max_daily_loss_pct,
                'max_drawdown_pct': self.limits.max_drawdown_pct,
                'max_consecutive_losses': self.limits.max_consecutive_losses
            },
            'status': {
                'trading_halted': self.state.trading_halted,
                'halt_reason': self.state.halt_reason,
                'halt_timestamp': self.state.halt_timestamp.isoformat() if self.state.halt_timestamp else None
            }
        }
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at market open)"""
        self.state.daily_start_equity = self.state.current_equity
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.consecutive_losses = 0
        
        logger.info("Daily risk stats reset")

# ==============================================================================
# SECTION 15: knowledge/llm_advisor.py
# ==============================================================================

"""
LLM Advisor
Multi-provider LLM integration with caching and knowledge extraction.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp

from loguru import logger


class LLMProvider(Enum):
    # Native SDK providers
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    # OpenAI-compatible providers
    OPENROUTER = "openrouter"
    GITHUB_MODELS = "github_models"
    XAI = "xai"
    ZAI = "zai"
    GROQ = "groq"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Rate limiting
    max_calls_per_minute: int = 10
    cache_responses: bool = True
    cache_ttl_seconds: int = 3600
    
    # Headers for OpenRouter
    site_url: Optional[str] = None
    app_name: Optional[str] = None


@dataclass
class LLMResponse:
    """LLM response with metadata"""
    content: str
    provider: LLMProvider
    model: str
    tokens_used: int = 0
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0


@dataclass
class CacheEntry:
    """Cache entry for LLM responses"""
    response: LLMResponse
    expires_at: datetime
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class LLMAdvisor:
    """
    Multi-provider LLM advisor with automatic failover.
    
    Features:
    - Multiple LLM providers with failover
    - Response caching
    - Rate limiting
    - Cost tracking
    - Knowledge extraction from responses
    """
    
    def __init__(self, configs: List[LLMConfig]):
        self.configs = {config.provider: config for config in configs}
        self.cache: Dict[str, CacheEntry] = {}
        self.call_counts: Dict[LLMProvider, int] = {}
        self.last_call_times: Dict[LLMProvider, datetime] = {}
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Stats
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'errors': 0,
            'tokens_used': 0
        }
    
    @classmethod
    def create_with_failover(cls, config: 'LLMConfig') -> Optional['LLMAdvisor']:
        """
        Create LLM advisor with automatic failover chain.
        
        This method creates a chain of LLM providers to try in order.
        """
        # This would be implemented to create the failover chain
        # For now, return None if no valid config
        if not config.api_key and config.provider != LLMProvider.OLLAMA:
            return None
        
        configs = [config]  # Simplified - would expand to failover chain
        return cls(configs)
    
    async def initialize(self) -> None:
        """Initialize HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_advice(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Optional[LLMResponse]:
        """
        Get advice from LLM with caching and failover.
        
        Args:
            prompt: The prompt to send
            context: Additional context for the prompt
            use_cache: Whether to use cached responses
        """
        if not self.session:
            await self.initialize()
        
        # Create cache key
        cache_key = self._create_cache_key(prompt, context)
        
        # Check cache first
        if use_cache and cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired:
                self.stats['cache_hits'] += 1
                response = entry.response
                response.cached = True
                return response
        
        # Try each provider in order
        for provider, config in self.configs.items():
            try:
                # Rate limiting check
                if not self._check_rate_limit(provider, config):
                    continue
                
                response = await self._call_provider(provider, config, prompt, context)
                
                if response:
                    self.stats['total_calls'] += 1
                    self.call_counts[provider] = self.call_counts.get(provider, 0) + 1
                    
                    # Cache response
                    if config.cache_responses:
                        expires_at = datetime.now() + timedelta(seconds=config.cache_ttl_seconds)
                        self.cache[cache_key] = CacheEntry(response, expires_at)
                    
                    return response
                    
            except Exception as e:
                logger.error(f"LLM call failed for {provider.value}: {e}")
                self.stats['errors'] += 1
                continue
        
        logger.error("All LLM providers failed")
        return None
    
    def _create_cache_key(self, prompt: str, context: Optional[Dict]) -> str:
        """Create cache key from prompt and context"""
        key_data = prompt
        if context:
            key_data += json.dumps(context, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _check_rate_limit(self, provider: LLMProvider, config: LLMConfig) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        last_call = self.last_call_times.get(provider)
        
        if last_call:
            time_since_last = (now - last_call).total_seconds()
            min_interval = 60 / config.max_calls_per_minute
            
            if time_since_last < min_interval:
                return False
        
        self.last_call_times[provider] = now
        return True
    
    async def _call_provider(
        self,
        provider: LLMProvider,
        config: LLMConfig,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[LLMResponse]:
        """Call specific LLM provider"""
        start_time = datetime.now()
        
        try:
            if provider == LLMProvider.OPENAI:
                return await self._call_openai(config, prompt, context)
            elif provider == LLMProvider.GEMINI:
                return await self._call_gemini(config, prompt, context)
            elif provider == LLMProvider.ANTHROPIC:
                return await self._call_anthropic(config, prompt, context)
            elif provider in [LLMProvider.OPENROUTER, LLMProvider.GITHUB_MODELS, 
                             LLMProvider.XAI, LLMProvider.ZAI, LLMProvider.GROQ]:
                return await self._call_openai_compatible(config, prompt, context)
            elif provider == LLMProvider.OLLAMA:
                return await self._call_ollama(config, prompt, context)
            else:
                logger.error(f"Unsupported provider: {provider}")
                return None
                
        except Exception as e:
            logger.error(f"Provider {provider.value} call failed: {e}")
            return None
        finally:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            logger.debug(f"LLM call to {provider.value} took {latency:.1f}ms")
    
    async def _call_openai(
        self, 
        config: LLMConfig, 
        prompt: str, 
        context: Optional[Dict]
    ) -> Optional[LLMResponse]:
        """Call OpenAI API"""
        import openai
        
        client = openai.AsyncOpenAI(api_key=config.api_key)
        
        messages = [{"role": "user", "content": prompt}]
        if context:
            system_msg = f"Context: {json.dumps(context)}"
            messages.insert(0, {"role": "system", "content": system_msg})
        
        response = await client.chat.completions.create(
            model=config.model or "gpt-4",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.OPENAI,
            model=response.model,
            tokens_used=tokens
        )
    
    async def _call_gemini(
        self, 
        config: LLMConfig, 
        prompt: str, 
        context: Optional[Dict]
    ) -> Optional[LLMResponse]:
        """Call Gemini API"""
        import google.generativeai as genai
        
        genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(config.model or "gemini-pro")
        
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {json.dumps(context)}\n\n{prompt}"
        
        response = await model.generate_content_async(full_prompt)
        
        return LLMResponse(
            content=response.text,
            provider=LLMProvider.GEMINI,
            model=config.model or "gemini-pro"
        )
    
    async def _call_anthropic(
        self, 
        config: LLMConfig, 
        prompt: str, 
        context: Optional[Dict]
    ) -> Optional[LLMResponse]:
        """Call Anthropic API"""
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=config.api_key)
        
        system_msg = None
        user_msg = prompt
        
        if context:
            system_msg = f"Context: {json.dumps(context)}"
        
        response = await client.messages.create(
            model=config.model or "claude-3-sonnet-20240229",
            max_tokens=1000,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}]
        )
        
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.ANTHROPIC,
            model=response.model,
            tokens_used=tokens
        )
    
    async def _call_openai_compatible(
        self, 
        config: LLMConfig, 
        prompt: str, 
        context: Optional[Dict]
    ) -> Optional[LLMResponse]:
        """Call OpenAI-compatible API"""
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        messages = [{"role": "user", "content": prompt}]
        if context:
            system_msg = f"Context: {json.dumps(context)}"
            messages.insert(0, {"role": "system", "content": system_msg})
        
        # Add headers for OpenRouter
        extra_headers = {}
        if config.provider == LLMProvider.OPENROUTER:
            if config.site_url:
                extra_headers["HTTP-Referer"] = config.site_url
            if config.app_name:
                extra_headers["X-Title"] = config.app_name
        
        response = await client.chat.completions.create(
            model=config.model or "gpt-4",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            extra_headers=extra_headers
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        
        return LLMResponse(
            content=content,
            provider=config.provider,
            model=response.model,
            tokens_used=tokens
        )
    
    async def _call_ollama(
        self, 
        config: LLMConfig, 
        prompt: str, 
        context: Optional[Dict]
    ) -> Optional[LLMResponse]:
        """Call Ollama (local) API"""
        url = config.base_url or "http://localhost:11434/v1/chat/completions"
        
        messages = [{"role": "user", "content": prompt}]
        if context:
            system_msg = f"Context: {json.dumps(context)}"
            messages.insert(0, {"role": "system", "content": system_msg})
        
        payload = {
            "model": config.model or "llama2",
            "messages": messages,
            "stream": False
        }
        
        async with self.session.post(url, json=payload) as resp:
            if resp.status != 200:
                raise Exception(f"Ollama API error: {resp.status}")
            
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.OLLAMA,
                model=config.model or "llama2"
            )
    
    def extract_knowledge(self, response: LLMResponse) -> Dict[str, Any]:
        """
        Extract structured knowledge from LLM response.
        
        This is a placeholder - would implement actual extraction logic
        based on the specific use case.
        """
        # Placeholder implementation
        return {
            'confidence': 0.5,
            'key_points': response.content.split('.'),
            'sentiment': 'neutral',
            'extracted_at': datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        return {
            'total_calls': self.stats['total_calls'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_calls'], 1),
            'errors': self.stats['errors'],
            'error_rate': self.stats['errors'] / max(self.stats['total_calls'], 1),
            'tokens_used': self.stats['tokens_used'],
            'provider_usage': dict(self.call_counts)
        }
    
    def clear_cache(self) -> None:
        """Clear response cache"""
        self.cache.clear()
        logger.info("LLM response cache cleared")

# ==============================================================================
# SECTION 16: knowledge/strategy_library.py
# ==============================================================================

"""
Strategy Library
Pre-built trading strategies for order flow analysis.
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

from core.data_structures import SignalType, Signal
from core.feature_engine import FeatureEngine


class StrategyType(Enum):
    ABSORPTION = "absorption"
    DELTA_DIVERGENCE = "delta_divergence"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    STACKED_IMBALANCE = "stacked_imbalance"
    VALUE_AREA = "value_area"


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_features: List[str]
    risk_multiplier: float = 1.0


@dataclass
class StrategyResult:
    """Strategy evaluation result"""
    signal: SignalType
    confidence: float
    reasoning: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float


# ==================== STRATEGY IMPLEMENTATIONS ====================

def absorption_strategy(features: Dict[str, float], params: Dict[str, Any]) -> StrategyResult:
    """
    Absorption Strategy
    
    Detects when large orders are being absorbed without price movement,
    indicating potential continuation or reversal.
    """
    absorption_strength = features.get('absorption_strength', 0)
    absorption_volume = features.get('absorption_volume', 0)
    absorption_side = features.get('absorption_side', 0)
    
    # Parameters
    min_strength = params.get('min_strength', 0.7)
    min_volume = params.get('min_volume', 1000)
    max_duration = params.get('max_duration', 30)
    
    duration = features.get('absorption_duration', 0)
    
    # Check conditions
    if (absorption_strength >= min_strength and 
        absorption_volume >= min_volume and 
        duration <= max_duration):
        
        # Determine direction based on absorbing side
        if absorption_side > 0:  # Buy side absorbing
            signal = SignalType.BUY
            confidence = min(absorption_strength, 0.9)
        elif absorption_side < 0:  # Sell side absorbing
            signal = SignalType.SELL
            confidence = min(absorption_strength, 0.9)
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.0
        
        # Calculate entry/exit levels
        current_price = features.get('mid_price', 0)
        entry_price = current_price
        
        # Stop loss: opposite of signal direction
        sl_distance = current_price * params.get('stop_loss_pct', 0.005)
        if signal == SignalType.BUY:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * params.get('risk_reward', 2)
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * params.get('risk_reward', 2)
        
        position_size = params.get('position_size', 1.0)
        
        return StrategyResult(
            signal=signal,
            confidence=confidence,
            reasoning=f"Absorption detected: strength={absorption_strength:.2f}, volume={absorption_volume:.0f}, side={absorption_side}",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size
        )
    
    return StrategyResult(
        signal=SignalType.NEUTRAL,
        confidence=0.0,
        reasoning="No absorption pattern detected",
        entry_price=0,
        stop_loss=0,
        take_profit=0,
        position_size=0
    )


def delta_divergence_strategy(features: Dict[str, float], params: Dict[str, Any]) -> StrategyResult:
    """
    Delta Divergence Strategy
    
    Looks for divergence between price movement and order flow delta,
    indicating potential reversal.
    """
    # Get delta features
    delta_15s = features.get('delta_15s', 0)
    delta_60s = features.get('delta_60s', 0)
    price_change_15s = features.get('price_change_pct_15s', 0)
    price_change_60s = features.get('price_change_pct_60s', 0)
    
    # Parameters
    divergence_threshold = params.get('divergence_threshold', 0.5)
    min_volume = params.get('min_volume', 500)
    
    total_volume_15s = features.get('total_volume_15s', 0)
    
    if total_volume_15s < min_volume:
        return StrategyResult(
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            reasoning="Insufficient volume for divergence analysis",
            entry_price=0, stop_loss=0, take_profit=0, position_size=0
        )
    
    # Calculate divergence score
    price_direction_15s = 1 if price_change_15s > 0 else -1
    price_direction_60s = 1 if price_change_60s > 0 else -1
    delta_direction_15s = 1 if delta_15s > 0 else -1
    delta_direction_60s = 1 if delta_60s > 0 else -1
    
    # Bullish divergence: price down but delta up (more buying)
    bullish_divergence = (price_direction_15s < 0 and delta_direction_15s > 0 and
                         price_direction_60s < 0 and delta_direction_60s > 0)
    
    # Bearish divergence: price up but delta down (more selling)
    bearish_divergence = (price_direction_15s > 0 and delta_direction_15s < 0 and
                         price_direction_60s > 0 and delta_direction_60s < 0)
    
    if bullish_divergence:
        signal = SignalType.BUY
        confidence = min(abs(delta_15s) / abs(price_change_15s + 1e-9), 0.8)
        confidence = max(confidence, divergence_threshold)
    elif bearish_divergence:
        signal = SignalType.SELL
        confidence = min(abs(delta_15s) / abs(price_change_15s + 1e-9), 0.8)
        confidence = max(confidence, divergence_threshold)
    else:
        signal = SignalType.NEUTRAL
        confidence = 0.0
    
    if signal != SignalType.NEUTRAL:
        current_price = features.get('mid_price', 0)
        entry_price = current_price
        
        # Stop loss and take profit
        sl_distance = current_price * params.get('stop_loss_pct', 0.005)
        if signal == SignalType.BUY:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * params.get('risk_reward', 2)
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * params.get('risk_reward', 2)
        
        position_size = params.get('position_size', 1.0)
        
        return StrategyResult(
            signal=signal,
            confidence=confidence,
            reasoning=f"Delta divergence detected: price_dir={price_direction_15s}, delta_dir={delta_direction_15s}",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size
        )
    
    return StrategyResult(
        signal=SignalType.NEUTRAL,
        confidence=0.0,
        reasoning="No delta divergence detected",
        entry_price=0, stop_loss=0, take_profit=0, position_size=0
    )


def liquidity_sweep_strategy(features: Dict[str, float], params: Dict[str, Any]) -> StrategyResult:
    """
    Liquidity Sweep Strategy
    
    Detects aggressive orders sweeping through resting liquidity,
    indicating potential momentum continuation.
    """
    sweep_strength = features.get('sweep_strength', 0)
    sweep_volume = features.get('sweep_volume', 0)
    sweep_direction = features.get('sweep_direction', 0)
    sweep_speed = features.get('sweep_speed', 0)
    
    # Parameters
    min_strength = params.get('min_strength', 0.6)
    min_volume = params.get('min_volume', 2000)
    max_speed = params.get('max_speed', 10)  # seconds
    
    # Check conditions
    if (sweep_strength >= min_strength and 
        sweep_volume >= min_volume and 
        sweep_speed <= max_speed):
        
        # Determine direction
        if sweep_direction > 0:
            signal = SignalType.BUY  # Bullish sweep
            confidence = min(sweep_strength, 0.85)
        elif sweep_direction < 0:
            signal = SignalType.SELL  # Bearish sweep
            confidence = min(sweep_strength, 0.85)
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.0
        
        if signal != SignalType.NEUTRAL:
            current_price = features.get('mid_price', 0)
            entry_price = current_price
            
            # Wider stops for momentum strategies
            sl_distance = current_price * params.get('stop_loss_pct', 0.01)
            if signal == SignalType.BUY:
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + sl_distance * params.get('risk_reward', 3)
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - sl_distance * params.get('risk_reward', 3)
            
            position_size = params.get('position_size', 1.0)
            
            return StrategyResult(
                signal=signal,
                confidence=confidence,
                reasoning=f"Liquidity sweep detected: strength={sweep_strength:.2f}, volume={sweep_volume:.0f}, direction={sweep_direction}",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size
            )
    
    return StrategyResult(
        signal=SignalType.NEUTRAL,
        confidence=0.0,
        reasoning="No liquidity sweep detected",
        entry_price=0, stop_loss=0, take_profit=0, position_size=0
    )


def stacked_imbalance_strategy(features: Dict[str, float], params: Dict[str, Any]) -> StrategyResult:
    """
    Stacked Imbalance Strategy
    
    Looks for multiple consecutive price levels with order book imbalances,
    indicating strong directional pressure.
    """
    imbalance_ratio = features.get('imbalance_ratio', 0)
    imbalance_side = features.get('imbalance_side', 0)
    imbalance_stacked = features.get('imbalance_stacked', 0)
    imbalance_stack_count = features.get('imbalance_stack_count', 0)
    
    # Parameters
    min_ratio = params.get('min_ratio', 2.0)
    require_stacked = params.get('require_stacked', True)
    min_stack_count = params.get('min_stack_count', 2)
    
    # Check conditions
    stacked_ok = not require_stacked or (imbalance_stacked and imbalance_stack_count >= min_stack_count)
    
    if imbalance_ratio >= min_ratio and stacked_ok:
        if imbalance_side > 0:
            signal = SignalType.BUY
            confidence = min(imbalance_ratio / 5.0, 0.8)  # Scale confidence
        elif imbalance_side < 0:
            signal = SignalType.SELL
            confidence = min(imbalance_ratio / 5.0, 0.8)
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.0
        
        if signal != SignalType.NEUTRAL:
            current_price = features.get('mid_price', 0)
            entry_price = current_price
            
            sl_distance = current_price * params.get('stop_loss_pct', 0.007)
            if signal == SignalType.BUY:
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + sl_distance * params.get('risk_reward', 2.5)
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - sl_distance * params.get('risk_reward', 2.5)
            
            position_size = params.get('position_size', 1.0)
            
            return StrategyResult(
                signal=signal,
                confidence=confidence,
                reasoning=f"Stacked imbalance detected: ratio={imbalance_ratio:.2f}, stacked={imbalance_stacked}, count={imbalance_stack_count}",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size
            )
    
    return StrategyResult(
        signal=SignalType.NEUTRAL,
        confidence=0.0,
        reasoning="No stacked imbalance detected",
        entry_price=0, stop_loss=0, take_profit=0, position_size=0
    )


def value_area_strategy(features: Dict[str, float], params: Dict[str, Any]) -> StrategyResult:
    """
    Value Area Strategy
    
    Trades breakouts from volume profile value areas,
    combined with order flow confirmation.
    """
    price_vs_poc = features.get('price_vs_poc_pct', 0)
    price_vs_vah = features.get('price_vs_vah_pct', 0)
    price_vs_val = features.get('price_vs_val_pct', 0)
    in_value_area = features.get('in_value_area', 0)
    
    # Order flow confirmation
    delta_30s = features.get('delta_30s', 0)
    volume_acceleration = features.get('volume_acceleration', 1.0)
    
    # Parameters
    breakout_threshold = params.get('breakout_threshold', 0.001)  # 0.1%
    min_volume_accel = params.get('min_volume_accel', 1.2)
    min_delta = params.get('min_delta', 100)
    
    # Check for breakout above VAH
    bullish_breakout = (price_vs_vah > breakout_threshold and 
                       volume_acceleration >= min_volume_accel and
                       delta_30s > min_delta)
    
    # Check for breakout below VAL
    bearish_breakout = (price_vs_val < -breakout_threshold and 
                       volume_acceleration >= min_volume_accel and
                       delta_30s < -min_delta)
    
    if bullish_breakout:
        signal = SignalType.BUY
        confidence = min(volume_acceleration / 2.0, 0.75)
        reasoning = f"VAH breakout with volume accel {volume_acceleration:.2f} and delta {delta_30s:.0f}"
        
    elif bearish_breakout:
        signal = SignalType.SELL
        confidence = min(volume_acceleration / 2.0, 0.75)
        reasoning = f"VAL breakout with volume accel {volume_acceleration:.2f} and delta {delta_30s:.0f}"
        
    else:
        return StrategyResult(
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            reasoning="No value area breakout detected",
            entry_price=0, stop_loss=0, take_profit=0, position_size=0
        )
    
    current_price = features.get('mid_price', 0)
    entry_price = current_price
    
    # Stop loss at recent value area boundary
    if signal == SignalType.BUY:
        stop_loss = features.get('val', current_price * 0.995)  # VAL or 0.5% below
        take_profit = entry_price + (entry_price - stop_loss) * params.get('risk_reward', 3)
    else:
        stop_loss = features.get('vah', current_price * 1.005)  # VAH or 0.5% above
        take_profit = entry_price - (stop_loss - entry_price) * params.get('risk_reward', 3)
    
    position_size = params.get('position_size', 1.0)
    
    return StrategyResult(
        signal=signal,
        confidence=confidence,
        reasoning=reasoning,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size
    )


# ==================== STRATEGY REGISTRY ====================

STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {
    StrategyType.ABSORPTION.value: {
        'function': absorption_strategy,
        'config': StrategyConfig(
            name='Absorption',
            description='Detects absorption of large orders without price movement',
            parameters={
                'min_strength': 0.7,
                'min_volume': 1000,
                'max_duration': 30,
                'stop_loss_pct': 0.005,
                'risk_reward': 2.0,
                'position_size': 1.0
            },
            required_features=[
                'absorption_strength', 'absorption_volume', 'absorption_duration', 
                'absorption_side', 'mid_price'
            ],
            risk_multiplier=1.0
        )
    },
    
    StrategyType.DELTA_DIVERGENCE.value: {
        'function': delta_divergence_strategy,
        'config': StrategyConfig(
            name='Delta Divergence',
            description='Detects divergence between price and order flow delta',
            parameters={
                'divergence_threshold': 0.5,
                'min_volume': 500,
                'stop_loss_pct': 0.005,
                'risk_reward': 2.0,
                'position_size': 1.0
            },
            required_features=[
                'delta_15s', 'delta_60s', 'price_change_pct_15s', 'price_change_pct_60s',
                'total_volume_15s', 'mid_price'
            ],
            risk_multiplier=1.2
        )
    },
    
    StrategyType.LIQUIDITY_SWEEP.value: {
        'function': liquidity_sweep_strategy,
        'config': StrategyConfig(
            name='Liquidity Sweep',
            description='Detects aggressive orders sweeping through resting liquidity',
            parameters={
                'min_strength': 0.6,
                'min_volume': 2000,
                'max_speed': 10,
                'stop_loss_pct': 0.01,
                'risk_reward': 3.0,
                'position_size': 1.0
            },
            required_features=[
                'sweep_strength', 'sweep_volume', 'sweep_direction', 
                'sweep_speed', 'mid_price'
            ],
            risk_multiplier=1.5
        )
    },
    
    StrategyType.STACKED_IMBALANCE.value: {
        'function': stacked_imbalance_strategy,
        'config': StrategyConfig(
            name='Stacked Imbalance',
            description='Detects multiple consecutive imbalanced price levels',
            parameters={
                'min_ratio': 2.0,
                'require_stacked': True,
                'min_stack_count': 2,
                'stop_loss_pct': 0.007,
                'risk_reward': 2.5,
                'position_size': 1.0
            },
            required_features=[
                'imbalance_ratio', 'imbalance_side', 'imbalance_stacked',
                'imbalance_stack_count', 'mid_price'
            ],
            risk_multiplier=1.3
        )
    },
    
    StrategyType.VALUE_AREA.value: {
        'function': value_area_strategy,
        'config': StrategyConfig(
            name='Value Area',
            description='Trades breakouts from volume profile value areas',
            parameters={
                'breakout_threshold': 0.001,
                'min_volume_accel': 1.2,
                'min_delta': 100,
                'risk_reward': 3.0,
                'position_size': 1.0
            },
            required_features=[
                'price_vs_poc_pct', 'price_vs_vah_pct', 'price_vs_val_pct',
                'in_value_area', 'delta_30s', 'volume_acceleration',
                'mid_price', 'poc', 'vah', 'val'
            ],
            risk_multiplier=1.4
        )
    }
}


def get_strategy(strategy_name: str) -> Optional[Callable]:
    """Get strategy function by name"""
    if strategy_name in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[strategy_name]['function']
    return None


def get_strategy_config(strategy_name: str) -> Optional[StrategyConfig]:
    """Get strategy configuration by name"""
    if strategy_name in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[strategy_name]['config']
    return None


def get_all_strategies() -> List[str]:
    """Get list of all available strategies"""
    return list(STRATEGY_REGISTRY.keys())


def validate_strategy_features(strategy_name: str, features: Dict[str, float]) -> bool:
    """Validate that required features are present for strategy"""
    config = get_strategy_config(strategy_name)
    if not config:
        return False
    
    return all(feature in features for feature in config.required_features)

# ==============================================================================
# SECTION 17: optimization/optuna_optimizer.py
# ==============================================================================

"""
Optuna Optimizer
Parameter optimization with warm starts and multi-objective support.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os

from loguru import logger


class OptimizationObjective(Enum):
    SHARPE = "sharpe"
    PROFIT = "profit"
    ROBUST = "robust"


@dataclass
class OptimizationResult:
    """Optimization result"""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    objective: OptimizationObjective
    study_name: str
    trials_df: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'objective': self.objective.value,
            'study_name': self.study_name,
            'timestamp': datetime.now().isoformat()
        }


@dataclass
class StrategyOptimizer:
    """
    Multi-objective parameter optimizer using Optuna.
    
    Features:
    - Warm starts from previous studies
    - Multiple optimization objectives
    - Parameter sensitivity analysis
    - Study persistence
    - Parallel optimization
    """
    
    strategy_name: str
    llm_advisor: Optional[Any] = None  # LLMAdvisor
    storage: str = "sqlite:///optuna_studies.db"
    study_name: Optional[str] = None
    
    # Optimization settings
    min_trades_required: int = 100
    n_startup_trials: int = 20
    n_warmup_steps: int = 50
    
    def __post_init__(self):
        if not self.study_name:
            self.study_name = f"{self.strategy_name}_optimization"
    
    def optimize(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: int = 200,
        n_jobs: int = 1,
        objective_type: str = "robust"
    ) -> OptimizationResult:
        """
        Run parameter optimization.
        
        Args:
            backtest_fn: Function that takes params dict and returns metrics dict
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            objective_type: "sharpe", "profit", or "robust"
        """
        objective = OptimizationObjective(objective_type)
        
        # Create or load study
        study = self._create_study(objective)
        
        # Define objective function
        def objective_fn(trial):
            params = self._suggest_parameters(trial)
            metrics = backtest_fn(params)
            
            # Store metrics in trial
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            
            # Calculate objective score
            score = self._calculate_objective_score(metrics, objective)
            
            return score
        
        # Run optimization
        logger.info(f"Starting optimization: {n_trials} trials, objective={objective.value}")
        
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=None,
            catch=(Exception,)
        )
        
        # Extract results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        # Get all trials data
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.TrialState.COMPLETE:
                trial_data = {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'user_attrs': trial.user_attrs
                }
                trials_data.append(trial_data)
        
        trials_df = pd.DataFrame(trials_data) if trials_data else None
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
            objective=objective,
            study_name=self.study_name,
            trials_df=trials_df
        )
        
        logger.info(f"Optimization complete. Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return result
    
    def _create_study(self, objective: OptimizationObjective) -> optuna.Study:
        """Create or load Optuna study"""
        try:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage
            )
            logger.info(f"Loaded existing study: {self.study_name}")
            
        except KeyError:
            # Create new study
            direction = optuna.study.StudyDirection.MAXIMIZE
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=direction,
                load_if_exists=False
            )
            logger.info(f"Created new study: {self.study_name}")
        
        return study
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameter values for the trial"""
        # This is strategy-specific - would be customized per strategy
        # For now, using generic parameters that work across strategies
        
        params = {}
        
        # Common parameters for order flow strategies
        params['min_strength'] = trial.suggest_float('min_strength', 0.5, 0.9, step=0.05)
        params['min_volume'] = trial.suggest_int('min_volume', 500, 5000, step=250)
        params['stop_loss_pct'] = trial.suggest_float('stop_loss_pct', 0.003, 0.015, step=0.001)
        params['risk_reward'] = trial.suggest_float('risk_reward', 1.5, 4.0, step=0.25)
        params['position_size'] = trial.suggest_float('position_size', 0.5, 2.0, step=0.1)
        
        # Strategy-specific parameters
        if 'absorption' in self.strategy_name:
            params['max_duration'] = trial.suggest_int('max_duration', 15, 60, step=5)
            
        elif 'delta_divergence' in self.strategy_name:
            params['divergence_threshold'] = trial.suggest_float('divergence_threshold', 0.3, 0.8, step=0.05)
            
        elif 'liquidity_sweep' in self.strategy_name:
            params['max_speed'] = trial.suggest_int('max_speed', 5, 30, step=5)
            
        elif 'stacked_imbalance' in self.strategy_name:
            params['min_ratio'] = trial.suggest_float('min_ratio', 1.5, 4.0, step=0.25)
            params['min_stack_count'] = trial.suggest_int('min_stack_count', 2, 5)
            
        elif 'value_area' in self.strategy_name:
            params['breakout_threshold'] = trial.suggest_float('breakout_threshold', 0.0005, 0.002, step=0.0001)
            params['min_volume_accel'] = trial.suggest_float('min_volume_accel', 1.1, 2.0, step=0.1)
        
        return params
    
    def _calculate_objective_score(
        self, 
        metrics: Dict[str, float], 
        objective: OptimizationObjective
    ) -> float:
        """
        Calculate optimization objective score.
        
        Handles different objective types with proper scaling and penalties.
        """
        # Check minimum trade requirement
        total_trades = metrics.get('total_trades', 0)
        if total_trades < self.min_trades_required:
            return -999  # Heavy penalty for insufficient trades
        
        if objective == OptimizationObjective.SHARPE:
            return metrics.get('sharpe_ratio', -999)
            
        elif objective == OptimizationObjective.PROFIT:
            # Profit with drawdown constraint
            total_return = metrics.get('total_return_pct', 0)
            max_drawdown = metrics.get('max_drawdown_pct', 1.0)
            
            if max_drawdown > 0.15:  # 15% max drawdown constraint
                return -999
            
            return total_return
            
        elif objective == OptimizationObjective.ROBUST:
            # Multi-factor robust score
            sharpe = metrics.get('sharpe_ratio', 0)
            profit_factor = metrics.get('profit_factor', 0)
            win_rate = metrics.get('win_rate', 0)
            max_drawdown = metrics.get('max_drawdown_pct', 1.0)
            
            # Normalize components
            sharpe_norm = min(max(sharpe, -2), 3) / 5 + 0.5  # Scale to 0-1
            profit_norm = min(profit_factor, 3) / 3  # Cap at 3
            win_rate_norm = win_rate
            drawdown_penalty = max(0, max_drawdown - 0.1) * 5  # Penalty for >10% DD
            
            # Weighted score
            score = (sharpe_norm * 0.4 + 
                    profit_norm * 0.3 + 
                    win_rate_norm * 0.2 - 
                    drawdown_penalty * 0.1)
            
            return max(score, -1)  # Floor at -1
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def analyze_parameter_sensitivity(self, result: OptimizationResult) -> Dict[str, Dict[str, float]]:
        """
        Analyze parameter sensitivity using partial dependence.
        
        Returns importance scores for each parameter.
        """
        if not result.trials_df or len(result.trials_df) < 20:
            return {}
        
        df = result.trials_df.dropna()
        if df.empty:
            return {}
        
        sensitivity = {}
        
        for param in result.best_params.keys():
            if param not in df['params'].iloc[0]:
                continue
            
            try:
                # Simple correlation-based importance
                param_values = df['params'].apply(lambda x: x.get(param, 0))
                scores = df['value']
                
                if param_values.std() > 0:
                    correlation = abs(np.corrcoef(param_values, scores)[0, 1])
                    sensitivity[param] = {
                        'importance': correlation,
                        'correlation': correlation,
                        'range': float(param_values.max() - param_values.min())
                    }
                    
            except Exception as e:
                logger.debug(f"Could not analyze sensitivity for {param}: {e}")
        
        # Sort by importance
        if sensitivity:
            sorted_params = sorted(sensitivity.items(), key=lambda x: x[1]['importance'], reverse=True)
            sensitivity = dict(sorted_params)
        
        return sensitivity
    
    def save_results(self, result: OptimizationResult, filepath: str) -> None:
        """Save optimization results to JSON file"""
        try:
            data = result.to_dict()
            
            # Add sensitivity analysis if available
            sensitivity = self.analyze_parameter_sensitivity(result)
            if sensitivity:
                data['parameter_sensitivity'] = sensitivity
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def load_results(self, filepath: str) -> Optional[OptimizationResult]:
        """Load optimization results from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            return OptimizationResult(
                best_params=data['best_params'],
                best_score=data['best_score'],
                n_trials=data['n_trials'],
                objective=OptimizationObjective(data['objective']),
                study_name=data['study_name']
            )
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return None
    
    def get_study_stats(self) -> Dict[str, Any]:
        """Get statistics about the optimization study"""
        try:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage
            )
            
            completed_trials = [t for t in study.trials if t.state == optuna.TrialState.COMPLETE]
            
            if not completed_trials:
                return {'status': 'no_completed_trials'}
            
            scores = [t.value for t in completed_trials]
            
            return {
                'study_name': self.study_name,
                'n_trials': len(study.trials),
                'n_completed': len(completed_trials),
                'best_score': study.best_value,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': min(scores),
                'max_score': max(scores)
            }
            
        except Exception as e:
            return {'error': str(e)}

# ==============================================================================
# SECTION 18: data_structures.py (duplicate)
# ==============================================================================

# This is a duplicate of core/data_structures.py - content already included above

# ==============================================================================
# SECTION 19: feature_engine.py (duplicate)
# ==============================================================================

# This is a duplicate of core/feature_engine.py - content already included above

# ==============================================================================
# END OF COMPREHENSIVE FILE
# ==============================================================================

# All project scripts have been gathered into this single file.
# Each section is clearly marked with the original filename.
# The file can be split back into individual files if needed. 



# ==============================================================================
# SECTION 1: main.py
# ==============================================================================

"""
Main Entry Point - Enhanced for Perfect Optimization
Orchestrates the complete order flow trading system.
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Optional, Dict, Any

from loguru import logger
import pandas as pd

# Configure logging
logger.add(
    "logs/orderflow_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)

from config.settings import settings, Settings
from core.feature_engine import FeatureEngine, FeatureConfig
from core.data_structures import OrderFlowState, Signal
from knowledge.llm_advisor import LLMAdvisor, LLMProvider
from knowledge.strategy_library import get_all_strategies, get_strategy
from optimization.optuna_optimizer import StrategyOptimizer
from backtesting.engine import BacktestEngine, WalkForwardValidator
from execution.risk_manager import RiskManager, RiskLimits
from execution.order_manager import OrderManager
from data.exchange_connector import ExchangeConnector, ExchangeConfig
from data.data_recorder import DataRecorder


class OrderFlowSystem:
    """
    Main system orchestrator with proper state management for optimization.
    
    Modes:
    1. Record - Record market data for backtesting
    2. Backtest - Run historical backtests
    3. Optimize - Optimize strategy parameters
    4. Validate - Walk-forward validation
    5. Paper - Paper trading (simulation)
    6. Live - Live trading
    7. Test - Connectivity test
    """
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        
        # Components (initialized lazily)
        self.llm_advisor: Optional[LLMAdvisor] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.exchange: Optional[ExchangeConnector] = None
        self.data_recorder: Optional[DataRecorder] = None
        
        # State
        self.running = False
        self.current_state: Optional[OrderFlowState] = None
    
    def _init_llm(self) -> None:
        """Initialize LLM advisor with automatic failover"""
        self.llm_advisor = LLMAdvisor.create_with_failover(self.settings.llm)
        
        if self.llm_advisor is None:
            logger.warning("LLM advisor not available - using hardcoded constants only")
        else:
            logger.info(f"LLM advisor ready: {self.llm_advisor.provider.value}")
    
    def _init_components(self, mode: str) -> None:
        """Initialize components based on mode"""
        # Feature engine always needed
        self.feature_engine = FeatureEngine(FeatureConfig(
            windows=self.settings.trading.feature_windows
        ))
        
        if mode in ['paper', 'live']:
            self.risk_manager = RiskManager(
                limits=RiskLimits(
                    max_position_size=self.settings.trading.max_position_size,
                    max_daily_loss_pct=self.settings.trading.max_daily_loss_pct
                ),
                initial_equity=self.settings.backtest.initial_capital
            )
            self.order_manager = OrderManager()
        
        if mode in ['record', 'paper', 'live']:
            self.exchange = ExchangeConnector(ExchangeConfig(
                exchange_id=self.settings.trading.exchange.value,
                testnet=(mode == 'paper')
            ))
        
        if mode == 'record':
            self.data_recorder = DataRecorder(
                symbol=self.settings.trading.symbol.replace('/', '')
            )
    
    def _get_feature_config(self) -> FeatureConfig:
        """Get standardized FeatureConfig for this system"""
        return FeatureConfig(
            windows=self.settings.trading.feature_windows,
            tick_size=self.settings.trading.tick_size  # <--- ADD THIS
        )
    
    # ==================== RECORD MODE ====================
    
    async def run_record(self, duration_hours: int = 24) -> None:
        """Record market data"""
        logger.info(f"Starting data recording for {duration_hours} hours")
        
        self._init_components('record')
        
        try:
            await self.exchange.connect()
            
            self.exchange.on_order_book_update = self._on_order_book_record
            self.exchange.on_trade = self._on_trade_record
            
            record_task = asyncio.create_task(self.data_recorder.start_recording())
            ws_task = asyncio.create_task(
                self.exchange.start_websocket(self.settings.trading.symbol)
            )
            
            await asyncio.sleep(duration_hours * 3600)
            
            await self.data_recorder.stop_recording()
            self.exchange.running = False
            
            logger.info("Recording complete")
        finally:
            await self.exchange.disconnect()
    
    async def _on_order_book_record(self, order_book: dict) -> None:
        """Callback for order book updates during recording"""
        self.data_recorder.record_order_book(order_book)
    
    async def _on_trade_record(self, trade: dict) -> None:
        """Callback for trade updates during recording"""
        self.data_recorder.record_trade(trade)
    
    # ==================== BACKTEST MODE ====================
    
    def run_backtest(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        params: dict = None,
        merged: bool = False
    ) -> dict:
        """Run backtest on historical data with proper state management"""
        logger.info(f"Running backtest: {strategy_name} from {start_date} to {end_date}")
        
        # Load data
        data = self._load_backtest_data(start_date, end_date, merged)
        
        if data.empty:
            logger.error("No data available for backtest period")
            return {}
        
        # Get strategy
        strategy = get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Unknown strategy: {strategy_name}")
            return {}
        
        # Run backtest with FeatureConfig (not feature_engine)
        engine = BacktestEngine(
            initial_capital=self.settings.backtest.initial_capital,
            fee_pct=self.settings.trading.fee_pct,
            slippage_pct=self.settings.trading.slippage_estimate_pct,
            feature_config=self._get_feature_config()  # FIX: pass config, not engine
        )
        
        metrics = engine.run(data, strategy, params)
        
        # Log results
        logger.info("Backtest Results:")
        logger.info(f"  Total Return: {metrics.total_return_pct*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Win Rate: {metrics.win_rate*100:.1f}%")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown_pct*100:.2f}%")
        logger.info(f"  Total Trades: {metrics.total_trades}")
        logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
        
        return {
            "total_return_pct": metrics.total_return_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "win_rate": metrics.win_rate,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "profit_factor": metrics.profit_factor,
            "total_trades": metrics.total_trades
        }
    
    def _load_backtest_data(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        merged: bool
    ) -> pd.DataFrame:
        """Load data for backtesting"""
        if merged:
            symbol_clean = self.settings.trading.symbol.replace('/', '')
            merged_file = f"./data/backtests/{symbol_clean}_{start_date.strftime('%Y%m%d')}_merged.parquet"
            
            logger.info(f"Loading pre-merged data: {merged_file}")
            try:
                data = pd.read_parquet(merged_file)
                data = data[
                    (data['timestamp'] >= pd.Timestamp(start_date, tz='UTC')) & 
                    (data['timestamp'] < pd.Timestamp(end_date + timedelta(days=1), tz='UTC'))
                ]
                return data
            except FileNotFoundError:
                logger.error(f"Merged file not found: {merged_file}")
                return pd.DataFrame()
        else:
            recorder = DataRecorder()
            return recorder.load_recorded_data(start_date, end_date, 'trades')
    
    # ==================== OPTIMIZE MODE (ENHANCED) ====================
    
    def run_optimization(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        n_trials: int = 200,
        merged: bool = False,
        objective: str = "robust"
    ) -> dict:
        """
        Run parameter optimization with enhanced objective function.
        
        Objectives:
        - "sharpe": Maximize Sharpe ratio only
        - "profit": Maximize total return with drawdown constraint
        - "robust": Multi-factor score (RECOMMENDED)
        """
        logger.info(f"Starting optimization: {strategy_name} (objective={objective})")
        
        self._init_llm()
        
        # Load data once
        data = self._load_backtest_data(start_date, end_date, merged)
        
        if data.empty:
            logger.error("No data available")
            return {}
        
        strategy = get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Unknown strategy: {strategy_name}")
            return {}
        
        # Create optimizer with enhanced objective
        optimizer = StrategyOptimizer(
            strategy_name=strategy_name,
            llm_advisor=self.llm_advisor,
            storage=self.settings.optuna.storage
        )
        
        # Get feature config once for all trials
        feature_config = self._get_feature_config()
        
        def backtest_fn(params: Dict[str, Any]) -> Dict[str, float]:
            # Each trial gets its own engine with fresh FeatureEngine
            engine = BacktestEngine(
                initial_capital=self.settings.backtest.initial_capital,
                fee_pct=self.settings.trading.fee_pct,
                slippage_pct=self.settings.trading.slippage_estimate_pct,
                feature_config=feature_config  # FIX: pass config, not engine
            )
            
            metrics = engine.run(data, strategy, params)
            
            # Check minimum trade count
            if metrics.total_trades < self.settings.optuna.min_trades_required:
                return {
                    "sharpe_ratio": -999,
                    "profit_factor": 0,
                    "win_rate": 0,
                    "max_drawdown_pct": 1.0,
                    "total_trades": 0,
                    "total_return_pct": 0,
                    "_penalized": True
                }
            
            return {
                "sharpe_ratio": metrics.sharpe_ratio,
                "profit_factor": metrics.profit_factor,
                "win_rate": metrics.win_rate,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "total_trades": metrics.total_trades,
                "total_return_pct": metrics.total_return_pct,
                "_penalized": False
            }
        
        # Run with selected objective
        result = optimizer.optimize(
            backtest_fn, 
            n_trials=n_trials,
            n_jobs=self.settings.optuna.n_jobs,
            objective_type=objective
        )
        
        # Save results
        output_path = f"./results/{strategy_name}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        optimizer.save_results(result, output_path)
        
        # Analyze parameter sensitivity
        sensitivity = optimizer.analyze_parameter_sensitivity(result)
        if sensitivity:
            logger.info("Parameter Sensitivity (importance):")
            for param, importance in sorted(sensitivity.items(), key=lambda x: -x[1]):
                logger.info(f"  {param}: {importance:.3f}")
        
        # Log analysis
        self._log_optimization_analysis(result)
        
        return {
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "objective": objective,
            "output_path": output_path
        }
    
    def _log_optimization_analysis(self, result) -> None:
        """Log analysis of optimization results"""
        if not result.all_trials:
            return
        
        valid_trials = [
            t for t in result.all_trials 
            if not t.get("user_attrs", {}).get("_penalized", False)
        ]
        
        logger.info(f"Optimization Analysis:")
        logger.info(f"  Valid trials: {len(valid_trials)} / {len(result.all_trials)}")
        
        if valid_trials:
            scores = [t["value"] for t in valid_trials if t["value"] is not None]
            trades = [t["user_attrs"].get("total_trades", 0) for t in valid_trials]
            
            if scores:
                logger.info(f"  Score range: {min(scores):.4f} to {max(scores):.4f}")
            if trades:
                logger.info(f"  Trade count range: {min(trades)} to {max(trades)}")
    
    # ==================== WALK-FORWARD VALIDATION (ENHANCED) ====================
    
    def run_walk_forward(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        merged: bool = False,
        n_trials_per_fold: int = 50
    ) -> dict:
        """
        Run walk-forward validation with enhanced metrics.
        """
        logger.info(f"Starting walk-forward validation: {strategy_name}")
        
        self._init_llm()
        
        data = self._load_backtest_data(start_date, end_date, merged)
        
        if data.empty:
            logger.error("No data available")
            return {}
        
        strategy = get_strategy(strategy_name)
        
        optimizer = StrategyOptimizer(
            strategy_name=strategy_name,
            llm_advisor=self.llm_advisor,
            storage=self.settings.optuna.storage
        )
        
        validator = WalkForwardValidator(
            train_days=self.settings.backtest.train_window_days,
            test_days=self.settings.backtest.test_window_days,
            step_days=self.settings.backtest.step_days,
            min_oos_trades=self.settings.optuna.min_trades_required
        )
        
        results = validator.validate(
            data=data,
            strategy=strategy,
            optimizer=optimizer,
            n_trials_per_fold=n_trials_per_fold,
            feature_config=self._get_feature_config()
        )
        
        # Enhanced logging
        summary = results.get('summary', {})
        logger.info("Walk-Forward Results:")
        logger.info(f"  Valid folds: {summary.get('valid_folds', 0)}/{summary.get('total_folds', 0)}")
        logger.info(f"  Avg OOS Sharpe: {summary.get('avg_oos_sharpe', 0):.2f}")
        logger.info(f"  Profitable Folds: {summary.get('pct_profitable_folds', 0)*100:.1f}%")
        
        if summary.get('skipped_folds', 0) > 0:
            logger.warning(f"  Skipped folds (insufficient trades): {summary['skipped_folds']}")
        
        # Log parameter stability
        stability = results.get('params_stability', {})
        if stability:
            logger.info("Parameter Stability (CV - lower is more stable):")
            for param, stats in sorted(stability.items(), key=lambda x: x[1].get('cv', 999)):
                cv = stats.get('cv', 0)
                stability_flag = "STABLE" if cv < 0.3 else "UNSTABLE" if cv > 0.7 else "MODERATE"
                logger.info(f"  {param}: CV={cv:.2f} [{stability_flag}]")
        
        return results
    
    # ==================== PAPER TRADING ====================
    
    async def run_paper(self, strategy_name: str, params: dict = None) -> None:
        """Run paper trading"""
        logger.info(f"Starting paper trading: {strategy_name}")
        
        self._init_components('paper')
        
        strategy = get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Unknown strategy: {strategy_name}")
            return
        
        await self.exchange.connect()
        
        self.exchange.on_order_book_update = lambda ob: self._on_market_data(ob, None, strategy, params)
        self.exchange.on_trade = lambda t: self._on_market_data(None, t, strategy, params)
        
        self.running = True
        
        await self.exchange.start_websocket(self.settings.trading.symbol)
    
    async def _on_market_data(
        self,
        order_book: dict,
        trade: dict,
        strategy,
        params: dict
    ) -> None:
        """Process market data update"""
        # Simplified - would need proper state management
        pass
    
    # ==================== LIVE TRADING ====================
    
    async def run_live(self, strategy_name: str, params: dict) -> None:
        """Run live trading - USE WITH EXTREME CAUTION"""
        logger.warning("=" * 50)
        logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
        logger.warning("=" * 50)
        
        confirm = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
        if confirm != "CONFIRM LIVE TRADING":
            logger.info("Live trading cancelled")
            return
        
        self._init_components('live')
        # Similar to paper trading but with real execution
        pass


async def run_connectivity_test(symbol: str = "XRP/USDT") -> None:
    """Test connectivity to REST and WebSocket endpoints."""
    print("\n" + "="*70)
    print("CONNECTIVITY TEST")
    print("="*70 + "\n")
    
    exchange = ExchangeConnector(ExchangeConfig(
        exchange_id="binance",
        testnet=False
    ))
    
    try:
        await exchange.connect()
        results = await exchange.test_connectivity(symbol)
        
        print(f"Symbol: {symbol}\n")
        
        print("REST API Connectivity:")
        rest_result = results.get("rest", {})
        if rest_result.get("ok"):
            print(f"  ✓ OK (latency: {rest_result.get('latency_ms', 0):.1f}ms)")
        else:
            print(f"  ✗ FAILED: {rest_result.get('error', 'Unknown error')}")
        
        print("\nWebSocket Connectivity:")
        ws_results = results.get("ws_urls", [])
        if ws_results:
            for i, ws_result in enumerate(ws_results, 1):
                url = ws_result.get("url", "")
                ok = ws_result.get("ok", False)
                latency = ws_result.get("latency_ms", 0)
                error = ws_result.get("error", "")
                
                try:
                    host = url.split('/')[2].split(':')[0]
                except:
                    host = "unknown"
                
                if ok:
                    print(f"  ✓ {host} OK (latency: {latency:.1f}ms)")
                else:
                    print(f"  ✗ {host} FAILED: {error}")
        else:
            print("  No WebSocket URLs tested")
        
        print("\n" + "="*70)
        all_ok = (results.get("rest", {}).get("ok", False) and 
                  all(r.get("ok", False) for r in ws_results))
        if all_ok:
            print("✓ ALL TESTS PASSED: Ready for production")
        else:
            print("✗ SOME TESTS FAILED: Check network, firewall, or DNS")
        print("="*70 + "\n")
        
    finally:
        await exchange.disconnect()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Order Flow Trading System")
    
    parser.add_argument(
        'mode',
        choices=['record', 'backtest', 'optimize', 'validate', 'paper', 'live', 'test'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='absorption',
        help='Strategy name (absorption, delta_divergence, liquidity_sweep, stacked_imbalance, value_area)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=24,
        help='Recording duration in hours'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=200,
        help='Number of optimization trials'
    )
    
    parser.add_argument(
        '--merged',
        action='store_true',
        help='Use pre-merged data from data_merger.py'
    )
    
    parser.add_argument(
        '--objective',
        type=str,
        choices=['sharpe', 'profit', 'robust'],
        default='robust',
        help='Optimization objective (default: robust)'
    )
    
    parser.add_argument(
        '--trials-per-fold',
        type=int,
        default=50,
        help='Number of optimization trials per walk-forward fold'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = (
        datetime.strptime(args.start_date, '%Y-%m-%d') 
        if args.start_date 
        else datetime.now() - timedelta(days=30)
    )
    end_date = (
        datetime.strptime(args.end_date, '%Y-%m-%d') 
        if args.end_date 
        else datetime.now()
    )
    
    # Create system
    system = OrderFlowSystem()
    
    # Run appropriate mode
    if args.mode == 'record':
        asyncio.run(system.run_record(args.duration))
    
    elif args.mode == 'backtest':
        result = system.run_backtest(args.strategy, start_date, end_date, merged=args.merged)
        if result:
            print("\n" + "="*50)
            print("BACKTEST SUMMARY")
            print("="*50)
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print("="*50)
    
    elif args.mode == 'optimize':
        result = system.run_optimization(
            args.strategy, 
            start_date, 
            end_date, 
            n_trials=args.trials, 
            merged=args.merged,
            objective=args.objective
        )
        if result:
            print("\n" + "="*50)
            print("OPTIMIZATION SUMMARY")
            print("="*50)
            print(f"  Objective: {args.objective}")
            print(f"  Best Score: {result['best_score']:.4f}")
            print(f"  Best Params:")
            for key, value in result['best_params'].items():
                print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
            print(f"  Output: {result.get('output_path', 'N/A')}")
            print("="*50)
    
    elif args.mode == 'validate':
        result = system.run_walk_forward(
            args.strategy, 
            start_date, 
            end_date, 
            merged=args.merged,
            n_trials_per_fold=args.trials_per_fold
        )
        if result:
            print("\n" + "="*50)
            print("WALK-FORWARD VALIDATION SUMMARY")
            print("="*50)
            summary = result.get('summary', {})
            print(f"  Valid Folds: {summary.get('valid_folds', 0)}/{summary.get('total_folds', 0)}")
            print(f"  Avg OOS Sharpe: {summary.get('avg_oos_sharpe', 0):.2f}")
            print(f"  Profitable Folds: {summary.get('pct_profitable_folds', 0)*100:.1f}%")
            if summary.get('error'):
                print(f"  ERROR: {summary['error']}")
            print("="*50)
    
    elif args.mode == 'paper':
        asyncio.run(system.run_paper(args.strategy))
    
    elif args.mode == 'live':
        asyncio.run(system.run_live(args.strategy, {}))
    
    elif args.mode == 'test':
        asyncio.run(run_connectivity_test(args.strategy))


if __name__ == "__main__":
    main()

# ==============================================================================
# SECTION 2: config/__init__.py
# ==============================================================================

# Empty file

# ==============================================================================
# SECTION 3: config/settings.py
# ==============================================================================

"""
Configuration and Settings
All parameters, API keys, and system configuration
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()


class Exchange(str, Enum):
    BINANCE = "binance"
    COINBASE = "coinbasepro"
    BYBIT = "bybit"
    OKX = "okx"


class LLMProvider(str, Enum):
    # Native SDK providers
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    # OpenAI-compatible providers
    OPENROUTER = "openrouter"
    GITHUB_MODELS = "github_models"
    XAI = "xai"
    ZAI = "zai"
    GROQ = "groq"
    OLLAMA = "ollama"


class TradingConfig(BaseModel):
    """Core trading parameters"""
    symbol: str = "XRP/USDT"
    exchange: Exchange = Exchange.BINANCE
    
    # Position limits
    max_position_size: float = 10000.0  # Was 1 for BTC Now 10000 for XRP
    max_position_value_pct: float = 0.25  # 25% of account
    
    # Risk limits
    max_daily_loss_pct: float = 0.02  # 2%
    max_drawdown_pct: float = 0.05  # 5%
    max_consecutive_losses: int = 5
    
    # Execution
    min_time_between_trades_sec: int = 30
    slippage_estimate_pct: float = 0.0005  # 0.05%
    fee_pct: float = 0.0004  # 0.04% maker
    
    #Add tick_size to the TradingConfig so it can be passed to the feature engine.
    tick_size: float = 0.0001  # <--- XRP typically uses 0.0001 or 0.01
    
    # Feature computation windows (seconds)
    feature_windows: List[int] = [15, 30, 60, 300, 600, 900]


class LLMConfig(BaseModel):
    """LLM API configuration"""
    provider: LLMProvider = LLMProvider.GEMINI
    model: Optional[str] = None  # None lets LLMAdvisor choose per-provider default
    
    # Native SDK API Keys (from environment)
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    
    # OpenAI-compatible providers API Keys (from environment)
    openrouter_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    github_token: Optional[str] = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))
    xai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("XAI_API_KEY"))
    zai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ZAI_API_KEY"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    
    # OpenRouter metadata (optional headers)
    openrouter_site_url: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", "http://localhost")
    )
    openrouter_app_name: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "OrderFlowTrader")
    )
    
    # Rate limiting
    max_calls_per_minute: int = 10
    cache_responses: bool = True
    cache_ttl_seconds: int = 3600
    
    # Failover configuration
    enable_failover: bool = True
    failover_providers: List[LLMProvider] = Field(
        default_factory=lambda: [
            LLMProvider.GEMINI,
            LLMProvider.GROQ,
            LLMProvider.OPENROUTER,
            LLMProvider.OLLAMA,
        ]
    )
    
    def get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Return API key for given provider"""
        key_map = {
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GEMINI: self.gemini_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENROUTER: self.openrouter_api_key,
            LLMProvider.GITHUB_MODELS: self.github_token,
            LLMProvider.XAI: self.xai_api_key,
            LLMProvider.ZAI: self.zai_api_key,
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.OLLAMA: "ollama",
        }
        return key_map.get(provider)
    
    def get_base_url(self, provider: LLMProvider) -> Optional[str]:
        """Return base URL for OpenAI-compatible providers"""
        urls = {
            LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
            LLMProvider.GITHUB_MODELS: "https://models.inference.ai.azure.com",
            LLMProvider.XAI: "https://api.x.ai/v1",
            LLMProvider.ZAI: "https://api.z.ai/api/paas/v4",
            LLMProvider.GROQ: "https://api.groq.com/openai/v1",
            LLMProvider.OLLAMA: "http://localhost:11434/v1",
        }
        return urls.get(provider)
    
    def get_failover_chain(self) -> List:
        """
        Returns ordered list of (provider, api_key, base_url) tuples to try.
        Priority: user preference first, then failover_providers list, then any remaining.
        """
        providers = []
        if self.provider not in self.failover_providers:
            providers.append(self.provider)
        providers.extend([p for p in self.failover_providers if p != self.provider])
        
        for p in LLMProvider:
            if p not in providers:
                providers.append(p)
        
        chain = []
        for provider in providers:
            api_key = self.get_api_key(provider)
            base_url = self.get_base_url(provider)
            
            if provider != LLMProvider.OLLAMA and not api_key:
                continue
                
            chain.append((provider, api_key, base_url))
        
        return chain


class OptunaConfig(BaseModel):
    """Optimization configuration - cloud optimized"""
    n_trials: int = 500
    n_startup_trials: int = 20  # Random exploration before Bayesian
    
    # Pruning
    enable_pruning: bool = True
    pruning_warmup_steps: int = 50
    
    # Study persistence
    # Use SQLite for single-instance, PostgreSQL for distributed cloud
    storage: str = "sqlite:///optuna_studies.db"
    study_name: str = "orderflow_optimization"
    
    # Objective
    primary_metric: str = "sharpe_ratio"
    min_trades_required: int = 100


class BacktestConfig(BaseModel):
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    
    # Walk-forward settings
    train_window_days: int = 60
    test_window_days: int = 14
    step_days: int = 7
    
    # Costs
    include_slippage: bool = True
    include_fees: bool = True
    
    # Data
    data_path: str = "./data/historical/"
    min_data_points: int = 100000


class Settings(BaseModel):
    """Master settings container"""
    trading: TradingConfig = TradingConfig()
    llm: LLMConfig = LLMConfig()
    optuna: OptunaConfig = OptunaConfig()
    backtest: BacktestConfig = BacktestConfig()


# Global settings instance
settings = Settings()

# ==============================================================================
# SECTION 4: core/data_structures.py
# ==============================================================================

"""
Core Data Structures
Defines all order flow concepts as proper data types
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
from datetime import datetime
import numpy as np


class Side(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()


class Regime(Enum):
    """
    v6.12 Full Regime Classification
    Used by strategies for entry filtering and SL/TP adaptation
    """
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    HIGH_VOLATILITY = auto()
    LOW_LIQUIDITY = auto()
    ACCUMULATION = auto()
    DISTRIBUTION = auto()
    BREAKOUT = auto()
    UNKNOWN = auto()


class SignalType(Enum):
    STRONG_BUY = auto()
    BUY = auto()
    WEAK_BUY = auto()
    NEUTRAL = auto()
    WEAK_SELL = auto()
    SELL = auto()
    STRONG_SELL = auto()


@dataclass
class PriceLevel:
    """Single price level in order book"""
    price: float
    size: float
    order_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderBook:
    """Full order book snapshot"""
    timestamp: datetime
    bids: List[PriceLevel]  # Sorted descending by price
    asks: List[PriceLevel]  # Sorted ascending by price
    
    @property
    def best_bid(self) -> Optional[PriceLevel]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[PriceLevel]:
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return 0.0
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points"""
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0
    
    @property
    def microprice(self) -> float:
        """Size-weighted mid price"""
        if self.best_bid and self.best_ask:
            total_size = self.best_bid.size + self.best_ask.size
            if total_size > 0:
                return (self.best_bid.price * self.best_ask.size + 
                        self.best_ask.price * self.best_bid.size) / total_size
        return self.mid_price


@dataclass
class Trade:
    """Single trade execution"""
    timestamp: datetime
    price: float
    size: float
    side: Side  # Aggressor side
    trade_id: Optional[str] = None
    
    @property
    def value(self) -> float:
        return self.price * self.size


@dataclass
class FootprintBar:
    """Footprint chart bar - volume at each price level"""
    timestamp: datetime
    duration_seconds: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    
    # Volume at price: {price: (buy_volume, sell_volume)}
    volume_at_price: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    
    @property
    def total_volume(self) -> float:
        return sum(buy + sell for buy, sell in self.volume_at_price.values())
    
    @property
    def delta(self) -> float:
        """Buy volume minus sell volume"""
        return sum(buy - sell for buy, sell in self.volume_at_price.values())
    
    @property
    def buy_volume(self) -> float:
        return sum(buy for buy, _ in self.volume_at_price.values())
    
    @property
    def sell_volume(self) -> float:
        return sum(sell for _, sell in self.volume_at_price.values())
    
    @property
    def poc_price(self) -> float:
        """Point of Control - price with highest volume"""
        if not self.volume_at_price:
            return self.close_price
        return max(self.volume_at_price.keys(), 
                   key=lambda p: sum(self.volume_at_price[p]))


@dataclass
class VolumeProfile:
    """Volume Profile / Market Profile statistics"""
    timestamp: datetime
    lookback_seconds: int
    
    # Volume distribution
    volume_at_price: Dict[float, float] = field(default_factory=dict)
    
    # Key levels (computed)
    poc: float = 0.0  # Point of Control
    vah: float = 0.0  # Value Area High
    val: float = 0.0  # Value Area Low
    
    # Additional statistics
    total_volume: float = 0.0
    value_area_pct: float = 0.70  # Default 70%
    
    def compute_value_area(self) -> None:
        """Calculate POC, VAH, VAL - O(N) with numpy"""
        if not self.volume_at_price:
            return
        
        prices = np.fromiter(self.volume_at_price.keys(), dtype=np.float64, 
                            count=len(self.volume_at_price))
        volumes = np.fromiter(self.volume_at_price.values(), dtype=np.float64,
                             count=len(self.volume_at_price))
        
        self.total_volume = volumes.sum()
        if self.total_volume == 0:
            return
        
        # POC: argmax is O(N) in C
        poc_idx = np.argmax(volumes)
        self.poc = float(prices[poc_idx])
        
        # Value Area calculation
        target = self.total_volume * self.value_area_pct
        sort_idx = np.argsort(volumes)[::-1]
        cumsum = np.cumsum(volumes[sort_idx])
        n_needed = np.searchsorted(cumsum, target) + 1
        
        va_prices = prices[sort_idx[:n_needed]]
        self.vah = float(va_prices.max())
        self.val = float(va_prices.min())


@dataclass
class Imbalance:
    """Detected imbalance at price level"""
    timestamp: datetime
    price: float
    buy_volume: float
    sell_volume: float
    imbalance_ratio: float  # buy/sell or sell/buy
    direction: Side  # Which side is dominant
    is_stacked: bool = False  # Part of stacked imbalance
    stack_count: int = 1  # How many consecutive levels


@dataclass
class Absorption:
    """Detected absorption event"""
    timestamp: datetime
    price: float
    absorbed_volume: float
    price_change_pct: float
    duration_seconds: float
    absorbing_side: Side  # Which side absorbed (held)
    strength: float  # 0-1 score
    
    @property
    def is_valid(self) -> bool:
        """Basic validation"""
        return self.strength > 0.5 and self.absorbed_volume > 0


@dataclass
class LiquiditySweep:
    """Detected liquidity sweep / stop hunt"""
    timestamp: datetime
    sweep_price: float  # Price swept to
    reversal_price: float  # Price after reversal
    direction: Side  # Direction of sweep
    volume_swept: float
    speed_seconds: float  # How fast the sweep occurred
    reversal_strength: float  # How strong the reversal


@dataclass
class IcebergOrder:
    """Detected iceberg order"""
    timestamp: datetime
    price: float
    visible_size: float
    estimated_hidden_size: float
    refill_count: int
    side: Side
    confidence: float  # Detection confidence


@dataclass 
class OrderFlowState:
    """Complete order flow state at a point in time"""
    timestamp: datetime
    
    # Order book state
    order_book: OrderBook
    
    # Recent trades
    recent_trades: List[Trade]
    
    # Computed metrics (will be populated by feature engine)
    features: Dict[str, float] = field(default_factory=dict)
    
    # Detected patterns
    absorptions: List[Absorption] = field(default_factory=list)
    imbalances: List[Imbalance] = field(default_factory=list)
    sweeps: List[LiquiditySweep] = field(default_factory=list)
    icebergs: List[IcebergOrder] = field(default_factory=list)
    
    # Market context
    regime: Regime = Regime.UNKNOWN
    volume_profile: Optional[VolumeProfile] = None
    
    # Signal
    signal: SignalType = SignalType.NEUTRAL
    signal_confidence: float = 0.0


@dataclass
class Signal:
    """Trading signal output"""
    timestamp: datetime
    signal_type: SignalType
    confidence: float  # 0-1
    
    # Trade parameters
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    
    # Reasoning
    primary_reason: str
    supporting_factors: List[str] = field(default_factory=list)
    
    # Risk metrics
    risk_reward_ratio: float = 0.0
    expected_value: float = 0.0
    
    @property
    def is_actionable(self) -> bool:
        return self.signal_type not in [SignalType.NEUTRAL] and self.confidence > 0.5

# ==============================================================================
# SECTION 5: core/feature_engine.py
# ==============================================================================

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
        
        return Regime.UNKNOWN


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