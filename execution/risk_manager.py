"""
Risk Manager
Enforces position limits, drawdown limits, and other risk controls.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from enum import Enum, auto

from loguru import logger

from core.data_structures import Signal, SignalType, Side


class RiskAction(Enum):
    ALLOW = auto()
    REDUCE_SIZE = auto()
    REJECT = auto()
    HALT_TRADING = auto()


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    # Position limits
    max_position_size: float = 1.0  # In base currency (e.g., BTC)
    max_position_value_pct: float = 0.25  # As % of account
    
    # Loss limits
    max_daily_loss_pct: float = 0.02  # 2%
    max_weekly_loss_pct: float = 0.05  # 5%
    max_drawdown_pct: float = 0.10  # 10%
    
    # Trade limits
    max_trades_per_day: int = 50
    max_trades_per_hour: int = 10
    min_time_between_trades_sec: int = 30
    max_consecutive_losses: int = 20
    
    # Exposure limits
    max_correlated_positions: int = 3
    max_sector_exposure_pct: float = 0.50


@dataclass
class RiskState:
    """Current risk state tracking"""
    # P&L tracking
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Trade tracking
    trades_today: int = 0
    trades_this_hour: int = 0
    consecutive_losses: int = 0
    last_trade_time: Optional[datetime] = None
    
    # Position tracking
    current_position_size: float = 0.0
    current_position_value: float = 0.0
    
    # Status
    trading_halted: bool = False
    halt_reason: str = ""
    
    # Timestamps
    day_start: Optional[datetime] = None
    week_start: Optional[datetime] = None
    hour_start: Optional[datetime] = None


class RiskManager:
    """
    Central risk management system.
    
    Responsibilities:
    1. Check all trades against risk limits before execution
    2. Track P&L and drawdown in real-time
    3. Enforce position sizing limits
    4. Halt trading when limits are breached
    5. Provide risk-adjusted position sizing
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None, initial_equity: float = 100000.0):
        self.limits = limits or RiskLimits()
        self.initial_equity = initial_equity
        
        self.state = RiskState(
            peak_equity=initial_equity,
            current_equity=initial_equity
        )
        
        self._reset_daily_tracking()
    
    def _reset_daily_tracking(self, current_timestamp: Optional[datetime] = None):
        """Reset daily counters using backtest timestamp (not wall-clock)"""
        now = current_timestamp or datetime.now()
        # Strip timezone info to avoid offset-naive/offset-aware comparison errors
        if hasattr(now, 'tzinfo') and now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        
        if self.state.day_start is None or now.date() > self.state.day_start.date():
            self.state.day_start = now.replace(hour=0, minute=0, second=0)
            self.state.daily_pnl = 0.0
            self.state.trades_today = 0
        
        if self.state.week_start is None or (now - self.state.week_start).days >= 7:
            self.state.week_start = now
            self.state.weekly_pnl = 0.0
        
        if self.state.hour_start is None or (now - self.state.hour_start).seconds >= 3600:
            self.state.hour_start = now
            self.state.trades_this_hour = 0
    
    def check_signal(self, signal: Signal, current_price: float, current_timestamp: Optional[datetime] = None) -> tuple:
        """
        Check if a signal passes all risk checks.
        
        Returns:
            (RiskAction, adjusted_signal or None, reason)
        """
        self._reset_daily_tracking(current_timestamp)
        
        # Check if trading is halted
        if self.state.trading_halted:
            return (RiskAction.HALT_TRADING, None, self.state.halt_reason)
        
        # Check daily loss limit
        daily_loss_pct = abs(self.state.daily_pnl) / self.initial_equity
        if self.state.daily_pnl < 0 and daily_loss_pct >= self.limits.max_daily_loss_pct:
            self._halt_trading("Daily loss limit reached")
            return (RiskAction.HALT_TRADING, None, "Daily loss limit reached")
        
        # Check weekly loss limit
        weekly_loss_pct = abs(self.state.weekly_pnl) / self.initial_equity
        if self.state.weekly_pnl < 0 and weekly_loss_pct >= self.limits.max_weekly_loss_pct:
            self._halt_trading("Weekly loss limit reached")
            return (RiskAction.HALT_TRADING, None, "Weekly loss limit reached")
        
        # Check drawdown limit
        if self.state.current_drawdown_pct >= self.limits.max_drawdown_pct:
            self._halt_trading("Max drawdown reached")
            return (RiskAction.HALT_TRADING, None, "Max drawdown reached")
        
        # Check consecutive losses
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            return (RiskAction.REJECT, None, "Too many consecutive losses")
        
        # Check trade frequency
        if self.state.trades_today >= self.limits.max_trades_per_day:
            return (RiskAction.REJECT, None, "Daily trade limit reached")
        
        if self.state.trades_this_hour >= self.limits.max_trades_per_hour:
            return (RiskAction.REJECT, None, "Hourly trade limit reached")
        
        # Check minimum time between trades
        if self.state.last_trade_time and current_timestamp:
            ts = current_timestamp
            if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            seconds_since_last = (ts - self.state.last_trade_time).total_seconds()
            if seconds_since_last < self.limits.min_time_between_trades_sec:
                return (RiskAction.REJECT, None, "Too soon since last trade")
        
        # Check and adjust position size
        adjusted_signal = self._adjust_position_size(signal, current_price)
        
        if adjusted_signal.position_size <= 0:
            return (RiskAction.REJECT, None, "Position size reduced to zero")
        
        if adjusted_signal.position_size < signal.position_size:
            return (RiskAction.REDUCE_SIZE, adjusted_signal, "Position size reduced for risk")
        
        return (RiskAction.ALLOW, adjusted_signal, "Passed all risk checks")
    
    def _adjust_position_size(self, signal: Signal, current_price: float) -> Signal:
        """Adjust position size based on risk limits"""
        import copy
        adjusted = copy.deepcopy(signal)
        
        # Calculate proposed position value
        proposed_value = self.state.current_equity * signal.position_size
        proposed_size = proposed_value / current_price
        
        # Check against max position size
        if proposed_size > self.limits.max_position_size:
            proposed_size = self.limits.max_position_size
            proposed_value = proposed_size * current_price
        
        # Check against max position value %
        max_value = self.state.current_equity * self.limits.max_position_value_pct
        if proposed_value > max_value:
            proposed_value = max_value
            proposed_size = proposed_value / current_price
        
        # Reduce size based on drawdown (dynamic sizing)
        drawdown_factor = 1.0 - (self.state.current_drawdown_pct / self.limits.max_drawdown_pct)
        drawdown_factor = max(0.25, min(1.0, drawdown_factor))  # Min 25% of normal size
        
        proposed_size *= drawdown_factor
        proposed_value = proposed_size * current_price
        
        # Reduce size based on consecutive losses
        if self.state.consecutive_losses > 0:
            loss_factor = 1.0 - (self.state.consecutive_losses * 0.1)  # -10% per loss
            loss_factor = max(0.5, loss_factor)  # Min 50%
            proposed_size *= loss_factor
        
        # Update signal
        adjusted.position_size = proposed_value / self.state.current_equity
        
        return adjusted
    
    def record_trade_opened(self, entry_price: float, size: float, side: Side, current_timestamp: Optional[datetime] = None):
        """Record that a trade was opened"""
        ts = current_timestamp or datetime.now()
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        self.state.last_trade_time = ts
        self.state.trades_today += 1
        self.state.trades_this_hour += 1
        
        self.state.current_position_size = size
        self.state.current_position_value = entry_price * size
        
        logger.debug(f"Trade opened: {side.name} {size:.4f} @ {entry_price:.2f}")
    
    def record_trade_closed(self, pnl: float):
        """Record that a trade was closed"""
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        
        # Update equity
        self.state.current_equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.current_equity)
        
        # Update drawdown
        if self.state.peak_equity > 0:
            self.state.current_drawdown_pct = (
                (self.state.peak_equity - self.state.current_equity) / self.state.peak_equity
            )
        
        # Track consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        # Reset position
        self.state.current_position_size = 0.0
        self.state.current_position_value = 0.0
        
        logger.debug(f"Trade closed: PnL={pnl:.2f}, Daily PnL={self.state.daily_pnl:.2f}")
    
    def _halt_trading(self, reason: str):
        """Halt all trading"""
        self.state.trading_halted = True
        self.state.halt_reason = reason
        logger.warning(f"TRADING HALTED: {reason}")
    
    def resume_trading(self):
        """Resume trading after halt"""
        self.state.trading_halted = False
        self.state.halt_reason = ""
        logger.info("Trading resumed")
    
    def get_status(self) -> Dict:
        """Get current risk status"""
        return {
            "trading_halted": self.state.trading_halted,
            "halt_reason": self.state.halt_reason,
            "daily_pnl": self.state.daily_pnl,
            "daily_pnl_pct": self.state.daily_pnl / self.initial_equity,
            "current_drawdown_pct": self.state.current_drawdown_pct,
            "consecutive_losses": self.state.consecutive_losses,
            "trades_today": self.state.trades_today,
            "current_position_size": self.state.current_position_size,
            "risk_capacity_pct": 1.0 - (self.state.current_drawdown_pct / self.limits.max_drawdown_pct)
        }