"""
Fee-Aware Signal Filter
=======================
PHASE 2 ENHANCEMENT

Filters out signals where the predicted price move doesn't cover trading costs.
This prevents "negative edge" trades that look profitable in backtest but lose money after fees.

Concept:
- Entry fee (maker/taker)
- Exit fee (maker/taker)
- Bid-ask spread
- Minimum profit target
- Total cost = entry_fee + exit_fee + spread + min_profit

If predicted_move < threshold(cost, confidence), reject signal.
Confidence-adjusted: lower confidence requires larger move to justify trade.
"""

from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from core.data_structures import Signal


@dataclass
class FeeConfig:
    """Fee configuration for a trading pair (Futures)"""
    maker_fee_pct: float = 0.0002      # 0.02% Binance futures maker
    taker_fee_pct: float = 0.0005      # 0.05% Binance futures taker
    expected_spread_pct: float = 0.0001  # 1 bps typical spread
    min_profit_target_pct: float = 0.0002  # 2 bps minimum profit margin


class FeeAwareFilter:
    """
    Filter signals based on whether the predicted price move covers trading costs.
    
    Only accepts signals if:
    - predicted_move_pct >= (entry_fee + exit_fee + spread + min_profit) / confidence
    
    This confidence-adjustment means:
    - High confidence (0.9) requires smaller move
    - Low confidence (0.5) requires larger move
    - Failed trades are caught earlier
    """
    
    def __init__(
        self,
        maker_fee_pct: float = 0.0002,
        taker_fee_pct: float = 0.0005,
        expected_spread_pct: float = 0.0001,
        min_profit_target_pct: float = 0.0002,
    ):
        """
        Initialize fee-aware filter.
        
        Args:
            maker_fee_pct: Maker fee as decimal (0.001 = 0.1%)
            taker_fee_pct: Taker fee as decimal (0.001 = 0.1%)
            expected_spread_pct: Expected bid-ask spread as decimal (0.0005 = 5 bps)
            min_profit_target_pct: Minimum profit margin target as decimal
        """
        self.entry_fee = maker_fee_pct
        self.exit_fee = taker_fee_pct  # Assume exit might use taker
        self.expected_spread = expected_spread_pct
        self.min_profit = min_profit_target_pct
        
        # Total cost threshold (all fees + minimum profit)
        self.total_cost = (
            self.entry_fee +
            self.exit_fee +
            self.expected_spread +
            self.min_profit
        )
        
        logger.debug(
            f"[FeeAwareFilter] Initialized with total cost threshold: {self.total_cost:.4%} "
            f"(entry={self.entry_fee:.4%}, exit={self.exit_fee:.4%}, "
            f"spread={self.expected_spread:.4%}, min_profit={self.min_profit:.4%})"
        )
    
    def should_ignore_signal(
        self,
        signal: Signal,
        predicted_price_move_pct: float,
        confidence: float = 0.7
    ) -> Tuple[bool, str]:
        """
        Determine if signal should be ignored based on fee coverage.
        
        Args:
            signal: The trading signal
            predicted_price_move_pct: Predicted price movement as decimal (0.01 = 1%)
            confidence: Model confidence 0.0-1.0 (default 0.7). 
                       Adjusted threshold = cost / confidence
        
        Returns:
            Tuple[bool, str]: (should_ignore, reason_string)
                - should_ignore=True means signal doesn't cover costs, reject it
                - reason_string explains why (for logging/analysis)
        
        Example:
            >>> faf = FeeAwareFilter()
            >>> ignore, reason = faf.should_ignore_signal(
            ...     signal=my_signal,
            ...     predicted_price_move_pct=0.002,
            ...     confidence=0.65
            ... )
            >>> if ignore:
            ...     logger.info(f"Skipped signal: {reason}")
        """
        # Clamp confidence to reasonable range
        confidence = max(0.3, min(confidence, 1.0))
        
        # Adjust threshold for confidence: lower confidence needs higher move
        # Example: cost=0.004 (0.4%), confidence=0.7 → threshold=0.4%/0.7=0.57%
        confidence_adjusted_threshold = self.total_cost / confidence
        
        # Check if predicted move covers the confidence-adjusted cost
        if predicted_price_move_pct < confidence_adjusted_threshold:
            reason = (
                f"Predicted move {predicted_price_move_pct:.4%} < "
                f"threshold {confidence_adjusted_threshold:.4%} "
                f"(base_cost={self.total_cost:.4%}, confidence={confidence:.2f}). "
                f"Signal would likely lose money after fees."
            )
            return True, reason
        
        reason = (
            f"Signal passes fee-aware filter. "
            f"Predicted move {predicted_price_move_pct:.4%} >= "
            f"threshold {confidence_adjusted_threshold:.4%}"
        )
        return False, reason
    
    def should_reject_by_spread(
        self,
        actual_spread_pct: float
    ) -> Tuple[bool, str]:
        """
        Reject signal if actual spread is much worse than expected (liquidity issue).
        
        Args:
            actual_spread_pct: Current bid-ask spread as decimal
        
        Returns:
            Tuple[bool, str]: (should_reject, reason)
        """
        # Reject if spread is >2x expected (likely illiquidity)
        spread_multiplier = actual_spread_pct / self.expected_spread
        
        if spread_multiplier > 2.0:
            reason = (
                f"Spread {actual_spread_pct:.4%} is {spread_multiplier:.1f}x "
                f"expected {self.expected_spread:.4%}. Likely illiquidity. Reject."
            )
            return True, reason
        
        return False, ""
    
    def filter_signals(
        self,
        signals: list,
        predicted_moves: dict,
        confidences: dict = None
    ) -> Tuple[list, dict]:
        """
        Filter a list of signals based on fee coverage.
        
        Args:
            signals: List of Signal objects
            predicted_moves: Dict mapping signal_id -> predicted_move_pct
            confidences: Dict mapping signal_id -> confidence (default 0.7 if missing)
        
        Returns:
            Tuple[filtered_signals, rejection_reasons]
                - filtered_signals: Signals that pass the filter
                - rejection_reasons: Dict of rejected signal_id -> reason
        """
        if confidences is None:
            confidences = {}
        
        filtered = []
        rejected = {}
        
        for signal in signals:
            signal_id = id(signal)  # Use object id as unique identifier
            predicted_move = predicted_moves.get(signal_id, 0.0)
            confidence = confidences.get(signal_id, 0.7)
            
            should_ignore, reason = self.should_ignore_signal(
                signal,
                predicted_move,
                confidence
            )
            
            if should_ignore:
                rejected[signal_id] = reason
                logger.debug(f"[filter_signals] Rejected signal: {reason}")
            else:
                filtered.append(signal)
        
        logger.info(
            f"[filter_signals] Filtered {len(signals)} signals -> "
            f"{len(filtered)} passed, {len(rejected)} rejected by fee filter"
        )
        
        return filtered, rejected
    
    def get_breakeven_move(self, confidence: float = 0.7) -> float:
        """
        Get the minimum predicted move required to cover costs at given confidence.
        
        Useful for debugging: "What's the minimum move needed for this trade to work?"
        
        Args:
            confidence: Model confidence 0.0-1.0
        
        Returns:
            Minimum predicted move as decimal (e.g., 0.004 = 0.4%)
        """
        confidence = max(0.3, min(confidence, 1.0))
        return self.total_cost / confidence


class LiveFeeAwareFilter(FeeAwareFilter):
    """
    Extended filter for live/paper trading with real-time spread checking.
    """
    
    def should_trade_with_spread(
        self,
        signal: Signal,
        predicted_price_move_pct: float,
        confidence: float,
        current_bid: float,
        current_ask: float,
        last_price: float
    ) -> Tuple[bool, str]:
        """
        Determine if trade should proceed with real-time spread data.
        
        Args:
            signal: Trading signal
            predicted_price_move_pct: Predicted move from model
            confidence: Model confidence
            current_bid: Current bid price
            current_ask: Current ask price
            last_price: Last traded price (for spread calculation)
        
        Returns:
            Tuple[bool, str]: (should_trade, reason)
        """
        # Calculate actual spread
        actual_spread = current_ask - current_bid
        actual_spread_pct = actual_spread / last_price if last_price > 0 else 0
        
        # Check spread first (quick liquidity filter)
        should_reject, spread_reason = self.should_reject_by_spread(actual_spread_pct)
        if should_reject:
            return False, spread_reason
        
        # Check fee coverage with fee-aware filter
        should_ignore, fee_reason = self.should_ignore_signal(
            signal,
            predicted_price_move_pct,
            confidence
        )
        if should_ignore:
            return False, fee_reason
        
        # All checks pass
        return True, (
            f"Trade approved. Spread={actual_spread_pct:.4%}, "
            f"predicted_move={predicted_price_move_pct:.4%}, "
            f"confidence={confidence:.2f}"
        )
