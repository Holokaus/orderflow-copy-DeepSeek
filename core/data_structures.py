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
    CRASH = auto()
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
        """
        Calculate POC, VAH, VAL using the standard Market Profile value-area
        algorithm (a.k.a. "market profile" / TPO expansion from POC).

        The old implementation sorted price levels by *volume* and took the
        top-N, which could yield a VAH/VAL spanning disconnected price
        clusters — useless for mean-reversion at a continuous value band.

        Standard algorithm: start at the POC, then repeatedly expand into the
        adjacent (higher or lower) price level with the larger volume, until
        the accumulated volume reaches `value_area_pct` of total. VAH/VAL are
        then the max/min of the contiguous band around the POC.
        """
        if not self.volume_at_price:
            return

        self.total_volume = sum(self.volume_at_price.values())
        if self.total_volume == 0:
            return

        # POC: price level with the highest volume.
        poc = max(self.volume_at_price, key=lambda p: self.volume_at_price[p])
        self.poc = float(poc)

        target = self.total_volume * self.value_area_pct

        # Build a sorted-by-price view so we can look up neighbours by index.
        sorted_prices = sorted(self.volume_at_price.keys())
        pos = sorted_prices.index(poc)

        lo = hi = pos
        acc = self.volume_at_price[poc]
        n = len(sorted_prices)

        while acc < target and (lo > 0 or hi < n - 1):
            # Volume of the level just below (lo-1) and just above (hi+1) the band.
            below_vol = self.volume_at_price[sorted_prices[lo - 1]] if lo > 0 else -1.0
            above_vol = self.volume_at_price[sorted_prices[hi + 1]] if hi < n - 1 else -1.0
            if below_vol < 0 and above_vol < 0:
                break
            if above_vol >= below_vol:  # ties expand upward (convention)
                hi += 1
                acc += above_vol
            else:
                lo -= 1
                acc += below_vol

        self.vah = float(sorted_prices[hi])
        self.val = float(sorted_prices[lo])


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