"""
Order Manager
Handles order execution, tracking, and exchange interaction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum, auto
import asyncio
import uuid

from loguru import logger

from core.fee_aware_filter import LiveFeeAwareFilter


class OrderStatus(Enum):
    PENDING = auto()
    SUBMITTED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


@dataclass
class Order:
    """Order representation"""
    id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: OrderType
    size: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None
    
    # Linked orders (for OCO, brackets)
    parent_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None


@dataclass
class Fill:
    """Trade fill/execution"""
    order_id: str
    fill_id: str
    price: float
    size: float
    fee: float
    timestamp: datetime


class OrderManager:
    """
    Manages order lifecycle and execution.
    
    Responsibilities:
    1. Submit orders to exchange
    2. Track order status
    3. Handle fills and partial fills
    4. Manage bracket orders (entry + SL + TP)
    5. Provide execution callbacks
    """
    
    def __init__(self, exchange_client=None):
        self.exchange = exchange_client
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.pending_orders: List[str] = []
        self.active_orders: List[str] = []
        
        # Fill tracking
        self.fills: List[Fill] = []
        
        # Callbacks
        self.on_fill: Optional[Callable[[Fill], None]] = None
        self.on_order_update: Optional[Callable[[Order], None]] = None

        # Fee-aware filter for live/paper trading (Futures fee structure)
        self.fee_filter = LiveFeeAwareFilter(
            maker_fee_pct=0.0002,
            taker_fee_pct=0.0005,
            expected_spread_pct=0.0001,
            min_profit_target_pct=0.0002
        )
    
    def create_market_order(
        self,
        symbol: str,
        side: str,
        size: float
    ) -> Order:
        """Create a market order"""
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            size=size
        )
        
        self.orders[order.id] = order
        self.pending_orders.append(order.id)
        
        return order
    
    def create_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float
    ) -> Order:
        """Create a limit order"""
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            size=size,
            price=price
        )
        
        self.orders[order.id] = order
        self.pending_orders.append(order.id)
        
        return order
    
    def create_bracket_order(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: Optional[float],
        stop_loss: float,
        take_profit: float,
        entry_type: OrderType = OrderType.MARKET
    ) -> tuple:
        """
        Create a bracket order (entry + stop loss + take profit).
        
        Returns:
            (entry_order, stop_loss_order, take_profit_order)
        """
        # Entry order
        entry_order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=entry_type,
            size=size,
            price=entry_price if entry_type == OrderType.LIMIT else None
        )
        
        # Stop loss order (opposite side)
        sl_side = "sell" if side == "buy" else "buy"
        sl_order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=sl_side,
            order_type=OrderType.STOP,
            size=size,
            stop_price=stop_loss,
            parent_order_id=entry_order.id
        )
        
        # Take profit order (opposite side, limit)
        tp_order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=sl_side,
            order_type=OrderType.LIMIT,
            size=size,
            price=take_profit,
            parent_order_id=entry_order.id
        )
        
        # Link orders
        entry_order.stop_loss_order_id = sl_order.id
        entry_order.take_profit_order_id = tp_order.id
        
        # Store orders
        self.orders[entry_order.id] = entry_order
        self.orders[sl_order.id] = sl_order
        self.orders[tp_order.id] = tp_order
        
        self.pending_orders.append(entry_order.id)
        
        return (entry_order, sl_order, tp_order)
    
    def validate_signal_with_fee_filter(
        self,
        signal,
        predicted_move_pct: float,
        confidence: float,
        current_bid: float,
        current_ask: float,
        last_price: float
    ) -> dict:
        """
        Validate a signal through the fee-aware filter before order creation.

        Args:
            signal: Signal object from strategy evaluation
            predicted_move_pct: Predicted price move as decimal
            confidence: Signal confidence 0.0-1.0
            current_bid: Current bid price
            current_ask: Current ask price
            last_price: Last traded price

        Returns:
            dict with 'status' ('APPROVED' or 'REJECTED') and 'reason'
        """
        should_trade, reason = self.fee_filter.should_trade_with_spread(
            signal,
            predicted_move_pct,
            confidence,
            current_bid,
            current_ask,
            last_price
        )

        if not should_trade:
            logger.warning(f"[OrderManager] Fee filter REJECTED: {reason}")
            return {'status': 'REJECTED', 'reason': reason}

        logger.debug(f"[OrderManager] Fee filter APPROVED: {reason}")
        return {'status': 'APPROVED', 'reason': reason}

    async def submit_order(self, order: Order) -> bool:
        """Submit order to exchange"""
        if not self.exchange:
            logger.warning("No exchange client configured - simulating submission")
            order.status = OrderStatus.SUBMITTED
            order.exchange_order_id = f"SIM_{order.id}"
            return True
        
        try:
            if order.order_type == OrderType.MARKET:
                result = await self.exchange.create_market_order(
                    order.symbol,
                    order.side,
                    order.size
                )
            elif order.order_type == OrderType.LIMIT:
                result = await self.exchange.create_limit_order(
                    order.symbol,
                    order.side,
                    order.size,
                    order.price
                )
            elif order.order_type == OrderType.STOP:
                result = await self.exchange.create_stop_order(
                    order.symbol,
                    order.side,
                    order.size,
                    order.stop_price
                )
            else:
                logger.error(f"Unsupported order type: {order.order_type}")
                return False
            
            order.exchange_order_id = result.get('id')
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            
            if order.id in self.pending_orders:
                self.pending_orders.remove(order.id)
            self.active_orders.append(order.id)
            
            logger.info(f"Order submitted: {order.id} -> {order.exchange_order_id}")
            
            if self.on_order_update:
                self.on_order_update(order)
            
            return True
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error(f"Order submission failed: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            logger.warning(f"Cannot cancel order in status: {order.status}")
            return False
        
        if self.exchange and order.exchange_order_id:
            try:
                await self.exchange.cancel_order(order.exchange_order_id, order.symbol)
            except Exception as e:
                logger.error(f"Failed to cancel order on exchange: {e}")
                return False
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        if order.id in self.active_orders:
            self.active_orders.remove(order.id)
        
        if self.on_order_update:
            self.on_order_update(order)
        
        return True
    
    def process_fill(self, order_id: str, price: float, size: float, fee: float = 0.0):
        """Process a fill for an order"""
        if order_id not in self.orders:
            logger.warning(f"Fill for unknown order: {order_id}")
            return
        
        order = self.orders[order_id]
        
        # Create fill record
        fill = Fill(
            order_id=order_id,
            fill_id=str(uuid.uuid4()),
            price=price,
            size=size,
            fee=fee,
            timestamp=datetime.now()
        )
        
        self.fills.append(fill)
        
        # Update order
        old_filled = order.filled_size
        order.filled_size += size
        
        # Update average fill price
        if order.filled_size > 0:
            order.avg_fill_price = (
                (old_filled * order.avg_fill_price + size * price) / order.filled_size
            )
        
        order.updated_at = datetime.now()
        
        # Update status
        if order.filled_size >= order.size:
            order.status = OrderStatus.FILLED
            if order.id in self.active_orders:
                self.active_orders.remove(order.id)
            
            # Activate bracket orders if this was an entry
            self._activate_bracket_orders(order)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        logger.info(f"Fill processed: {order_id}, {size}@{price}, total filled: {order.filled_size}/{order.size}")
        
        # Callbacks
        if self.on_fill:
            self.on_fill(fill)
        
        if self.on_order_update:
            self.on_order_update(order)
    
    def _activate_bracket_orders(self, entry_order: Order):
        """Activate stop loss and take profit orders after entry fills"""
        if entry_order.stop_loss_order_id:
            sl_order = self.orders.get(entry_order.stop_loss_order_id)
            if sl_order:
                self.pending_orders.append(sl_order.id)
                logger.debug(f"Stop loss order queued: {sl_order.id}")
        
        if entry_order.take_profit_order_id:
            tp_order = self.orders.get(entry_order.take_profit_order_id)
            if tp_order:
                self.pending_orders.append(tp_order.id)
                logger.debug(f"Take profit order queued: {tp_order.id}")
    
    def cancel_bracket_orders(self, entry_order_id: str):
        """Cancel the bracket orders when one side fills"""
        entry_order = self.orders.get(entry_order_id)
        if not entry_order:
            return
        
        # Cancel stop loss
        if entry_order.stop_loss_order_id:
            asyncio.create_task(self.cancel_order(entry_order.stop_loss_order_id))
        
        # Cancel take profit
        if entry_order.take_profit_order_id:
            asyncio.create_task(self.cancel_order(entry_order.take_profit_order_id))
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return [
            self.orders[oid] for oid in self.active_orders
            if self.orders[oid].status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)