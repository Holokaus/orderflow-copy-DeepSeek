"""
Exchange Connector
Handles connection to exchanges for market data and trading.

Properly manages a local order book via REST snapshots + incremental
WebSocket diff application, per the Binance documentation:
  1. Fetch a REST depth snapshot on connect / reconnect.
  2. Buffer WS diff events until the snapshot is obtained.
  3. Apply diffs incrementally: upsert levels, delete when qty == 0.
  4. Track sequence IDs (`U`, `u`, `pu`) to detect missed events.
  5. If a gap is detected, re-fetch a REST snapshot and re-sync.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import json

from loguru import logger

try:
    import ccxt.async_support as ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    exchange_id: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    rate_limit: bool = True
    ws_base_url: str = ""  # Optional override for WebSocket base URL
    use_futures_stream: bool = False  # Use futures stream instead of spot (higher liquidity)


class ExchangeConnector:
    """
    Unified exchange connector supporting REST and WebSocket.

    Features:
    1. REST API for order management
    2. WebSocket for real-time market data
    3. Proper local order book maintenance (snapshot + incremental diffs)
    4. Trade stream processing with exchange timestamps
    5. Connection health monitoring with automatic reconnect
    6. Sequence validation to detect missed messages
    """

    # --------------- Connection health constants ---------------
    MAX_RECONNECT_DELAY = 30          # seconds
    INITIAL_RECONNECT_DELAY = 1       # seconds
    STALE_CONNECTION_TIMEOUT = 60     # seconds – force reconnect if silent
    BOOK_DEPTH_LIMIT = 1000           # levels to request in REST snapshot
    EMIT_BOOK_DEPTH = 20              # levels to emit to callbacks

    def __init__(self, config: ExchangeConfig):
        self.config = config

        if not HAS_CCXT:
            raise ImportError("ccxt not installed. Run: pip install ccxt")

        # Initialize REST client
        exchange_class = getattr(ccxt, config.exchange_id, None)
        if not exchange_class:
            raise ValueError(f"Unknown exchange: {config.exchange_id}")

        self.rest_client = exchange_class({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'sandbox': config.testnet,
            'enableRateLimit': config.rate_limit,
            'options': {
                'defaultType': 'future' if 'future' in config.exchange_id else 'spot'
            }
        })

        # WebSocket state
        self.ws_connection = None
        self.running = False

        # ---------- Local order book (price -> size dicts) ----------
        self._local_bids: Dict[float, float] = {}   # price -> size
        self._local_asks: Dict[float, float] = {}   # price -> size
        self._last_update_id: int = 0                # from REST snapshot
        self._prev_final_update_id: int = 0          # `u` of last applied diff
        self._book_initialised: bool = False
        self._event_buffer: List[Dict] = []          # buffer while waiting for snapshot
        self._snapshot_lock = asyncio.Lock()

        # Public-facing order_book dict (backward compatible)
        self.order_book: Dict[str, Any] = {}
        self.recent_trades: List[Dict] = []

        # Connection health
        self._last_message_time: float = 0.0
        self._reconnect_delay: float = self.INITIAL_RECONNECT_DELAY
        self._ws_symbol: str = ""
        self._ws_url_index: int = 0  # Track current URL in fallback list
        self._ws_urls: List[str] = []  # List of URLs to try

        # Callbacks
        self.on_order_book_update: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
        self.on_ticker: Optional[Callable] = None
        self.on_health_alert: Optional[Callable] = None  # Health monitoring callback

        # Health metrics for data quality
        self._health_metrics = {
            'gaps_detected': 0,
            'resyncs': 0,
            'events_received': 0,
            'events_applied': 0,
            'avg_latency_ms': 0,
            'last_gap_time': None
        }
        self._pending_gap: Optional[tuple] = None
        self._last_resync_time: float = 0.0  # Monotonic time of last REST re-sync

    async def connect(self):
        """Initialize connection"""
        await self.rest_client.load_markets()
        logger.info(f"Connected to {self.config.exchange_id}")

    async def disconnect(self):
        """Close all connections properly."""
        self.running = False
        
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        
        # CRITICAL: Close REST client to prevent unclosed session warning
        if self.rest_client:
            await self.rest_client.close()
        
        logger.info("Disconnected from exchange")

    # ==================== REST API Methods ====================

    async def get_balance(self) -> Dict[str, float]:
        """Get account balances"""
        try:
            balance = await self.rest_client.fetch_balance()
            return {
                currency: data['free']
                for currency, data in balance.items()
                if isinstance(data, dict) and data.get('free', 0) > 0
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker"""
        try:
            return await self.rest_client.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch ticker: {e}")
            return {}

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book snapshot"""
        try:
            return await self.rest_client.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            return {'bids': [], 'asks': []}

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        try:
            return await self.rest_client.fetch_trades(symbol, limit=limit)
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float
    ) -> Dict:
        """Create market order"""
        try:
            order = await self.rest_client.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            logger.info(f"Market order created: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to create market order: {e}")
            raise

    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float
    ) -> Dict:
        """Create limit order"""
        try:
            order = await self.rest_client.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
            logger.info(f"Limit order created: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to create limit order: {e}")
            raise

    async def create_stop_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float
    ) -> Dict:
        """Create stop order"""
        try:
            order = await self.rest_client.create_order(
                symbol=symbol,
                type='stop_market',
                side=side,
                amount=amount,
                params={'stopPrice': stop_price}
            )
            logger.info(f"Stop order created: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to create stop order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            await self.rest_client.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            return await self.rest_client.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    # ==================== Local Order Book Management ====================

    async def _fetch_rest_snapshot(self, symbol: str):
        """Fetch snapshot and find valid starting point in buffer."""
        async with self._snapshot_lock:
            try:
                snap = await self.rest_client.fetch_order_book(
                    symbol, limit=self.BOOK_DEPTH_LIMIT
                )

                self._local_bids = {
                    float(price): float(size)
                    for price, size in snap.get('bids', [])
                    if float(size) > 0
                }
                self._local_asks = {
                    float(price): float(size)
                    for price, size in snap.get('asks', [])
                    if float(size) > 0
                }

                self._last_update_id = snap.get('nonce', 0)
                self._prev_final_update_id = 0
                self._pending_gap = None

                logger.info(
                    f"Snapshot loaded: {len(self._local_bids)}b/{len(self._local_asks)}a, "
                    f"lastUpdateId={self._last_update_id}"
                )

                # Search buffer for valid first event
                target = self._last_update_id + 1
                valid_idx = None
                
                for idx, evt in enumerate(self._event_buffer):
                    u_first = evt.get('U', 0)
                    u_final = evt.get('u', 0)
                    
                    if u_final <= self._last_update_id:
                        continue
                    
                    if u_first <= target <= u_final:
                        valid_idx = idx
                        logger.info(f"Valid event at buffer[{idx}]: U={u_first}, u={u_final}")
                        break
                    elif u_first > target:
                        gap_size = u_first - target
                        logger.warning(f"Gap in buffer: missing {gap_size} updates, clearing")
                        self._event_buffer.clear()
                        break
                
                if valid_idx is not None:
                    applied = 0
                    for evt in self._event_buffer[valid_idx:]:
                        if not self._apply_diff_if_valid(evt):
                            break
                        applied += 1
                    logger.info(f"Applied {applied} buffered events")
                
                if self._event_buffer:
                    logger.debug(f"Discarding {len(self._event_buffer)} stale buffered events after snapshot")
                self._event_buffer.clear()
                self._book_initialised = True
                logger.info(
                    f"Book re-initialized. Total resyncs so far: {self._health_metrics['resyncs']}"
                )

            except Exception as e:
                logger.error(f"Snapshot failed: {e}")
                self._book_initialised = False

    def _apply_diff_if_valid(self, data: Dict) -> bool:
        """
        Apply Binance Spot depthUpdate with strict validation.
        
        Rules:
        - First event: U <= lastUpdateId+1 AND u >= lastUpdateId+1
        - Subsequent: U == prev_u + 1 (strict continuity)
        
        Returns True if applied, False if rejected (triggers re-sync or gap fill).
        """
        event_first_id = data.get('U', 0)
        event_final_id = data.get('u', 0)
        event_time_ms = data.get('E', 0)
        
        # Calculate receive latency (informational only, does NOT affect validation)
        recv_latency_ms = (time.time() * 1000) - event_time_ms if event_time_ms else 0
        if recv_latency_ms > 5000:  # Only warn for genuinely high latency (>5s)
            logger.debug(f"WS event age: {recv_latency_ms:.0f}ms (may be stale buffered event)")
        
        # Drop events entirely before snapshot
        if event_final_id <= self._last_update_id:
            return False
        
        # Validate first event after snapshot
        if self._prev_final_update_id == 0:
            if not (event_first_id <= self._last_update_id + 1 <= event_final_id):
                logger.debug(
                    f"First event rejected: U={event_first_id}, u={event_final_id}, "
                    f"need U<={self._last_update_id + 1} <= u"
                )
                return False
        else:
            # Strict continuity check
            if event_first_id != self._prev_final_update_id + 1:
                gap_size = event_first_id - self._prev_final_update_id - 1
                logger.error(
                    f"GAP DETECTED: expected U={self._prev_final_update_id + 1}, "
                    f"got U={event_first_id}, missing {gap_size} updates"
                )
                self._health_metrics['gaps_detected'] += 1
                self._health_metrics['last_gap_time'] = datetime.now(timezone.utc)
                self._pending_gap = (self._prev_final_update_id + 1, event_first_id - 1)
                return False
        
        # Apply diffs
        for price_str, size_str in data.get('b', []):
            price, size = float(price_str), float(size_str)
            if size == 0:
                self._local_bids.pop(price, None)
            else:
                self._local_bids[price] = size

        for price_str, size_str in data.get('a', []):
            price, size = float(price_str), float(size_str)
            if size == 0:
                self._local_asks.pop(price, None)
            else:
                self._local_asks[price] = size

        self._prev_final_update_id = event_final_id
        self._health_metrics['events_applied'] += 1
        return True

    async def _backfill_gap(self, symbol: str, start_id: int, end_id: int):
        """
        Attempt to fill gap using REST API.
        Note: Binance doesn't provide historical diff updates via REST.
        Only trades can be backfilled. For L2 gaps, we must re-sync.
        """
        logger.warning(f"Cannot backfill L2 gap {start_id}-{end_id}, forcing re-sync")
        # Binance lacks historical order book diff API
        # Alternative: fetch recent trades to maintain some data continuity
        try:
            trades = await self.rest_client.fetch_trades(symbol, since=start_id)
            logger.info(f"Backfilled {len(trades)} trades for gap period")
            # Store these trades separately as 'gap_fill' records
        except Exception as e:
            logger.error(f"Trade backfill failed: {e}")
        
        # Force full re-sync
        self._book_initialised = False

    def _build_sorted_book(self) -> Dict[str, Any]:
        """
        Build a sorted, trimmed order book from the local state for emission.
        Bids: sorted descending by price. Asks: sorted ascending by price.
        Only positive-size levels are included.
        """
        sorted_bids = sorted(
            ((p, s) for p, s in self._local_bids.items() if s > 0),
            key=lambda x: x[0],
            reverse=True
        )[:self.EMIT_BOOK_DEPTH]

        sorted_asks = sorted(
            ((p, s) for p, s in self._local_asks.items() if s > 0),
            key=lambda x: x[0]
        )[:self.EMIT_BOOK_DEPTH]

        return {
            'bids': [[p, s] for p, s in sorted_bids],
            'asks': [[p, s] for p, s in sorted_asks],
        }

    # ==================== WebSocket Methods ====================

    async def start_websocket(self, symbol: str):
        """Start WebSocket with multi-endpoint failover and health monitoring."""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets not installed. Run: pip install websockets")
        
        self.running = True
        self._ws_symbol = symbol
        self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
        self._ws_urls = self._get_ws_urls(symbol)
        self._ws_url_index = 0
        consecutive_failures = 0
        
        while self.running:
            ws_url = self._ws_urls[self._ws_url_index]
            
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    open_timeout=15,
                    #additional_headers={"User-Agent": "orderflow-recorder/1.0"},
                    #extra_headers={"User-Agent": "orderflow-recorder/1.0"},
                ) as ws:
                    self.ws_connection = ws
                    logger.info(f"WebSocket connected: {ws_url}")
                    
                    # Reset on successful connection
                    consecutive_failures = 0
                    self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
                    
                    # Reset book state and fetch fresh REST snapshot
                    self._book_initialised = False
                    self._event_buffer.clear()
                    self._prev_final_update_id = 0
                    await self._fetch_rest_snapshot(symbol)
                    
                    self._last_message_time = time.monotonic()
                    
                    await self._subscribe(ws, symbol)
                    
                    watchdog_task = asyncio.create_task(
                        self._stale_connection_watchdog(ws)
                    )
                    
                    try:
                        async for message in ws:
                            if not self.running:
                                break
                            self._last_message_time = time.monotonic()
                            await self._process_message(message, symbol)
                    finally:
                        watchdog_task.cancel()
                        try:
                            await watchdog_task
                        except asyncio.CancelledError:
                            pass
            
            except websockets.exceptions.ConnectionClosed as e:
                consecutive_failures += 1
                logger.warning(
                    f"WebSocket closed (code={e.code}) on {ws_url}, "
                    f"failure #{consecutive_failures}"
                )
            
            except Exception as e:
                consecutive_failures += 1
                error_msg = str(e)
                logger.error(
                    f"WebSocket error on {ws_url}: {error_msg}, "
                    f"failure #{consecutive_failures}"
                )
            
            if not self.running:
                break
            
            # Rotate to next URL after 2 consecutive failures on current one
            if consecutive_failures >= 2:
                old_url = self._ws_urls[self._ws_url_index]
                self._ws_url_index = (self._ws_url_index + 1) % len(self._ws_urls)
                new_url = self._ws_urls[self._ws_url_index]
                logger.warning(
                    f"Rotating endpoint: {old_url} -> {new_url} "
                    f"(after {consecutive_failures} failures)"
                )
                consecutive_failures = 0
                # Short delay before trying new endpoint
                await asyncio.sleep(1)
            else:
                # Backoff before retrying same endpoint
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self.MAX_RECONNECT_DELAY
                )

    async def _stale_connection_watchdog(self, ws):
        """Force-close the WS if no message is received for STALE_CONNECTION_TIMEOUT."""
        while True:
            await asyncio.sleep(5)
            elapsed = time.monotonic() - self._last_message_time
            if elapsed > self.STALE_CONNECTION_TIMEOUT:
                logger.warning(
                    f"No WS message for {elapsed:.1f}s, forcing reconnect"
                )
                await ws.close()
                return

    def _get_ws_urls(self, symbol: str) -> List[str]:
        """
        Return a LIST of WebSocket URLs to try, in priority order.
        Uses multiple Binance endpoints for resilience.
        No API key needed for public market data streams.
        """
        symbol_lower = symbol.replace('/', '').lower()
        streams = f"{symbol_lower}@depth@100ms/{symbol_lower}@aggTrade"
        
        if self.config.exchange_id != "binance":
            raise ValueError(f"WebSocket not configured for: {self.config.exchange_id}")
        
        if self.config.testnet:
            logger.warning(
                "Testnet has low liquidity and frequent gaps. "
                "Use production endpoints (testnet=False) for realistic data."
            )
            return [
                f"wss://stream.testnet.binance.vision/stream?streams={streams}",
            ]
        
        # Production endpoints - multiple for failover.
        # Port 443 is standard HTTPS, less likely blocked than 9443.
        # data-stream.binance.vision is an alternative domain.
        urls = []
        
        # If user provided a custom override, try it first
        if hasattr(self.config, 'ws_base_url') and self.config.ws_base_url:
            urls.append(f"{self.config.ws_base_url}/stream?streams={streams}")
        
        urls.extend([
            f"wss://stream.binance.com:443/stream?streams={streams}",
            f"wss://stream.binance.com:9443/stream?streams={streams}",
            f"wss://data-stream.binance.vision/stream?streams={streams}",
            f"wss://stream1.binance.com:443/stream?streams={streams}",
            f"wss://stream2.binance.com:443/stream?streams={streams}",
            f"wss://stream3.binance.com:443/stream?streams={streams}",
        ])
        
        return urls

    async def _subscribe(self, ws, symbol: str):
        """Subscribe to market data streams"""
        if self.config.exchange_id == "binance":
            # Binance uses combined streams in URL, no subscription needed
            pass
        elif self.config.exchange_id == "bybit":
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    f"orderbook.50.{symbol.replace('/', '')}",
                    f"publicTrade.{symbol.replace('/', '')}"
                ]
            }
            await ws.send(json.dumps(subscribe_msg))

    async def _process_message(self, message: str, symbol: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Unwrap combined stream envelope if present
            # Combined streams send: {"stream": "btcusdt@depth@100ms", "data": {...}}
            if 'stream' in data and 'data' in data:
                data = data['data']

            if self.config.exchange_id == "binance":
                if 'e' in data:
                    if data['e'] == 'depthUpdate':
                        await self._handle_depth_update(data, symbol)
                    elif data['e'] == 'aggTrade':
                        await self._handle_trade(data)
            elif self.config.exchange_id == "bybit":
                if 'topic' in data:
                    if 'orderbook' in data['topic']:
                        await self._handle_depth_update(data['data'], symbol)
                    elif 'publicTrade' in data['topic']:
                        await self._handle_trade(data['data'])

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    # ==================== Message Handlers ====================

    async def _handle_depth_update(self, data: Dict, symbol: str):
        """Handle depth update with gap detection and recovery.
        
        Key invariant: after fetching a REST snapshot, we MUST discard all
        buffered WS events that are older than the snapshot before processing
        new ones. Otherwise we enter a re-sync death spiral.
        """
        if not self._book_initialised:
            self._event_buffer.append(data)
            if len(self._event_buffer) > 1000:
                self._event_buffer = self._event_buffer[-500:]
            return

        self._health_metrics['events_received'] += 1
        
        event_final_id = data.get('u', 0)
        
        # CRITICAL: silently drop events that are older than our snapshot.
        # After a re-sync, many stale events may be queued in the WS buffer.
        # These must be skipped without triggering another re-sync.
        if event_final_id <= self._last_update_id:
            return  # Stale event, just skip it
        
        applied = self._apply_diff_if_valid(data)
        
        if not applied:
            # Only re-sync if we haven't JUST re-synced.
            # Use a cooldown to prevent rapid-fire re-syncs.
            now = time.monotonic()
            time_since_last_resync = now - self._last_resync_time
            
            if time_since_last_resync < 5.0:
                # Recently re-synced — this is likely a stale event from before
                # the resync. Just skip it silently.
                logger.debug(
                    f"Skipping rejected diff (U={data.get('U')}, u={event_final_id}) "
                    f"— last resync was {time_since_last_resync:.1f}s ago"
                )
                return
            
            if hasattr(self, '_pending_gap') and self._pending_gap:
                await self._backfill_gap(symbol, *self._pending_gap)
                self._pending_gap = None
            else:
                logger.warning(
                    f"Diff rejected (U={data.get('U')}, u={event_final_id}, "
                    f"lastUpdateId={self._last_update_id}), re-syncing from REST"
                )
                self._health_metrics['resyncs'] += 1
                self._book_initialised = False
                self._last_resync_time = now
                await self._fetch_rest_snapshot(symbol)
            return

        # Successfully applied — build and emit book
        sorted_book = self._build_sorted_book()
        exchange_ts_ms = data.get('E')
        
        # Only compute latency if the event is reasonably fresh
        latency_ms = None
        if exchange_ts_ms:
            latency_ms = (time.time() * 1000) - exchange_ts_ms
            # If latency is negative or absurdly high, it's a stale/clock-skew event
            if latency_ms < 0 or latency_ms > 10000:  # >10 seconds or negative (clock skew)
                latency_ms = None  # Don't report misleading latency
        
        sorted_book.update({
            'timestamp': datetime.fromtimestamp(exchange_ts_ms / 1000, tz=timezone.utc) if exchange_ts_ms else datetime.now(timezone.utc),
            'exchange_timestamp_ms': exchange_ts_ms,
            'last_update_id': self._prev_final_update_id,
            'latency_ms': latency_ms
        })
        
        self.order_book = sorted_book
        
        if self.on_order_book_update:
            await self.on_order_book_update(sorted_book)

    async def _handle_trade(self, data: Dict):
        """Handle trade update, using exchange timestamps."""
        # Validate input
        if not isinstance(data, dict):
            logger.warning(f"Invalid trade data: not a dict")
            return
        
        if 'p' not in data and 'price' not in data:
            logger.warning(f"Invalid trade data: missing price field")
            return
        
        try:
            if isinstance(data, dict):
                # Extract exchange trade time from Binance aggTrade `T` field (ms)
                trade_ts_ms = data.get('T', None)
                if trade_ts_ms is not None:
                    trade_dt = datetime.fromtimestamp(
                        trade_ts_ms / 1000.0, tz=timezone.utc
                    )
                else:
                    trade_dt = datetime.now(timezone.utc)

                price = float(data.get('p', data.get('price', 0)))
                size = float(data.get('q', data.get('size', 0)))
                
                # Skip trades with invalid price or size
                if price <= 0 or size <= 0:
                    logger.debug(f"Skipping invalid trade: price={price}, size={size}")
                    return

                trade = {
                    'price': price,
                    'size': size,
                    'side': 'sell' if data.get('m', data.get('side')) else 'buy',
                    'timestamp': trade_dt,
                    'exchange_timestamp_ms': trade_ts_ms,
                }
            else:
                trade = data
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse trade data: {e}")
            return

        self.recent_trades.append(trade)

        # Keep only recent trades (ring-buffer style)
        if len(self.recent_trades) > 1000:
            self.recent_trades = self.recent_trades[-500:]

        if self.on_trade:
            await self.on_trade(trade)

    def get_health_report(self) -> Dict:
        """Return data quality metrics."""
        return {
            **self._health_metrics,
            'book_initialized': self._book_initialised,
            'buffer_size': len(self._event_buffer),
            'last_update_id': self._prev_final_update_id
        }

    async def test_connectivity(self, symbol: str = "BTC/USDT") -> Dict:
        """Test REST and WebSocket connectivity to diagnose connection issues.
        
        Returns:
            Dict with connectivity status for REST and each WS URL:
            {
                "rest": {"ok": true, "latency_ms": 45},
                "ws_urls": [
                    {"url": "wss://...", "ok": true, "latency_ms": 120},
                    ...
                ]
            }
        """
        result = {"rest": {}, "ws_urls": []}
        
        # Test REST connectivity
        try:
            start = time.time()
            ticker = await self.rest_client.fetch_ticker(symbol)
            latency_ms = (time.time() - start) * 1000
            result["rest"] = {"ok": True, "latency_ms": latency_ms}
            logger.info(f"✓ REST API OK (latency: {latency_ms:.1f}ms)")
        except Exception as e:
            result["rest"] = {"ok": False, "error": str(e)}
            logger.error(f"✗ REST API FAILED: {e}")
        
        # Test WebSocket URLs
        symbol_lower = symbol.replace('/', '').lower()
        ws_urls_to_test = self._get_ws_urls(symbol)
        
        for ws_url in ws_urls_to_test:
            try:
                start = time.time()
                async with websockets.connect(
                    ws_url,
                    open_timeout=15,
                    close_timeout=5,
                ) as ws:
                    # Wait for first message (up to 2 seconds)
                    message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    latency_ms = (time.time() - start) * 1000
                    result["ws_urls"].append({"url": ws_url, "ok": True, "latency_ms": latency_ms})
                    logger.info(f"✓ WebSocket {ws_url.split('/')[2]} OK (latency: {latency_ms:.1f}ms)")
            except asyncio.TimeoutError:
                latency_ms = (time.time() - start) * 1000
                result["ws_urls"].append({"url": ws_url, "ok": False, "error": "Timeout waiting for data", "latency_ms": latency_ms})
                logger.warning(f"⚠ WebSocket {ws_url.split('/')[2]} timeout (no data in 2s)")
            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                result["ws_urls"].append({"url": ws_url, "ok": False, "error": str(e), "latency_ms": latency_ms})
                logger.error(f"✗ WebSocket {ws_url.split('/')[2]} FAILED: {e}")
        
        return result