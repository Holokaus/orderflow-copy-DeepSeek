# DeepSeek OrderFlow Project — Full Knowledge Document

> Generated: 2026-06-06  
> Purpose: Portable session handoff — all context, code, logs, decisions, and experiments  
> Branch: `post-cooldown-both-edits-for-paper-trade` (merged from `test-paper-fix-local-internet-orderbook-data-gaps`)

---

## 1. PROJECT OVERVIEW

**Goal:** Real-time order-flow trading system for Binance ICP/USDT.  
**Strategies:** Absorption + StackedImbalance (post-cooldown variants).  
**Trading mode:** Paper trading (simulated execution on live data).  
**Capital:** $100, 1x leverage, no margin.  
**Exchange:** Binance Spot (data) + Futures fee structure (fees).

### Architecture

```
User Script (run_paper_*.py)
  └── OrderFlowSystem (main.py)
        ├── ExchangeConnector (data/exchange_connector.py)
        │     ├── REST API (ccxt) — balance, orders
        │     └── WebSocket (websockets) — depth@100ms + aggTrade
        ├── FeatureEngine (core/feature_engine.py)
        │     └── FeaturePrecomputer (core/feature_precomputer.py)
        ├── RiskManager (execution/risk_manager.py)
        ├── OrderManager (execution/order_manager.py)
        │     └── LiveFeeAwareFilter (core/fee_aware_filter.py)
        └── Strategies (knowledge/strategy_library.py)
              ├── Absorption
              └── StackedImbalance
```

### Data Flow

```
Binance WS ──→ ExchangeConnector._process_message()
                ├── depthUpdate → _handle_depth_update() → callback on_order_book_update
                └── aggTrade   → _handle_trade()       → callback on_trade

_on_paper_order_book (every ~100ms):
  1. Build OrderBook from raw depth
  2. Convert buffered trades → List[Trade]
  3. Update FeatureEngine(state)
  4. Mark-to-market open positions
  5. Check exits (SL/TP/flow-based)
  6. Evaluate strategies (if flat / under max concurrent limit)
  7. Risk check → Fee filter → Entry execution
```

---

## 2. KEY FILES & ROLES

| File | Role |
|---|---|
| `main.py` | Core `OrderFlowSystem` — all paper trading logic |
| `data/exchange_connector.py` | Binance WS + REST, local order book, diagnostic counters |
| `core/fee_aware_filter.py` | `FeeAwareFilter` + `LiveFeeAwareFilter` — spread/fee checks |
| `execution/order_manager.py` | `OrderManager` with `LiveFeeAwareFilter` instance |
| `execution/risk_manager.py` | `RiskManager` + `RiskLimits` |
| `core/data_structures.py` | `Side`, `SignalType`, `Signal`, `Trade`, `OrderBook`, `PriceLevel` |
| `core/feature_engine.py` | `FeatureEngine` + `FeatureConfig` |
| `core/feature_precomputer.py` | `FeaturePrecomputer` — precompute features for backtest speed |
| `knowledge/strategy_library.py` | Strategy definitions (Absorption, StackedImbalance) |
| `backtesting/engine.py` | `BacktestEngine` — historical backtesting engine |
| `run_paper_both.py` | Paper trading runner (single position, relaxed spread) |
| `run_paper_multitrade.py` | Paper trading runner (up to 2 concurrent positions) |
| `run_post_cooldown_both.py` | Backtest runner (original, single position, 0.01% spread) |
| `test_multi_trade_relaxed_spread.py` | One-time backtest (0.05% spread + multi-position) |

---

## 3. BRANCH HISTORY

```
* d66e086 (HEAD -> post-cooldown-both-edits-for-paper-trade) fix
* 22348ed Update log
* 64f1e31 Update log
* 0e753c4 switch to spot mode (use_futures=False)
* e80af96 diagnostic: WS message type counters + trade trace logging
* cf6eb41 not effective changes
* c99de5d fix-local-unternet-orderbook-data-gap
* 8268cea future-stream
* df60868 Update log
* 87c617d /post-cooldown-both-edits-for-paper-trade
* f6572b7 paper trading pipeline: multi-strategy support + loss streak cooldown
* af3420d add runner script for Post-cooldown Both Strategies
* 7998a53 Post-cooldown Absorption / SI / Both
* 6b6758d AI report
* 7788dc8 cleaning
...
```

Key branch: `post-cooldown-both-edits-for-paper-trade` — contains all paper trading + diagnostics + spot mode fix.

---

## 4. THE 0-TRADES PROBLEM (ROOT CAUSE & FIX)

### Problem

Paper trading showed `TradesRcv:0` after 39 minutes — 16,000 depth updates, zero `aggTrade` events.

### Investigation

1. **Hypothesis 1: Field name mismatch (AI misdiagnosis)**  
   The AI claimed field names like `p`, `q`, `m`, `T` from Binance were not being mapped correctly.  
   **Reality:** The `_handle_trade()` method correctly maps `p→price`, `q→size`, `m→side`, `T→timestamp`. The AI was wrong.

2. **Hypothesis 2: Futures domain has no trade data**  
   Connected to `fstream.binance.com:443` (Binance Futures).  
   **Test:** Added `_msg_counters` (depthUpdate/aggTrade/other) and `_trade_log_count` trace logging.  
   **Result:** After 39 minutes: 16,000 depth updates, 0 aggTrade. **Confirmed: ICPUSDT futures has no trade volume.**

3. **Hypothesis 3: User error with Binance testnet**  
   Verified `testnet=False` — connecting to production. Not the issue.

### Fix

**Switch from Binance Futures to Binance Spot** (`use_futures=False`):

```python
# In ExchangeConfig and _get_ws_urls:
# Before (futures):
primary_domain = "fstream.binance.com"  # No aggTrade for ICP

# After (spot):
primary_domain = "stream.binance.com"   # aggTrade flows normally
```

Changed in `_init_components()` (main.py:109):
```python
def _init_components(self, mode, testnet=None, use_futures=None):
    is_futures = use_futures if use_futures is not None else True  # default futures
    # ...
    ExchangeConfig(use_futures=is_futures)
```

All runners explicitly pass `use_futures=False`.

**Result (spot):**
```
[TRACE TRADE #1] p=2.42600000 q=66.34000000 m=False
[TRACE TRADE #2] p=2.42500000 q=2.63000000 m=True
[TRACE TRADE #3] p=2.42600000 q=5.67000000 m=False
→ 550 aggTrade in ~25 min on Tokyo server
```

---

## 5. THE FEE FILTER PROBLEM

### Problem

With spot data working, strategy signals fired but the **fee filter rejected every trade**:

```
Fee filter REJECTED: Spread 0.0412% is 4.1x expected 0.0100%. Likely illiquidity.
```

### Root Cause

The `LiveFeeAwareFilter` in `order_manager.py` was initialized with futures-level expected spread (0.01%). ICP spot has naturally wider spreads (~0.04%) because spot has less liquidity than futures perpetuals.

**Filter code** (`core/fee_aware_filter.py:138-161`):
```python
def should_reject_by_spread(self, actual_spread_pct):
    spread_multiplier = actual_spread_pct / self.expected_spread
    if spread_multiplier > 2.0:
        return True, f"Spread {actual_spread_pct:.4%} is {spread_multiplier:.1f}x expected {self.expected_spread:.4%}"
    return False, ""
```

### Fix

Relax the expected spread to 0.05% in runner scripts:

```python
# In run_paper_both.py and run_paper_multitrade.py:
if system.order_manager and hasattr(system.order_manager, 'fee_filter'):
    fee_filter = system.order_manager.fee_filter
    fee_filter.expected_spread = 0.0005       # 0.05% (was 0.01%)
    fee_filter.total_cost = (                 # Recalc total cost threshold
        fee_filter.entry_fee + fee_filter.exit_fee +
        fee_filter.expected_spread + fee_filter.min_profit
    )
```

**Only the runners are modified — no source files changed.**

---

## 6. DIAGNOSTICS ADDED

### In `data/exchange_connector.py`

**Message type counters** (printed every 500 WS messages):
```python
_msg_counters = {'depthUpdate': 0, 'aggTrade': 0, 'other': 0, 'total': 0}
# In _process_message():
if self._msg_counters['total'] % 500 == 0:
    logger.info(f"[WS MSGS] total={c['total']} depth={c['depthUpdate']} aggTrade={c['aggTrade']} other={c['other']}")
```

**Trade trace logging** (first 3 trades + every 500th):
```python
_trade_log_count = 0
# In _handle_trade():
self._trade_log_count += 1
if self._trade_log_count <= 3 or self._trade_log_count % 500 == 0:
    logger.info(f"[TRACE TRADE #{self._trade_log_count}] keys={list(data.keys())} p={data.get('p')} q={data.get('q')} m={data.get('m')}")
```

### In `main.py`

**Paper trading health counters:**
```python
self._paper_tick_count = 0
self._paper_total_trades_received = 0
self._paper_skipped_no_book = 0
self._paper_skipped_has_pos = 0
self._paper_skipped_cooldown = 0
self._paper_skipped_loss_streak = 0
self._paper_signals_evaluated = 0
```

**Health report** (every 1000 ticks):
```python
logger.info(
    f"[HEALTH] Ticks:{count} TradesRcv:{trades} Features:{fcount} "
    f"Regime:{regime} Mid:{mid:.4f} Pos:{'OPEN' if pos else 'FLAT'} "
    f"ClTrades:{closed} ConsecLoss:{losses} SigEval:{signals} "
    f"NoBook:{no_book} LossSkip:{loss_skip}"
)
```

---

## 7. RUNNER SCRIPTS COMPARISON

### `run_paper_both.py` (194 lines)

| Feature | Value |
|---|---|
| Max positions | 1 (single) |
| Spread threshold | 0.05% (relaxed) |
| Data stream | Spot (`use_futures=False`) |
| Fee structure | Futures (0.05% taker) |
| Position tracking | `system.paper_position` (single dict) |
| Exit handling | `_close_paper_trade()` |
| Status display | Single position info |
| Risk model | Same as original |

**How it works:**
1. Creates `OrderFlowSystem`
2. Calls `_init_components('paper', testnet=False, use_futures=False)` 
3. Relaxes fee filter spread to 0.05%
4. Calls `system.run_paper(STRATEGIES, use_futures=False)`
5. The `run_paper` method in `main.py` handles everything (single-position logic)

**Known issue:** Because `_init_components` is called in the runner AND in `run_paper()` (guarded by `if not self.exchange`), the first call pre-initializes the exchange. The second call inside `run_paper` skips `_init_components` but still creates `self.paper_feature_engine` and sets up callbacks.

### `run_paper_multitrade.py` (359 lines)

| Feature | Value |
|---|---|
| Max positions | **2 (concurrent)** |
| Spread threshold | 0.05% (relaxed) |
| Data stream | Spot (`use_futures=False`) |
| Fee structure | Futures (0.05% taker) |
| Position tracking | `system.paper_positions` (list of dicts) |
| Exit handling | `_close_multi_paper_trade(pos)` — takes a specific position |
| Status display | Lists all open positions |
| Method patching | `types.MethodType` to replace `system._on_paper_order_book` |

**How it works:**
1. Creates `OrderFlowSystem`
2. Initializes `system.paper_positions = []` (new attribute)
3. **Monkey-patches** `system._on_paper_order_book` with `_multi_on_paper_order_book` using `types.MethodType`
4. Also patches `system._close_multi_paper_trade`
5. Calls `_init_components`, relaxes fee filter, runs

**Key differences in `_multi_on_paper_order_book`:**
- Loops over ALL positions in `self.paper_positions` to update P&L, trailing stops, and check exits
- Signal evaluation gate: `if len(positions) >= MAX_CONCURRENT` instead of `if self.paper_position`
- Entry: `self.paper_positions.append(new_pos)` instead of `self.paper_position = {...}`
- Exit: `self._close_multi_paper_trade(state, ts, reason, pos)` removes the specific position from list

**Bug fixed during session:** Missing `from execution.risk_manager import RiskAction` — added after first run on Tokyo server failed with `NameError: name 'RiskAction' is not defined`.

---

## 8. EXCHANGE CONNECTOR DETAILS

### Connection Flow

```
connect() → rest_client (ccxt) → start_websocket(symbol)
  └─ _get_ws_urls(symbol) → ["wss://stream.binance.com:443/stream?streams=icpusdt@depth@100ms/icpusdt@aggTrade", ...]
  └─ websockets.connect(ws_url)
      └─ _fetch_rest_snapshot(symbol) — loads full depth (1000 levels)
      └─ _subscribe(ws, symbol) — Binance uses URL-based subscription (pass)
      └─ Async loop: recv → _process_message(message, symbol)
```

### Stream URLs (from `_get_ws_urls`)

```python
symbol_lower = symbol.replace('/', '').lower()  # "ICPUSDT" → "icpusdt"
streams = f"{symbol_lower}@depth@100ms/{symbol_lower}@aggTrade"
# Spot:  wss://stream.binance.com:443/stream?streams=icpusdt@depth@100ms/icpusdt@aggTrade
# Futures: wss://fstream.binance.com:443/stream?streams=icpusdt@depth@100ms/icpusdt@aggTrade
```

### Key Observation: ICPUSDT on Futures Has No Trades

- Tested 39 min on `fstream.binance.com`: 16,000 depth, 0 aggTrade
- Tested 5 min on `stream.binance.com`: 69 aggTrade (and rising)
- Conclusion: ICPUSDT perpetual futures lacks trade volume on Binance
- **Always use spot** for ICP trade data

### Order Book Syncing

```python
1. _fetch_rest_snapshot(symbol):
   - HTTP GET order book (1000 levels)
   - Store in _local_bids / _local_asks dicts
   - Set _last_update_id

2. _handle_depth_update(data):
   - Buffer events if snapshot not ready
   - Apply diff to local book
   - Check sequence gaps (lenient — tolerates gaps to avoid re-sync death spiral)

3. _build_sorted_book():
   - Sorted top 20 bids (desc) + top 20 asks (asc)
   - Emit via callback
```

---

## 9. FEE-AWARE FILTER (FULL LOGIC)

### `FeeAwareFilter` (`core/fee_aware_filter.py`)

```python
# Default config
FeeConfig:
    maker_fee_pct: float = 0.0002       # 0.02%
    taker_fee_pct: float = 0.0005       # 0.05%
    expected_spread_pct: float = 0.0001 # 0.01% (changed to 0.0005 = 0.05% in runners)
    min_profit_target_pct: float = 0.0002 # 0.02%

class FeeAwareFilter:
    def __init__(self, maker_fee_pct=0.0002, taker_fee_pct=0.0005, 
                 expected_spread_pct=0.0001, min_profit_target_pct=0.0002):
        self.entry_fee = maker_fee_pct
        self.exit_fee = taker_fee_pct
        self.expected_spread = expected_spread_pct
        self.min_profit = min_profit_target_pct
        self.total_cost = entry_fee + exit_fee + expected_spread + min_profit

    def should_ignore_signal(self, signal, predicted_move_pct, confidence=0.7):
        # Reject if predicted move < total_cost / confidence
        threshold = self.total_cost / confidence
        if predicted_move_pct < threshold:
            return True, "Predicted move too small"

    def should_reject_by_spread(self, actual_spread_pct):
        # Reject if actual spread > 2x expected_spread
        multiplier = actual_spread_pct / self.expected_spread
        if multiplier > 2.0:
            return True, f"Spread too wide: {multiplier:.1f}x expected"
        return False, ""
```

### `LiveFeeAwareFilter` (subclass, `core/fee_aware_filter.py:228-279`)

```python
class LiveFeeAwareFilter(FeeAwareFilter):
    def should_trade_with_spread(self, signal, predicted_move_pct, confidence,
                                  current_bid, current_ask, last_price):
        actual_spread_pct = (current_ask - current_bid) / last_price
        # 1. Spread check → should_reject_by_spread
        # 2. Fee coverage → should_ignore_signal
        # 3. Both pass → return True
```

### Instantiation in `order_manager.py:100-106`

```python
self.fee_filter = LiveFeeAwareFilter(
    maker_fee_pct=0.0002,        # 0.02%
    taker_fee_pct=0.0005,        # 0.05%
    expected_spread_pct=0.0001,  # 0.01% (overridden in runners to 0.05%)
    min_profit_target_pct=0.0002 # 0.02%
)
```

### Instantiation in `backtesting/engine.py:203-209`

```python
self.fee_filter = FeeAwareFilter(
    maker_fee_pct=0.0002,
    taker_fee_pct=0.0005,
    expected_spread_pct=0.0001,  # ← Changeme for backtest with relaxed spread
    min_profit_target_pct=0.0002
)
```

---

## 10. PAPER TRADING POSITION MANAGEMENT

### Single-Position (`run_paper_both.py` / `main.py`)

**Tracked as:** `self.paper_position` (single dict or None)

```python
self.paper_position = {
    'strategy': strat_name,         # "absorption" | "stacked_imbalance"
    'entry_time': ts,               # datetime
    'entry_price': entry_price,     # float
    'size': size,                   # float (units)
    'allocated': pos_value,         # float (capital allocated)
    'stop_loss': sig.stop_loss,     # float
    'take_profit': sig.take_profit, # float
    'trailing_activation': pct,     # float
    'trailing_active': False,       # bool
    'trailing_stop_price': 0.0,     # float
    'highest_price': entry_price,   # float
    'entry_fee': fee,               # float
    'unrealized_pnl': 0.0,          # float (added during mark-to-market)
}
```

**Signal evaluation gate:**
```python
# main.py line 632:
if self.paper_position:      # ← blocks if ANY position is open
    self._paper_skipped_has_pos += 1
    return
```

### Multi-Position (`run_paper_multitrade.py`)

**Tracked as:** `self.paper_positions` (list of dicts)

```python
self.paper_positions = []  # ← list of position dicts (same structure as above)

# In _multi_on_paper_order_book:
for pos in list(self.paper_positions):  # iterate ALL positions
    # Update P&L
    # Check trailing
    # Check exits

# Signal evaluation gate:
if len(self.positions) >= MAX_CONCURRENT:  # ← allow if < 2
    return
```

**Key change:** The `_close_multi_paper_trade(state, ts, reason, pos)` function takes a specific position object and removes it from the list via `self.paper_positions.remove(pos)`.

### Exit Conditions

Same for both versions:

```python
# Hard stop loss (always active)
if exit_price <= pos['stop_loss']:
    return 'stop_loss'

# Take profit (always active)
if exit_price >= pos['take_profit']:
    return 'take_profit'

# Flow-based exits (after 1200s = 20 min minimum hold):
# - book_pressure_collapse: net_pressure < -0.5 AND bid/ask depth ratio < 0.3
# (sweep exits disabled in paper trading)
```

### Position Sizing

```python
pos_value = self.paper_capital * signal.position_size  # e.g., 0.25 → 25% of capital
size = pos_value / entry_price
entry_fee = pos_value * fee_pct  # 0.05%
allocated = pos_value + entry_fee
```

---

## 11. BACKTEST RESULTS

### Dataset

- **Files:** `data/backtests/ICPUSDT_20260531_processed.parquet` + `ICPUSDT_20260601_processed.parquet`
- **Period:** 2026-05-31 16:36 → 2026-06-01 14:40 (0.9 days, 37,575 ticks)
- **10 depth levels** available per tick

### Original Backtest (`run_post_cooldown_both.py`)

Settings: `fee_pct=0.0005`, `slippage_pct=0.0003`, `expected_spread=0.0001` (0.01%), **single position**.

| Strategy | Trades | Win Rate | Return | DD | Profit Factor |
|---|---|---|---|---|---|
| Absorption | — | — | — | — | — |
| StackedImbalance | — | — | — | — | — |
| Combined | — | — | — | — | — |

*(Note: run_post_cooldown_both.py was not re-executed in this session. The results below are from the multi-trade experiment.)*

### Multi-Trade Backtest (`test_multi_trade_relaxed_spread.py`)

Settings: `fee_pct=0.0005`, `slippage_pct=0.0003`, `expected_spread=0.0005` (0.05%), **max 2 concurrent trades**.

**Key code in the test script:**
```python
# Monkey-patch FeeAwareFilter default BEFORE engine creation:
FeeAwareFilter.__init__.__defaults__ = (0.0002, 0.0005, 0.0005, 0.0002)
```

#### Absorption Strategy (multi, 15 trades)

| # | Time | PnL | Exit Reason |
|---|---|---|---|
| 1 | 05/31 17:31 | +0.422% | book_pressure_collapse |
| 2 | 05/31 17:31 | +0.460% | book_pressure_collapse |
| 3 | 05/31 20:04 | +0.578% | stop_loss |
| 4 | 05/31 20:04 | +0.578% | stop_loss |
| 5 | 05/31 21:05 | +1.546% | stop_loss |
| 6 | 05/31 21:05 | +1.470% | stop_loss |
| 7 | 05/31 21:31 | +1.145% | book_pressure_collapse |
| 8 | 05/31 21:31 | +1.108% | book_pressure_collapse |
| 9 | 05/31 21:45 | -0.869% | stop_loss |
| 10 | 05/31 22:03 | -0.870% | stop_loss |
| 11 | 05/31 22:15 | +0.478% | book_pressure_collapse |
| 12 | 05/31 23:03 | +0.073% | book_pressure_collapse |
| 13 | 05/31 23:03 | +0.036% | book_pressure_collapse |
| 14 | 05/31 23:21 | -0.906% | stop_loss |
| 15 | 05/31 23:21 | -0.870% | stop_loss |

**Summary: 73.3% win, +4.42% return ($4.42), 2.90% max DD, PF=2.246**

#### StackedImbalance Strategy (multi, 15 trades)

| # | Time | PnL | Exit Reason |
|---|---|---|---|
| 1 | 05/31 17:31 | +0.460% | book_pressure_collapse |
| 2 | 05/31 17:31 | +0.422% | book_pressure_collapse |
| 3 | 05/31 20:04 | +0.578% | stop_loss |
| 4 | 05/31 20:04 | +0.502% | stop_loss |
| 5 | 05/31 21:05 | +1.280% | stop_loss |
| 6 | 05/31 21:05 | +1.167% | stop_loss |
| 7 | 05/31 21:31 | +1.108% | book_pressure_collapse |
| 8 | 05/31 21:31 | +1.108% | book_pressure_collapse |
| 9 | 05/31 22:15 | +0.257% | book_pressure_collapse |
| 10 | 05/31 22:15 | +0.294% | book_pressure_collapse |
| 11 | 05/31 23:03 | +0.404% | book_pressure_collapse |
| 12 | 05/31 23:03 | +0.367% | book_pressure_collapse |
| 13 | 05/31 23:22 | -0.870% | stop_loss |
| 14 | 05/31 23:23 | -0.871% | stop_loss |
| 15 | 05/31 23:39 | -0.875% | stop_loss |

**Summary: 80.0% win, +5.43% return ($5.43), 2.59% max DD, PF=3.037**

#### Combined (30 trades, merged chronologically)

**76.7% win, +10.08% return ($110.08), 4.32% max DD, PF=2.583**

Observations:
- Trades enter in pairs at the same timestamp (multi-position working)
- Most exits are `book_pressure_collapse` (flow-based) or `stop_loss` (SL hit)
- 0 take-profit exits — TP was likely too far (+10%) compared to SL (-0.78%)
- Average win: +0.69%, average loss: -0.88%
- 3 losses at the end (23:21-23:39) triggered loss streak cooldown

---

## 12. LIVE PAPER TRADING RUNS

### Tokyo Server Run (`run_paper_both.py`)

```
16:18:24 | Book synchronized
16:18:40 | [TRACE TRADE #1] p=2.361 q=20.90 m=True
16:18:52 | [TRACE TRADE #2] p=2.361 q=15.08 m=True
16:18:54 | [TRACE TRADE #3] p=2.362 q=174.04 m=False
16:18:57 | [Stacked Imbalance] STRONG_BUY @ 2.3660 | Score=8.13
16:18:57 | PAPER ENTRY @ 2.3682 | SL: 2.2960 | TP: 3.3660 | Size: 0.9991
16:20:34 | [WS MSGS] total=500 depth=416 aggTrade=84 other=0
16:23:19 | [WS MSGS] total=1000 depth=891 aggTrade=109 other=0
16:26:19 | [WS MSGS] total=1500 depth=1332 aggTrade=168 other=0
16:28:16 | [WS MSGS] total=2000 depth=1760 aggTrade=240 other=0
16:30:56 | [WS MSGS] total=2500 depth=2225 aggTrade=275 other=0
16:33:02 | [WS MSGS] total=3000 depth=2694 aggTrade=306 other=0
16:35:27 | [WS MSGS] total=3500 depth=3148 aggTrade=352 other=0
16:38:11 | [WS MSGS] total=4000 depth=3542 aggTrade=458 other=0
16:40:59 | [WS MSGS] total=4500 depth=4003 aggTrade=497 other=0
16:41:08 | [TRACE TRADE #500] p=2.365 q=166.59 m=False
16:43:46 | [WS MSGS] total=5000 depth=4450 aggTrade=550 other=0
```

**Observation:** Trade opened at 16:18:57, never closed by 16:43:46 (~25 min).  
SL was at 2.2960 (-3%), price stayed in 2.361-2.365 range.

### Tokyo Server Run (`run_paper_multitrade.py` — after RiskAction fix)

```
16:58:09 | Book synchronized
16:58:16 | [TRACE TRADE #1] p=2.352 q=94.34 m=False
16:58:16 | [TRACE TRADE #2] p=2.352 q=27.98 m=True
16:58:23 | [TRACE TRADE #3] p=2.353 q=4.55 m=False
16:59:25 | [WS MSGS] total=7500 depth=6776 aggTrade=724 other=0
17:00:14 | [Stacked Imbalance] STRONG_BUY @ 2.3585 | Score=4.50
17:00:14 | ERROR: NameError: name 'RiskAction' is not defined  ← BUG
(Repeated every signal until script fixed)
```

**Observation:** Multi-trade version hit the `RiskAction` import bug. After adding the import, signals should execute normally.

---

## 13. CONFIGURATION VALUES

### `PaperSettings` (from `run_paper_both.py`)

```python
settings = PaperSettings(
    trading=TradingConfig(
        symbol="ICPUSDT",
        exchange=Exchange.BINANCE,
        max_position_size=1000.0,
        max_position_value_pct=0.25,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
        max_consecutive_losses=20,
        min_time_between_trades_sec=30,
        slippage_estimate_pct=0.0005,
        fee_pct=0.0005,
        tick_size=0.001,
        feature_windows=[15, 30, 60, 300, 600, 900],
    ),
)
```

### `BacktestEngine` defaults (from `run_post_cooldown_both.py`)

```python
BacktestEngine(
    initial_capital=100.0,
    fee_pct=0.0005,
    slippage_pct=0.0003,
    sl_extra_slippage_pct=0.0003,
    warmup_seconds=60.0,
    min_time_between_trades_sec=30.0,
    risk_limits=RiskLimits(
        max_position_size=10000.0,
        max_position_value_pct=0.25,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
        max_trades_per_day=50,
        max_trades_per_hour=10,
        min_time_between_trades_sec=30,
        max_consecutive_losses=20,
    ),
)
```

---

## 14. SIGNAL QUALITY LOG

From the multi-trade backtest run, the `FeatureEngine` processes compute **216 features**. Key regimes detected:
- **DISTRIBUTION** (selling pressure)
- **ACCUMULATION** (buying pressure)  
- **UNKNOWN** (insufficient data, especially early in the run)

**Signal scores observed:** 3.49 → 12.00  
**Highest signal: Absorption BUY @ 12.00** (Score=12.00 at 2.7365)  
**Most frequent: StackedImbalance signals with Score=4.50**

**Common patterns:**
- Absorption generates BUY and STRONG_BUY signals
- StackedImbalance also generates BUY and STRONG_BUY
- No SELL signals (long-only strategy)
- Win rate is high (73-80%) but AVG WIN < AVG LOSS means position sizing matters

---

## 15. THINKING CHAIN (SESSION LOG)

### Session Start — Paper Trading Not Working

**Observation:** `TradesRcv:0` after long run.  
**Action:** Added WS message counters + trade trace logging.  
**Result:** Confirmed aggTrade=0 on futures domain.

### Step 2 — Switch to Spot

**Decision:** Changed `use_futures=False` in runner.  
**Result:** Trades arrived immediately. But fee filter rejected every signal.

### Step 3 — Fix Fee Filter

**Decision:** Relaxed `expected_spread` from 0.01% to 0.05% in runner.  
**Result:** Signals pass fee filter, trade entered.

### Step 4 — Merge Branches

**Problem:** User's working branch (`post-cooldown-both-edits-for-paper-trade`) didn't have my changes.  
**Action:** Branch merge with worktree (locked log file issue on Windows).  
**Result:** All diagnostics + spot mode on `post-cooldown-both-edits-for-paper-trade`.

### Step 5 — Backtest Experiment

**Request:** "Re-backtest with 0.05% spread + allow 2 concurrent trades."  
**Action:** Created `test_multi_trade_relaxed_spread.py` — standalone script, no source changes.  
**Result:** 30 trades, +10.08% return, PF=2.58.

### Step 6 — Multi-Trade Paper Script

**Request:** "Create paper trading script applying that."  
**Action:** Created `run_paper_multitrade.py` — monkey-patches `_on_paper_order_book` for multi-position.  
**Bug:** Missing `RiskAction` import → `NameError` on first run.  
**Fix:** Added `from execution.risk_manager import RiskAction`.

---

## 16. KNOWN ISSUES & EDGE CASES

| Issue | Description | Status |
|---|---|---|
| **WS message counters reset on reconnect** | `_msg_counters` is a class var, resets on `ExchangeConnector` re-creation | Cosmetic |
| **Multi-trade position tracking tied to instance method** | `_close_multi_paper_trade` is patched via `types.MethodType` — fragile if `run_paper` re-creates components | Workaround works |
| **No HEALT lines in Tokyo logs** | The user's `post-cooldown-both-edits-for-paper-trade` branch may not have the diagnostic commits before merge | Verify branch state |
| **Backtest FeeAwareFilter default** | `backtesting/engine.py:207` hardcodes `expected_spread_pct=0.0001` — test script overrides via `__defaults__` tuple | Works |
| **FeatureEngine 216 features, Regime=UNKNOWN** | Without trade data (futures), regime classifier returns UNKNOWN. Fixed by spot mode | Resolved |
| **9-10 win rate with losses >0.87% avg loss** | The multi-trade backtest shows avg loss (-0.88%) > avg win (+0.69%). Risk: position sizing needs careful tuning | Monitoring |

---

## 17. FILE INVENTORY — CREATED/MODIFIED THIS SESSION

### New Files (standalone, no source modifications)

| File | Description | Safe to delete |
|---|---|---|
| `run_paper_multitrade.py` | Multi-position paper trading runner | No (keep) |
| `test_multi_trade_relaxed_spread.py` | One-time backtest (0.05% + multi-position) | Yes |
| `test_paper_debug.py` | Debug script for paper trading issue | Yes |

### Modified Files (source changes merged from `test-paper-fix-...`)

| File | Change |
|---|---|
| `data/exchange_connector.py` | Added `_msg_counters` (per-500 log), `_trade_log_count` (trace log), removed docstrings |
| `main.py` | Added health counters, `use_futures` param to `_init_components`/`run_paper`, skip double-init, trade size fallback to 'q' field |
| `run_paper_both.py` | Switched `use_futures=True→False`, added fee filter relaxation |

---

## 18. QUICK COMMANDS

```bash
# Paper trading (single position, relaxed spread):
python run_paper_both.py

# Paper trading (up to 2 concurrent positions):
python run_paper_multitrade.py

# Backtest (original — single position, 0.01% spread):
python run_post_cooldown_both.py

# Backtest (multi-position experiment):
python test_multi_trade_relaxed_spread.py

# Test connectivity:
python test_connection.py
```

---

## 19. RAW CODE SNIPPETS — CRITICAL FUNCTIONS

### ExchangeConnector._handle_trade (with diagnostics)

```python
_trade_log_count = 0

async def _handle_trade(self, data: Dict):
    if not isinstance(data, dict):
        return
    if 'p' not in data and 'price' not in data:
        return
    self._trade_log_count += 1
    if self._trade_log_count <= 3 or self._trade_log_count % 500 == 0:
        logger.info(
            f"[TRACE TRADE #{self._trade_log_count}] "
            f"keys={list(data.keys())} "
            f"p={data.get('p')} q={data.get('q')} m={data.get('m')}"
        )
    try:
        trade_ts_ms = data.get('T', None)
        trade_dt = datetime.fromtimestamp(trade_ts_ms / 1000.0, tz=timezone.utc) if trade_ts_ms else datetime.now(timezone.utc)
        price = float(data.get('p', data.get('price', 0)))
        size = float(data.get('q', data.get('size', 0)))
        if price <= 0 or size <= 0:
            return
        trade = {
            'price': price, 'size': size,
            'side': 'sell' if data.get('m', data.get('side')) else 'buy',
            'timestamp': trade_dt, 'exchange_timestamp_ms': trade_ts_ms,
        }
    except (ValueError, TypeError):
        return
    self.recent_trades.append(trade)
    if len(self.recent_trades) > 1000:
        self.recent_trades = self.recent_trades[-500:]
    if self.on_trade:
        await self.on_trade(trade)
```

### ExchangeConnector._process_message (with WS MSGS)

```python
_msg_counters = {'depthUpdate': 0, 'aggTrade': 0, 'other': 0, 'total': 0}

async def _process_message(self, message: str, symbol: str):
    try:
        data = json.loads(message)
        if 'stream' in data and 'data' in data:
            data = data['data']
        if self.config.exchange_id == "binance":
            if 'e' in data:
                self._msg_counters['total'] += 1
                if data['e'] == 'depthUpdate':
                    self._msg_counters['depthUpdate'] += 1
                    await self._handle_depth_update(data, symbol)
                elif data['e'] == 'aggTrade':
                    self._msg_counters['aggTrade'] += 1
                    await self._handle_trade(data)
                else:
                    self._msg_counters['other'] += 1
            else:
                self._msg_counters['other'] += 1
        if self._msg_counters['total'] % 500 == 0:
            c = self._msg_counters
            logger.info(f"[WS MSGS] total={c['total']} depth={c['depthUpdate']} aggTrade={c['aggTrade']} other={c['other']}")
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON message: {message[:100]}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")
```

### Fee Filter Relaxation Pattern (used in both runners)

```python
if system.order_manager and hasattr(system.order_manager, 'fee_filter'):
    ff = system.order_manager.fee_filter
    ff.expected_spread = 0.0005  # 0.05%
    # IMPORTANT: total_cost includes expected_spread, must recalc:
    ff.total_cost = ff.entry_fee + ff.exit_fee + ff.expected_spread + ff.min_profit
```

### Multi-Position Monkey-Patch Pattern

```python
import types

# Define replacement function
async def _multi_on_paper_order_book(self, order_book):
    # ... multi-position logic ...

# Bind it to the instance
system._on_paper_order_book = types.MethodType(_multi_on_paper_order_book, system)

# system.run_paper() will use this patched method because:
# self.exchange.on_order_book_update = self._on_paper_order_book  (resolved at call time)
```

---

## 20. CONCLUSION

The project is now in a working state with two paper trading runners:
1. `run_paper_both.py` — single position, relaxed spread, proven working
2. `run_paper_multitrade.py` — up to 2 concurrent positions, spread relaxed, needs RiskAction import fix on target machine

Both use spot data stream (for reliable ICP trade data) with futures-equivalent fee calculations. The backtest experiment shows +10% return over 0.9 days with 30 trades at 76.7% win rate.

**Next session start point:**
- Verify `run_paper_multitrade.py` works on Tokyo server after the `RiskAction` import fix
- Compare live paper trading vs backtest results
- Tune parameters further if needed
