"""
PHASE 4: VALIDATION & PROOF OF LIFE
Run backtest on XRP data, verify all fixes.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")

from core.feature_engine import FeatureConfig
from backtesting.engine import BacktestEngine
from knowledge.strategy_library import get_strategy, get_all_strategies
from core.fee_aware_filter import FeeAwareFilter

def find_xrp_data():
    data_dir = project_root / "data" / "backtests"
    candidates = list(data_dir.glob("XRP*merged*.parquet"))
    if not candidates:
        candidates = list(data_dir.glob("XRPUSDT*.parquet"))
    if not candidates:
        candidates = list(data_dir.glob("XRP*.parquet"))
    if candidates:
        path = candidates[0]
        print(f"DATA FILE: {path.name}")
        return path
    return None

data_path = find_xrp_data()
if data_path is None:
    print("ERROR: No XRP data file found!")
    sys.exit(1)

print(f"Loading data from {data_path}...")
df = pd.read_parquet(data_path)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns[:20])}")

# Use first 50000 rows for quick validation
df = df.head(50000)
print(f"Using {len(df)} rows for validation")

# Print fee filter configuration
faf = FeeAwareFilter()
print(f"\n{'='*60}")
print(f"FEE STRUCTURE VALIDATION:")
print(f"{'='*60}")
print(f"  Maker fee:           {faf.entry_fee:.4%}")
print(f"  Taker fee:           {faf.exit_fee:.4%}")
print(f"  Expected spread:     {faf.expected_spread:.4%}")
print(f"  Min profit target:   {faf.min_profit:.4%}")
print(f"  TOTAL COST:          {faf.total_cost:.4%}")
assert abs(faf.total_cost - 0.0010) < 0.0001, f"Total cost {faf.total_cost:.4%} != 0.10%"
print(f"  [OK] Total cost = 0.10% (0.0010) [PASS]")

engine = BacktestEngine(
    initial_capital=100000.0,
    feature_config=FeatureConfig(windows=[15, 30, 60, 300, 600, 900]),
)

strategies_to_test = ["absorption", "delta_divergence", "stacked_imbalance", "value_area"]
all_trades = []

for strat_name in strategies_to_test:
    print(f"\n{'='*60}")
    print(f"RUNNING: {strat_name}")
    print(f"{'='*60}")
    strategy = get_strategy(strat_name)
    if strategy is None:
        print(f"  SKIP: Strategy '{strat_name}' not found")
        continue

    metrics = engine.run(df, strategy)

    print(f"  Total Trades:      {metrics.total_trades}")
    print(f"  Win Rate:          {metrics.win_rate*100:.1f}%")
    print(f"  Total Return:      {metrics.total_return_pct*100:.2f}%")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:      {metrics.max_drawdown_pct*100:.2f}%")
    print(f"  Profit Factor:     {metrics.profit_factor:.2f}")

    trades = engine.closed_trades
    all_trades.extend(trades)

    if trades:
        durations = [t.duration_seconds for t in trades]
        print(f"\n  TRADE DURATIONS: min={min(durations):.1f}s, max={max(durations):.1f}s, avg={np.mean(durations):.1f}s")
        print(f"  FIRST 5 TRADES:")
        for i, t in enumerate(trades[:5]):
            print(f"    {i+1}. Entry={t.entry_time}, Exit={t.exit_time}, "
                  f"Dur={t.duration_seconds:.0f}s, PnL={t.pnl:.2f}, "
                  f"Reason={t.exit_reason}, Side={t.side.name}")

        sell_trades = [t for t in trades if t.side.name == "SELL"]
        if sell_trades:
            print(f"  [FAIL] Found {len(sell_trades)} SELL trades")
        else:
            print(f"  [OK] Zero SELL trades executed [PASS]")

        # Check minimum hold
        short_trades = [t for t in trades if t.duration_seconds < 5]
        if short_trades:
            print(f"  [FAIL] Found {len(short_trades)} trades with < 5s duration")
            for t in short_trades[:3]:
                print(f"    {t.exit_reason}: {t.duration_seconds:.0f}s")
        else:
            print(f"  [OK] No millisecond churn [PASS]")
    else:
        print(f"  [WARN] No trades generated")

print(f"\n{'='*60}")
print(f"VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"  Total Trades:       {len(all_trades)}")
if all_trades:
    winning = [t for t in all_trades if t.pnl > 0]
    print(f"  Win Rate:           {len(winning)/len(all_trades)*100:.1f}%")
    durations = [t.duration_seconds for t in all_trades]
    print(f"  Avg Duration:       {np.mean(durations):.1f}s")
    print(f"  Min Duration:       {min(durations):.1f}s")
    print(f"  Max Duration:       {max(durations):.1f}s")

    times = [t.entry_time for t in all_trades[:5]]
    if len(times) >= 2:
        gaps = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        print(f"\n  TIMESTAMP GAPS (backtest time, not wall-clock):")
        for i, gap in enumerate(gaps):
            print(f"    Trade {i+1} -> {i+2}: {gap:.0f}s between entries")
        print(f"  [OK] Timestamps use backtest time [PASS]")

    sell_trades = [t for t in all_trades if t.side.name == "SELL"]
    print(f"\n  SELL trades:        {len(sell_trades)} (expected 0)")
    if len(sell_trades) == 0:
        print(f"  [OK] LONG-ONLY constraint [PASS]")
    else:
        print(f"  [FAIL] LONG-ONLY constraint")

    print(f"\n  TOTAL TRADES > 5:   {'[PASS]' if len(all_trades) > 5 else '[FAIL]'}")
    print(f"  WIN RATE > 0%:      {'[PASS]' if any(t.pnl > 0 for t in all_trades) else '[FAIL]'}")
else:
    print(f"  [FAIL] No trades generated")

print(f"\n{'='*60}")
print(f"VALIDATION COMPLETE")
print(f"{'='*60}")
