"""
Full backtest: MIN_HOLD=1200s, TP=0.50%, SL=0.35%
All 4 active strategies on full dataset.
"""

import sys, time
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from loguru import logger
logger.remove()

df = pd.read_parquet(project_root / "data" / "backtests" / "XRPUSDT_market_data_20260328.parquet")
print(f"Full data: {len(df):,} rows")
print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import (
    create_absorption_strategy, create_delta_divergence_strategy,
    create_stacked_imbalance_strategy, create_value_area_strategy,
)

strategies = {
    "Absorption": create_absorption_strategy(),
    "DeltaDivergence": create_delta_divergence_strategy(),
    "StackedImbalance": create_stacked_imbalance_strategy(),
    "ValueAreaMR": create_value_area_strategy(),
}

for name, s in strategies.items():
    s.sl_mult_high_vol = 3.5
    s.sl_mult_low_vol = 3.5
    s.sl_mult_trending = 3.5
    s.tp_mult_high_vol = 5.0
    s.tp_mult_low_vol = 5.0
    s.tp_mult_trending = 5.0

risk_limits = RiskLimits(
    max_position_size=10000.0, max_position_value_pct=0.25,
    max_daily_loss_pct=0.02, max_drawdown_pct=0.10,
    max_trades_per_day=50, max_trades_per_hour=10,
    min_time_between_trades_sec=30, max_consecutive_losses=20,
)

all_results = {}
all_closed = []

SEP = "=" * 70
DASH = "-" * 70

for sname, strat in strategies.items():
    print(f"\n{SEP}")
    print(f"  {sname}")
    print(SEP)

    eng = BacktestEngine(
        initial_capital=100000.0, fee_pct=0.0005, slippage_pct=0.0003,
        sl_extra_slippage_pct=0.0003, warmup_seconds=60.0,
        min_time_between_trades_sec=30.0,
        risk_limits=risk_limits,
    )

    t0 = time.time()
    metrics = eng.run(df, strat)
    elapsed = time.time() - t0

    print(f"  Time: {elapsed:.0f}s | Trades: {metrics.total_trades}")

    if metrics.total_trades > 0:
        wins = sum(1 for t in eng.closed_trades if t.pnl > 0)
        losses = metrics.total_trades - wins
        wr = wins / metrics.total_trades * 100
        pnls = [t.pnl_pct * 100 for t in eng.closed_trades]
        avg_pnl = float(np.mean(pnls))
        durations = [t.duration_seconds / 60 for t in eng.closed_trades]
        avg_hold = float(np.mean(durations))
        win_pnls = [t.pnl_pct * 100 for t in eng.closed_trades if t.pnl > 0]
        loss_pnls = [t.pnl_pct * 100 for t in eng.closed_trades if t.pnl <= 0]
        avg_win = float(np.mean(win_pnls)) if win_pnls else 0
        avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0

        reasons = {}
        for t in eng.closed_trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        print(f"  W/L: {wins}/{losses} | WR: {wr:.1f}% | EV: {avg_pnl:+.4f}%")
        print(f"  AvgWin: {avg_win:.3f}% | AvgLoss: {avg_loss:.3f}% | Hold: {avg_hold:.1f}min")
        print(f"  Return: {metrics.total_return_pct*100:.4f}% | Sharpe: {metrics.sharpe_ratio:.3f} | PF: {metrics.profit_factor:.3f}")
        print(f"  MaxDD: {metrics.max_drawdown_pct*100:.2f}% | Trades/Day: {metrics.trades_per_day:.2f}")
        print(f"  RiskRej: {metrics.signals_rejected_by_risk} | FeeRej: {metrics.signals_rejected_by_fee_filter}")
        print(f"  Exits: {reasons}")

        for t in eng.closed_trades:
            all_closed.append({
                "strategy": sname,
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "entry_price": round(t.entry_price, 5),
                "exit_price": round(t.exit_price, 5),
                "pnl_pct": round(t.pnl_pct * 100, 4),
                "exit_reason": t.exit_reason,
                "duration_min": round(t.duration_seconds / 60, 1),
                "sl": round(t.stop_loss, 5) if t.stop_loss else None,
                "tp": round(t.take_profit, 5) if t.take_profit else None,
            })
    else:
        print(f"  ZERO TRADES")

    all_results[sname] = {"metrics": metrics, "trades": list(eng.closed_trades)}

print(f"\n{SEP}")
print(f"  COMBINED SUMMARY")
print(SEP)
total_trades = sum(r["metrics"].total_trades for r in all_results.values())
print(f"  Total trades: {total_trades}")
if total_trades > 0 and all_closed:
    all_pnls = [t["pnl_pct"] for t in all_closed]
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    wr = n_wins / len(all_pnls) * 100
    ev = float(np.mean(all_pnls))
    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = float(np.mean(losses)) if losses else 0
    all_holds = [t["duration_min"] for t in all_closed]
    med_hold = float(np.median(all_holds))
    avg_hold = float(np.mean(all_holds))

    reasons_all = {}
    for t in all_closed:
        reasons_all[t["exit_reason"]] = reasons_all.get(t["exit_reason"], 0) + 1

    print(f"  Combined WR: {wr:.1f}% ({n_wins}W/{n_losses}L)")
    print(f"  Combined EV: {ev:+.4f}%")
    print(f"  AvgWin: {avg_win:+.3f}% | AvgLoss: {avg_loss:+.3f}%")
    print(f"  AvgHold: {avg_hold:.1f} min | MedHold: {med_hold:.1f} min")
    print(f"  Exit breakdown: {reasons_all}")

    print(f"\n{DASH}")
    print(f"  PER-STRATEGY BREAKDOWN")
    print(DASH)
    header = f"  {'Strategy':>20} {'Trades':>7} {'WR%':>7} {'EV%':>9} {'AvgWin%':>9} {'AvgLoss%':>9} {'Hold(m)':>8} {'PF':>7} {'Sharpe':>7}"
    print(header)
    print(DASH)
    for sname in strategies:
        r = all_results[sname]
        m = r["metrics"]
        tlist = r["trades"]
        if m.total_trades > 0:
            win_p = [x.pnl_pct * 100 for x in tlist if x.pnl > 0]
            loss_p = [x.pnl_pct * 100 for x in tlist if x.pnl <= 0]
            h = float(np.mean([x.duration_seconds / 60 for x in tlist]))
            ev_s = float(np.mean([x.pnl_pct * 100 for x in tlist]))
            a_w = float(np.mean(win_p)) if win_p else 0
            a_l = float(np.mean(loss_p)) if loss_p else 0
            print(f"  {sname:>20} {m.total_trades:>7,d} {m.win_rate*100:>6.1f}% "
                  f"{ev_s:>+8.4f}% {a_w:>+8.3f}% {a_l:>+8.3f}% {h:>7.1f} "
                  f"{m.profit_factor:>6.3f} {m.sharpe_ratio:>6.3f}")
        else:
            print(f"  {sname:>20}       0      N/A       N/A       N/A       N/A      N/A      N/A       N/A")

    print(f"\n{DASH}")
    print(f"  ALL CLOSED TRADES (sorted by entry)")
    print(DASH)
    all_closed_sorted = sorted(all_closed, key=lambda x: x["entry_time"])
    print(f"  {'#':>4} {'Strategy':>16} {'EntryTime':>20} {'Entry':>9} {'Exit':>9} {'Pnl%':>8} {'Hold':>6} {'Reason':>25}")
    print(DASH)
    for i, t in enumerate(all_closed_sorted):
        print(f"  {i+1:>4d} {t['strategy']:>16} {t['entry_time'][11:19]:>20} {t['entry_price']:>9.5f} {t['exit_price']:>9.5f} "
              f"{t['pnl_pct']:>+7.3f}% {t['duration_min']:>5.1f} {t['exit_reason']:>25}")

else:
    print(f"  No trades executed across any strategy")

print(f"\n  Engine MIN_HOLD = {eng.MIN_HOLD_BEFORE_FLOW_EXIT_SEC}s ({eng.MIN_HOLD_BEFORE_FLOW_EXIT_SEC/60:.0f} min)")
print(f"  TP target: 0.50% | SL target: 0.35%")
print(f"  Fee: 0.05%/side | Slippage: 0.03%/side")
print(f"\nDone.")
