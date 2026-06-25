"""Full baseline sweep - now fast (~25min). Writes incrementally.
Establishes the post-Step-1 baseline that all future changes are measured against."""
import os
os.environ["LOGURU_LEVEL"] = "CRITICAL"
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd, numpy as np
from loguru import logger; logger.remove()
from backtesting.engine import BacktestEngine
from execution.risk_manager import RiskLimits
from knowledge.strategy_library import (
    create_absorption_strategy, create_stacked_imbalance_strategy,
    create_delta_divergence_strategy, create_value_area_strategy,
    create_trend_following_strategy,
)
from core.data_structures import Side

DATA = Path(__file__).parent / "data" / "backtests"
OUT = Path(__file__).parent / "_baseline_results.txt"
STRATS = [
    ("absorption", create_absorption_strategy),
    ("stacked_imbalance", create_stacked_imbalance_strategy),
    ("delta_divergence", create_delta_divergence_strategy),
    ("value_area", create_value_area_strategy),
    ("trend_following", create_trend_following_strategy),
]
def make():
    return BacktestEngine(initial_capital=100.0, fee_pct=0.0005, slippage_pct=0.0003,
        sl_extra_slippage_pct=0.0003, warmup_seconds=60.0, min_time_between_trades_sec=30.0,
        risk_limits=RiskLimits(max_position_size=10000.0, max_position_value_pct=0.25,
            max_daily_loss_pct=0.02, max_drawdown_pct=0.10, max_trades_per_day=50,
            max_trades_per_hour=10, min_time_between_trades_sec=30, max_consecutive_losses=3))
def log(m):
    print(m, flush=True)
    with open(OUT, "a", encoding="utf-8") as f: f.write(m+"\n")

open(OUT, "w").close()
log("="*80); log("FULL BASELINE (post Step-1 speedup, glm5.2 branch)"); log("="*80)
DATES = [
    ("May 31 (up +3.81%)", "ICPUSDT_20260531_processed.parquet"),
    ("Jun 1  (up +2.70%)", "ICPUSDT_20260601_processed.parquet"),
    ("Jun 8  (dn -2.26%)", "ICPUSDT_20260608_processed.parquet"),
    ("Jun 9  (dn -1.05%)", "ICPUSDT_20260609_processed.parquet"),
]
rows = []
for dl, fn in DATES:
    df = pd.read_parquet(DATA / fn)
    log(f"\n### {dl}: {len(df)} ticks ###")
    day = []
    for name, fn2 in STRATS:
        t0=time.time(); e=make(); e.run(df, fn2()); el=time.time()-t0
        tr=e.closed_trades
        if tr:
            pnls=[t.pnl_pct for t in tr]; ret=(np.prod([1+p for p in pnls])-1)*100
            wins=sum(1 for p in pnls if p>0); lp=[p for p in pnls if p<=0]
            gp=sum(p for p in pnls if p>0); gl=abs(sum(lp)); pf=gp/gl if gl>0 else float('inf')
            L=sum(1 for t in tr if t.side==Side.BUY); S=sum(1 for t in tr if t.side==Side.SELL)
            wr=wins/len(pnls)*100
            log(f"  {name:20s} n={len(tr):3d} WR={wr:5.1f}% ret={ret:>+7.3f}% PF={pf:.2f} L/S={L}/{S} ({el:.0f}s)")
            rows.append((dl[:6],name,len(tr),wr,ret,pf,L,S))
        else:
            log(f"  {name:20s} n=  0  ({el:.0f}s)")
            rows.append((dl[:6],name,0,0,0,0,0,0))
        day.extend(tr)
    if day:
        pnls=[t.pnl_pct for t in day]; ret=(np.prod([1+p for p in pnls])-1)*100
        wins=sum(1 for p in pnls if p>0)
        L=sum(1 for t in day if t.side==Side.BUY); S=sum(1 for t in day if t.side==Side.SELL)
        log(f"  {'ALL':20s} n={len(pnls):3d} WR={wins/len(pnls)*100:5.1f}% ret={ret:>+7.3f}% L/S={L}/{S}")

log("\n"+"="*80); log("SUMMARY"); log("="*80)
log(f"{'Date':<8} {'Strategy':<20} {'n':>4} {'WR%':>6} {'ret%':>8} {'PF':>5} {'L':>3} {'S':>3}")
log("-"*62)
for r in rows:
    d,nm,n,wr,ret,pf,ln,s=r
    log(f"{d:<8} {nm:<20} {n:>4} {wr:>6.1f} {ret:>+8.3f} {pf:>5.2f} {ln:>3} {s:>3}")
log("\nDONE")
