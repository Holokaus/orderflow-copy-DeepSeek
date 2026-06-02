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
    level="DEBUG"
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
            feature_config=self._get_feature_config(),
            risk_limits=RiskLimits(
                max_position_size=self.settings.trading.max_position_size,  # 10000
                max_position_value_pct=self.settings.trading.max_position_value_pct,# FIX: pass config, not engine
            )
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
# SECTION 2: backtesting/engine.py
# ==============================================================================

"""
Backtesting Engine v5 — Integrated with Enhanced FeatureEngine
=================================================================
Fixes from v4:
1. Accepts FeatureConfig from caller (tick_size=0.5, regime params, etc.)
2. Removes duplicate footprint bar system (FeatureEngine is single source)
3. Removes duplicate regime detection (RegimeClassifier is single source)
4. WalkForwardValidator properly isolates trials for parallel execution
5. Min-trades filtering in OOS results

Performance optimisations from v4 preserved:
- Pre-extracted NumPy arrays
- Lightweight _Row container
- Pattern/volume profile throttling
- Equity curve downsampling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import copy
import math

from loguru import logger

from core.data_structures import (
    OrderFlowState, Signal, SignalType, Side, Trade,
    OrderBook, PriceLevel, FootprintBar, Regime
)
from core.feature_engine import FeatureEngine, FeatureConfig
from knowledge.strategy_library import StrategyDefinition
from execution.risk_manager import RiskManager, RiskLimits, RiskAction


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Track an open position"""
    entry_time: datetime
    entry_price: float
    allocated_capital: float
    size: float
    side: Side
    stop_loss: float
    take_profit: float
    signal: Signal
    strategy_name: str
    trailing_stop_activation_pct: float
    risk_action: str = ""

    entry_fee: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0


@dataclass
class ClosedTrade:
    """Record of a completed trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    side: Side
    pnl: float
    pnl_pct: float
    exit_reason: str
    duration_seconds: float
    signal_confidence: float
    risk_action: str = ""


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    avg_trade_duration_seconds: float = 0.0
    max_consecutive_losses: int = 0

    trades_per_day: float = 0.0
    calmar_ratio: float = 0.0

    signals_rejected_by_risk: int = 0
    signals_reduced_by_risk: int = 0
    halts_triggered: int = 0
    suspicious_pnl_rejected: int = 0


# ---------------------------------------------------------------------------
# Lightweight row container
# ---------------------------------------------------------------------------

class _Row:
    """Ultra-lightweight row accessor backed by raw numpy arrays."""
    __slots__ = (
        'timestamp', 'bid_price', 'ask_price', 'bid_size', 'ask_size',
        'trade_price', 'trade_size', 'trade_side',
        '_depth_bids_p', '_depth_bids_s', '_depth_asks_p', '_depth_asks_s',
        '_has_depth', '_n_depth',
    )

    def __init__(
        self,
        timestamp, bid_price, ask_price, bid_size, ask_size,
        trade_price, trade_size, trade_side,
        depth_bids_p=None, depth_bids_s=None,
        depth_asks_p=None, depth_asks_s=None,
    ):
        self.timestamp = timestamp
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.trade_price = trade_price
        self.trade_size = trade_size
        self.trade_side = trade_side
        self._depth_bids_p = depth_bids_p
        self._depth_bids_s = depth_bids_s
        self._depth_asks_p = depth_asks_p
        self._depth_asks_s = depth_asks_s
        self._has_depth = depth_bids_p is not None
        self._n_depth = len(depth_bids_p) if depth_bids_p is not None else 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven backtesting engine — integrated with enhanced FeatureEngine.
    """

    # Throttling knobs
    PATTERN_DETECTION_EVERY_N = 25
    EQUITY_SAMPLE_EVERY_N = 50
    VOLUME_PROFILE_EVERY_N = 100

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_pct: float = 0.0004,
        slippage_pct: float = 0.0005,
        sl_extra_slippage_pct: float = 0.0003,
        warmup_seconds: float = 60.0,
        min_time_between_trades_sec: float = 30.0,
        equity_floor_pct: float = 0.50,
        suspicious_pnl_pct: float = 0.10,
        risk_limits: Optional[RiskLimits] = None,
        feature_config: Optional[FeatureConfig] = None,  # NEW: accept config
    ):
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.sl_extra_slippage_pct = sl_extra_slippage_pct
        self.warmup_seconds = warmup_seconds
        self.min_time_between_trades_sec = min_time_between_trades_sec
        self.equity_floor_pct = equity_floor_pct
        self.equity_floor = initial_capital * equity_floor_pct
        self.suspicious_pnl_pct = suspicious_pnl_pct

        # FIX: Store feature config for creating FeatureEngine
        self._feature_config = feature_config or FeatureConfig()

        self.risk_limits = risk_limits or RiskLimits(
            max_position_size=10000.0, #1.0 for BTC change to 10000 for XRP
            max_position_value_pct=0.25,
            max_daily_loss_pct=0.02,
            max_weekly_loss_pct=0.05,
            max_drawdown_pct=0.10,
            max_trades_per_day=50,
            max_trades_per_hour=10,
            min_time_between_trades_sec=int(min_time_between_trades_sec),
            max_consecutive_losses=5,
        )

        # State (initialized in reset())
        self.capital: float = 0.0
        self.position: Optional[Position] = None
        self.closed_trades: List[ClosedTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.feature_engine: Optional[FeatureEngine] = None
        self.risk_manager: Optional[RiskManager] = None

        self._last_trade_close_time: Optional[datetime] = None
        self._last_signal_strategy: Optional[str] = None
        self._last_signal_time: Optional[datetime] = None
        self._current_day = None

        # Tracking
        self._signals_rejected = 0
        self._signals_reduced = 0
        self._halts = 0
        self._suspicious_rejected = 0

    def reset(self) -> None:
        """Reset state for new backtest."""
        self.capital = self.initial_capital
        self.position = None
        self.closed_trades = []
        self.equity_curve = []
        
        # FIX: Create FeatureEngine with proper config
        self.feature_engine = FeatureEngine(self._feature_config)
        
        self.risk_manager = RiskManager(
            limits=self.risk_limits,
            initial_equity=self.initial_capital,
        )
        self._last_trade_close_time = None
        self._last_signal_strategy = None
        self._last_signal_time = None
        self._current_day = None
        self._signals_rejected = 0
        self._signals_reduced = 0
        self._halts = 0
        self._suspicious_rejected = 0
        self._entry_tick_idx: int = -1 # ADD Initialize entry tick tracker

    # ------------------------------------------------------------------
    # DataFrame → numpy pre-extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(data: pd.DataFrame) -> List[_Row]:
        """Convert DataFrame to lightweight _Row objects."""
        
        # === FIX: Map _0 columns to base columns ===
        # (Because we dropped the redundant ones in the Parquet conversion)
        if 'bid_price' not in data.columns and 'bid_price_0' in data.columns:
            data = data.copy()
            data['bid_price'] = data['bid_price_0']
            data['ask_price'] = data['ask_price_0']
            data['bid_size'] = data['bid_size_0']
            data['ask_size'] = data['ask_size_0']
        # ============================================
        
        timestamps = pd.to_datetime(data['timestamp']).dt.to_pydatetime()

        bid_price = data['bid_price'].values if 'bid_price' in data.columns else np.zeros(len(data))
        ask_price = data['ask_price'].values if 'ask_price' in data.columns else np.zeros(len(data))
        bid_size = data['bid_size'].values if 'bid_size' in data.columns else np.zeros(len(data))
        ask_size = data['ask_size'].values if 'ask_size' in data.columns else np.zeros(len(data))

        trade_price = data['trade_price'].values if 'trade_price' in data.columns else np.full(len(data), np.nan)
        trade_size = data['trade_size'].values if 'trade_size' in data.columns else np.zeros(len(data))

        if 'trade_side' in data.columns:
            trade_side = data['trade_side'].astype(str).str.lower().values
        else:
            trade_side = np.full(len(data), 'buy')

        has_depth = 'bid_price_0' in data.columns
        depth_bids_p = depth_bids_s = depth_asks_p = depth_asks_s = None

        if has_depth:
            n_depth = 0
            while f'bid_price_{n_depth}' in data.columns:
                n_depth += 1

            zeros = np.zeros(len(data))
            depth_bids_p = np.column_stack([
                data.get(f'bid_price_{i}', zeros) for i in range(n_depth)
            ])
            depth_bids_s = np.column_stack([
                data.get(f'bid_size_{i}', zeros) for i in range(n_depth)
            ])
            depth_asks_p = np.column_stack([
                data.get(f'ask_price_{i}', zeros) for i in range(n_depth)
            ])
            depth_asks_s = np.column_stack([
                data.get(f'ask_size_{i}', zeros) for i in range(n_depth)
            ])

        n = len(data)
        rows: List[_Row] = []
        for i in range(n):
            ts = timestamps[i]
            rows.append(_Row(
                timestamp=ts,
                bid_price=float(bid_price[i]),
                ask_price=float(ask_price[i]),
                bid_size=float(bid_size[i]),
                ask_size=float(ask_size[i]),
                trade_price=float(trade_price[i]),
                trade_size=float(trade_size[i]),
                trade_side=str(trade_side[i]),
                depth_bids_p=depth_bids_p[i] if has_depth else None,
                depth_bids_s=depth_bids_s[i] if has_depth else None,
                depth_asks_p=depth_asks_p[i] if has_depth else None,
                depth_asks_s=depth_asks_s[i] if has_depth else None,
            ))

        return rows

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        strategy: StrategyDefinition,
        params: Optional[Dict[str, Any]] = None,
    ) -> BacktestMetrics:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with columns: timestamp, bid_price, ask_price,
                  bid_size, ask_size, trade_price, trade_size, trade_side
            strategy: Strategy to test
            params: Override strategy parameters

        Returns:
            BacktestMetrics
        """
        self.reset()

        if params:
            strategy = self._apply_params(strategy, params)

        rows = self._preprocess(data)
        n_rows = len(rows)
        logger.debug(f"Running backtest with {n_rows} data points")
        
        # DEPTH CHECK - Verify data loading
        if len(rows) > 0:
            sample = rows[0]
            logger.info(f"DEPTH CHECK: Has depth={sample._has_depth}, Levels={sample._n_depth}")
            if not sample._has_depth:
                logger.error("WARNING: No depth data detected in backtest data! Order book will have only best bid/ask.")
            else:
                logger.info(f"✓ Depth data loaded successfully ({sample._n_depth} levels per row)")

        first_timestamp: Optional[datetime] = None
        state: Optional[OrderFlowState] = None
        timestamp = None

        pattern_every = self.PATTERN_DETECTION_EVERY_N
        equity_every = self.EQUITY_SAMPLE_EVERY_N
        vp_every = self.VOLUME_PROFILE_EVERY_N

        for tick_idx in range(n_rows):
            row = rows[tick_idx]
            timestamp = row.timestamp
            
            # DEBUG: First 5 iterations - verify order book construction
            if tick_idx < 5:
                logger.debug(f"DEBUG Row {tick_idx}: Has depth={row._has_depth}, Levels={row._n_depth}, "
                           f"Best bid={row.bid_price}, Best ask={row.ask_price}")

            if first_timestamp is None:
                first_timestamp = timestamp
                self.equity_curve.append((first_timestamp, self.initial_capital))

            # Daily reset
            self._handle_daily_reset(timestamp)

            # Equity floor check
            current_equity = self._calculate_equity_fast()
            if current_equity < self.equity_floor:
                logger.warning(f"Equity floor breached. Halting.")
                break

            # Build order book
            order_book = self._build_order_book_fast(row, timestamp)

            # Build trades
            trades = self._build_trades_fast(row, timestamp)

            # FIX: REMOVED _accumulate_footprint() — FeatureEngine handles this

            # Update feature engine (single source of truth)
            run_patterns = (tick_idx % pattern_every == 0)
            run_vp = (tick_idx % vp_every == 0)

            state = self.feature_engine.update(
                order_book, trades,
                detect_patterns=run_patterns,
                compute_volume_profile=run_vp,
            )

            # FIX: REMOVED state.regime override — FeatureEngine's classifier is used

            # Update open position
            if self.position:
                self._update_position(state)
                self._update_trailing_stop(state)

                if tick_idx > self._entry_tick_idx:  # Only gate the EXIT check
                    exit_reason = self._check_exit_conditions(state, timestamp)
                    if exit_reason:
                        self._close_position(state, timestamp, exit_reason, strategy)

            # Warmup gate
            elapsed = (timestamp - first_timestamp).total_seconds()
            if elapsed < self.warmup_seconds:
                if tick_idx % equity_every == 0:
                    self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                continue

            # Signal evaluation (only when flat)
            if not self.position:
                if not self._cooldown_passed(timestamp):
                    if tick_idx % equity_every == 0:
                        self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                    continue

                signal = strategy.evaluate(state)

                if signal and signal.is_actionable:
                    if self._is_duplicate_signal(signal, strategy, timestamp):
                        if tick_idx % equity_every == 0:
                            self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                        continue

                    risk_action, adjusted_signal, reason = self.risk_manager.check_signal(
                        signal, state.order_book.mid_price
                    )

                    if risk_action == RiskAction.HALT_TRADING:
                        self._halts += 1
                        continue
                    elif risk_action == RiskAction.REJECT:
                        self._signals_rejected += 1
                        continue
                    elif risk_action == RiskAction.REDUCE_SIZE:
                        self._signals_reduced += 1
                        self._open_position(adjusted_signal, state, timestamp, strategy, risk_action.name)
                        self._entry_tick_idx = tick_idx #ADD THIS: Initialize entry tick tracker
                    elif risk_action == RiskAction.ALLOW:
                        self._open_position(adjusted_signal or signal, state, timestamp, strategy, risk_action.name)
                        self._entry_tick_idx = tick_idx #ADD THIS: Initialize entry tick tracker

            # Equity curve (down-sampled)
            if tick_idx % equity_every == 0:
                self.equity_curve.append((timestamp, self._calculate_equity_fast()))

        # Close remaining position
        if self.position and state is not None and timestamp is not None:
            self._close_position(state, timestamp, "end_of_backtest", strategy)

        if timestamp is not None:
            self.equity_curve.append((timestamp, self._calculate_equity_fast()))

        return self._calculate_metrics()

    # ------------------------------------------------------------------
    # Parameter application
    # ------------------------------------------------------------------

    def _apply_params(
        self,
        strategy: StrategyDefinition,
        params: Dict[str, Any],
    ) -> StrategyDefinition:
        """Apply parameter overrides to strategy"""
        strategy = copy.deepcopy(strategy)

        for key, value in params.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
            else:
                for condition in strategy.entry_conditions:
                    if condition.feature == key or f"{condition.feature}_threshold" == key:
                        condition.threshold = value

        return strategy

    # ------------------------------------------------------------------
    # Fast order book builder
    # ------------------------------------------------------------------

    def _build_order_book_fast(self, row: _Row, timestamp: datetime) -> OrderBook:
        """Build OrderBook from pre-extracted _Row."""
        best_bid = row.bid_price
        best_ask = row.ask_price
        best_bid_size = row.bid_size
        best_ask_size = row.ask_size

        if best_bid == 0 and best_ask == 0:
            tp = row.trade_price
            if not math.isnan(tp) and tp > 0:
                spread_est = tp * 0.0001
                best_bid = tp - spread_est / 2
                best_ask = tp + spread_est / 2
                best_bid_size = row.trade_size if row.trade_size > 0 else 1.0
                best_ask_size = best_bid_size

        n_levels = 20
        bids: List[PriceLevel] = []
        asks: List[PriceLevel] = []

        if row._has_depth:
            for i in range(min(n_levels, row._n_depth)):
                bp = float(row._depth_bids_p[i])
                bs = float(row._depth_bids_s[i])
                ap = float(row._depth_asks_p[i])
                a_s = float(row._depth_asks_s[i])
                if bp > 0 and bs > 0:
                    bids.append(PriceLevel(price=bp, size=bs, timestamp=timestamp))
                if ap > 0 and a_s > 0:
                    asks.append(PriceLevel(price=ap, size=a_s, timestamp=timestamp))
        else:
            # NO SYNTHETIC DEPTH - Data missing real depth levels
            logger.warning(f"No depth data available at {timestamp}. Book contains only best bid/ask.")
            if best_bid > 0:
                bids.append(PriceLevel(price=best_bid, size=best_bid_size, timestamp=timestamp))
            if best_ask > 0:
                asks.append(PriceLevel(price=best_ask, size=best_ask_size, timestamp=timestamp))

        bids.sort(key=lambda l: l.price, reverse=True)
        asks.sort(key=lambda l: l.price)

        return OrderBook(timestamp=timestamp, bids=bids, asks=asks)

    # ------------------------------------------------------------------
    # Fast trade builder
    # ------------------------------------------------------------------

    def _build_trades_fast(self, row: _Row, timestamp: datetime) -> List[Trade]:
        """Build Trade list from pre-extracted _Row."""
        if math.isnan(row.trade_price) or row.trade_price <= 0:
            return []
        if math.isnan(row.trade_size) or row.trade_size <= 0:
            return []

        side = Side.BUY if str(row.trade_side).lower().strip() == 'buy' else Side.SELL
        return [Trade(
            timestamp=timestamp,
            price=row.trade_price,
            size=row.trade_size,
            side=side,
        )]

    # NOTE: _accumulate_footprint() REMOVED — FeatureEngine handles this

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _update_position(self, state: OrderFlowState) -> None:
        """Update position P&L using best bid/ask."""
        if not self.position:
            return

        book = state.order_book
        mid = book.mid_price

        if self.position.side == Side.BUY:
            mark_price = book.best_bid.price if book.best_bid else mid
            self.position.unrealized_pnl = (mark_price - self.position.entry_price) * self.position.size
        else:
            mark_price = book.best_ask.price if book.best_ask else mid
            self.position.unrealized_pnl = (self.position.entry_price - mark_price) * self.position.size

        self.position.highest_price = max(self.position.highest_price, mid)
        if self.position.lowest_price <= 0:
            self.position.lowest_price = mid
        else:
            self.position.lowest_price = min(self.position.lowest_price, mid)

    def _update_trailing_stop(self, state: OrderFlowState) -> None:
        """Activate and ratchet trailing stop."""
        if not self.position:
            return

        book = state.order_book
        activation = self.position.trailing_stop_activation_pct
        if activation <= 0:
            return

        if self.position.side == Side.BUY:
            mark = book.best_bid.price if book.best_bid else book.mid_price
            move_pct = (mark - self.position.entry_price) / self.position.entry_price
            if move_pct >= activation:
                self.position.trailing_stop_active = True
                trail_distance = activation * 0.5
                new_stop = mark * (1 - trail_distance)
                self.position.trailing_stop_price = max(
                    self.position.trailing_stop_price, new_stop
                )
                if self.position.trailing_stop_price > self.position.stop_loss:
                    self.position.stop_loss = self.position.trailing_stop_price
        else:
            mark = book.best_ask.price if book.best_ask else book.mid_price
            move_pct = (self.position.entry_price - mark) / self.position.entry_price
            if move_pct >= activation:
                self.position.trailing_stop_active = True
                trail_distance = activation * 0.5
                new_stop = mark * (1 + trail_distance)
                if self.position.trailing_stop_price <= 0:
                    self.position.trailing_stop_price = new_stop
                else:
                    self.position.trailing_stop_price = min(
                        self.position.trailing_stop_price, new_stop
                    )
                if self.position.trailing_stop_price < self.position.stop_loss:
                    self.position.stop_loss = self.position.trailing_stop_price

    # ------------------------------------------------------------------
    # Exit conditions
    # ------------------------------------------------------------------

    def _check_exit_conditions(
        self,
        state: OrderFlowState,
        timestamp: datetime,
    ) -> Optional[str]:
        """Check if position should be closed."""
        if not self.position:
            return None

        book = state.order_book
        features = state.features

        if self.position.side == Side.BUY:
            exit_check_price = book.best_bid.price if book.best_bid else book.mid_price
        else:
            exit_check_price = book.best_ask.price if book.best_ask else book.mid_price

        # 1) Hard stop loss
        if self.position.side == Side.BUY:
            if exit_check_price <= self.position.stop_loss:
                return "stop_loss"
        else:
            if exit_check_price >= self.position.stop_loss:
                return "stop_loss"

        # 2) Take profit
        if self.position.side == Side.BUY:
            if exit_check_price >= self.position.take_profit:
                return "take_profit"
        else:
            if exit_check_price <= self.position.take_profit:
                return "take_profit"

        # 3a) Absorption against
        if state.absorptions:
            latest_abs = state.absorptions[-1]
            if (self.position.side == Side.BUY and
                    latest_abs.absorbing_side == Side.SELL and
                    latest_abs.strength >= 0.6):
                return "absorption_against"
            if (self.position.side == Side.SELL and
                    latest_abs.absorbing_side == Side.BUY and
                    latest_abs.strength >= 0.6):
                return "absorption_against"

        # 3b) Delta divergence against
        delta_div = features.get("delta_divergence_60s", 0)
        if delta_div == 1.0:
            delta_60 = features.get("delta_60s", 0)
            if self.position.side == Side.BUY and delta_60 < 0:
                return "delta_divergence_against"
            if self.position.side == Side.SELL and delta_60 > 0:
                return "delta_divergence_against"

        # 3c) Exhaustion
        if self.position.side == Side.BUY:
            if features.get("buying_exhaustion", 0) >= 1.0:
                return "buying_exhaustion"
        else:
            if features.get("selling_exhaustion", 0) >= 1.0:
                return "selling_exhaustion"

        # 3d) Sweep against
        if state.sweeps:
            latest_sweep = state.sweeps[-1]
            if (self.position.side == Side.BUY and
                    latest_sweep.direction == Side.SELL and
                    latest_sweep.reversal_strength < 0.4):
                return "sweep_against_long"
            if (self.position.side == Side.SELL and
                    latest_sweep.direction == Side.BUY and
                    latest_sweep.reversal_strength < 0.4):
                return "sweep_against_short"

        # 3e) Book pressure collapse
        net_pressure = features.get("net_pressure", 0)
        if self.position.side == Side.BUY and net_pressure < -0.5:
            bid_depth = features.get("bid_depth_10", 0)
            ask_depth = features.get("ask_depth_10", 0)
            if ask_depth > 0 and bid_depth / (ask_depth + 1e-9) < 0.3:
                return "book_pressure_collapse"
        if self.position.side == Side.SELL and net_pressure > 0.5:
            bid_depth = features.get("bid_depth_10", 0)
            ask_depth = features.get("ask_depth_10", 0)
            if bid_depth > 0 and ask_depth / (bid_depth + 1e-9) < 0.3:
                return "book_pressure_collapse"

        return None

    # ------------------------------------------------------------------
    # Open/Close position
    # ------------------------------------------------------------------

    def _open_position(
        self,
        signal: Signal,
        state: OrderFlowState,
        timestamp: datetime,
        strategy: StrategyDefinition,
        risk_action: str = "ALLOW",
    ) -> None:
        """Open a new position using best bid/ask fill model."""
        book = state.order_book
        best_bid = book.best_bid.price if book.best_bid else book.mid_price
        best_ask = book.best_ask.price if book.best_ask else book.mid_price

        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            entry_price = best_ask * (1 + self.slippage_pct)
            side = Side.BUY
        else:
            entry_price = best_bid * (1 - self.slippage_pct)
            side = Side.SELL

        position_value = self.capital * signal.position_size
        size = position_value / entry_price

        entry_fee = position_value * self.fee_pct
        allocated = position_value + entry_fee

        if allocated > self.capital:
            allocated = self.capital * 0.95
            position_value = allocated / (1 + self.fee_pct)
            size = position_value / entry_price
            entry_fee = position_value * self.fee_pct

        self.capital -= allocated

        self.position = Position(
            entry_time=timestamp,
            entry_price=entry_price,
            allocated_capital=allocated,
            size=size,
            side=side,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal=signal,
            strategy_name=strategy.name,
            trailing_stop_activation_pct=strategy.trailing_stop_activation_pct,
            risk_action=risk_action,
            highest_price=entry_price,
            lowest_price=entry_price,
            entry_fee=entry_fee,
        )

        self.risk_manager.record_trade_opened(entry_price, size, side)
        self._last_signal_time = timestamp
        self._last_signal_strategy = strategy.name

    def _close_position(
        self,
        state: OrderFlowState,
        timestamp: datetime,
        reason: str,
        strategy: StrategyDefinition,
    ) -> None:
        """Close current position using best bid/ask fill model."""
        if not self.position:
            return

        book = state.order_book
        best_bid = book.best_bid.price if book.best_bid else book.mid_price
        best_ask = book.best_ask.price if book.best_ask else book.mid_price

        adverse_slip = (self.slippage_pct + self.sl_extra_slippage_pct 
                       if reason == "stop_loss" else self.slippage_pct)

        if self.position.side == Side.BUY:
            exit_price = best_bid * (1 - adverse_slip)
            gross_pnl = (exit_price - self.position.entry_price) * self.position.size
        else:
            exit_price = best_ask * (1 + adverse_slip)
            gross_pnl = (self.position.entry_price - exit_price) * self.position.size

        exit_value = exit_price * self.position.size
        exit_fee = exit_value * self.fee_pct
        pnl = gross_pnl - exit_fee

        notional = self.position.entry_price * self.position.size
        pnl_pct = pnl / notional if notional > 0 else 0.0

        # Suspicious PnL rejection
        if abs(pnl_pct) > self.suspicious_pnl_pct and reason != "end_of_backtest":
            self._suspicious_rejected += 1
            self.capital += self.position.allocated_capital
            self.position = None
            return

        self.capital += (self.position.allocated_capital - self.position.entry_fee) + pnl

        duration = (timestamp - self.position.entry_time).total_seconds()

        self.closed_trades.append(ClosedTrade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            size=self.position.size,
            side=self.position.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            duration_seconds=duration,
            signal_confidence=self.position.signal.confidence,
            risk_action=self.position.risk_action,
        ))

        self.risk_manager.record_trade_closed(pnl)
        self._last_trade_close_time = timestamp
        self.position = None

    # ------------------------------------------------------------------
    # Signal de-duplication / cooldown
    # ------------------------------------------------------------------

    def _is_duplicate_signal(
        self,
        signal: Signal,
        strategy: StrategyDefinition,
        timestamp: datetime,
    ) -> bool:
        if self._last_signal_time is None:
            return False
        if self._last_signal_strategy != strategy.name:
            return False
        elapsed = (timestamp - self._last_signal_time).total_seconds()
        return elapsed < 5.0

    def _cooldown_passed(self, timestamp: datetime) -> bool:
        if self._last_trade_close_time is None:
            return True
        elapsed = (timestamp - self._last_trade_close_time).total_seconds()
        return elapsed >= self.min_time_between_trades_sec

    # ------------------------------------------------------------------
    # Daily resets
    # ------------------------------------------------------------------

    def _handle_daily_reset(self, timestamp: datetime) -> None:
        day = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        if self._current_day is None:
            self._current_day = day
            return

        if day != self._current_day:
            self._current_day = day
            self.risk_manager.state.daily_pnl = 0.0
            self.risk_manager.state.trades_today = 0
            self.risk_manager.state.trades_this_hour = 0

            if (self.risk_manager.state.trading_halted and
                    "Daily" in self.risk_manager.state.halt_reason):
                self.risk_manager.resume_trading()
                logger.info("New trading day — daily halt lifted.")

    # ------------------------------------------------------------------
    # Equity helpers
    # ------------------------------------------------------------------

    def _calculate_equity_fast(self) -> float:
        equity = self.capital
        if self.position:
            equity += self.position.unrealized_pnl
            equity += (self.position.allocated_capital - self.position.entry_fee)
        return equity

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()

        if not self.closed_trades:
            return metrics

        metrics.total_trades = len(self.closed_trades)

        final_equity = self._calculate_equity_fast()
        metrics.total_return = final_equity - self.initial_capital
        metrics.total_return_pct = metrics.total_return / self.initial_capital

        winning = [t for t in self.closed_trades if t.pnl > 0]
        losing = [t for t in self.closed_trades if t.pnl <= 0]

        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = len(winning) / len(self.closed_trades)

        metrics.avg_win = float(np.mean([t.pnl for t in winning])) if winning else 0
        metrics.avg_loss = float(np.mean([t.pnl for t in losing])) if losing else 0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        pnls = [t.pnl for t in self.closed_trades]
        metrics.expectancy = float(np.mean(pnls))

        durations = [t.duration_seconds for t in self.closed_trades]
        metrics.avg_trade_duration_seconds = float(np.mean(durations))

        max_consec = 0
        current_consec = 0
        for trade in self.closed_trades:
            if trade.pnl <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        metrics.max_consecutive_losses = max_consec

        if len(self.equity_curve) > 1:
            timestamps_eq = [e[0] for e in self.equity_curve]
            equities = [e[1] for e in self.equity_curve]

            peak = equities[0]
            max_dd = 0.0
            for eq in equities:
                peak = max(peak, eq)
                dd = (peak - eq) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            metrics.max_drawdown_pct = max_dd
            metrics.max_drawdown = max_dd * self.initial_capital

            equity_series = pd.Series(
                equities,
                index=pd.DatetimeIndex(timestamps_eq),
            )
            daily_equity = equity_series.resample('D').last().dropna()

            if len(daily_equity) > 2:
                daily_returns = daily_equity.pct_change().dropna().values

                if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                    metrics.sharpe_ratio = float(
                        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                    )

                    downside = daily_returns[daily_returns < 0]
                    if len(downside) > 0:
                        downside_std = np.std(downside)
                        if downside_std > 0:
                            metrics.sortino_ratio = float(
                                np.mean(daily_returns) / downside_std * np.sqrt(252)
                            )

            if metrics.max_drawdown_pct > 0:
                metrics.calmar_ratio = metrics.total_return_pct / metrics.max_drawdown_pct

            total_seconds = (timestamps_eq[-1] - timestamps_eq[0]).total_seconds()
            total_days = total_seconds / 86400.0
            if total_days > 0:
                metrics.trades_per_day = metrics.total_trades / total_days
                metrics.annualized_return = metrics.total_return_pct * (365.0 / total_days)

        metrics.signals_rejected_by_risk = self._signals_rejected
        metrics.signals_reduced_by_risk = self._signals_reduced
        metrics.halts_triggered = self._halts
        metrics.suspicious_pnl_rejected = self._suspicious_rejected

        return metrics


# ======================================================================
# Walk-Forward Validator — Fixed for Parallel Safety
# ======================================================================

class WalkForwardValidator:
    """
    Walk-forward validation with proper isolation for parallel optimization.
    """

    def __init__(
        self,
        train_days: int = 60,
        test_days: int = 14,
        step_days: int = 7,
        min_oos_trades: int = 10,  # NEW: filter out useless folds
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.min_oos_trades = min_oos_trades

    def generate_splits(
        self,
        data: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits"""
        data = data.sort_values('timestamp').reset_index(drop=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        start_date = data['timestamp'].min()
        end_date = data['timestamp'].max()

        splits = []
        current_start = start_date

        while True:
            train_end = current_start + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)

            if test_end > end_date:
                break

            train_data = data[
                (data['timestamp'] >= current_start) &
                (data['timestamp'] < train_end)
            ]

            test_data = data[
                (data['timestamp'] >= train_end) &
                (data['timestamp'] < test_end)
            ]

            if len(train_data) > 100 and len(test_data) > 10:
                splits.append((train_data, test_data))

            current_start += timedelta(days=self.step_days)

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def validate(
        self,
        data: pd.DataFrame,
        strategy: StrategyDefinition,
        optimizer,
        n_trials_per_fold: int = 50,
        feature_config: Optional[FeatureConfig] = None,  # NEW: pass config
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation with proper trial isolation.

        FIX: Each trial creates its own BacktestEngine to avoid race conditions
        when n_jobs > 1 in optimization.
        """
        splits = self.generate_splits(data)

        results: Dict[str, Any] = {
            "folds": [],
            "oos_metrics": [],
            "params_stability": {},
            "skipped_folds": 0,  # NEW: track filtered folds
        }

        all_params: List[Dict] = []
        valid_oos_sharpes: List[float] = []  # NEW: for proper averaging

        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Processing fold {i+1}/{len(splits)}")

            # FIX: Create engine INSIDE backtest_fn for thread safety
            # When n_jobs > 1, multiple trials run in parallel
            def train_backtest(params, _train_data=train_data, _fc=feature_config):
                # Each trial gets its own engine instance
                engine = BacktestEngine(feature_config=_fc)
                metrics = engine.run(_train_data, strategy, params)
                return {
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "profit_factor": metrics.profit_factor,
                    "win_rate": metrics.win_rate,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "total_trades": metrics.total_trades,
                }

            # Optimize with n_jobs=1 for walk-forward (safer, clearer logs)
            # The inner optimization can still be fast with TPE
            opt_result = optimizer.optimize(
                train_backtest, 
                n_trials=n_trials_per_fold,
                n_jobs=1  # FIX: Force sequential for fold stability
            )
            best_params = opt_result.best_params
            all_params.append(best_params)

            # OOS test with fresh engine
            oos_engine = BacktestEngine(feature_config=feature_config)
            oos_metrics = oos_engine.run(test_data, strategy, best_params)

            # FIX: Filter out folds with insufficient trades
            if oos_metrics.total_trades < self.min_oos_trades:
                logger.warning(
                    f"Fold {i+1} skipped: OOS trades ({oos_metrics.total_trades}) "
                    f"< min ({self.min_oos_trades})"
                )
                results["skipped_folds"] += 1
                continue

            fold_result = {
                "fold": i,
                "train_score": opt_result.best_score,
                "oos_sharpe": oos_metrics.sharpe_ratio,
                "oos_return": oos_metrics.total_return_pct,
                "oos_win_rate": oos_metrics.win_rate,
                "oos_trades": oos_metrics.total_trades,
                "oos_max_dd": oos_metrics.max_drawdown_pct,
                "best_params": best_params,
            }

            results["folds"].append(fold_result)
            results["oos_metrics"].append(oos_metrics)
            valid_oos_sharpes.append(oos_metrics.sharpe_ratio)

            logger.info(
                f"Fold {i+1}: Train={opt_result.best_score:.2f}, "
                f"OOS Sharpe={oos_metrics.sharpe_ratio:.2f}, "
                f"OOS Trades={oos_metrics.total_trades}"
            )

        # Parameter stability
        if all_params:
            param_names = list(all_params[0].keys())
            for param in param_names:
                values = [p.get(param, 0) for p in all_params]
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                results["params_stability"][param] = {
                    "mean": mean_val,
                    "std": std_val,
                    "cv": std_val / (abs(mean_val) + 1e-9),
                }

        # FIX: Use only valid folds for summary
        if valid_oos_sharpes:
            results["summary"] = {
                "avg_oos_sharpe": float(np.mean(valid_oos_sharpes)),
                "std_oos_sharpe": float(np.std(valid_oos_sharpes)),
                "min_oos_sharpe": float(np.min(valid_oos_sharpes)),
                "max_oos_sharpe": float(np.max(valid_oos_sharpes)),
                "pct_profitable_folds": sum(1 for s in valid_oos_sharpes if s > 0) / len(valid_oos_sharpes),
                "total_folds": len(splits),
                "valid_folds": len(valid_oos_sharpes),
                "skipped_folds": results["skipped_folds"],
            }
        else:
            results["summary"] = {
                "avg_oos_sharpe": 0.0,
                "std_oos_sharpe": 0.0,
                "pct_profitable_folds": 0.0,
                "total_folds": len(splits),
                "valid_folds": 0,
                "skipped_folds": results["skipped_folds"],
                "error": "No valid folds (all had insufficient OOS trades)",
            }

        logger.info(f"Walk-Forward Summary:")
        logger.info(f"  Valid folds: {results['summary']['valid_folds']}/{results['summary']['total_folds']}")
        logger.info(f"  Avg OOS Sharpe: {results['summary']['avg_oos_sharpe']:.2f}")
        logger.info(f"  Profitable folds: {results['summary']['pct_profitable_folds']*100:.1f}%")

        return results
    
    
# ==============================================================================
# SECTION 3: core/feature_engine.py
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
        
        return Regime.RANGING


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
        
        # Build traded prices set (spoofing filter)
        traded_prices: set = set()
        for trade in self.trade_history[-100:]:
            bucket = round(trade.price / self.config.tick_size) * self.config.tick_size
            traded_prices.add(bucket)
        
        for price, data in refill_counts.items():
            if data["refill_count"] >= self.config.iceberg_refill_threshold:
                price_bucket = round(price / self.config.tick_size) * self.config.tick_size
                if price_bucket not in traded_prices:
                    continue  # Spoofing filter
                
                icebergs.append(IcebergOrder(
                    timestamp=self._current_timestamp or datetime.now(),
                    price=price,
                    visible_size=data["visible_size"],
                    estimated_hidden_size=data["total_refilled"],
                    refill_count=data["refill_count"],
                    side=data["side"],
                    confidence=min(data["refill_count"] / 10, 1.0)
                ))
        
        return icebergs
    
    def _track_refills(
        self,
        prev_levels: List[PriceLevel],
        curr_levels: List[PriceLevel],
        refill_counts: Dict[float, Dict],
        side: Side
    ) -> None:
        """Track order refills at price levels"""
        prev_by_price = {l.price: l.size for l in prev_levels[:20]}
        curr_by_price = {l.price: l.size for l in curr_levels[:20]}
        
        for price in prev_by_price:
            if price in curr_by_price:
                prev_size = prev_by_price[price]
                curr_size = curr_by_price[price]
                
                if prev_size < self.config.iceberg_size_threshold and curr_size > prev_size * 2:
                    if price not in refill_counts:
                        refill_counts[price] = {
                            "refill_count": 0,
                            "total_refilled": 0,
                            "visible_size": curr_size,
                            "side": side
                        }
                    
                    refill_counts[price]["refill_count"] += 1
                    refill_counts[price]["total_refilled"] += curr_size - prev_size
                    refill_counts[price]["visible_size"] = curr_size
    
    # ==================== PATTERN-TO-FEATURE BRIDGE (CRITICAL FIX) ====================
    
    def _compute_pattern_features(self, state: OrderFlowState) -> Dict[str, float]:
        """
        Bridge detected patterns into named features that strategies consume.
        
        WITHOUT this method, strategies reference features like
        'recent_absorption_strength' that never exist → no signals fire.
        """
        features: Dict[str, float] = {}
        now = self._current_timestamp or datetime.now()
        
        # ── Absorption features ──
        recent_abs = [
            a for a in state.absorptions
            if (now - a.timestamp).total_seconds() < 60
        ]
        if recent_abs:
            latest = recent_abs[-1]
            features["recent_absorption_strength"] = latest.strength
            features["recent_absorption_detected"] = 1.0
            features["recent_absorption_direction"] = (
                1.0 if latest.absorbing_side == Side.BUY else -1.0
            )
            features["max_absorption_strength_60s"] = max(
                a.strength for a in recent_abs
            )
            features["absorption_count_60s"] = float(len(recent_abs))
            features["avg_absorption_volume"] = sum(
                a.absorbed_volume for a in recent_abs
            ) / len(recent_abs)
        else:
            features["recent_absorption_strength"] = 0.0
            features["recent_absorption_detected"] = 0.0
            features["recent_absorption_direction"] = 0.0
            features["max_absorption_strength_60s"] = 0.0
            features["absorption_count_60s"] = 0.0
            features["avg_absorption_volume"] = 0.0
        
        # ── Sweep features ──
        recent_sw = [
            s for s in state.sweeps
            if (now - s.timestamp).total_seconds() < 120
        ]
        if recent_sw:
            latest = recent_sw[-1]
            features["recent_sweep_detected"] = 1.0
            features["recent_sweep_reversal_strength"] = latest.reversal_strength
            features["recent_sweep_direction"] = (
                1.0 if latest.direction == Side.BUY else -1.0
            )
            features["recent_sweep_volume"] = latest.volume_swept
            features["sweep_count_120s"] = float(len(recent_sw))
        else:
            features["recent_sweep_detected"] = 0.0
            features["recent_sweep_reversal_strength"] = 0.0
            features["recent_sweep_direction"] = 0.0
            features["recent_sweep_volume"] = 0.0
            features["sweep_count_120s"] = 0.0
        
        # ── Stacked imbalance features ──
        stacked = [i for i in state.imbalances if i.is_stacked]
        if stacked:
            features["stacked_imbalance_count"] = float(len(stacked))
            features["max_stack_size"] = float(max(i.stack_count for i in stacked))
            buy_imb = sum(1 for i in stacked if i.direction == Side.BUY)
            sell_imb = sum(1 for i in stacked if i.direction == Side.SELL)
            features["imbalance_direction_ratio"] = (buy_imb - sell_imb) / (
                buy_imb + sell_imb + 1e-9
            )
        else:
            features["stacked_imbalance_count"] = 0.0
            features["max_stack_size"] = 0.0
            features["imbalance_direction_ratio"] = 0.0
        
        if state.imbalances:
            features["avg_imbalance_ratio"] = sum(
                i.imbalance_ratio for i in state.imbalances
            ) / len(state.imbalances)
        else:
            features["avg_imbalance_ratio"] = 0.0
        
        # ── Iceberg features ──
        if state.icebergs:
            best = max(state.icebergs, key=lambda x: x.confidence)
            features["iceberg_detected"] = 1.0
            features["iceberg_confidence"] = best.confidence
            features["iceberg_direction"] = (
                1.0 if best.side == Side.BUY else -1.0
            )
            features["iceberg_estimated_size"] = best.estimated_hidden_size
            features["iceberg_count"] = float(len(state.icebergs))
        else:
            features["iceberg_detected"] = 0.0
            features["iceberg_confidence"] = 0.0
            features["iceberg_direction"] = 0.0
            features["iceberg_estimated_size"] = 0.0
            features["iceberg_count"] = 0.0
        
        return features
    
    # ==================== COMPOSITE FEATURES (ALL 8 STRATEGIES NEED) ====================
    
    def _compute_composite_features(self, base_features: Dict[str, float]) -> Dict[str, float]:
        """
        Derived features combining multiple base signals.
        Every feature referenced by any strategy is computed here.
        """
        features: Dict[str, float] = {}
        
        # 1. Volume acceleration: 60s vs 300s normalized
        short_vol = base_features.get("total_volume_60s", 0)
        long_vol = base_features.get("total_volume_300s", 0)
        features["volume_acceleration"] = (short_vol * 5) / (long_vol + 1e-9)
        
        # 2. Delta divergence: price and delta disagree
        price_dir_60 = np.sign(base_features.get("price_change_pct_60s", 0))
        delta_dir_60 = np.sign(base_features.get("delta_pct_60s", 0))
        features["delta_divergence_60s"] = float(
            price_dir_60 != delta_dir_60 and price_dir_60 != 0
        )
        
        # 3. CVD divergence: cumulative delta vs price
        cvd_sign = np.sign(base_features.get("cvd", 0))
        price_dir_300 = np.sign(base_features.get("price_change_pct_300s", 0))
        features["cvd_price_divergence"] = float(
            cvd_sign != price_dir_300 and price_dir_300 != 0
        )
        
        # 4. Exhaustion score (from footprint)
        buying_ex = base_features.get("buying_exhaustion", 0)
        selling_ex = base_features.get("selling_exhaustion", 0)
        features["exhaustion_score"] = max(buying_ex, selling_ex)
        
        # 5. Pressure confirmed: book pressure aligns with delta
        net_pressure = base_features.get("net_pressure", 0)
        delta_60 = base_features.get("delta_60s", 0)
        features["pressure_confirmed"] = float(
            np.sign(net_pressure) == np.sign(delta_60) and net_pressure != 0
        )
        
        # 6. Book-trade agreement: depth imbalance aligns with trade flow
        book_bias = base_features.get("depth_imbalance_10", 0)
        trade_bias = base_features.get("delta_pct_60s", 0)
        features["book_trade_agreement"] = float(
            np.sign(book_bias) == np.sign(trade_bias) and book_bias != 0
        )
        
        # 7. VA breakout potential: outside VA with rising volume
        in_va = base_features.get("in_value_area", 0)
        vol_accel = features.get("volume_acceleration", 1.0)
        features["va_breakout_potential"] = float(in_va == 0 and vol_accel > 1.5)
        
        # 8. Price vs VWAP (alias for vwap_deviation)
        features["price_vs_vwap_pct"] = base_features.get("vwap_deviation_60s", 0)
        
        return features
    
    # ==================== TEMPORAL FEATURE TRACKING ====================
    
    def _store_feature_snapshot(
        self, 
        timestamp: datetime, 
        features: Dict[str, float]
    ) -> None:
        """Store snapshot of key features for temporal look-back."""
        _TRACKED = [
            "recent_absorption_strength", "recent_absorption_direction",
            "depth_imbalance_10", "delta_60s", "delta_pct_60s",
            "footprint_imbalance_count", "volume_acceleration",
            "recent_sweep_detected", "recent_sweep_reversal_strength",
            "net_pressure", "spread_bps", "mid_price",
        ]
        snapshot = {k: features.get(k, 0.0) for k in _TRACKED}
        self._feature_snapshots.append((timestamp, snapshot))
        
        # Keep only last 60 seconds
        cutoff = timestamp - timedelta(seconds=60)
        while self._feature_snapshots and self._feature_snapshots[0][0] < cutoff:
            self._feature_snapshots.pop(0)
    
    def get_feature_at_lag(self, feature_name: str, seconds_ago: float) -> float:
        """Retrieve a feature value from N seconds ago."""
        if not self._feature_snapshots or not self._current_timestamp:
            return 0.0
        
        target = self._current_timestamp - timedelta(seconds=seconds_ago)
        best_snap = None
        best_diff = float("inf")
        
        for ts, snap in self._feature_snapshots:
            diff = abs((ts - target).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_snap = snap
        
        if best_snap and best_diff < seconds_ago * 0.5:
            return best_snap.get(feature_name, 0.0)
        return 0.0
    
    
# ==============================================================================
# SECTION 4: knowledge/strategy_library.py
# ==============================================================================
"""
Strategy Library
Complete definitions of order flow trading strategies.
Each strategy encapsulates domain knowledge in executable form.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
import numpy as np
from loguru import logger

from core.data_structures import (
    OrderFlowState, Signal, SignalType, Side, Regime
)


class StrategyCategory(Enum):
    ABSORPTION = auto()
    MOMENTUM = auto()
    REVERSAL = auto()
    BREAKOUT = auto()
    MEAN_REVERSION = auto()


@dataclass
class StrategyCondition:
    """Single condition for strategy entry/exit"""
    feature: str
    operator: str  # ">", "<", ">=", "<=", "==", "between"
    threshold: float
    threshold_high: Optional[float] = None  # For "between" operator
    weight: float = 1.0  # Importance weight
    required: bool = False  # Must be satisfied
    
    def evaluate(self, features: Dict[str, float]) -> tuple:
        """Returns (satisfied: bool, score: float)"""
        if self.feature not in features:
            return (False, 0.0)
        
        value = features[self.feature]
        
        if self.operator == ">":
            satisfied = value > self.threshold
        elif self.operator == "<":
            satisfied = value < self.threshold
        elif self.operator == ">=":
            satisfied = value >= self.threshold
        elif self.operator == "<=":
            satisfied = value <= self.threshold
        elif self.operator == "==":
            satisfied = abs(value - self.threshold) < 1e-9
        elif self.operator == "between":
            satisfied = self.threshold <= value <= self.threshold_high
        else:
            satisfied = False
        
        # Score is how far beyond threshold (normalized)
        if satisfied and self.operator in [">", ">="]:
            score = min((value - self.threshold) / (abs(self.threshold) + 1e-9), 2.0)
        elif satisfied and self.operator in ["<", "<="]:
            score = min((self.threshold - value) / (abs(self.threshold) + 1e-9), 2.0)
        elif satisfied:
            score = 1.0
        else:
            score = 0.0
        
        return (satisfied, score * self.weight)


@dataclass
class StrategyDefinition:
    """Complete strategy definition with all rules"""
    name: str
    category: StrategyCategory
    description: str
    
    # Entry conditions
    entry_conditions: List[StrategyCondition] = field(default_factory=list)
    min_conditions_satisfied: int = 3
    min_score_threshold: float = 2.0
    
    # Exit conditions
    stop_loss_atr_mult: float = 3.5 # Was 2.0 - need wider stops for XRP
    take_profit_atr_mult: float = 5.0
    max_holding_seconds: int = 3600
    trailing_stop_activation_pct: float = 0.005
    
    # Filters (conditions that must NOT be true)
    filters: List[StrategyCondition] = field(default_factory=list)
    
    # Regime filters
    allowed_regimes: List[Regime] = field(default_factory=lambda: list(Regime))
    
    # Position sizing
    base_position_pct: float = 0.1  # 10% of account
    max_position_pct: float = 0.25  # 25% of account
    scale_with_score: bool = True
    
    def evaluate(self, state: OrderFlowState) -> Optional[Signal]:
        """Evaluate strategy conditions and return signal if triggered
        
        FIX 3: Added regime-based SL/TP adaptation.
        """
        features = state.features
        
        # FIX 3: Check for LOW_LIQUIDITY regime first - return None
        if state.regime == Regime.LOW_LIQUIDITY:
            logger.debug(f"[{self.name}] REJECTED: LOW_LIQUIDITY regime detected")
            return None
        
        # Check filters first
        for filter_cond in self.filters:
            satisfied, _ = filter_cond.evaluate(features)
            if satisfied:
                logger.debug(f"[{self.name}] REJECTED by filter: {filter_cond.feature} {filter_cond.operator} {filter_cond.threshold}")
                return None  # Filtered out
        
        # Check regime
        if state.regime not in self.allowed_regimes:
            logger.debug(f"[{self.name}] REJECTED: Regime {state.regime} not in allowed {self.allowed_regimes}")
            return None
        
        # Evaluate entry conditions
        satisfied_count = 0
        total_score = 0.0
        required_satisfied = True
        reasons = []
        failed_conditions = []
        
        for condition in self.entry_conditions:
            satisfied, score = condition.evaluate(features)
            
            if satisfied:
                satisfied_count += 1
                total_score += score
                reasons.append(f"{condition.feature} {condition.operator} {condition.threshold}")
            else:
                if condition.required:
                    failed_conditions.append(f"REQUIRED: {condition.feature} {condition.operator} {condition.threshold}")
                else:
                    failed_conditions.append(f"{condition.feature} {condition.operator} {condition.threshold}")
            
            if condition.required and not satisfied:
                required_satisfied = False
        
        # Check if entry conditions met
        if not required_satisfied:
            logger.debug(f"[{self.name}] REJECTED: Required condition failed - {failed_conditions}")
            return None
        
        if satisfied_count < self.min_conditions_satisfied:
            logger.debug(f"[{self.name}] REJECTED: Only {satisfied_count}/{self.min_conditions_satisfied} conditions met. Failed: {failed_conditions}")
            return None
        
        if total_score < self.min_score_threshold:
            logger.debug(f"[{self.name}] REJECTED: Score {total_score:.2f} < threshold {self.min_score_threshold}")
            return None
        
        # Determine direction from dominant conditions
        direction = self._determine_direction(state, total_score)
        
        # ============================================
        # FIX: Return None if no clear direction
        # ============================================
        if direction == SignalType.NEUTRAL:
            logger.debug(f"[{self.name}] REJECTED: No clear direction (NEUTRAL). Delta: {state.features.get('delta_60s', 0)}, Imbalance: {state.features.get('depth_imbalance_10', 0)}, Pressure: {state.features.get('net_pressure', 0)}")
            return None
        
        # Calculate position size
        position_pct = self.base_position_pct
        if self.scale_with_score:
            score_mult = min(total_score / self.min_score_threshold, 2.0)
            position_pct = min(self.base_position_pct * score_mult, self.max_position_pct)
        
        # FIX 3: Store original multipliers for regime adaptation
        original_stop_loss_atr_mult = self.stop_loss_atr_mult
        original_take_profit_atr_mult = self.take_profit_atr_mult
        
        # FIX 3: Adapt SL/TP based on regime
        temp_stop_mult = self.stop_loss_atr_mult
        temp_tp_mult = self.take_profit_atr_mult
        
        if state.regime == Regime.HIGH_VOLATILITY:
            # Wider stops and targets in high volatility
            temp_stop_mult = self.stop_loss_atr_mult * 1.5
            temp_tp_mult = self.take_profit_atr_mult * 1.5
        elif state.regime == Regime.RANGING:
            # Tighter targets in ranging markets
            temp_tp_mult = self.take_profit_atr_mult * 0.7
        
        # Calculate stops using temporary regime-adapted multipliers
        atr = self._estimate_atr(state)
        mid_price = state.order_book.mid_price
        
        if direction == SignalType.BUY or direction == SignalType.STRONG_BUY:
            stop_loss = mid_price - atr * temp_stop_mult
            take_profit = mid_price + atr * temp_tp_mult
        else:
            stop_loss = mid_price + atr * temp_stop_mult
            take_profit = mid_price - atr * temp_tp_mult
        
        # ENFORCE MINIMUM $250 TP (0.37% at $67,815)
        min_tp_dist = mid_price * 0.0037
        actual_tp_dist = abs(take_profit - mid_price)
        if actual_tp_dist < min_tp_dist:
            if direction in (SignalType.BUY, SignalType.STRONG_BUY):
                take_profit = mid_price + min_tp_dist
            else:
                take_profit = mid_price - min_tp_dist
        
        # Confidence based on score
        confidence = min(total_score / (self.min_score_threshold * 2), 1.0)
        
        logger.info(f"[{self.name}] SIGNAL GENERATED: {direction.name} | Confidence: {confidence:.2%} | Score: {total_score:.2f} | Reasons: {', '.join(reasons)}")
        
        signal = Signal(
            timestamp=state.timestamp,
            signal_type=direction,
            confidence=confidence,
            entry_price=mid_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_pct,
            primary_reason=reasons[0] if reasons else self.name,
            supporting_factors=reasons[1:5],
            risk_reward_ratio=abs(take_profit - mid_price) / abs(mid_price - stop_loss)
        )
        
        # FIX 3: Restore original multipliers (no side effects)
        self.stop_loss_atr_mult = original_stop_loss_atr_mult
        self.take_profit_atr_mult = original_take_profit_atr_mult
        
        return signal
    
    def _determine_direction(self, state: OrderFlowState, score: float) -> SignalType:
        """Determine signal direction from state"""
        features = state.features
        
        # Use delta and imbalance to determine direction
        delta = features.get("delta_60s", 0)
        imbalance = features.get("depth_imbalance_10", 0)
        pressure = features.get("net_pressure", 0)
        
        bullish_score = (
            (1 if delta > 0 else 0) +
            (1 if imbalance > 0.1 else 0) + # Use 0.1 not 0.0 — filters noise Filters micro-fluctuation
            (1 if pressure > 0 else 0)
        )
        
        if bullish_score >= 2:
            return SignalType.STRONG_BUY if score > self.min_score_threshold * 1.5 else SignalType.BUY
        elif bullish_score <= 1:
            return SignalType.STRONG_SELL if score > self.min_score_threshold * 1.5 else SignalType.SELL
        else:
            return SignalType.NEUTRAL
    
    def _estimate_atr(self, state: OrderFlowState, default: float = 100.0) -> float:
        """Estimate ATR using True Range from recent trades.
        
        FIX 1: Calculate True Range (High-Low) from last 500-1000 trades
        with a 0.3% minimum floor.
        """
        mid_price = state.order_book.mid_price
        
        # Fallback if no mid price
        if mid_price <= 0:
            return default
        
        # Get recent trades - prefer state.recent_trades, fallback to empty list
        trades = state.recent_trades if state.recent_trades else []
        
        # Need at least some trades to calculate range
        if len(trades) < 2:
            return mid_price * 0.003  # 0.3% fallback
        
        # Use last 500-1000 trades
        sample_trades = trades[-1000:] if len(trades) > 1000 else trades[-500:] if len(trades) >= 500 else trades
        
        # Calculate True Range: max(price) - min(price)
        prices = [t.price for t in sample_trades]
        if not prices:
            return mid_price * 0.003
        
        price_range = max(prices) - min(prices)
        
        # Apply 0.3% minimum floor
        min_atr = mid_price * 0.005 #0.003 for BTC it's too tight for XRP make the stop inside the spread Change it to 0.5%
        atr = max(price_range, min_atr)
        
        return atr


# ==================== PRE-DEFINED STRATEGIES ====================

def create_absorption_strategy() -> StrategyDefinition:
    """
    Absorption Strategy
    
    Enters when large volume is absorbed without price movement.
    Indicates strong buying/selling interest that will likely move price.
    """
    return StrategyDefinition(
        name="Absorption",
        category=StrategyCategory.ABSORPTION,
        description="Trade after detecting absorption of aggressive orders",
        
        entry_conditions=[
            # Must have recent absorption detected
            StrategyCondition(
                feature="recent_absorption_strength",
                operator=">=",
                threshold=0.30,  # changed from 0.45 to 30% of aggressive volume absorbed
                weight=2.0,
                required=True
            ),
            # Volume must be elevated
            StrategyCondition(
                feature="volume_acceleration",
                operator=">",
                threshold=1.0, # Volume increasing from 1.2 to 1.0compared to recent average
                weight=1.5
            ),
            # Price should be stable
            StrategyCondition(
                feature="price_change_pct_60s",
                operator="<",
                threshold=0.001,
                weight=1.0
            ),
            # Delta should show the absorption direction
            StrategyCondition(
                feature="abs_delta_60s",
                operator=">",
                threshold=0,  # Will be parameterized
                weight=1.5
            ),
            # Book should support the direction
            StrategyCondition(
                feature="depth_imbalance_10",
                operator=">",  # Or < for shorts
                threshold=0.05,  # 5% imbalance in top 10 levels
                weight=1.0
            ),
            # Near POC is good
            StrategyCondition(
                feature="price_vs_poc_pct",
                operator="between",
                threshold=-0.005,
                threshold_high=0.005,
                weight=0.5
            ),
        ],
        
        filters=[
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION, Regime.TRENDING_UP, Regime.TRENDING_DOWN]
    )


def create_delta_divergence_strategy() -> StrategyDefinition:
    """
    Delta Divergence Strategy
    
    Enters when price and delta disagree, indicating potential reversal.
    """
    return StrategyDefinition(
        name="Delta Divergence",
        category=StrategyCategory.REVERSAL,
        description="Trade reversals when price diverges from cumulative delta",
        
        entry_conditions=[
            # Divergence detected
            StrategyCondition(
                feature="delta_divergence_60s",
                operator="==",
                threshold=1.0,
                weight=2.5,
                required=True
            ),
            # CVD divergence confirms
            StrategyCondition(
                feature="cvd_price_divergence",
                operator="==",
                threshold=1.0,
                weight=2.0
            ),
            # Exhaustion signals
            StrategyCondition(
                feature="exhaustion_score",
                operator=">",
                threshold=0.3,
                weight=1.5
            ),
            # Volume declining
            StrategyCondition(
                feature="volume_acceleration",
                operator="<",
                threshold=0.8,
                weight=1.0
            ),
            # At value area extreme
            StrategyCondition(
                feature="in_value_area",
                operator="==",
                threshold=0,  # Outside value area
                weight=1.0
            ),
            # FIX 5: Strengthen delta threshold - ensure divergence has volume behind it
            StrategyCondition(
                feature="abs_delta_60s",
                operator=">",
                threshold=8.0,
                weight=1.5,
                required=True  # Significant order flow imbalance required
            ),
        ],
        
        filters=[
            # Don't fight strong momentum
            StrategyCondition(
                feature="delta_pct_300s",
                operator=">",
                threshold=0.7
            ),
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.RANGING]
    )


def create_liquidity_sweep_strategy() -> StrategyDefinition:
    """
    Liquidity Sweep Strategy
    
    Enters after a stop hunt / liquidity sweep reverses.
    """
    return StrategyDefinition(
        name="Liquidity Sweep",
        category=StrategyCategory.REVERSAL,
        description="Fade liquidity sweeps after reversal confirmation",
        
        entry_conditions=[
            # Sweep detected
            StrategyCondition(
                feature="recent_sweep_detected",
                operator="==",
                threshold=1.0,
                weight=2.5,
                required=True
            ),
            # Strong reversal
            StrategyCondition(
                feature="recent_sweep_reversal_strength",
                operator=">",
                threshold=0.35,
                weight=2.0
            ),
            # Volume spike during sweep
            StrategyCondition(
                feature="volume_acceleration",
                operator=">",
                threshold=1.5,
                weight=1.5
            ),
            # Price back inside range
            StrategyCondition(
                feature="in_value_area",
                operator="==",
                threshold=1.0,
                weight=1.0
            ),
            # Book shifted in reversal direction
            StrategyCondition(
                feature="book_trade_agreement",
                operator="==",
                threshold=1.0,
                weight=2.5,      # High weight but NOT required
                required=False   # Let it influence the score, not veto the trade
            ),
        ],
        
        filters=[
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        # Note: LOW_LIQUIDITY regime handled by FIX 3 in evaluate()
        allowed_regimes=[Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION]
    )


def create_stacked_imbalance_strategy() -> StrategyDefinition:
    """
    Stacked Imbalance Strategy
    
    Enters in direction of stacked footprint imbalances.
    """
    return StrategyDefinition(
        name="Stacked Imbalance",
        category=StrategyCategory.MOMENTUM,
        description="Trade momentum when multiple price levels show same-side imbalance",
        
        entry_conditions=[
            # Stacked imbalance detected
            StrategyCondition(
                feature="footprint_imbalance_count",
                operator=">=",
                threshold=2,
                weight=2.0,
                required=True
            ),
            # Delta confirms direction
            StrategyCondition(
                feature="delta_pct_60s",
                operator=">",
                threshold=0.2,
                weight=1.5
            ),
            # Book supports
            StrategyCondition(
                feature="depth_imbalance_10",
                operator=">",
                threshold=0.15,
                weight=1.5
            ),
            # Pressure confirms
            StrategyCondition(
                feature="pressure_confirmed",
                operator="==",
                threshold=1.0,
                weight=1.0
            ),
            # Not overextended
            StrategyCondition(
                feature="price_vs_vwap_pct",
                operator="between",
                threshold=-0.003,
                threshold_high=0.003,
                weight=0.5
            ),
        ],
        
        filters=[
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.BREAKOUT, Regime.ACCUMULATION, Regime.DISTRIBUTION] #Add Regime.ACCUMULATION, Regime.DISTRIBUTION
    )


def create_value_area_strategy() -> StrategyDefinition:
    """
    Value Area Strategy
    
    Mean reversion trades at value area boundaries.
    """
    return StrategyDefinition(
        name="Value Area Mean Reversion",
        category=StrategyCategory.MEAN_REVERSION,
        description="Fade moves to value area boundaries",
        
        entry_conditions=[
            # At value area boundary
            StrategyCondition(
                feature="va_breakout_potential",
                operator="==",
                threshold=0,  # NOT breaking out = mean reversion setup
                weight=1.5
            ),
            # Price at VAH or VAL
            StrategyCondition(
                feature="price_vs_vah_pct",
                operator="between",
                threshold=-0.002,
                threshold_high=0.002,
                weight=2.0
            ),
            # Delta showing rejection
            StrategyCondition(
                feature="delta_divergence_60s",
                operator="==",
                threshold=1.0,
                weight=1.5
            ),
            # Volume declining
            StrategyCondition(
                feature="volume_acceleration",
                operator="<",
                threshold=1.0,
                weight=1.0
            ),
            # Book shifting
            StrategyCondition(
                feature="slope_asymmetry",
                operator=">",
                threshold=0,
                weight=0.5
            ),
        ],
        
        filters=[
            # Don't fade strong momentum
            StrategyCondition(
                feature="trade_intensity_60s",
                operator=">",
                threshold=80  # Too active
            ),
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.RANGING]
    )


# Strategy registry
STRATEGY_LIBRARY = {
    "absorption": create_absorption_strategy,
    "delta_divergence": create_delta_divergence_strategy,
    "liquidity_sweep": create_liquidity_sweep_strategy,
    "stacked_imbalance": create_stacked_imbalance_strategy,
    "value_area": create_value_area_strategy,
}


def get_all_strategies() -> Dict[str, StrategyDefinition]:
    """Get all strategy definitions"""
    return {name: factory() for name, factory in STRATEGY_LIBRARY.items()}


def get_strategy(name: str) -> Optional[StrategyDefinition]:
    """Get a specific strategy by name"""
    if name in STRATEGY_LIBRARY:
        return STRATEGY_LIBRARY[name]()
    return None

# ==============================================================================
# SECTION 5: knowledge/llm_advisor.py
# ==============================================================================

"""
LLM Advisor
Interface to external LLM APIs for knowledge extraction and guidance.
Supports 9 LLM providers: OpenAI, Gemini, Anthropic, OpenRouter, GitHub Models, xAI, Z.ai, Groq, Ollama.
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from functools import lru_cache
import os

from loguru import logger

# Import LLMProvider from config as single source of truth
from config.settings import LLMProvider, settings

# API clients - import conditionally
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Re-export LLMProvider for backward compatibility with main.py
__all__ = ['LLMAdvisor', 'LLMProvider', 'LLMResponse', 'ResponseCache', 
           'ABSORPTION_KNOWLEDGE', 'DELTA_DIVERGENCE_KNOWLEDGE', 
           'LIQUIDITY_SWEEP_KNOWLEDGE', 'STACKED_IMBALANCE_KNOWLEDGE']


@dataclass
class LLMResponse:
    """Structured response from LLM"""
    content: str
    parsed: Optional[Dict] = None
    provider: str = ""
    model: str = ""
    tokens_used: int = 0
    cached: bool = False


class ResponseCache:
    """Simple in-memory cache for LLM responses (includes provider+model in hash to prevent collisions)"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple] = {}  # hash -> (response, timestamp)
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def _hash_prompt(self, prompt: str, provider: str = "", model: str = "") -> str:
        """Hash prompt with provider+model to prevent cross-provider collisions"""
        combined = f"{provider}:{model}:{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, prompt: str, provider: str = "", model: str = "") -> Optional[str]:
        key = self._hash_prompt(prompt, provider, model)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, prompt: str, response: str, provider: str = "", model: str = ""):
        key = self._hash_prompt(prompt, provider, model)
        self.cache[key] = (response, datetime.now())


class LLMAdvisor:
    """
    Interface to LLM APIs for trading knowledge extraction.
    
    Supports 9 providers:
    - OPENAI: OpenAI GPT models
    - GEMINI: Google Gemini
    - ANTHROPIC: Anthropic Claude
    - OPENROUTER: OpenRouter (OpenAI-compatible)
    - GITHUB_MODELS: GitHub Models (OpenAI-compatible)
    - XAI: xAI Grok (OpenAI-compatible)
    - ZAI: Z.ai/Zhipu (OpenAI-compatible)
    - GROQ: Groq (OpenAI-compatible)
    - OLLAMA: Local Ollama (OpenAI-compatible, no API key)
    
    Uses LLMs for:
    1. Feature engineering guidance
    2. Parameter range suggestions
    3. Strategy rule extraction
    4. Pattern interpretation
    5. Weak labeling of setups
    """
    
    # Provider configuration: compatible providers mapping
    _PROVIDER_CONFIG = {
        # OpenAI-compatible providers
        LLMProvider.OPENAI: {
            "type": "openai",
            "base_url": None,
            "default_model": "gpt-4o",
        },
        LLMProvider.OPENROUTER: {
            "type": "openai",
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": "anthropic/claude-3.5-sonnet",
        },
        LLMProvider.GITHUB_MODELS: {
            "type": "openai",
            "base_url": "https://models.inference.ai.azure.com",
            "default_model": "gpt-4o",
        },
        LLMProvider.XAI: {
            "type": "openai",
            "base_url": "https://api.x.ai/v1",
            "default_model": "grok-2-latest",
        },
        LLMProvider.ZAI: {
            "type": "openai",
            "base_url": "https://api.z.ai/api/paas/v4",
            "default_model": "glm-5",
        },
        LLMProvider.GROQ: {
            "type": "openai",
            "base_url": "https://api.groq.com/openai/v1",
            "default_model": "llama-3.3-70b-versatile",
        },
        LLMProvider.OLLAMA: {
            "type": "openai",
            "base_url": "http://localhost:11434/v1",
            "default_model": "llama3",
        },
        # Native SDK providers
        LLMProvider.GEMINI: {
            "type": "gemini",
            "default_model": "gemini-2.0-flash",
        },
        LLMProvider.ANTHROPIC: {
            "type": "anthropic",
            "default_model": "claude-sonnet-4-20250514",
        },
    }
    
    def __init__(
        self,
        provider: Union[LLMProvider, str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_responses: bool = True,
        base_url: Optional[str] = None,
    ):
        """
        Initialize LLM Advisor with specified provider.
        
        Args:
            provider: LLMProvider enum or string value
            model: Model name (optional, uses provider default if None)
            api_key: API key (optional, reads from settings/env vars if None)
            cache_responses: Enable response caching
            base_url: Custom base URL for OpenAI-compatible providers
        """
        # Handle provider input
        if provider is None:
            provider = settings.llm.provider
        elif isinstance(provider, str):
            provider = LLMProvider(provider)
        
        self.provider = provider
        self.cache = ResponseCache(settings.llm.cache_ttl_seconds) if cache_responses else None
        
        # Get provider config
        if provider not in self._PROVIDER_CONFIG:
            raise ValueError(f"Unknown provider: {provider}")
        
        config = self._PROVIDER_CONFIG[provider]
        self.model = model or settings.llm.model or config["default_model"]
        
        # Resolve API key with priority: arg > settings > env var
        resolved_key = self._resolve_api_key(provider, api_key)
        
        # Initialize client based on provider type
        if config["type"] == "openai":
            self._init_openai_compatible(provider, config, resolved_key, base_url)
        elif config["type"] == "gemini":
            self._init_gemini(resolved_key)
        elif config["type"] == "anthropic":
            self._init_anthropic(resolved_key)
    
    @classmethod
    def create_with_failover(cls, llm_config: 'LLMConfig') -> Optional['LLMAdvisor']:
        """
        Factory method that attempts to initialize LLM advisor with automatic failover.
        
        Tries providers in order from llm_config.get_failover_chain() until one succeeds.
        Performs a test API call to verify each provider works before returning.
        
        Args:
            llm_config: LLMConfig instance with failover configuration
            
        Returns:
            Initialized LLMAdvisor instance, or None if all providers fail
            
        Example:
            advisor = LLMAdvisor.create_with_failover(settings.llm)
            if advisor is None:
                logger.warning("No LLM available, using fallback constants")
        """
        if not llm_config.enable_failover:
            # Simple initialization without failover
            try:
                api_key = llm_config.get_api_key(llm_config.provider)
                base_url = llm_config.get_base_url(llm_config.provider)
                return cls(
                    provider=llm_config.provider,
                    model=llm_config.model,
                    api_key=api_key,
                    cache_responses=llm_config.cache_responses,
                    base_url=base_url
                )
            except Exception as e:
                logger.error(f"LLM initialization failed: {e}")
                return None
        
        # Failover mode: try each provider in chain
        chain = llm_config.get_failover_chain()
        
        for provider, api_key, base_url in chain:
            try:
                logger.info(f"Attempting LLM provider: {provider.value}")
                
                advisor = cls(
                    provider=provider,
                    model=llm_config.model,
                    api_key=api_key,
                    cache_responses=llm_config.cache_responses,
                    base_url=base_url
                )
                
                # Verify with test call
                test_response = advisor._call_llm("Test connection")
                
                if test_response and test_response.content:
                    logger.info(f"LLM initialized successfully: {provider.value}")
                    return advisor
                else:
                    raise ValueError("Empty response from test call")
                    
            except Exception as e:
                error_msg = str(e)
                # Truncate long error messages (e.g., Gemini quota errors)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                logger.warning(f"{provider.value} failed: {error_msg}")
                continue  # Try next provider
        
        logger.error("All LLM providers exhausted, no LLM available")
        return None
    
    def _resolve_api_key(self, provider: LLMProvider, explicit_key: Optional[str]) -> Optional[str]:
        """Resolve API key with priority: explicit arg > settings > env var"""
        if explicit_key:
            return explicit_key
        
        # Try settings first
        llm_config = settings.llm
        key_map = {
            LLMProvider.OPENAI: llm_config.openai_api_key,
            LLMProvider.GEMINI: llm_config.gemini_api_key,
            LLMProvider.ANTHROPIC: llm_config.anthropic_api_key,
            LLMProvider.OPENROUTER: llm_config.openrouter_api_key,
            LLMProvider.GITHUB_MODELS: llm_config.github_token,
            LLMProvider.XAI: llm_config.xai_api_key,
            LLMProvider.ZAI: llm_config.zai_api_key,
            LLMProvider.GROQ: llm_config.groq_api_key,
            LLMProvider.OLLAMA: "ollama",  # Dummy value - no auth needed
        }
        
        if provider in key_map:
            key = key_map[provider]
            if key:
                return key
        
        # Fallback to env var
        env_map = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.GEMINI: "GEMINI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
            LLMProvider.GITHUB_MODELS: "GITHUB_TOKEN",
            LLMProvider.XAI: "XAI_API_KEY",
            LLMProvider.ZAI: "ZAI_API_KEY",
            LLMProvider.GROQ: "GROQ_API_KEY",
            LLMProvider.OLLAMA: None,
        }
        
        env_var = env_map.get(provider)
        if env_var:
            return os.getenv(env_var)
        
        if provider == LLMProvider.OLLAMA:
            return "ollama"  # Dummy value
        
        return None
    
    def _init_openai_compatible(
        self,
        provider: LLMProvider,
        config: Dict,
        api_key: Optional[str],
        custom_base_url: Optional[str]
    ):
        """Initialize OpenAI-compatible provider"""
        if not HAS_OPENAI:
            raise ImportError(
                f"openai package not installed. Install with: pip install openai"
            )
        
        # Use custom base URL if provided, otherwise use config
        base_url = custom_base_url or config.get("base_url")
        
        # Prepare initialization kwargs
        init_kwargs = {"api_key": api_key or "dummy"}  # Dummy key if none provided
        if base_url:
            init_kwargs["base_url"] = base_url
        
        self.client = openai.OpenAI(**init_kwargs)
        
        # Store provider-specific headers for OpenRouter
        self.provider_headers = {}
        if provider == LLMProvider.OPENROUTER:
            if settings.llm.openrouter_site_url:
                self.provider_headers["HTTP-Referer"] = settings.llm.openrouter_site_url
            if settings.llm.openrouter_app_name:
                self.provider_headers["X-Title"] = settings.llm.openrouter_app_name
    
    def _init_gemini(self, api_key: Optional[str]):
        """Initialize Google Gemini"""
        if not HAS_GEMINI:
            raise ImportError(
                f"google-generativeai package not installed. Install with: pip install google-generativeai"
            )
        
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Store safety settings for later use
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Create Gemini client
        self.client = genai.GenerativeModel(self.model)
    
    def _init_anthropic(self, api_key: Optional[str]):
        """Initialize Anthropic Claude"""
        if not HAS_ANTHROPIC:
            raise ImportError(
                f"anthropic package not installed. Install with: pip install anthropic"
            )
        
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Make API call to LLM"""
        
        # Check cache
        if self.cache:
            cached = self.cache.get(prompt, self.provider.value, self.model)
            if cached:
                logger.debug(f"Using cached LLM response (provider={self.provider.value}, model={self.model})")
                return LLMResponse(content=cached, cached=True, provider=self.provider.value, model=self.model)
        
        try:
            # Route to appropriate provider
            config = self._PROVIDER_CONFIG[self.provider]
            
            if config["type"] == "openai":
                # OpenAI-compatible provider
                content, tokens = self._call_openai_compatible(prompt, system_prompt)
            elif config["type"] == "gemini":
                # Google Gemini
                content, tokens = self._call_gemini(prompt, system_prompt)
            elif config["type"] == "anthropic":
                # Anthropic Claude
                content, tokens = self._call_anthropic(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown provider type: {config['type']}")
            
            # Cache response
            if self.cache:
                self.cache.set(prompt, content, self.provider.value, self.model)
            
            return LLMResponse(
                content=content,
                provider=self.provider.value,
                model=self.model,
                tokens_used=tokens
            )
            
        except Exception as e:
            logger.error(f"LLM API call failed ({self.provider.value}): {e}")
            raise
    
    def _call_openai_compatible(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        """Call OpenAI-compatible API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare kwargs for API call
        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }
        
        # Add provider-specific headers for OpenRouter
        if self.provider == LLMProvider.OPENROUTER and self.provider_headers:
            call_kwargs["extra_headers"] = self.provider_headers
        
        response = self.client.chat.completions.create(**call_kwargs)
        content = response.choices[0].message.content
        tokens = getattr(response.usage, 'total_tokens', 0)
        
        return content, tokens
    
    def _call_gemini(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        """Call Google Gemini API"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = self.client.generate_content(
            full_prompt,
            safety_settings=self.safety_settings
        )
        content = response.text
        tokens = 0  # Gemini doesn't always return token count
        
        return content, tokens
    
    def _call_anthropic(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        """Call Anthropic Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return content, tokens
    
    def _parse_json_response(self, response: LLMResponse) -> Dict:
        """Extract JSON from LLM response"""
        content = response.content
        
        # Try to find JSON block
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return {}
    
    # ==================== KNOWLEDGE EXTRACTION METHODS ====================
    
    def get_parameter_ranges(self, pattern_name: str) -> Dict[str, Dict[str, float]]:
        """
        Get suggested parameter ranges for a specific pattern.
        
        Returns dict like:
        {
            "volume_threshold": {"min": 1.5, "max": 4.0, "default": 2.0},
            "price_threshold": {"min": 0.0005, "max": 0.002, "default": 0.001},
            ...
        }
        """
        system_prompt = """You are an expert quantitative trader specializing in order flow analysis.
        Provide parameter ranges based on your domain knowledge of market microstructure.
        Be specific and practical. Return valid JSON only."""
        
        prompt = f"""For detecting "{pattern_name}" patterns in BTCUSD order flow data, 
        what are the recommended parameter ranges?
        
        Consider:
        - Typical values used by professional traders
        - Ranges that work across different market conditions
        - Conservative defaults that avoid false positives
        
        Return JSON in this exact format:
        {{
            "parameter_name": {{
                "min": <minimum_value>,
                "max": <maximum_value>,
                "default": <recommended_default>,
                "description": "<what this parameter controls>"
            }},
            ...
        }}
        
        Include all relevant parameters for {pattern_name} detection."""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def get_feature_importance(self, strategy_type: str) -> Dict[str, float]:
        """
        Get feature importance weights for a strategy type.
        
        Returns dict like:
        {
            "delta_imbalance": 0.8,
            "book_pressure": 0.6,
            ...
        }
        """
        system_prompt = """You are an expert in order flow trading strategies.
        Provide feature importance scores based on domain expertise.
        Scores should be 0-1 where 1 is most important."""
        
        prompt = f"""For a "{strategy_type}" order flow trading strategy on BTCUSD,
        rate the importance of each feature (0-1 scale).
        
        Consider these feature categories:
        1. Order book features (depth, imbalance, spread, microprice)
        2. Trade flow features (delta, volume, aggressor ratio)
        3. Volume profile features (POC, value area, VWAP)
        4. Footprint features (delta at price, stacked imbalances)
        5. Pattern detection (absorption, sweeps, icebergs)
        
        Return JSON with feature names as keys and importance scores as values.
        Include at least 20 features."""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def get_strategy_rules(self, strategy_name: str) -> Dict[str, Any]:
        """
        Extract explicit trading rules for a strategy.
        
        Returns structured rules that can be encoded into logic.
        """
        system_prompt = """You are an expert order flow trader.
        Provide specific, actionable trading rules that can be implemented in code.
        Be precise about conditions, thresholds, and logic."""
        
        prompt = f"""Define the trading rules for a "{strategy_name}" strategy.
        
        Provide:
        1. Entry conditions (specific feature thresholds)
        2. Exit conditions (stop loss, take profit logic)
        3. Position sizing rules
        4. Filter conditions (when NOT to trade)
        5. Regime-specific adjustments
        
        Return JSON in this format:
        {{
            "entry_conditions": [
                {{"feature": "...", "operator": ">", "threshold": ..., "weight": ...}},
                ...
            ],
            "exit_conditions": {{
                "stop_loss": {{"type": "...", "value": ...}},
                "take_profit": {{"type": "...", "value": ...}}
            }},
            "filters": [...],
            "position_sizing": {{...}},
            "regime_adjustments": {{...}}
        }}"""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def label_setup(self, features: Dict[str, float], context: str = "") -> Dict[str, Any]:
        """
        Get LLM to label a trading setup for weak supervision.
        
        Returns quality score and reasoning.
        """
        system_prompt = """You are an expert order flow analyst evaluating trading setups.
        Rate the quality of this setup based on the provided features.
        Be specific about why this is or isn't a good setup."""
        
        # Format features for prompt
        feature_str = "\n".join([f"- {k}: {v:.4f}" for k, v in features.items()])
        
        prompt = f"""Evaluate this order flow setup for BTCUSD:
        
        Features:
        {feature_str}
        
        Context: {context if context else "No additional context"}
        
        Return JSON:
        {{
            "quality_score": <0-100>,
            "direction": "<LONG/SHORT/NEUTRAL>",
            "confidence": <0-100>,
            "primary_signal": "<main reason for or against>",
            "supporting_signals": ["<additional positive factors>"],
            "warning_signals": ["<concerns or red flags>"],
            "suggested_stop_distance_pct": <0.001-0.01>,
            "suggested_target_distance_pct": <0.001-0.02>
        }}"""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def get_initial_parameters(self, strategy_type: str) -> Dict[str, float]:
        """
        Get initial/default parameter values for Optuna warm start.
        
        Returns a single set of "best guess" parameters.
        """
        system_prompt = """You are an expert quantitative trader.
        Provide your best estimate for initial parameter values.
        These will be used as starting points for optimization."""
        
        prompt = f"""For a "{strategy_type}" order flow trading strategy on BTCUSD,
        provide your best initial parameter values.
        
        Consider:
        - BTCUSD typical volatility and spread
        - Common institutional order flow patterns
        - Conservative values that work across conditions
        
        Return JSON with parameter names and values:
        {{
            "absorption_volume_mult": <value>,
            "absorption_price_threshold": <value>,
            "imbalance_ratio_threshold": <value>,
            "delta_threshold": <value>,
            "lookback_seconds": <value>,
            "min_holding_seconds": <value>,
            "stop_loss_atr_mult": <value>,
            "take_profit_atr_mult": <value>,
            ...include all relevant parameters...
        }}"""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def analyze_performance(
        self, 
        metrics: Dict[str, float],
        recent_trades: List[Dict]
    ) -> Dict[str, Any]:
        """
        Get LLM analysis of strategy performance and suggestions.
        """
        system_prompt = """You are a trading performance analyst.
        Analyze the provided metrics and trades to identify issues and improvements."""
        
        metrics_str = "\n".join([f"- {k}: {v:.4f}" for k, v in metrics.items()])
        trades_str = json.dumps(recent_trades[:20], indent=2)
        
        prompt = f"""Analyze this trading strategy performance:
        
        Metrics:
        {metrics_str}
        
        Recent trades (sample):
        {trades_str}
        
        Provide:
        1. Overall assessment
        2. Key issues identified
        3. Specific parameter adjustments to consider
        4. Market conditions where strategy underperforms
        
        Return JSON:
        {{
            "assessment": "<overall evaluation>",
            "grade": "<A/B/C/D/F>",
            "issues": ["<issue 1>", ...],
            "suggested_adjustments": {{
                "<parameter>": {{"current_implied": ..., "suggested": ..., "reason": "..."}}
            }},
            "avoid_conditions": ["<condition where strategy fails>", ...]
        }}"""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)


# ==================== DOMAIN KNOWLEDGE CONSTANTS ====================
# Pre-extracted knowledge to avoid repeated API calls

ABSORPTION_KNOWLEDGE = {
    "parameter_ranges": {
        "volume_multiplier": {"min": 1.5, "max": 5.0, "default": 2.5},
        "price_threshold_pct": {"min": 0.0003, "max": 0.002, "default": 0.001},
        "min_duration_seconds": {"min": 1.0, "max": 10.0, "default": 3.0},
        "strength_threshold": {"min": 0.4, "max": 0.8, "default": 0.6},
    },
    "rules": {
        "entry": "Volume > multiplier * avg AND price_change < threshold AND duration > min_duration",
        "confirmation": "Look for failed auction (price returns to absorption level)",
        "invalidation": "Price breaks through absorption level with increasing delta"
    }
}

DELTA_DIVERGENCE_KNOWLEDGE = {
    "parameter_ranges": {
        "lookback_bars": {"min": 3, "max": 20, "default": 10},
        "price_threshold_pct": {"min": 0.001, "max": 0.01, "default": 0.003},
        "delta_threshold_pct": {"min": 0.1, "max": 0.5, "default": 0.25},
    },
    "rules": {
        "bullish_divergence": "Price makes lower low BUT delta makes higher low",
        "bearish_divergence": "Price makes higher high BUT delta makes lower high",
        "confirmation": "Wait for price to cross back above/below trigger level"
    }
}

LIQUIDITY_SWEEP_KNOWLEDGE = {
    "parameter_ranges": {
        "sweep_speed_max_seconds": {"min": 1.0, "max": 10.0, "default": 5.0},
        "sweep_depth_pct": {"min": 0.001, "max": 0.01, "default": 0.003},
        "reversal_threshold_pct": {"min": 0.3, "max": 0.8, "default": 0.5},
        "volume_spike_multiplier": {"min": 2.0, "max": 10.0, "default": 4.0},
    },
    "rules": {
        "identification": "Fast move through liquidity zone followed by reversal",
        "entry": "Enter on reversal confirmation (price reclaims key level)",
        "stop": "Place stop beyond the sweep extreme",
        "target": "Target opposite liquidity zone or POC"
    }
}

STACKED_IMBALANCE_KNOWLEDGE = {
    "parameter_ranges": {
        "imbalance_ratio": {"min": 2.0, "max": 5.0, "default": 3.0},
        "min_stack_levels": {"min": 2, "max": 5, "default": 3},
        "volume_threshold": {"min": 0.5, "max": 2.0, "default": 1.0},
    },
    "rules": {
        "identification": "3+ consecutive price levels with same-side imbalance",
        "interpretation": "Indicates strong directional conviction",
        "entry": "Trade in direction of imbalance on pullback to stack",
        "invalidation": "Stack levels are traded through with opposing delta"
    }
}


# ==============================================================================
# SECTION 6: optimization/optuna_optimizer.py
# ==============================================================================

"""
Optuna Optimizer - Enhanced for Perfect Optimization
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from loguru import logger

from knowledge.llm_advisor import LLMAdvisor, ABSORPTION_KNOWLEDGE, DELTA_DIVERGENCE_KNOWLEDGE
from knowledge.strategy_library import StrategyDefinition, get_strategy

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    """Results from optimization run"""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    study_name: str
    timestamp: datetime
    all_trials: List[Dict] = None
    objective_type: str = "robust"


class StrategyOptimizer:
    """
    Enhanced strategy parameter optimizer.
    
    Features:
    1. Warm-start from LLM domain knowledge
    2. Multiple objective functions (sharpe, profit, robust)
    3. Minimum trade count enforcement
    4. Correlation-aware parameter constraints
    5. Advanced pruning
    """
    
    def __init__(
        self,
        strategy_name: str,
        llm_advisor: Optional[LLMAdvisor] = None,
        storage: str = "sqlite:///optuna_studies.db"
    ):
        self.strategy_name = strategy_name
        self.llm_advisor = llm_advisor
        self.storage = storage
        
        self.base_strategy = get_strategy(strategy_name)
        if not self.base_strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        self.param_ranges = self._get_parameter_ranges()
        self.warm_start_params = self._get_warm_start_params()
    
    def _get_parameter_ranges(self) -> Dict[str, Dict]:
        """Get parameter ranges with LLM enhancement"""
        if self.strategy_name == "absorption":
            ranges = ABSORPTION_KNOWLEDGE["parameter_ranges"].copy()
        elif self.strategy_name == "delta_divergence":
            ranges = DELTA_DIVERGENCE_KNOWLEDGE["parameter_ranges"].copy()
        else:
            ranges = {}
        
        common_ranges = {
            "lookback_seconds": {"min": 30, "max": 600, "default": 120},
            "min_score_threshold": {"min": 1.0, "max": 5.0, "default": 3.0},
            "stop_loss_atr_mult": {"min": 0.5, "max": 3.0, "default": 1.5},
            "take_profit_atr_mult": {"min": 1.0, "max": 8.0, "default": 4.5},  # Higher for 3:1 RR
            "min_conditions_satisfied": {"min": 2, "max": 6, "default": 3},
            "base_position_pct": {"min": 0.05, "max": 0.25, "default": 0.1},
        }
        
        ranges.update(common_ranges)
        
        if self.llm_advisor:
            try:
                llm_ranges = self.llm_advisor.get_parameter_ranges(self.strategy_name)
                for param, values in llm_ranges.items():
                    if param not in ranges:
                        ranges[param] = values
            except Exception as e:
                logger.warning(f"Failed to get LLM parameter ranges: {e}")
        
        return ranges
    
    def _get_warm_start_params(self) -> Dict[str, float]:
        """Get initial parameter values for warm start"""
        warm_start = {}
        
        for param, range_dict in self.param_ranges.items():
            if "default" in range_dict:
                warm_start[param] = range_dict["default"]
        
        # Enforce logical constraints in warm start
        if "stop_loss_atr_mult" in warm_start and "take_profit_atr_mult" in warm_start:
            # Ensure 2:1 minimum risk/reward
            warm_start["take_profit_atr_mult"] = max(
                warm_start["take_profit_atr_mult"],
                warm_start["stop_loss_atr_mult"] * 2
            )
        
        if self.llm_advisor:
            try:
                llm_params = self.llm_advisor.get_initial_parameters(self.strategy_name)
                for param, value in llm_params.items():
                    if param in self.param_ranges:
                        range_dict = self.param_ranges[param]
                        if range_dict["min"] <= value <= range_dict["max"]:
                            warm_start[param] = value
            except Exception as e:
                logger.warning(f"Failed to get LLM initial params: {e}")
        
        return warm_start
    
    def create_objective(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        objective_type: str = "robust"
    ) -> Callable[[optuna.Trial], float]:
        """
        Create Optuna objective function with selected objective type.
        
        Args:
            backtest_fn: Function that takes params and returns metrics
            objective_type: 
                - "sharpe": Pure Sharpe ratio maximization
                - "profit": Return maximization with drawdown penalty
                - "robust": Multi-factor composite (RECOMMENDED)
        """
        def objective(trial: optuna.Trial) -> float:
            params = {}
            
            for param_name, range_dict in self.param_ranges.items():
                min_val = range_dict["min"]
                max_val = range_dict["max"]
                
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            # Enforce constraints
            params = self._enforce_constraints(params, trial)
            
            try:
                metrics = backtest_fn(params)
            except Exception as e:
                logger.debug(f"Backtest failed: {e}")
                return float('-inf')
            
            # Check if penalized (too few trades)
            if metrics.get("_penalized", False):
                trial.set_user_attr("penalized", True)
                return float('-inf')
            
            # Store all metrics
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            
            # Calculate objective based on type
            if objective_type == "sharpe":
                score = metrics.get("sharpe_ratio", 0)
            elif objective_type == "profit":
                score = self._profit_objective(metrics)
            elif objective_type == "robust":
                score = self._robust_objective(metrics)
            else:
                score = metrics.get("sharpe_ratio", 0)
            
            # Report for pruning
            trial.report(score, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
        
        return objective
    
    def _enforce_constraints(
        self, 
        params: Dict[str, Any], 
        trial: optuna.Trial
    ) -> Dict[str, Any]:
        """Enforce logical parameter constraints"""
        # Ensure take_profit > stop_loss (minimum 2:1 RR)
        if "stop_loss_atr_mult" in params and "take_profit_atr_mult" in params:
            min_tp = params["stop_loss_atr_mult"] * 2
            if params["take_profit_atr_mult"] < min_tp:
                params["take_profit_atr_mult"] = trial.suggest_float(
                    "take_profit_atr_mult_constrained",
                    min_tp,
                    self.param_ranges["take_profit_atr_mult"]["max"]
                )
        
        return params
    
    def _profit_objective(self, metrics: Dict[str, float]) -> float:
        """
        Profit-focused objective with drawdown penalty.
        
        Score = Return - 3 * MaxDrawdown
        """
        total_return = metrics.get("total_return_pct", 0) * 100  # Convert to %
        max_drawdown = metrics.get("max_drawdown_pct", 0) * 100
        
        return total_return - 3 * max_drawdown
    
    def _robust_objective(self, metrics: Dict[str, float]) -> float:
        """
        Robust multi-factor objective.
        
        Combines:
        - Sharpe ratio (risk-adjusted returns)
        - Profit factor (win/loss ratio)
        - Win rate (consistency)
        - Drawdown penalty (risk control)
        
        This objective prefers strategies that are:
        - Profitable (positive return)
        - Consistent (high win rate)
        - Risk-controlled (low drawdown)
        - Efficient (good profit factor)
        """
        sharpe = metrics.get("sharpe_ratio", 0)
        profit_factor = metrics.get("profit_factor", 0)
        win_rate = metrics.get("win_rate", 0)
        max_drawdown = metrics.get("max_drawdown_pct", 0)
        total_trades = metrics.get("total_trades", 0)
        
        # Component 1: Sharpe (normalized to 0-1 range, cap at 3)
        sharpe_component = min(sharpe / 3.0, 1.0)
        
        # Component 2: Profit factor (normalized, cap at 3)
        pf_component = min(profit_factor / 3.0, 1.0) if profit_factor > 0 else 0
        
        # Component 3: Win rate (direct, but weighted lower)
        win_rate_component = win_rate * 0.8  # 80% win rate = 0.64
        
        # Component 4: Drawdown penalty (exponential decay)
        # 0% DD = 1.0, 10% DD = 0.37, 20% DD = 0.14
        drawdown_penalty = np.exp(-max_drawdown * 10)
        
        # Component 5: Trade count sufficiency
        # Sigmoid function: approaches 1 around 100 trades
        trade_sufficiency = 1 / (1 + np.exp(-0.05 * (total_trades - 100)))
        
        # Weighted combination
        weights = {
            "sharpe": 0.30,
            "profit_factor": 0.25,
            "win_rate": 0.15,
            "drawdown": 0.20,
            "trade_count": 0.10
        }
        
        score = (
            weights["sharpe"] * sharpe_component +
            weights["profit_factor"] * pf_component +
            weights["win_rate"] * win_rate_component +
            weights["drawdown"] * drawdown_penalty +
            weights["trade_count"] * trade_sufficiency
        )
        
        # Hard penalty: negative return = bad
        if metrics.get("total_return_pct", 0) < 0:
            score *= 0.5
        
        return score
    
    def optimize(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: int = 200,
        n_jobs: int = -1,
        timeout: Optional[int] = None,
        objective_type: str = "robust"
    ) -> OptimizationResult:
        """
        Run optimization with warm start and enhanced objective.
        """
        study_name = f"{self.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Enhanced sampler with warm start
        sampler = TPESampler(
            n_startup_trials=20,
            multivariate=True,  # Consider parameter correlations
            warn_independent_sampling=False
        )
        
        # Hyperband pruner for aggressive early stopping
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=1,
            reduction_factor=3
        )
        
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        # Enqueue warm-start trial
        if self.warm_start_params:
            valid_warm_start = {
                k: v for k, v in self.warm_start_params.items()
                if k in self.param_ranges
            }
            if valid_warm_start:
                study.enqueue_trial(valid_warm_start)
                logger.info(f"Enqueued warm-start trial with {len(valid_warm_start)} params")
        
        objective = self.create_objective(backtest_fn, objective_type)
        
        logger.info(f"Starting optimization: {n_trials} trials, objective={objective_type}, n_jobs={n_jobs}")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=False  # Cleaner logs
        )
        
        # Filter out penalized trials for analysis
        valid_trials = [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "user_attrs": t.user_attrs
            }
            for t in study.trials
            if t.value is not None and not t.user_attrs.get("_penalized", False)
        ]
        
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study_name=study_name,
            timestamp=datetime.now(),
            all_trials=valid_trials,
            objective_type=objective_type
        )
        
        logger.info(f"Optimization complete.")
        logger.info(f"  Best score ({objective_type}): {result.best_score:.4f}")
        logger.info(f"  Valid trials: {len(valid_trials)} / {len(study.trials)}")
        logger.info(f"  Best params: {result.best_params}")
        
        return result
    
    def save_results(self, result: OptimizationResult, path: str) -> None:
        """Save optimization results to file"""
        output = {
            "strategy_name": self.strategy_name,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "valid_trials": len(result.all_trials) if result.all_trials else 0,
            "timestamp": result.timestamp.isoformat(),
            "objective_type": result.objective_type,
            "param_ranges": self.param_ranges,
            "warm_start_params": self.warm_start_params
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved results to {path}")
    
    def analyze_parameter_sensitivity(self, result: OptimizationResult) -> Dict[str, float]:
        """
        Analyze which parameters most affect the objective.
        Returns importance scores for each parameter.
        """
        if not result.all_trials or len(result.all_trials) < 10:
            return {}
        
        # Get parameter names
        param_names = list(result.all_trials[0]["params"].keys())
        
        importance = {}
        
        for param in param_names:
            # Get param values and corresponding scores
            pairs = [
                (t["params"].get(param), t["value"])
                for t in result.all_trials
                if param in t["params"] and t["value"] is not None
            ]
            
            if len(pairs) < 5:
                continue
            
            values = np.array([p[0] for p in pairs])
            scores = np.array([p[1] for p in pairs])
            
            # Calculate correlation as importance
            if np.std(values) > 0 and np.std(scores) > 0:
                correlation = np.corrcoef(values, scores)[0, 1]
                importance[param] = abs(correlation)
            else:
                importance[param] = 0.0
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    
# ==============================================================================
# SECTION 7: config/settings.py
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
    n_jobs: int = -1  # Auto-detect all CPU cores (use -1 for cloud)
    
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