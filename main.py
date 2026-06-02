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
import sys

from loguru import logger
import pandas as pd

logger.remove()

# Configure logging
logger.add(
    "logs/orderflow_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)

logger.add(
    sys.stderr,              # Standard error stream (terminal/console)
    level="INFO",            # Only INFO and above
    format="{time:HH:mm:ss} | {level} | {message}"  # Compact format
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
        self.onchain_connector: Optional[Any] = None
        self.onchain_filter: Optional[Any] = None
        
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
        
        # On-chain filter (optional, lazy import — no error if missing deps)
        if self.settings.onchain.enabled:
            try:
                from data.onchain_connector import GlassnodeConnector, OnChainRegimeFilter
                self.onchain_connector = GlassnodeConnector(
                    api_key=self.settings.onchain.api_key,
                    asset=self.settings.trading.symbol.split('/')[0]
                )
                self.onchain_filter = OnChainRegimeFilter()
                logger.info("On-chain regime filter enabled")
            except ImportError as e:
                logger.warning(f"On-chain filter disabled — missing dependency: {e}")
                logger.warning("  pip install requests  # required for on-chain")
    
    def validate_optional_features(self) -> dict:
        """Check optional feature health and return status dict"""
        status = {}
        # On-chain
        if self.settings.onchain.enabled:
            if self.onchain_connector:
                status["onchain"] = "ready"
            else:
                status["onchain"] = "disabled (import failed)"
        else:
            status["onchain"] = "disabled (config)"
        # ML ensemble
        if self.settings.ml_ensemble.enabled:
            try:
                from prediction.ml_ensemble import MLEnsemble
                status["ml_ensemble"] = "ready"
            except ImportError as e:
                status["ml_ensemble"] = f"missing deps: {e}"
        else:
            status["ml_ensemble"] = "disabled (config)"
        # Fee-aware filter
        if self.settings.fee_aware_filter.enabled:
            status["fee_aware_filter"] = "enabled"
        else:
            status["fee_aware_filter"] = "disabled"
        # Data filtering
        if self.settings.data_filtering.enabled:
            status["data_filtering"] = "enabled"
        else:
            status["data_filtering"] = "disabled"
        return status

    def _print_startup_dashboard(self) -> None:
        """Print startup dashboard with feature status"""
        print("\n" + "=" * 60)
        print("  ORDER FLOW TRADING SYSTEM — STARTUP DASHBOARD")
        print("=" * 60)
        print(f"  Trading Pair : {self.settings.trading.symbol}")
        print(f"  Exchange     : {self.settings.trading.exchange.value}")
        print(f"  Mode         : paper/live recording")
        print("-" * 60)
        status = self.validate_optional_features()
        for feature, state in status.items():
            if "disabled" in state:
                print(f"  [-] {feature:20s}  {state}")
            else:
                print(f"  [+] {feature:20s}  {state}")
        print("=" * 60 + "\n")

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
        # On-chain regime check (gated by config)
        if self.settings.onchain.enabled and self.onchain_connector and self.onchain_filter:
            try:
                metrics = self.onchain_connector.get_latest_metrics()
                if metrics:
                    should_halt, reason = self.onchain_filter.should_halt_new_positions(metrics)
                    if should_halt:
                        logger.warning(f"On-chain filter halted trading: {reason}")
                        await asyncio.sleep(60)  # Wait 1 minute before retry
                        return
            except Exception as e:
                logger.error(f"On-chain check failed: {e}")
        
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
    
    # Print startup dashboard (non-test modes only)
    if args.mode != 'test':
        system._print_startup_dashboard()
    
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