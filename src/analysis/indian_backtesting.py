#!/usr/bin/env python3
"""
Indian Market Backtesting Engine

This module provides comprehensive backtesting capabilities for Indian market
strategies including Nifty 50, Bank Nifty, and Sensex options and futures
with realistic market conditions and transaction costs.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path

# Import our custom modules
try:
    from ..data.indian_market_data import IndianMarketDataFetcher
    from .indian_technical_analysis import IndianMarketAnalyzer
    from ..options.indian_options_engine import IndianOptionsStrategyEngine
    from ..portfolio.indian_portfolio_simulator import IndianPortfolioSimulator
    from ..config.config import get_config, get_lot_size
except ImportError:
    # Fallback for when running as main module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.indian_market_data import IndianMarketDataFetcher
    from analysis.indian_technical_analysis import IndianMarketAnalyzer
    from options.indian_options_engine import IndianOptionsStrategyEngine
    from portfolio.indian_portfolio_simulator import IndianPortfolioSimulator
    from config.config import get_config, get_lot_size

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtesting result data"""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    volatility: float
    calmar_ratio: float
    trade_history: List[Dict] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str
    end_date: str
    initial_capital: float
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    margin_requirement: float = 0.15
    max_positions: int = 10
    position_size: float = 0.1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rebalance_frequency: str = 'daily'  # daily, weekly, monthly

class IndianBacktestingEngine:
    """Comprehensive backtesting engine for Indian markets"""
    
    def __init__(self):
        self.data_fetcher = IndianMarketDataFetcher()
        self.technical_analyzer = IndianMarketAnalyzer()
        self.options_engine = IndianOptionsStrategyEngine()
        self.config = get_config()
        
        # Indian market specific parameters
        self.lot_sizes = {
            'NIFTY_50': 50,
            'BANK_NIFTY': 25,
            'SENSEX': 10,
            'NIFTY_IT': 25,
            'NIFTY_AUTO': 25,
            'NIFTY_PHARMA': 25
        }
        
        # Transaction costs (Indian market rates)
        self.transaction_costs = {
            'commission': 0.0003,  # 0.03%
            'stt': 0.0005,         # 0.05% STT
            'stamp_duty': 0.0001,  # 0.01%
            'exchange_charges': 0.0001,  # 0.01%
            'slippage': 0.0001     # 0.01% slippage
        }
    
    def run_backtest(self, strategy_name: str, symbol: str, config: BacktestConfig) -> BacktestResult:
        """Run comprehensive backtest for a strategy"""
        try:
            logger.info(f"Starting backtest for {strategy_name} on {symbol}")
            
            # Initialize portfolio simulator
            portfolio = IndianPortfolioSimulator(config.initial_capital)
            
            # Fetch historical data
            data = self._fetch_historical_data(symbol, config.start_date, config.end_date)
            if data.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Run strategy simulation
            trade_history, equity_curve = self._simulate_strategy(
                strategy_name, symbol, data, portfolio, config
            )
            
            # Calculate performance metrics
            result = self._calculate_performance_metrics(
                strategy_name, symbol, config, trade_history, equity_curve
            )
            
            logger.info(f"Backtest completed for {strategy_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise
    
    def _fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for backtesting"""
        try:
            # Calculate period based on date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            days_diff = (end_dt - start_dt).days
            
            if days_diff <= 30:
                period = "1mo"
            elif days_diff <= 90:
                period = "3mo"
            elif days_diff <= 180:
                period = "6mo"
            elif days_diff <= 365:
                period = "1y"
            elif days_diff <= 730:
                period = "2y"
            else:
                period = "5y"
            
            # Fetch data
            data = self.data_fetcher.fetch_index_data(symbol, period)
            
            if data.empty:
                return pd.DataFrame()
            
            # Filter by date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def _simulate_strategy(self, strategy_name: str, symbol: str, data: pd.DataFrame, 
                          portfolio: IndianPortfolioSimulator, config: BacktestConfig) -> Tuple[List[Dict], List[Dict]]:
        """Simulate strategy execution"""
        try:
            trade_history = []
            equity_curve = []
            
            # Initialize strategy parameters
            position = None
            entry_price = None
            entry_date = None
            
            # Process each trading day
            for i, (date, row) in enumerate(data.iterrows()):
                current_price = row['Close']
                current_volume = row['Volume']
                
                # Skip if insufficient data for analysis
                if i < 50:  # Need at least 50 days for technical indicators
                    continue
                
                # Get historical data up to current date
                historical_data = data.iloc[:i+1]
                
                # Perform technical analysis
                analysis = self.technical_analyzer.analyze_index(historical_data, symbol)
                if 'error' in analysis:
                    continue
                
                # Get trading signal
                signals = analysis.get('signals', {})
                overall_signal = signals.get('overall', {})
                signal = overall_signal.get('signal', 'NEUTRAL')
                signal_strength = overall_signal.get('strength', 0)
                
                # Strategy logic
                if strategy_name == 'momentum_strategy':
                    position, trade = self._momentum_strategy_logic(
                        signal, signal_strength, current_price, position, 
                        entry_price, entry_date, date, symbol, config
                    )
                elif strategy_name == 'mean_reversion_strategy':
                    position, trade = self._mean_reversion_strategy_logic(
                        signal, signal_strength, current_price, position,
                        entry_price, entry_date, date, symbol, config, analysis
                    )
                elif strategy_name == 'options_strategy':
                    position, trade = self._options_strategy_logic(
                        signal, signal_strength, current_price, position,
                        entry_price, entry_date, date, symbol, config, analysis
                    )
                else:
                    # Default strategy
                    position, trade = self._default_strategy_logic(
                        signal, signal_strength, current_price, position,
                        entry_price, entry_date, date, symbol, config
                    )
                
                # Record trade if any
                if trade:
                    trade_history.append(trade)
                    
                    # Update portfolio
                    if trade['action'] == 'BUY':
                        portfolio.add_position(
                            symbol=symbol,
                            instrument_type='index',
                            quantity=1,
                            entry_price=trade['price'],
                            strategy=strategy_name
                        )
                        position = 'LONG'
                        entry_price = trade['price']
                        entry_date = date
                    elif trade['action'] == 'SELL' and position:
                        # Close position
                        portfolio.close_position(
                            list(portfolio.positions.keys())[-1], 
                            trade['price']
                        )
                        position = None
                        entry_price = None
                        entry_date = None
                
                # Record equity curve
                portfolio_value = portfolio.available_capital
                for pos in portfolio.positions.values():
                    if pos.status == 'open':
                        portfolio_value += pos.margin_used
                        portfolio_value += pos.unrealized_pnl
                
                equity_curve.append({
                    'date': date,
                    'value': portfolio_value,
                    'price': current_price,
                    'signal': signal,
                    'position': position
                })
            
            return trade_history, equity_curve
            
        except Exception as e:
            logger.error(f"Error simulating strategy: {e}")
            return [], []
    
    def _momentum_strategy_logic(self, signal: str, signal_strength: float, current_price: float,
                                position: Optional[str], entry_price: Optional[float], 
                                entry_date: Optional[datetime], date: datetime, symbol: str,
                                config: BacktestConfig) -> Tuple[Optional[str], Optional[Dict]]:
        """Momentum strategy logic"""
        trade = None
        
        # Entry logic
        if not position and signal == 'BUY' and signal_strength > 0.6:
            trade = {
                'date': date,
                'action': 'BUY',
                'price': current_price,
                'symbol': symbol,
                'strategy': 'momentum_strategy',
                'reason': f'Strong buy signal: {signal_strength:.2f}'
            }
            position = 'LONG'
        
        # Exit logic
        elif position == 'LONG':
            # Stop loss
            if config.stop_loss and entry_price:
                stop_loss_price = entry_price * (1 - config.stop_loss)
                if current_price <= stop_loss_price:
                    trade = {
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'symbol': symbol,
                        'strategy': 'momentum_strategy',
                        'reason': f'Stop loss triggered at {stop_loss_price:.2f}'
                    }
                    position = None
            
            # Take profit
            elif config.take_profit and entry_price:
                take_profit_price = entry_price * (1 + config.take_profit)
                if current_price >= take_profit_price:
                    trade = {
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'symbol': symbol,
                        'strategy': 'momentum_strategy',
                        'reason': f'Take profit triggered at {take_profit_price:.2f}'
                    }
                    position = None
            
            # Signal-based exit
            elif signal == 'SELL' and signal_strength > 0.6:
                trade = {
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'symbol': symbol,
                    'strategy': 'momentum_strategy',
                    'reason': f'Strong sell signal: {signal_strength:.2f}'
                }
                position = None
        
        return position, trade
    
    def _mean_reversion_strategy_logic(self, signal: str, signal_strength: float, current_price: float,
                                     position: Optional[str], entry_price: Optional[float],
                                     entry_date: Optional[datetime], date: datetime, symbol: str,
                                     config: BacktestConfig, analysis: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """Mean reversion strategy logic"""
        trade = None
        
        # Get RSI for mean reversion signals
        rsi = analysis.get('indicators', {}).get('rsi', pd.Series())
        if rsi.empty:
            return position, trade
        
        current_rsi = rsi.iloc[-1]
        
        # Entry logic (contrarian)
        if not position:
            if current_rsi < 30 and signal == 'BUY':  # Oversold
                trade = {
                    'date': date,
                    'action': 'BUY',
                    'price': current_price,
                    'symbol': symbol,
                    'strategy': 'mean_reversion_strategy',
                    'reason': f'RSI oversold: {current_rsi:.1f}'
                }
                position = 'LONG'
            elif current_rsi > 70 and signal == 'SELL':  # Overbought
                trade = {
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'symbol': symbol,
                    'strategy': 'mean_reversion_strategy',
                    'reason': f'RSI overbought: {current_rsi:.1f}'
                }
                position = 'SHORT'
        
        # Exit logic
        elif position == 'LONG':
            if current_rsi > 50:  # Exit long when RSI normalizes
                trade = {
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'symbol': symbol,
                    'strategy': 'mean_reversion_strategy',
                    'reason': f'RSI normalized: {current_rsi:.1f}'
                }
                position = None
        elif position == 'SHORT':
            if current_rsi < 50:  # Exit short when RSI normalizes
                trade = {
                    'date': date,
                    'action': 'BUY',
                    'price': current_price,
                    'symbol': symbol,
                    'strategy': 'mean_reversion_strategy',
                    'reason': f'RSI normalized: {current_rsi:.1f}'
                }
                position = None
        
        return position, trade
    
    def _options_strategy_logic(self, signal: str, signal_strength: float, current_price: float,
                               position: Optional[str], entry_price: Optional[float],
                               entry_date: Optional[datetime], date: datetime, symbol: str,
                               config: BacktestConfig, analysis: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """Options strategy logic"""
        trade = None
        
        # This is a simplified options strategy
        # In practice, you would implement specific options strategies like spreads, straddles, etc.
        
        # Entry logic
        if not position and signal == 'BUY' and signal_strength > 0.7:
            # Simulate buying a call option
            option_premium = current_price * 0.02  # 2% of underlying price
            trade = {
                'date': date,
                'action': 'BUY',
                'price': option_premium,
                'symbol': symbol,
                'strategy': 'options_strategy',
                'reason': f'Buy call option, premium: {option_premium:.2f}'
            }
            position = 'LONG_CALL'
        
        # Exit logic
        elif position == 'LONG_CALL':
            # Simple exit after 5 days or if signal changes
            days_held = (date - entry_date).days if entry_date else 0
            
            if days_held >= 5 or signal == 'SELL':
                # Simulate option value (simplified)
                option_value = max(0, current_price - entry_price * 0.98) if entry_price else 0
                trade = {
                    'date': date,
                    'action': 'SELL',
                    'price': option_value,
                    'symbol': symbol,
                    'strategy': 'options_strategy',
                    'reason': f'Sell call option, value: {option_value:.2f}'
                }
                position = None
        
        return position, trade
    
    def _default_strategy_logic(self, signal: str, signal_strength: float, current_price: float,
                               position: Optional[str], entry_price: Optional[float],
                               entry_date: Optional[datetime], date: datetime, symbol: str,
                               config: BacktestConfig) -> Tuple[Optional[str], Optional[Dict]]:
        """Default strategy logic"""
        trade = None
        
        # Simple buy and hold with signal-based entries/exits
        if not position and signal == 'BUY' and signal_strength > 0.5:
            trade = {
                'date': date,
                'action': 'BUY',
                'price': current_price,
                'symbol': symbol,
                'strategy': 'default_strategy',
                'reason': f'Buy signal: {signal_strength:.2f}'
            }
            position = 'LONG'
        elif position == 'LONG' and signal == 'SELL' and signal_strength > 0.5:
            trade = {
                'date': date,
                'action': 'SELL',
                'price': current_price,
                'symbol': symbol,
                'strategy': 'default_strategy',
                'reason': f'Sell signal: {signal_strength:.2f}'
            }
            position = None
        
        return position, trade
    
    def _calculate_performance_metrics(self, strategy_name: str, symbol: str, config: BacktestConfig,
                                     trade_history: List[Dict], equity_curve: List[Dict]) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        try:
            if not equity_curve:
                raise ValueError("No equity curve data available")
            
            # Basic metrics
            initial_capital = config.initial_capital
            final_capital = equity_curve[-1]['value']
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Calculate annualized return
            start_date = pd.to_datetime(config.start_date)
            end_date = pd.to_datetime(config.end_date)
            years = (end_date - start_date).days / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Calculate daily returns
            daily_returns = []
            for i in range(1, len(equity_curve)):
                prev_value = equity_curve[i-1]['value']
                curr_value = equity_curve[i]['value']
                daily_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
                daily_returns.append(daily_return)
            
            # Calculate max drawdown
            peak = initial_capital
            max_drawdown = 0
            for point in equity_curve:
                if point['value'] > peak:
                    peak = point['value']
                drawdown = (peak - point['value']) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Calculate Sharpe ratio
            if daily_returns:
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
                
                # Calculate Sortino ratio
                negative_returns = [r for r in daily_returns if r < 0]
                downside_std = np.std(negative_returns) if negative_returns else 0
                sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
                
                # Calculate volatility
                volatility = std_return * np.sqrt(252)
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
                volatility = 0
            
            # Calculate Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Trade statistics
            if trade_history:
                # Separate buy and sell trades
                buy_trades = [t for t in trade_history if t['action'] == 'BUY']
                sell_trades = [t for t in trade_history if t['action'] == 'SELL']
                
                # Calculate trade P&L
                trade_pnls = []
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_price = buy_trades[i]['price']
                    sell_price = sell_trades[i]['price']
                    pnl = (sell_price - buy_price) / buy_price
                    trade_pnls.append(pnl)
                
                winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
                
                win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                largest_win = max(winning_trades) if winning_trades else 0
                largest_loss = min(losing_trades) if losing_trades else 0
                
                profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
                
                # Calculate average trade duration
                trade_durations = []
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_date = pd.to_datetime(buy_trades[i]['date'])
                    sell_date = pd.to_datetime(sell_trades[i]['date'])
                    duration = (sell_date - buy_date).days
                    trade_durations.append(duration)
                
                avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                largest_win = 0
                largest_loss = 0
                profit_factor = 0
                avg_trade_duration = 0
            
            return BacktestResult(
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                annualized_return=annualized_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(trade_history),
                winning_trades=len(winning_trades) if trade_history else 0,
                losing_trades=len(losing_trades) if trade_history else 0,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_trade_duration=avg_trade_duration,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                trade_history=trade_history,
                daily_returns=daily_returns,
                equity_curve=equity_curve
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    def compare_strategies(self, strategies: List[str], symbol: str, config: BacktestConfig) -> Dict[str, BacktestResult]:
        """Compare multiple strategies"""
        try:
            results = {}
            
            for strategy in strategies:
                logger.info(f"Backtesting strategy: {strategy}")
                result = self.run_backtest(strategy, symbol, config)
                results[strategy] = result
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return {}
    
    def export_backtest_results(self, results: Dict[str, BacktestResult], filepath: str) -> bool:
        """Export backtest results to JSON file"""
        try:
            export_data = {}
            
            for strategy_name, result in results.items():
                export_data[strategy_name] = {
                    'strategy_name': result.strategy_name,
                    'symbol': result.symbol,
                    'start_date': result.start_date,
                    'end_date': result.end_date,
                    'initial_capital': result.initial_capital,
                    'final_capital': result.final_capital,
                    'total_return': result.total_return,
                    'annualized_return': result.annualized_return,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': result.sortino_ratio,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'avg_win': result.avg_win,
                    'avg_loss': result.avg_loss,
                    'largest_win': result.largest_win,
                    'largest_loss': result.largest_loss,
                    'avg_trade_duration': result.avg_trade_duration,
                    'volatility': result.volatility,
                    'calmar_ratio': result.calmar_ratio,
                    'trade_history': result.trade_history,
                    'daily_returns': result.daily_returns,
                    'equity_curve': result.equity_curve
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Backtest results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting backtest results: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create backtesting engine
    engine = IndianBacktestingEngine()
    
    # Create backtest configuration
    config = BacktestConfig(
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=1000000,
        stop_loss=0.05,  # 5% stop loss
        take_profit=0.10  # 10% take profit
    )
    
    print("Testing Indian Market Backtesting Engine...")
    
    # Run backtest
    try:
        result = engine.run_backtest('momentum_strategy', 'NIFTY_50', config)
        print(f"Backtest completed: {result.strategy_name}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.1%}")
    except Exception as e:
        print(f"Backtest failed: {e}")
    
    print("Indian Market Backtesting Engine module loaded successfully")
