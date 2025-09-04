#!/usr/bin/env python3
"""
Indian Portfolio Simulator

This module provides comprehensive portfolio simulation and backtesting capabilities
specifically designed for Indian market instruments including Nifty 50, Bank Nifty,
and Sensex options and futures with Indian market-specific parameters.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Individual position in the portfolio"""
    symbol: str
    instrument_type: str  # 'option', 'future', 'index'
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    strategy: str
    expiry: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # 'call' or 'put'
    lot_size: int = 50
    margin_used: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'expired'

@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    symbol: str
    instrument_type: str
    action: str  # 'BUY', 'SELL'
    quantity: int
    price: float
    timestamp: datetime
    strategy: str
    expiry: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None
    lot_size: int = 50
    commission: float = 0.0
    pnl: float = 0.0

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    total_pnl: float
    total_pnl_percent: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    margin_available: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

class IndianPortfolioSimulator:
    """Comprehensive portfolio simulator for Indian markets"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
        # Indian market specific parameters
        self.lot_sizes = {
            'NIFTY_50': 50,
            'BANK_NIFTY': 25,
            'SENSEX': 10,
            'NIFTY_IT': 25,
            'NIFTY_AUTO': 25,
            'NIFTY_PHARMA': 25
        }
        
        # Commission and charges (Indian market rates)
        self.commission_rate = 0.0003  # 0.03% per trade
        self.stt_rate = 0.0005  # 0.05% STT
        self.stamp_duty = 0.0001  # 0.01% stamp duty
        self.exchange_charges = 0.0001  # 0.01% exchange charges
        
        # Margin requirements
        self.margin_requirements = {
            'NIFTY_50': 0.15,  # 15% of notional value
            'BANK_NIFTY': 0.15,
            'SENSEX': 0.15,
            'NIFTY_IT': 0.15,
            'NIFTY_AUTO': 0.15,
            'NIFTY_PHARMA': 0.15
        }
        
        # Risk management parameters
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.max_daily_loss = 0.05  # 5% daily loss limit
        self.max_drawdown_limit = 0.20  # 20% maximum drawdown
        
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        
    def add_position(self, symbol: str, instrument_type: str, quantity: int, 
                    entry_price: float, strategy: str, expiry: Optional[str] = None,
                    strike: Optional[float] = None, option_type: Optional[str] = None) -> bool:
        """Add a new position to the portfolio"""
        try:
            # Get lot size
            lot_size = self.lot_sizes.get(symbol, 50)
            
            # Calculate notional value
            notional_value = quantity * lot_size * entry_price
            
            # Calculate margin required
            margin_rate = self.margin_requirements.get(symbol, 0.15)
            margin_required = notional_value * margin_rate
            
            # Check if sufficient capital available
            if margin_required > self.available_capital:
                logger.warning(f"Insufficient capital for {symbol} position. Required: {margin_required}, Available: {self.available_capital}")
                return False
            
            # Create position
            position_id = f"{symbol}_{instrument_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            position = Position(
                symbol=symbol,
                instrument_type=instrument_type,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                entry_time=datetime.now(),
                strategy=strategy,
                expiry=expiry,
                strike=strike,
                option_type=option_type,
                lot_size=lot_size,
                margin_used=margin_required,
                status='open'
            )
            
            # Add to portfolio
            self.positions[position_id] = position
            self.available_capital -= margin_required
            
            logger.info(f"Added position: {symbol} {quantity} lots at {entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def update_position_prices(self, price_data: Dict[str, float]) -> None:
        """Update current prices for all positions"""
        try:
            for position_id, position in self.positions.items():
                if position.status == 'open':
                    # Get current price for the position
                    if position.instrument_type == 'option':
                        # For options, we need to calculate based on underlying price and Greeks
                        underlying_price = price_data.get(position.symbol, position.current_price)
                        position.current_price = self._calculate_option_price(
                            underlying_price, position.strike, position.option_type, position.expiry
                        )
                    else:
                        position.current_price = price_data.get(position.symbol, position.current_price)
                    
                    # Calculate unrealized P&L
                    position.unrealized_pnl = self._calculate_position_pnl(position)
                    # Calculate unrealized P&L percentage
                    position.unrealized_pnl_percent = (position.unrealized_pnl / (position.entry_price * position.quantity * position.lot_size)) * 100 if position.entry_price > 0 else 0
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"Error updating position prices: {e}")
    
    def close_position(self, position_id: str, exit_price: float) -> bool:
        """Close a position and realize P&L"""
        try:
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} not found")
                return False
            
            position = self.positions[position_id]
            if position.status != 'open':
                logger.warning(f"Position {position_id} is not open")
                return False
            
            # Calculate P&L
            pnl = self._calculate_position_pnl(position, exit_price)
            
            # Calculate commission and charges
            notional_value = position.quantity * position.lot_size * exit_price
            commission = notional_value * (self.commission_rate + self.stt_rate + self.stamp_duty + self.exchange_charges)
            
            # Net P&L after charges
            net_pnl = pnl - commission
            
            # Update position
            position.status = 'closed'
            position.realized_pnl = net_pnl
            position.current_price = exit_price
            
            # Release margin
            self.available_capital += position.margin_used
            
            # Record trade
            trade = Trade(
                trade_id=f"CLOSE_{position_id}",
                symbol=position.symbol,
                instrument_type=position.instrument_type,
                action='SELL',
                quantity=position.quantity,
                price=exit_price,
                timestamp=datetime.now(),
                strategy=position.strategy,
                expiry=position.expiry,
                strike=position.strike,
                option_type=position.option_type,
                lot_size=position.lot_size,
                commission=commission,
                pnl=net_pnl
            )
            
            self.trades.append(trade)
            
            logger.info(f"Closed position {position_id}: P&L = {net_pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def _calculate_position_pnl(self, position: Position, exit_price: Optional[float] = None) -> float:
        """Calculate P&L for a position"""
        try:
            current_price = exit_price if exit_price else position.current_price
            
            if position.instrument_type == 'option':
                # For options, P&L calculation depends on option type
                if position.option_type == 'call':
                    pnl = max(0, current_price - position.strike) - position.entry_price
                else:  # put
                    pnl = max(0, position.strike - current_price) - position.entry_price
            else:
                # For futures/index, simple price difference
                pnl = (current_price - position.entry_price) / position.entry_price
            
            return pnl * position.quantity * position.lot_size
            
        except Exception as e:
            logger.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def _calculate_option_price(self, underlying_price: float, strike: float, 
                               option_type: str, expiry: str) -> float:
        """Calculate option price using simplified Black-Scholes"""
        try:
            # Simplified option pricing (in practice, use actual options pricing model)
            time_to_expiry = (pd.to_datetime(expiry) - datetime.now()).days / 365
            
            if time_to_expiry <= 0:
                # Expired option
                if option_type == 'call':
                    return max(0, underlying_price - strike)
                else:
                    return max(0, strike - underlying_price)
            
            # Simplified intrinsic value calculation
            if option_type == 'call':
                intrinsic_value = max(0, underlying_price - strike)
            else:
                intrinsic_value = max(0, strike - underlying_price)
            
            # Add some time value (simplified)
            time_value = max(10, intrinsic_value * 0.1 * time_to_expiry)
            
            return intrinsic_value + time_value
            
        except Exception as e:
            logger.error(f"Error calculating option price: {e}")
            return 0.0
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio performance metrics"""
        try:
            # Calculate total portfolio value
            total_value = self.available_capital
            total_unrealized_pnl = 0
            total_margin_used = 0
            
            for position in self.positions.values():
                if position.status == 'open':
                    total_value += position.margin_used
                    total_unrealized_pnl += position.unrealized_pnl
                    total_margin_used += position.margin_used
            
            # Calculate total P&L
            total_realized_pnl = sum(trade.pnl for trade in self.trades)
            total_pnl = total_realized_pnl + total_unrealized_pnl
            total_pnl_percent = (total_pnl / self.initial_capital) * 100
            
            # Update peak value and drawdown
            if total_value > self.peak_value:
                self.peak_value = total_value
            
            current_drawdown = (self.peak_value - total_value) / self.peak_value
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Calculate other metrics
            if self.daily_returns:
                sharpe_ratio = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252) if np.std(self.daily_returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Trade statistics
            winning_trades = [trade for trade in self.trades if trade.pnl > 0]
            losing_trades = [trade for trade in self.trades if trade.pnl < 0]
            
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            avg_win = np.mean([trade.pnl for trade in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([trade.pnl for trade in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(sum(trade.pnl for trade in winning_trades) / sum(trade.pnl for trade in losing_trades)) if losing_trades else float('inf')
            
            # Store metrics
            self.portfolio_metrics = PortfolioMetrics(
                total_value=total_value,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl=total_realized_pnl,
                margin_used=total_margin_used,
                margin_available=self.available_capital,
                max_drawdown=self.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(self.trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=max([trade.pnl for trade in winning_trades]) if winning_trades else 0,
                largest_loss=min([trade.pnl for trade in losing_trades]) if losing_trades else 0
            )
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            self._update_portfolio_metrics()
            
            # Position summary
            open_positions = [pos for pos in self.positions.values() if pos.status == 'open']
            closed_positions = [pos for pos in self.positions.values() if pos.status == 'closed']
            
            # Strategy breakdown
            strategy_pnl = {}
            for trade in self.trades:
                strategy = trade.strategy
                if strategy not in strategy_pnl:
                    strategy_pnl[strategy] = 0
                strategy_pnl[strategy] += trade.pnl
            
            # Symbol breakdown
            symbol_pnl = {}
            for trade in self.trades:
                symbol = trade.symbol
                if symbol not in symbol_pnl:
                    symbol_pnl[symbol] = 0
                symbol_pnl[symbol] += trade.pnl
            
            return {
                'total_value': self.portfolio_metrics.total_value,
                'total_pnl': self.portfolio_metrics.total_pnl,
                'total_pnl_percent': self.portfolio_metrics.total_pnl_percent,
                'unrealized_pnl': self.portfolio_metrics.unrealized_pnl,
                'unrealized_pnl_percent': (self.portfolio_metrics.unrealized_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0,
                'portfolio_metrics': self.portfolio_metrics.__dict__,
                'positions': {
                    'total': len(self.positions),
                    'open': len(open_positions),
                    'closed': len(closed_positions)
                },
                'strategy_performance': strategy_pnl,
                'symbol_performance': symbol_pnl,
                'risk_metrics': {
                    'max_position_size': self.max_position_size,
                    'max_daily_loss': self.max_daily_loss,
                    'max_drawdown_limit': self.max_drawdown_limit,
                    'current_drawdown': self.max_drawdown
                },
                'capital_utilization': {
                    'initial_capital': self.initial_capital,
                    'available_capital': self.available_capital,
                    'margin_used': self.portfolio_metrics.margin_used,
                    'utilization_percent': (self.portfolio_metrics.margin_used / self.initial_capital) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_portfolio_history(self) -> Dict[str, Any]:
        """Get portfolio performance history for charts"""
        try:
            # Initialize portfolio history if it doesn't exist
            if not hasattr(self, 'portfolio_history'):
                self.portfolio_history = []
            
            # Get current portfolio metrics
            self._update_portfolio_metrics()
            
            # Create current history entry
            current_entry = {
                'timestamp': datetime.now().isoformat(),
                'total_value': self.portfolio_metrics.total_value,
                'total_pnl': self.portfolio_metrics.total_pnl,
                'total_pnl_percent': self.portfolio_metrics.total_pnl_percent,
                'unrealized_pnl': self.portfolio_metrics.unrealized_pnl,
                'realized_pnl': self.portfolio_metrics.realized_pnl,
                'margin_used': self.portfolio_metrics.margin_used,
                'available_capital': self.available_capital,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self.portfolio_metrics.sharpe_ratio,
                'win_rate': self.portfolio_metrics.win_rate,
                'total_trades': len(self.trades),
                'open_positions': len([pos for pos in self.positions.values() if pos.status == 'open'])
            }
            
            # Add to history
            self.portfolio_history.append(current_entry)
            
            # Keep only last 100 entries to prevent memory issues
            if len(self.portfolio_history) > 100:
                self.portfolio_history = self.portfolio_history[-100:]
            
            # Return formatted history for charts
            return {
                'history': self.portfolio_history,
                'current_metrics': self.portfolio_metrics.__dict__,
                'summary': {
                    'total_value': self.portfolio_metrics.total_value,
                    'total_pnl': self.portfolio_metrics.total_pnl,
                    'total_pnl_percent': self.portfolio_metrics.total_pnl_percent,
                    'max_drawdown': self.max_drawdown,
                    'sharpe_ratio': self.portfolio_metrics.sharpe_ratio,
                    'win_rate': self.portfolio_metrics.win_rate,
                    'total_trades': len(self.trades),
                    'open_positions': len([pos for pos in self.positions.values() if pos.status == 'open'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return {
                'history': [],
                'current_metrics': {},
                'summary': {
                    'total_value': self.initial_capital,
                    'total_pnl': 0,
                    'total_pnl_percent': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'open_positions': 0
                }
            }
    
    def export_portfolio_data(self, filepath: str) -> bool:
        """Export portfolio data to JSON file"""
        try:
            portfolio_data = {
                'initial_capital': self.initial_capital,
                'available_capital': self.available_capital,
                'positions': {pid: pos.__dict__ for pid, pos in self.positions.items()},
                'trades': [trade.__dict__ for trade in self.trades],
                'portfolio_history': self.portfolio_history,
                'metrics': self.portfolio_metrics.__dict__ if hasattr(self, 'portfolio_metrics') else {}
            }
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            portfolio_data = convert_datetime(portfolio_data)
            
            with open(filepath, 'w') as f:
                json.dump(portfolio_data, f, indent=2)
            
            logger.info(f"Portfolio data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting portfolio data: {e}")
            return False
    
    def import_portfolio_data(self, filepath: str) -> bool:
        """Import portfolio data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                portfolio_data = json.load(f)
            
            # Restore basic attributes
            self.initial_capital = portfolio_data.get('initial_capital', 1000000)
            self.available_capital = portfolio_data.get('available_capital', self.initial_capital)
            
            # Restore positions
            self.positions = {}
            for pid, pos_data in portfolio_data.get('positions', {}).items():
                # Convert datetime strings back to datetime objects
                if 'entry_time' in pos_data:
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                
                position = Position(**pos_data)
                self.positions[pid] = position
            
            # Restore trades
            self.trades = []
            for trade_data in portfolio_data.get('trades', []):
                if 'timestamp' in trade_data:
                    trade_data['timestamp'] = datetime.fromisoformat(trade_data['timestamp'])
                
                trade = Trade(**trade_data)
                self.trades.append(trade)
            
            # Restore portfolio history
            self.portfolio_history = portfolio_data.get('portfolio_history', [])
            
            logger.info(f"Portfolio data imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing portfolio data: {e}")
            return False
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """Check if portfolio is within risk limits"""
        try:
            self._update_portfolio_metrics()
            
            risk_alerts = []
            
            # Check drawdown limit
            if self.max_drawdown > self.max_drawdown_limit:
                risk_alerts.append({
                    'type': 'max_drawdown',
                    'message': f'Maximum drawdown {self.max_drawdown:.2%} exceeds limit {self.max_drawdown_limit:.2%}',
                    'severity': 'high'
                })
            
            # Check daily loss limit
            if self.daily_returns and self.daily_returns[-1] < -self.max_daily_loss:
                risk_alerts.append({
                    'type': 'daily_loss',
                    'message': f'Daily loss {abs(self.daily_returns[-1]):.2%} exceeds limit {self.max_daily_loss:.2%}',
                    'severity': 'high'
                })
            
            # Check position size limits
            for position in self.positions.values():
                if position.status == 'open':
                    position_value = position.quantity * position.lot_size * position.current_price
                    position_percent = position_value / self.initial_capital
                    
                    if position_percent > self.max_position_size:
                        risk_alerts.append({
                            'type': 'position_size',
                            'message': f'Position {position.symbol} size {position_percent:.2%} exceeds limit {self.max_position_size:.2%}',
                            'severity': 'medium'
                        })
            
            # Check margin utilization
            margin_utilization = self.portfolio_metrics.margin_used / self.initial_capital
            if margin_utilization > 0.8:  # 80% margin utilization
                risk_alerts.append({
                    'type': 'margin_utilization',
                    'message': f'Margin utilization {margin_utilization:.2%} is high',
                    'severity': 'medium'
                })
            
            return {
                'risk_alerts': risk_alerts,
                'risk_score': len([alert for alert in risk_alerts if alert['severity'] == 'high']) * 10 + 
                             len([alert for alert in risk_alerts if alert['severity'] == 'medium']) * 5,
                'within_limits': len(risk_alerts) == 0
            }
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {'risk_alerts': [], 'risk_score': 0, 'within_limits': True}
    
    def update_all_positions_with_live_data(self, data_fetcher) -> Dict[str, Any]:
        """Update all positions with live market data for paper trading"""
        try:
            updated_positions = []
            total_unrealized_pnl = 0
            total_investment = 0
            
            for position_id, position in self.positions.items():
                if position.status == 'open':
                    try:
                        # Get current market price based on symbol
                        symbol_mapping = {
                            'NIFTY_50': 'NIFTY_50',
                            'BANK_NIFTY': 'BANK_NIFTY', 
                            'SENSEX': 'SENSEX'
                        }
                        
                        symbol_key = symbol_mapping.get(position.symbol, position.symbol)
                        current_data = data_fetcher.fetch_index_data(symbol_key, "1d", "1m")
                        
                        if not current_data.empty:
                            current_price = current_data['Close'].iloc[-1]
                            position.current_price = current_price
                            
                            # Calculate P&L
                            if position.position_type == 'Long':
                                pnl = (current_price - position.entry_price) * position.quantity * position.lot_size
                            else:  # Short
                                pnl = (position.entry_price - current_price) * position.quantity * position.lot_size
                            
                            position.unrealized_pnl = pnl
                            position.unrealized_pnl_percent = (pnl / (position.entry_price * position.quantity * position.lot_size)) * 100 if position.entry_price > 0 else 0
                            
                            total_unrealized_pnl += pnl
                            total_investment += position.entry_price * position.quantity * position.lot_size
                            
                            updated_positions.append({
                                'position_id': position_id,
                                'symbol': position.symbol,
                                'quantity': position.quantity,
                                'entry_price': position.entry_price,
                                'current_price': current_price,
                                'position_type': position.position_type,
                                'unrealized_pnl': pnl,
                                'unrealized_pnl_percent': position.unrealized_pnl_percent,
                                'lot_size': position.lot_size,
                                'timestamp': position.timestamp,
                                'status': position.status
                            })
                        else:
                            # Use last known price if no new data
                            updated_positions.append({
                                'position_id': position_id,
                                'symbol': position.symbol,
                                'quantity': position.quantity,
                                'entry_price': position.entry_price,
                                'current_price': position.current_price,
                                'position_type': position.position_type,
                                'unrealized_pnl': position.unrealized_pnl,
                                'unrealized_pnl_percent': position.unrealized_pnl_percent,
                                'lot_size': position.lot_size,
                                'timestamp': position.timestamp,
                                'status': position.status
                            })
                            
                    except Exception as e:
                        logger.error(f"Error updating position {position_id}: {e}")
                        # Keep existing position data
                        updated_positions.append({
                            'position_id': position_id,
                            'symbol': position.symbol,
                            'quantity': position.quantity,
                            'entry_price': position.entry_price,
                            'current_price': position.current_price,
                            'position_type': position.position_type,
                            'unrealized_pnl': position.unrealized_pnl,
                            'unrealized_pnl_percent': position.unrealized_pnl_percent,
                            'lot_size': position.lot_size,
                            'timestamp': position.timestamp,
                            'status': position.status
                        })
            
            # Update portfolio metrics
            self.portfolio_metrics.total_pnl = total_unrealized_pnl
            self.portfolio_metrics.total_pnl_percent = (total_unrealized_pnl / total_investment) * 100 if total_investment > 0 else 0
            self.portfolio_metrics.total_value = self.initial_capital + total_unrealized_pnl
            
            return {
                'positions': updated_positions,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_investment': total_investment,
                'total_value': self.portfolio_metrics.total_value,
                'total_pnl_percent': self.portfolio_metrics.total_pnl_percent,
                'updated_at': datetime.now(),
                'open_positions_count': len([p for p in updated_positions if p['status'] == 'open'])
            }
            
        except Exception as e:
            logger.error(f"Error updating positions with live data: {e}")
            return {'positions': [], 'error': str(e)}
    
    def close_position(self, position_id: str, exit_price: float = None) -> Dict[str, Any]:
        """Close a position and calculate realized P&L"""
        try:
            if position_id not in self.positions:
                return {'success': False, 'error': 'Position not found'}
            
            position = self.positions[position_id]
            
            if position.status != 'open':
                return {'success': False, 'error': 'Position is not open'}
            
            # Use current price if exit price not provided
            if exit_price is None:
                exit_price = position.current_price
            
            # Calculate realized P&L
            if position.position_type == 'Long':
                realized_pnl = (exit_price - position.entry_price) * position.quantity * position.lot_size
            else:  # Short
                realized_pnl = (position.entry_price - exit_price) * position.quantity * position.lot_size
            
            # Update position
            position.status = 'closed'
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            position.realized_pnl = realized_pnl
            
            # Update portfolio metrics
            self.portfolio_metrics.total_pnl += realized_pnl
            self.portfolio_metrics.available_capital += realized_pnl
            
            return {
                'success': True,
                'position_id': position_id,
                'realized_pnl': realized_pnl,
                'exit_price': exit_price,
                'exit_time': position.exit_time
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create portfolio simulator
    simulator = IndianPortfolioSimulator(initial_capital=1000000)
    
    print("Testing Indian Portfolio Simulator...")
    
    # Add some sample positions
    simulator.add_position('NIFTY_50', 'option', 1, 150, 'bull_call_spread', 
                          expiry='2024-12-26', strike=24000, option_type='call')
    simulator.add_position('BANK_NIFTY', 'option', 1, 200, 'bear_put_spread',
                          expiry='2024-12-26', strike=50000, option_type='put')
    
    # Update prices
    price_data = {'NIFTY_50': 24100, 'BANK_NIFTY': 49800}
    simulator.update_position_prices(price_data)
    
    # Get portfolio summary
    summary = simulator.get_portfolio_summary()
    print(f"Portfolio Summary: {summary}")
    
    print("Indian Portfolio Simulator module loaded successfully")
