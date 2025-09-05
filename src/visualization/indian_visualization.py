#!/usr/bin/env python3
"""
Indian Market Visualization

This module provides comprehensive visualization capabilities specifically designed
for Indian market data including Nifty 50, Bank Nifty, and Sensex with
market-specific charts, indicators, and analysis displays.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class IndianMarketVisualizer:
    """Comprehensive visualization class for Indian market analysis"""
    
    def __init__(self):
        # Indian market color scheme
        self.color_scheme = {
            'nifty_50': '#FF6B35',
            'bank_nifty': '#004E89',
            'sensex': '#00A8CC',
            'bullish': '#00C851',
            'bearish': '#FF4444',
            'neutral': '#FF8800',
            'background': '#F8F9FA',
            'grid': '#E9ECEF',
            'text': '#2C3E50',
            'success': '#28A745',
            'warning': '#FFC107',
            'danger': '#DC3545',
            'info': '#17A2B8'
        }
        
        # Indian market specific settings
        self.market_symbols = {
            'NIFTY_50': {'name': 'Nifty 50', 'color': self.color_scheme['nifty_50']},
            'BANK_NIFTY': {'name': 'Bank Nifty', 'color': self.color_scheme['bank_nifty']},
            'SENSEX': {'name': 'Sensex', 'color': self.color_scheme['sensex']},
            'NIFTY_IT': {'name': 'Nifty IT', 'color': '#6F42C1'},
            'NIFTY_AUTO': {'name': 'Nifty Auto', 'color': '#E83E8C'},
            'NIFTY_PHARMA': {'name': 'Nifty Pharma', 'color': '#20C997'}
        }
    
    def create_indian_market_dashboard(self, market_data: Dict[str, pd.DataFrame], 
                                     analysis_data: Dict[str, Any]) -> go.Figure:
        """Create comprehensive Indian market dashboard"""
        try:
            # Create subplots for dashboard
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Nifty 50 Price Chart', 'Bank Nifty Price Chart',
                    'Market Overview', 'Sector Performance',
                    'Options Analysis', 'Portfolio Performance'
                ),
                specs=[
                    [{"secondary_y": True}, {"secondary_y": True}],
                    [{"type": "indicator"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )
            
            # Add Nifty 50 chart
            if 'NIFTY_50' in market_data:
                self._add_price_chart(fig, market_data['NIFTY_50'], 1, 1, 'NIFTY_50')
            
            # Add Bank Nifty chart
            if 'BANK_NIFTY' in market_data:
                self._add_price_chart(fig, market_data['BANK_NIFTY'], 1, 2, 'BANK_NIFTY')
            
            # Add market overview indicators
            self._add_market_overview(fig, analysis_data.get('market_overview', {}), 2, 1)
            
            # Add sector performance
            self._add_sector_performance(fig, analysis_data.get('sector_performance', {}), 2, 2)
            
            # Add options analysis
            self._add_options_analysis(fig, analysis_data.get('options_analysis', {}), 3, 1)
            
            # Add portfolio performance
            self._add_portfolio_performance(fig, analysis_data.get('portfolio_metrics', {}), 3, 2)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Indian Market Analysis Dashboard',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': self.color_scheme['text']}
                },
                height=1200,
                showlegend=True,
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor='white',
                font=dict(family="Arial", size=12, color=self.color_scheme['text'])
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Indian market dashboard: {e}")
            return go.Figure()
    
    def _add_price_chart(self, fig: go.Figure, data: pd.DataFrame, row: int, col: int, symbol: str) -> None:
        """Add price chart for Indian market index"""
        try:
            symbol_info = self.market_symbols.get(symbol, {'name': symbol, 'color': '#000000'})
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=f"{symbol_info['name']} Price",
                    increasing_line_color=self.color_scheme['bullish'],
                    decreasing_line_color=self.color_scheme['bearish']
                ),
                row=row, col=col
            )
            
            # Add moving averages
            if len(data) >= 20:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'].rolling(20).mean(),
                        name='SMA 20',
                        line=dict(color='orange', width=2)
                    ),
                    row=row, col=col
                )
            
            if len(data) >= 50:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'].rolling(50).mean(),
                        name='SMA 50',
                        line=dict(color='blue', width=2)
                    ),
                    row=row, col=col
                )
            
            # Add volume
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.6
                ),
                row=row, col=col, secondary_y=True
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Price", row=row, col=col)
            fig.update_yaxes(title_text="Volume", secondary_y=True, row=row, col=col)
            
        except Exception as e:
            logger.error(f"Error adding price chart for {symbol}: {e}")
    
    def _add_market_overview(self, fig: go.Figure, overview_data: Dict[str, Any], row: int, col: int) -> None:
        """Add market overview indicators"""
        try:
            # Create gauge charts for key metrics
            metrics = [
                ('Nifty 50', overview_data.get('NIFTY_50', {}).get('change_percent', 0)),
                ('Bank Nifty', overview_data.get('BANK_NIFTY', {}).get('change_percent', 0)),
                ('Sensex', overview_data.get('SENSEX', {}).get('change_percent', 0))
            ]
            
            for i, (name, value) in enumerate(metrics):
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=value,
                        domain={'x': [0, 0.33], 'y': [0.5 + i*0.15, 0.65 + i*0.15]},
                        title={'text': name},
                        gauge={
                            'axis': {'range': [-5, 5]},
                            'bar': {'color': self.color_scheme['bullish'] if value >= 0 else self.color_scheme['bearish']},
                            'steps': [
                                {'range': [-5, -2], 'color': self.color_scheme['bearish']},
                                {'range': [-2, 2], 'color': self.color_scheme['neutral']},
                                {'range': [2, 5], 'color': self.color_scheme['bullish']}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0
                            }
                        }
                    ),
                    row=row, col=col
                )
            
        except Exception as e:
            logger.error(f"Error adding market overview: {e}")
    
    def _add_sector_performance(self, fig: go.Figure, sector_data: Dict[str, Any], row: int, col: int) -> None:
        """Add sector performance bar chart"""
        try:
            sectors = list(sector_data.keys())
            performance = [sector_data[sector].get('change_percent', 0) for sector in sectors]
            
            colors = [self.color_scheme['bullish'] if p >= 0 else self.color_scheme['bearish'] for p in performance]
            
            fig.add_trace(
                go.Bar(
                    x=sectors,
                    y=performance,
                    name='Sector Performance',
                    marker_color=colors,
                    text=[f"{p:.2f}%" for p in performance],
                    textposition='auto'
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Sectors", row=row, col=col)
            fig.update_yaxes(title_text="Change %", row=row, col=col)
            
        except Exception as e:
            logger.error(f"Error adding sector performance: {e}")
    
    def _add_options_analysis(self, fig: go.Figure, options_data: Dict[str, Any], row: int, col: int) -> None:
        """Add options analysis visualization"""
        try:
            # IV Skew chart
            if 'iv_skew' in options_data:
                skew_data = options_data['iv_skew']
                strikes = list(skew_data.keys())
                ivs = list(skew_data.values())
                
                fig.add_trace(
                    go.Scatter(
                        x=strikes,
                        y=ivs,
                        mode='lines+markers',
                        name='IV Skew',
                        line=dict(color=self.color_scheme['info'], width=3),
                        marker=dict(size=8)
                    ),
                    row=row, col=col
                )
            
            # Put-Call Ratio
            if 'put_call_ratio' in options_data:
                pcr = options_data['put_call_ratio']
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1, 2, 3],
                        y=[pcr, pcr, pcr, pcr],
                        mode='lines',
                        name=f'PCR: {pcr:.2f}',
                        line=dict(color=self.color_scheme['warning'], width=2, dash='dash')
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="Strike Price", row=row, col=col)
            fig.update_yaxes(title_text="Implied Volatility", row=row, col=col)
            
        except Exception as e:
            logger.error(f"Error adding options analysis: {e}")
    
    def _add_portfolio_performance(self, fig: go.Figure, portfolio_data: Dict[str, Any], row: int, col: int) -> None:
        """Add portfolio performance visualization"""
        try:
            # Portfolio value over time
            if 'portfolio_history' in portfolio_data:
                history = portfolio_data['portfolio_history']
                dates = [h['date'] for h in history]
                values = [h['value'] for h in history]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color=self.color_scheme['success'], width=3)
                    ),
                    row=row, col=col
                )
            
            # P&L distribution
            if 'pnl_distribution' in portfolio_data:
                pnl_data = portfolio_data['pnl_distribution']
                fig.add_trace(
                    go.Histogram(
                        x=pnl_data,
                        name='P&L Distribution',
                        marker_color=self.color_scheme['info'],
                        opacity=0.7
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Portfolio Value", row=row, col=col)
            
        except Exception as e:
            logger.error(f"Error adding portfolio performance: {e}")
    
    def create_market_overview_chart(self, overview_data: Dict[str, Any]) -> go.Figure:
        """Create market overview chart showing key indices performance"""
        try:
            if not overview_data:
                return None
            
            # Prepare data for the chart
            indices = []
            prices = []
            changes = []
            colors = []
            
            for symbol, data in overview_data.items():
                symbol_info = self.market_symbols.get(symbol, {'name': symbol, 'color': '#1f77b4'})
                indices.append(symbol_info['name'])
                prices.append(data.get('current_price', 0))
                changes.append(data.get('change_percent', 0))
                
                # Color based on performance
                if data.get('change_percent', 0) > 0:
                    colors.append('#00ff00')  # Green for positive
                elif data.get('change_percent', 0) < 0:
                    colors.append('#ff0000')  # Red for negative
                else:
                    colors.append('#808080')  # Gray for neutral
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=indices,
                    y=prices,
                    marker_color=colors,
                    text=[f"₹{price:,.0f}<br>({change:+.2f}%)" for price, change in zip(prices, changes)],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.0f}<br>Change: %{customdata:+.2f}%<extra></extra>',
                    customdata=changes
                )
            ])
            
            fig.update_layout(
                title={
                    'text': 'Indian Market Overview - Key Indices',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                xaxis_title='Indices',
                yaxis_title='Price (₹)',
                showlegend=False,
                height=400,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Add horizontal line for reference
            if prices:
                avg_price = sum(prices) / len(prices)
                fig.add_hline(
                    y=avg_price,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Avg: ₹{avg_price:,.0f}",
                    annotation_position="top right"
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating market overview chart: {e}")
            return None
    
    def create_indian_options_strategy_chart(self, strategy_data: Dict[str, Any]) -> go.Figure:
        """Create options strategy payoff diagram for Indian markets"""
        try:
            fig = go.Figure()
            
            strategy_name = strategy_data.get('name', 'Options Strategy')
            legs = strategy_data.get('legs', [])
            
            # Calculate payoff at different underlying prices
            underlying_prices = np.arange(
                strategy_data.get('min_price', 20000),
                strategy_data.get('max_price', 30000),
                50
            )
            
            payoffs = []
            for price in underlying_prices:
                payoff = self._calculate_strategy_payoff(price, legs)
                payoffs.append(payoff)
            
            # Plot payoff diagram
            fig.add_trace(
                go.Scatter(
                    x=underlying_prices,
                    y=payoffs,
                    mode='lines',
                    name=strategy_name,
                    line=dict(color=self.color_scheme['info'], width=3),
                    fill='tonexty'
                )
            )
            
            # Add breakeven points
            breakeven_points = strategy_data.get('breakeven_points', [])
            for be_point in breakeven_points:
                # Convert numpy float64 to regular float for display
                be_value = float(be_point)
                fig.add_vline(
                    x=be_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"BE: ₹{be_value:,.2f}"
                )
            
            # Add current price line
            current_price = strategy_data.get('current_price', 25000)
            fig.add_vline(
                x=current_price,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Current: {current_price}"
            )
            
            # Update layout
            fig.update_layout(
                title=f"{strategy_name} Payoff Diagram",
                xaxis_title="Underlying Price",
                yaxis_title="Profit/Loss",
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor='white',
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating options strategy chart: {e}")
            return go.Figure()
    
    def _calculate_strategy_payoff(self, underlying_price: float, legs: List[Dict[str, Any]]) -> float:
        """Calculate strategy payoff at given underlying price"""
        try:
            total_payoff = 0
            
            for leg in legs:
                action = leg.get('action', 'BUY')
                option_type = leg.get('option_type', 'call')
                strike = leg.get('strike', 0)
                premium = leg.get('premium', 0)
                quantity = leg.get('quantity', 1)
                
                # Calculate option payoff
                if option_type == 'call':
                    option_payoff = max(0, underlying_price - strike) - premium
                else:  # put
                    option_payoff = max(0, strike - underlying_price) - premium
                
                # Apply action (buy/sell)
                if action == 'SELL':
                    option_payoff = -option_payoff
                
                total_payoff += option_payoff * quantity
            
            return total_payoff
            
        except Exception as e:
            logger.error(f"Error calculating strategy payoff: {e}")
            return 0.0
    
    def create_indian_technical_analysis_chart(self, data: pd.DataFrame, indicators: Dict[str, Any], 
                                             symbol: str) -> go.Figure:
        """Create comprehensive technical analysis chart for Indian markets"""
        try:
            # Validate input data
            if data is None or data.empty:
                logger.error("Data is None or empty")
                return self._create_error_chart("No data available for chart creation")
            
            if indicators is None:
                logger.error("Indicators is None")
                return self._create_error_chart("No indicators data available")
            
            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return self._create_error_chart(f"Missing required columns: {missing_columns}")
            
            symbol_info = self.market_symbols.get(symbol, {'name': symbol, 'color': '#000000'})
            
            # Debug logging
            logger.info(f"Creating chart for {symbol} with {len(data)} data points")
            logger.info(f"Available indicators: {list(indicators.keys()) if indicators else 'None'}")
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f'{symbol_info["name"]} Price & Indicators',
                    'RSI',
                    'MACD',
                    'Volume'
                ),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Main price chart with proper date/time formatting
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=self.color_scheme['bullish'],
                    decreasing_line_color=self.color_scheme['bearish']
                ),
                row=1, col=1
            )
            
            # Add moving averages with enhanced hover information
            if 'ema_12' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['ema_12'],
                        name='EMA 12',
                        line=dict(color='orange', width=2),
                        hovertemplate='<b>%{x}</b><br>EMA 12: ₹%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if 'ema_26' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['ema_26'],
                        name='EMA 26',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>%{x}</b><br>EMA 26: ₹%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Add SMA 50 if available
            if 'sma_50' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['sma_50'],
                        name='SMA 50',
                        line=dict(color='purple', width=2, dash='dash'),
                        hovertemplate='<b>%{x}</b><br>SMA 50: ₹%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if all(key in indicators for key in ['bb_upper', 'bb_middle', 'bb_lower']):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['bb_upper'],
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['bb_lower'],
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty'
                    ),
                    row=1, col=1
                )
            
            # VWAP Price Channel - Enhanced with professional styling
            if all(key in indicators for key in ['vwap_upper', 'vwap_middle', 'vwap_lower']):
                # VWAP Upper Band - Red line like in the reference chart
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['vwap_upper'],
                        name='VWAP Upper',
                        line=dict(color='#FF4444', width=2.5, shape='spline'),
                        opacity=0.8,
                        hovertemplate='<b>VWAP Upper</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # VWAP Middle Line - Dynamic color based on trend
                vwap_middle = indicators['vwap_middle']
                current_price = data['Close'].iloc[-1]
                vwap_color = '#00C851' if current_price > vwap_middle.iloc[-1] else '#FF4444'
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=vwap_middle,
                        name='VWAP',
                        line=dict(color=vwap_color, width=3, shape='spline'),
                        opacity=0.9,
                        hovertemplate='<b>VWAP</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # VWAP Lower Band - Green line with fill
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['vwap_lower'],
                        name='VWAP Lower',
                        line=dict(color='#00C851', width=2.5, shape='spline'),
                        opacity=0.8,
                        fill='tonexty',
                        fillcolor='rgba(0, 200, 81, 0.1)',
                        hovertemplate='<b>VWAP Lower</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # RSI with enhanced formatting
            if 'rsi' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['rsi'],
                        name='RSI',
                        line=dict(color=self.color_scheme['info'], width=2),
                        hovertemplate='<b>%{x}</b><br>RSI: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Add RSI overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought (70)", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold (30)", row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                             annotation_text="Neutral (50)", row=2, col=1)
            
            # MACD with enhanced formatting
            if all(key in indicators for key in ['macd', 'macd_signal', 'macd_hist']):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['macd'],
                        name='MACD',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>%{x}</b><br>MACD: %{y:.4f}<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['macd_signal'],
                        name='Signal',
                        line=dict(color='red', width=2),
                        hovertemplate='<b>%{x}</b><br>Signal: %{y:.4f}<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                # MACD Histogram with enhanced formatting
                colors = ['green' if val >= 0 else 'red' for val in indicators['macd_hist']]
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=indicators['macd_hist'],
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.6,
                        hovertemplate='<b>%{x}</b><br>Histogram: %{y:.4f}<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                # Add zero line for MACD
                fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
            
            # Volume with enhanced formatting
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.6,
                    hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>'
                ),
                row=4, col=1
            )
            
            # Get current timestamp and data range
            current_time = datetime.now()
            data_start = data.index[0] if not data.empty else current_time
            data_end = data.index[-1] if not data.empty else current_time
            
            # Update layout with comprehensive date/time formatting and timestamps
            fig.update_layout(
                title=f'{symbol_info["name"]} Technical Analysis<br><sub>Last Updated: {current_time.strftime("%Y-%m-%d %H:%M:%S IST")} | Data Range: {data_start.strftime("%Y-%m-%d")} to {data_end.strftime("%Y-%m-%d")}</sub>',
                height=1200,
                showlegend=True,
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor='white',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                annotations=[
                    dict(
                        text=f"Data Points: {len(data)} | Generated: {current_time.strftime('%H:%M:%S')}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="gray",
                        borderwidth=1
                    )
                ]
            )
            
            # Update axes with proper date/time formatting
            fig.update_xaxes(
                title_text="Date & Time",
                row=4, col=1,
                tickformat="%Y-%m-%d<br>%H:%M",
                tickangle=45,
                showgrid=True,
                gridcolor=self.color_scheme['grid']
            )
            
            # Update all x-axes to show proper date/time
            for i in range(1, 5):
                fig.update_xaxes(
                    tickformat="%Y-%m-%d<br>%H:%M",
                    tickangle=45,
                    showgrid=True,
                    gridcolor=self.color_scheme['grid'],
                    row=i, col=1
                )
            
            # Update y-axes with proper formatting
            fig.update_yaxes(
                title_text="Price (₹)",
                row=1, col=1,
                tickformat="₹.2f",
                showgrid=True,
                gridcolor=self.color_scheme['grid']
            )
            fig.update_yaxes(
                title_text="RSI",
                row=2, col=1,
                range=[0, 100],
                showgrid=True,
                gridcolor=self.color_scheme['grid']
            )
            fig.update_yaxes(
                title_text="MACD",
                row=3, col=1,
                showgrid=True,
                gridcolor=self.color_scheme['grid']
            )
            fig.update_yaxes(
                title_text="Volume",
                row=4, col=1,
                tickformat=".2s",
                showgrid=True,
                gridcolor=self.color_scheme['grid']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating technical analysis chart: {e}")
            return go.Figure()
    
    def create_indian_market_heatmap(self, sector_data: Dict[str, Any]) -> go.Figure:
        """Create sector performance heatmap for Indian markets"""
        try:
            # Prepare data for heatmap
            sectors = list(sector_data.keys())
            performance = [sector_data[sector].get('change_percent', 0) for sector in sectors]
            
            # Create color scale
            colors = []
            for p in performance:
                if p > 2:
                    colors.append(self.color_scheme['success'])
                elif p > 0:
                    colors.append(self.color_scheme['bullish'])
                elif p > -2:
                    colors.append(self.color_scheme['neutral'])
                else:
                    colors.append(self.color_scheme['bearish'])
            
            fig = go.Figure(data=go.Bar(
                x=sectors,
                y=performance,
                marker_color=colors,
                text=[f"{p:.2f}%" for p in performance],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Indian Market Sector Performance',
                xaxis_title='Sectors',
                yaxis_title='Change %',
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor='white',
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating market heatmap: {e}")
            return go.Figure()
    
    def create_indian_portfolio_performance_chart(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Create portfolio performance chart for Indian markets"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Portfolio Value Over Time',
                    'P&L Distribution',
                    'Strategy Performance',
                    'Risk Metrics'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "histogram"}],
                    [{"type": "bar"}, {"type": "indicator"}]
                ]
            )
            
            # Portfolio value over time
            if 'portfolio_history' in portfolio_data:
                history = portfolio_data['portfolio_history']
                dates = [h['date'] for h in history]
                values = [h['value'] for h in history]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color=self.color_scheme['success'], width=3)
                    ),
                    row=1, col=1
                )
            
            # P&L distribution
            if 'pnl_distribution' in portfolio_data:
                pnl_data = portfolio_data['pnl_distribution']
                fig.add_trace(
                    go.Histogram(
                        x=pnl_data,
                        name='P&L Distribution',
                        marker_color=self.color_scheme['info'],
                        opacity=0.7
                    ),
                    row=1, col=2
                )
            
            # Strategy performance
            if 'strategy_performance' in portfolio_data:
                strategies = list(portfolio_data['strategy_performance'].keys())
                pnl_values = list(portfolio_data['strategy_performance'].values())
                
                colors = [self.color_scheme['bullish'] if p >= 0 else self.color_scheme['bearish'] for p in pnl_values]
                
                fig.add_trace(
                    go.Bar(
                        x=strategies,
                        y=pnl_values,
                        name='Strategy P&L',
                        marker_color=colors
                    ),
                    row=2, col=1
                )
            
            # Risk metrics
            if 'risk_metrics' in portfolio_data:
                risk_data = portfolio_data['risk_metrics']
                sharpe_ratio = risk_data.get('sharpe_ratio', 0)
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=sharpe_ratio,
                        title={'text': 'Sharpe Ratio'},
                        gauge={
                            'axis': {'range': [-2, 2]},
                            'bar': {'color': self.color_scheme['success'] if sharpe_ratio > 1 else self.color_scheme['warning']},
                            'steps': [
                                {'range': [-2, 0], 'color': self.color_scheme['bearish']},
                                {'range': [0, 1], 'color': self.color_scheme['neutral']},
                                {'range': [1, 2], 'color': self.color_scheme['bullish']}
                            ]
                        }
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Indian Market Portfolio Performance',
                height=800,
                showlegend=True,
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating portfolio performance chart: {e}")
            return go.Figure()
    
    def create_live_prediction_chart(self, prediction_data: Dict[str, Any]) -> go.Figure:
        """Create live prediction chart with price forecasts"""
        try:
            current_price = prediction_data.get('current_price', 0)
            pred_5m = prediction_data.get('predicted_price_5m', current_price)
            pred_10m = prediction_data.get('predicted_price_10m', current_price)
            confidence_5m = prediction_data.get('confidence_5m', 0.5)
            confidence_10m = prediction_data.get('confidence_10m', 0.5)
            timestamp = prediction_data.get('timestamp', datetime.now())
            
            # Create time points
            times = [timestamp, timestamp + timedelta(minutes=5), timestamp + timedelta(minutes=10)]
            prices = [current_price, pred_5m, pred_10m]
            
            # Create main prediction line
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=times,
                y=prices,
                mode='lines+markers',
                name='Price Prediction',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            # Add confidence bands
            if confidence_5m > 0:
                uncertainty_5m = current_price * 0.01 * (1 - confidence_5m)  # 1% uncertainty
                fig.add_trace(go.Scatter(
                    x=[times[0], times[1]],
                    y=[current_price + uncertainty_5m, pred_5m + uncertainty_5m],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[times[0], times[1]],
                    y=[current_price - uncertainty_5m, pred_5m - uncertainty_5m],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    name=f'5m Confidence ({confidence_5m:.1%})',
                    hoverinfo='skip'
                ))
            
            if confidence_10m > 0:
                uncertainty_10m = current_price * 0.015 * (1 - confidence_10m)  # 1.5% uncertainty
                fig.add_trace(go.Scatter(
                    x=[times[1], times[2]],
                    y=[pred_5m + uncertainty_10m, pred_10m + uncertainty_10m],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[times[1], times[2]],
                    y=[pred_5m - uncertainty_10m, pred_10m - uncertainty_10m],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name=f'10m Confidence ({confidence_10m:.1%})',
                    hoverinfo='skip'
                ))
            
            # Add current price line
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Current: ₹{current_price:,.2f}"
            )
            
            fig.update_layout(
                title="Live Nifty 50 Price Prediction",
                xaxis_title="Time",
                yaxis_title="Price (₹)",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating live prediction chart: {e}")
            return self._create_error_chart("Error creating prediction chart")
    
    def create_entry_signal_chart(self, signal_data: Dict[str, Any]) -> go.Figure:
        """Create entry signal visualization"""
        try:
            signal = signal_data.get('signal', 'HOLD')
            confidence = signal_data.get('confidence', 0.5)
            target_price = signal_data.get('target_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            current_price = signal_data.get('current_price', 0)
            risk_reward = signal_data.get('risk_reward_ratio', 0)
            
            # Create gauge chart for signal strength
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Entry Signal: {signal}"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(
                title="Market Entry Signal Strength",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating entry signal chart: {e}")
            return self._create_error_chart("Error creating signal chart")
    
    def create_vwap_price_channel_chart(self, data: pd.DataFrame, indicators: Dict[str, Any], 
                                       symbol: str) -> go.Figure:
        """Create a dedicated VWAP Price Channel chart similar to professional trading platforms"""
        try:
            symbol_info = self.market_symbols.get(symbol, {'name': symbol, 'color': '#000000'})
            
            # Create subplots for main chart and volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f'{symbol_info["name"]} - VWAP Price Channel',
                    'Volume'
                ),
                row_heights=[0.8, 0.2]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=self.color_scheme['bullish'],
                    decreasing_line_color=self.color_scheme['bearish']
                ),
                row=1, col=1
            )
            
            # VWAP Price Channel with enhanced styling
            if all(key in indicators for key in ['vwap_upper', 'vwap_middle', 'vwap_lower']):
                # Check for valid data
                vwap_upper = indicators['vwap_upper'].dropna()
                vwap_middle = indicators['vwap_middle'].dropna()
                vwap_lower = indicators['vwap_lower'].dropna()
                
                if len(vwap_upper) > 0 and len(vwap_middle) > 0 and len(vwap_lower) > 0:
                    # VWAP Upper Band - Red line
                    fig.add_trace(
                        go.Scatter(
                            x=vwap_upper.index,
                            y=vwap_upper,
                            name='VWAP Upper',
                            line=dict(color='#FF4444', width=2.5, shape='spline'),
                            opacity=0.8,
                            hovertemplate='<b>VWAP Upper</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                
                    # VWAP Middle Line - Dynamic color
                    current_price = data['Close'].iloc[-1]
                    vwap_color = '#00C851' if current_price > vwap_middle.iloc[-1] else '#FF4444'
                    
                    fig.add_trace(
                        go.Scatter(
                            x=vwap_middle.index,
                            y=vwap_middle,
                            name='VWAP',
                            line=dict(color=vwap_color, width=3, shape='spline'),
                            opacity=0.9,
                            hovertemplate='<b>VWAP</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # VWAP Lower Band - Green line with fill
                    fig.add_trace(
                        go.Scatter(
                            x=vwap_lower.index,
                            y=vwap_lower,
                            name='VWAP Lower',
                            line=dict(color='#00C851', width=2.5, shape='spline'),
                            opacity=0.8,
                            fill='tonexty',
                            fillcolor='rgba(0, 200, 81, 0.1)',
                            hovertemplate='<b>VWAP Lower</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
            
            # Volume bars
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='lightgray',
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            # Update layout with professional styling
            # Get current timestamp and data range
            current_time = datetime.now()
            data_start = data.index[0] if not data.empty else current_time
            data_end = data.index[-1] if not data.empty else current_time
            
            fig.update_layout(
                title={
                    'text': f'{symbol_info["name"]} - VWAP Price Channel Analysis<br><sub>Last Updated: {current_time.strftime("%Y-%m-%d %H:%M:%S IST")} | Data Range: {data_start.strftime("%Y-%m-%d")} to {data_end.strftime("%Y-%m-%d")}</sub>',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': self.color_scheme['text']}
                },
                height=800,
                showlegend=True,
                plot_bgcolor='#1e1e1e',  # Dark background like professional charts
                paper_bgcolor='#1e1e1e',
                font=dict(family="Arial", size=12, color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                annotations=[
                    dict(
                        text=f"Data Points: {len(data)} | Generated: {current_time.strftime('%H:%M:%S')}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="lightgray"),
                        bgcolor="rgba(30,30,30,0.8)",
                        bordercolor="gray",
                        borderwidth=1
                    )
                ]
            )
            
            # Update axes with dark theme
            fig.update_xaxes(
                title_text="Date",
                gridcolor='#333333',
                showgrid=True,
                row=2, col=1
            )
            fig.update_yaxes(
                title_text="Price (₹)",
                gridcolor='#333333',
                showgrid=True,
                row=1, col=1
            )
            fig.update_yaxes(
                title_text="Volume",
                gridcolor='#333333',
                showgrid=True,
                row=2, col=1
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating VWAP Price Channel chart: {e}")
            return self._create_error_chart("Error creating VWAP Price Channel chart")
    
    def create_channel_comparison_chart(self, data: pd.DataFrame, indicators: Dict[str, Any], 
                                      symbol: str) -> go.Figure:
        """Create a comparison chart showing Traditional Donchian Channel vs VWAP Price Channel"""
        try:
            symbol_info = self.market_symbols.get(symbol, {'name': symbol, 'color': '#000000'})
            
            # Create subplots for main chart and volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f'{symbol_info["name"]} - Channel Comparison: Donchian vs VWAP Price Channel',
                    'Volume'
                ),
                row_heights=[0.8, 0.2]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=self.color_scheme['bullish'],
                    decreasing_line_color=self.color_scheme['bearish']
                ),
                row=1, col=1
            )
            
            # Traditional Donchian Channel - Blue stepped lines
            if all(key in indicators for key in ['donchian_upper', 'donchian_middle', 'donchian_lower']):
                # Donchian Upper Band - Blue stepped line
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['donchian_upper'],
                        name='Traditional Donchian Channel',
                        line=dict(color='#1f77b4', width=2, shape='hv'),  # hv = horizontal-vertical (stepped)
                        opacity=0.8,
                        hovertemplate='<b>Donchian Upper</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Donchian Middle Line - Red line
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['donchian_middle'],
                        name='Donchian Middle',
                        line=dict(color='#FF4444', width=2, shape='hv'),
                        opacity=0.8,
                        hovertemplate='<b>Donchian Middle</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Donchian Lower Band - Blue stepped line with fill
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['donchian_lower'],
                        name='Donchian Lower',
                        line=dict(color='#1f77b4', width=2, shape='hv'),
                        opacity=0.8,
                        fill='tonexty',
                        fillcolor='rgba(31, 119, 180, 0.1)',
                        hovertemplate='<b>Donchian Lower</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # VWAP Price Channel - Smooth curves
            if all(key in indicators for key in ['vwap_upper', 'vwap_middle', 'vwap_lower']):
                # Check for valid VWAP data
                vwap_upper = indicators['vwap_upper'].dropna()
                vwap_middle = indicators['vwap_middle'].dropna()
                vwap_lower = indicators['vwap_lower'].dropna()
                
                if len(vwap_upper) > 0 and len(vwap_middle) > 0 and len(vwap_lower) > 0:
                    # VWAP Upper Band - Red smooth line
                    fig.add_trace(
                        go.Scatter(
                            x=vwap_upper.index,
                            y=vwap_upper,
                            name='Corners Cut with Anchored VWAPs',
                            line=dict(color='#FF4444', width=2.5, shape='spline'),
                            opacity=0.8,
                            hovertemplate='<b>VWAP Upper</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                
                    # VWAP Middle Line - Dynamic color
                    current_price = data['Close'].iloc[-1]
                    vwap_color = '#00C851' if current_price > vwap_middle.iloc[-1] else '#FF4444'
                    
                    fig.add_trace(
                        go.Scatter(
                            x=vwap_middle.index,
                            y=vwap_middle,
                            name='VWAP Middle',
                            line=dict(color=vwap_color, width=3, shape='spline'),
                            opacity=0.9,
                            hovertemplate='<b>VWAP Middle</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # VWAP Lower Band - Green smooth line with fill
                    fig.add_trace(
                        go.Scatter(
                            x=vwap_lower.index,
                            y=vwap_lower,
                            name='VWAP Lower',
                            line=dict(color='#00C851', width=2.5, shape='spline'),
                            opacity=0.8,
                            fill='tonexty',
                            fillcolor='rgba(0, 200, 81, 0.15)',
                            hovertemplate='<b>VWAP Lower</b><br>Price: ₹%{y:,.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
            
            # Volume bars
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='lightgray',
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            # Add annotations for channel labels
            if len(data) > 0:
                # Traditional Donchian Channel label
                fig.add_annotation(
                    x=data.index[len(data)//3],
                    y=indicators['donchian_upper'].iloc[len(data)//3] if 'donchian_upper' in indicators else data['High'].iloc[len(data)//3],
                    text="Traditional Donchian Channel",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='#1f77b4',
                    font=dict(color='#1f77b4', size=12),
                    row=1, col=1
                )
                
                # VWAP Price Channel label
                fig.add_annotation(
                    x=data.index[2*len(data)//3],
                    y=indicators['vwap_lower'].iloc[2*len(data)//3] if 'vwap_lower' in indicators else data['Low'].iloc[2*len(data)//3],
                    text="Corners Cut with Anchored VWAPs",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='#00C851',
                    font=dict(color='#00C851', size=12),
                    row=1, col=1
                )
            
            # Get current timestamp and data range
            current_time = datetime.now()
            data_start = data.index[0] if not data.empty else current_time
            data_end = data.index[-1] if not data.empty else current_time
            
            # Update layout with professional styling and timestamps
            fig.update_layout(
                title={
                    'text': f'{symbol_info["name"]} - Channel Comparison Analysis<br><sub>Last Updated: {current_time.strftime("%Y-%m-%d %H:%M:%S IST")} | Data Range: {data_start.strftime("%Y-%m-%d")} to {data_end.strftime("%Y-%m-%d")}</sub>',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': self.color_scheme['text']}
                },
                height=800,
                showlegend=True,
                plot_bgcolor='#1e1e1e',  # Dark background
                paper_bgcolor='#1e1e1e',
                font=dict(family="Arial", size=12, color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                annotations=[
                    dict(
                        text=f"Data Points: {len(data)} | Generated: {current_time.strftime('%H:%M:%S')}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="lightgray"),
                        bgcolor="rgba(30,30,30,0.8)",
                        bordercolor="gray",
                        borderwidth=1
                    )
                ]
            )
            
            # Update axes with dark theme
            fig.update_xaxes(
                title_text="Date",
                gridcolor='#333333',
                showgrid=True,
                row=2, col=1
            )
            fig.update_yaxes(
                title_text="Price (₹)",
                gridcolor='#333333',
                showgrid=True,
                row=1, col=1
            )
            fig.update_yaxes(
                title_text="Volume",
                gridcolor='#333333',
                showgrid=True,
                row=2, col=1
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating channel comparison chart: {e}")
            return self._create_error_chart("Error creating channel comparison chart")
    
    def create_simple_test_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create a simple test chart to verify basic functionality"""
        try:
            fig = go.Figure()
            
            # Add simple line chart
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Get current timestamp and data range
            current_time = datetime.now()
            data_start = data.index[0] if not data.empty else current_time
            data_end = data.index[-1] if not data.empty else current_time
            
            fig.update_layout(
                title=f'{symbol} - Simple Test Chart<br><sub>Last Updated: {current_time.strftime("%Y-%m-%d %H:%M:%S IST")} | Data Range: {data_start.strftime("%Y-%m-%d")} to {data_end.strftime("%Y-%m-%d")}</sub>',
                xaxis_title='Date & Time',
                yaxis_title='Price (₹)',
                height=400,
                annotations=[
                    dict(
                        text=f"Data Points: {len(data)} | Generated: {current_time.strftime('%H:%M:%S')}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="gray",
                        borderwidth=1
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating simple test chart: {e}")
            return self._create_error_chart(f"Error creating test chart: {e}")
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create a simple error chart when visualization fails"""
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16
        )
        fig.update_layout(
            title="Visualization Error",
            height=400,
            showlegend=False
        )
        return fig

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create visualizer
    visualizer = IndianMarketVisualizer()
    
    print("Testing Indian Market Visualization...")
    print("Indian Market Visualization module loaded successfully")
