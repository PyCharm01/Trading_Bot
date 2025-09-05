#!/usr/bin/env python3
"""
Indian Trading Application

Main application interface for Indian market trading analysis including
Nifty 50, Bank Nifty, and Sensex with comprehensive features for
technical analysis, options strategies, and portfolio management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import plotly.graph_objects as go
import time

# Import our custom modules
from indian_market_data import IndianMarketDataFetcher, INDIAN_MARKET_SYMBOLS
from indian_technical_analysis import IndianMarketAnalyzer
from indian_options_engine import IndianOptionsStrategyEngine
from indian_portfolio_simulator import IndianPortfolioSimulator
from indian_visualization import IndianMarketVisualizer
from config import get_config, get_indian_market_strategies, get_lot_size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

class IndianTradingApp:
    """Main Indian Trading Application"""
    
    def __init__(self):
        self.data_fetcher = IndianMarketDataFetcher()
        self.technical_analyzer = IndianMarketAnalyzer()
        self.options_engine = IndianOptionsStrategyEngine()
        self.portfolio_simulator = IndianPortfolioSimulator(config.DEFAULT_INITIAL_CAPITAL)
        self.visualizer = IndianMarketVisualizer()
        
        # Initialize session state
        if 'portfolio_simulator' not in st.session_state:
            st.session_state.portfolio_simulator = self.portfolio_simulator
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = 'NIFTY_50'
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="Indian Market Trading Analysis",
            page_icon="🇮🇳",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Live refresh functionality
        self._setup_live_refresh()
        
        # Main title
        st.title("🇮🇳 Indian Market Trading Analysis Platform")
        st.markdown("Comprehensive analysis for Nifty 50, Bank Nifty, and Sensex with advanced options strategies and portfolio management.")
        
        # Sidebar
        self._create_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Market Overview", 
            "📈 Technical Analysis", 
            "🎯 Options Strategies", 
            "💼 Portfolio Management", 
            "📋 Reports"
        ])
        
        with tab1:
            self._show_market_overview()
        
        with tab2:
            self._show_technical_analysis()
        
        with tab3:
            self._show_options_strategies()
        
        with tab4:
            self._show_portfolio_management()
        
        with tab5:
            self._show_reports()
    
    def _setup_live_refresh(self):
        """Setup live refresh functionality"""
        # Add refresh controls to sidebar
        st.sidebar.header("🔄 Live Refresh")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh (Every 5 seconds)", value=True)
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now", type="primary"):
            st.rerun()
        
        # Refresh interval selector
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval",
            options=[1, 2, 5, 10, 30, 60],
            index=2,  # Default to 5 seconds
            format_func=lambda x: f"{x} seconds"
        )
        
        # Store refresh settings in session state
        st.session_state.auto_refresh = auto_refresh
        st.session_state.refresh_interval = refresh_interval
        
        # Auto-refresh logic
        if auto_refresh:
            # Create a placeholder for the countdown
            countdown_placeholder = st.sidebar.empty()
            
            # Get current time
            current_time = datetime.now()
            
            # Check if we need to refresh
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = current_time
                st.session_state.refresh_counter = 0
            
            # Calculate time since last refresh
            time_since_refresh = (current_time - st.session_state.last_refresh).total_seconds()
            
            # Update countdown
            remaining_time = max(0, refresh_interval - time_since_refresh)
            countdown_placeholder.metric(
                "Next Refresh", 
                f"{remaining_time:.0f}s",
                delta=f"Last: {st.session_state.refresh_counter}"
            )
            
            # Trigger refresh if interval has passed
            if time_since_refresh >= refresh_interval:
                st.session_state.last_refresh = current_time
                st.session_state.refresh_counter += 1
                st.rerun()
        
        # Add live status indicator
        st.sidebar.markdown("---")
        status_color = "🟢" if auto_refresh else "🔴"
        st.sidebar.markdown(f"**Status:** {status_color} {'Live' if auto_refresh else 'Static'}")
        
        # Market status
        try:
            market_status = self.data_fetcher.get_market_status()
            market_icon = "🟢" if market_status.get('is_market_open', False) else "🔴"
            st.sidebar.markdown(f"**Market:** {market_icon} {'Open' if market_status.get('is_market_open', False) else 'Closed'}")
        except:
            st.sidebar.markdown("**Market:** ⚪ Unknown")
    
    def _create_sidebar(self):
        """Create sidebar with controls"""
        st.sidebar.header("🎛️ Analysis Controls")
        
        # Symbol selection
        symbol_options = {symbol: info.name for symbol, info in INDIAN_MARKET_SYMBOLS.items()}
        selected_symbol = st.sidebar.selectbox(
            "Select Market Index",
            options=list(symbol_options.keys()),
            format_func=lambda x: symbol_options[x],
            index=list(symbol_options.keys()).index(st.session_state.selected_symbol)
        )
        st.session_state.selected_symbol = selected_symbol
        
        # Time period selection
        time_period = st.sidebar.selectbox(
            "Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
        
        # Analysis parameters
        st.sidebar.subheader("📊 Analysis Parameters")
        
        # Technical indicators
        show_rsi = st.sidebar.checkbox("RSI", value=True)
        show_macd = st.sidebar.checkbox("MACD", value=True)
        show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
        show_volume = st.sidebar.checkbox("Volume Analysis", value=True)
        
        # Options analysis
        st.sidebar.subheader("🎯 Options Analysis")
        show_iv_analysis = st.sidebar.checkbox("IV Analysis", value=True)
        show_pcr = st.sidebar.checkbox("Put-Call Ratio", value=True)
        show_strategy_recommendations = st.sidebar.checkbox("Strategy Recommendations", value=True)
        
        # Portfolio settings
        st.sidebar.subheader("💼 Portfolio Settings")
        initial_capital = st.sidebar.number_input(
            "Initial Capital (₹)",
            min_value=100000.0,
            max_value=10000000.0,
            value=config.DEFAULT_INITIAL_CAPITAL,
            step=100000.0
        )
        
        # Update portfolio simulator if capital changed
        if initial_capital != st.session_state.portfolio_simulator.initial_capital:
            st.session_state.portfolio_simulator = IndianPortfolioSimulator(initial_capital)
        
        # Store parameters in session state
        st.session_state.analysis_params = {
            'time_period': time_period,
            'show_rsi': show_rsi,
            'show_macd': show_macd,
            'show_bollinger': show_bollinger,
            'show_volume': show_volume,
            'show_iv_analysis': show_iv_analysis,
            'show_pcr': show_pcr,
            'show_strategy_recommendations': show_strategy_recommendations
        }
        
        # Market status
        self._show_market_status()
    
    def _show_market_status(self):
        """Show current market status"""
        st.sidebar.subheader("🕐 Market Status")
        
        try:
            market_status = self.data_fetcher.get_market_status()
            
            if market_status.get('is_market_open'):
                st.sidebar.success("🟢 Market Open")
            else:
                st.sidebar.warning("🔴 Market Closed")
            
            st.sidebar.write(f"**Next Event:** {market_status.get('next_event', 'Unknown')}")
            st.sidebar.write(f"**Time:** {market_status.get('next_event_time', 'Unknown')}")
            
            # Trading hours
            with st.sidebar.expander("Trading Hours"):
                trading_hours = market_status.get('trading_hours', {})
                for session, hours in trading_hours.items():
                    st.write(f"**{session.replace('_', ' ').title()}:** {hours}")
        
        except Exception as e:
            st.sidebar.error(f"Error fetching market status: {e}")
    
    def _show_market_overview(self):
        """Show market overview dashboard"""
        st.header("📊 Indian Market Overview")
        
        # Live timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        st.markdown(f"**Last Updated:** {current_time}")
        
        try:
            # Fetch market overview data
            with st.spinner("Fetching market data..."):
                overview_data = self.data_fetcher.fetch_market_overview()
                sector_data = self.data_fetcher.fetch_sector_performance()
            
            if not overview_data:
                st.error("Unable to fetch market data. Please try again later.")
                return
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'NIFTY_50' in overview_data:
                    nifty_data = overview_data['NIFTY_50']
                    st.metric(
                        "Nifty 50",
                        f"₹{nifty_data['current_price']:,.0f}",
                        f"{nifty_data['change_percent']:+.2f}%"
                    )
            
            with col2:
                if 'BANK_NIFTY' in overview_data:
                    bank_nifty_data = overview_data['BANK_NIFTY']
                    st.metric(
                        "Bank Nifty",
                        f"₹{bank_nifty_data['current_price']:,.0f}",
                        f"{bank_nifty_data['change_percent']:+.2f}%"
                    )
            
            with col3:
                if 'SENSEX' in overview_data:
                    sensex_data = overview_data['SENSEX']
                    st.metric(
                        "Sensex",
                        f"₹{sensex_data['current_price']:,.0f}",
                        f"{sensex_data['change_percent']:+.2f}%"
                    )
            
            with col4:
                # Market sentiment
                positive_sectors = sum(1 for sector in sector_data.values() if sector.get('change_percent', 0) > 0)
                total_sectors = len(sector_data)
                sentiment = "Bullish" if positive_sectors > total_sectors // 2 else "Bearish"
                st.metric(
                    "Market Sentiment",
                    sentiment,
                    f"{positive_sectors}/{total_sectors} sectors up"
                )
            
            # Sector performance chart
            if sector_data:
                st.subheader("📈 Sector Performance")
                sector_chart = self.visualizer.create_indian_market_heatmap(sector_data)
                st.plotly_chart(sector_chart, use_container_width=True)
            
            # Market overview chart
            st.subheader("📊 Market Overview Dashboard")
            
            # Fetch data for charts
            symbols = ['NIFTY_50', 'BANK_NIFTY', 'SENSEX']
            market_data = {}
            
            for symbol in symbols:
                data = self.data_fetcher.fetch_index_data(symbol, st.session_state.analysis_params['time_period'])
                if not data.empty:
                    market_data[symbol] = data
            
            if market_data:
                # Create dashboard
                dashboard_data = {
                    'market_overview': overview_data,
                    'sector_performance': sector_data,
                    'options_analysis': {},
                    'portfolio_metrics': {}
                }
                
                dashboard_chart = self.visualizer.create_indian_market_dashboard(market_data, dashboard_data)
                st.plotly_chart(dashboard_chart, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error displaying market overview: {e}")
            logger.error(f"Error in market overview: {e}")
    
    def _show_technical_analysis(self):
        """Show technical analysis"""
        st.header("📈 Technical Analysis")
        
        # Live timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        st.markdown(f"**Last Updated:** {current_time}")
        
        symbol = st.session_state.selected_symbol
        time_period = st.session_state.analysis_params['time_period']
        
        try:
            # Fetch data
            with st.spinner(f"Analyzing {INDIAN_MARKET_SYMBOLS[symbol].name}..."):
                data = self.data_fetcher.fetch_index_data(symbol, time_period)
                
                if data.empty:
                    st.error(f"No data available for {INDIAN_MARKET_SYMBOLS[symbol].name}")
                    return
                
                # Perform technical analysis
                analysis = self.technical_analyzer.analyze_index(data, symbol)
            
            if 'error' in analysis:
                st.error(f"Analysis error: {analysis['error']}")
                return
            
            # Display current metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"₹{analysis['current_price']:,.0f}",
                    f"RSI: {analysis['indicators']['rsi'].iloc[-1]:.1f}"
                )
            
            with col2:
                signals = analysis['signals']
                overall_signal = signals.get('overall', {})
                signal_color = {
                    'BUY': 'green',
                    'SELL': 'red',
                    'NEUTRAL': 'orange'
                }.get(overall_signal.get('signal', 'NEUTRAL'), 'gray')
                
                st.metric(
                    "Overall Signal",
                    overall_signal.get('signal', 'NEUTRAL'),
                    f"Strength: {overall_signal.get('strength', 0):.1%}"
                )
            
            with col3:
                market_regime = analysis['market_regime']
                st.metric(
                    "Market Regime",
                    market_regime.get('trend', 'Unknown'),
                    f"Volatility: {market_regime.get('volatility', 'Unknown')}"
                )
            
            with col4:
                support_resistance = analysis['support_resistance']
                st.metric(
                    "Support/Resistance",
                    f"Support: ₹{support_resistance['support_1']:,.0f}",
                    f"Resistance: ₹{support_resistance['resistance_1']:,.0f}"
                )
            
            # Technical analysis chart
            st.subheader("📊 Technical Analysis Chart")
            tech_chart = self.visualizer.create_indian_technical_analysis_chart(
                data, analysis['indicators'], symbol
            )
            st.plotly_chart(tech_chart, use_container_width=True)
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Technical Signals")
                for signal_name, signal_data in signals.items():
                    if signal_name != 'overall':
                        signal_color = {
                            'BUY': '🟢',
                            'SELL': '🔴',
                            'NEUTRAL': '🟡'
                        }.get(signal_data.get('signal', 'NEUTRAL'), '⚪')
                        
                        st.write(f"{signal_color} **{signal_name.upper()}:** {signal_data.get('signal', 'NEUTRAL')} - {signal_data.get('reasoning', 'No reasoning')}")
            
            with col2:
                st.subheader("📊 Market Analysis")
                st.write(f"**Trend:** {market_regime.get('trend', 'Unknown')}")
                st.write(f"**Volatility:** {market_regime.get('volatility', 'Unknown')}")
                st.write(f"**Momentum:** {market_regime.get('momentum', 'Unknown')}")
                
                # Volatility analysis
                vol_analysis = analysis['volatility_analysis']
                st.write(f"**Daily Volatility:** {vol_analysis.get('daily_volatility', 0):.2%}")
                st.write(f"**Annualized Volatility:** {vol_analysis.get('annualized_volatility', 0):.2%}")
                st.write(f"**Volatility Regime:** {vol_analysis.get('volatility_regime', 'Unknown')}")
            
            # Analysis summary
            st.subheader("📋 Analysis Summary")
            st.write(analysis['summary'])
        
        except Exception as e:
            st.error(f"Error in technical analysis: {e}")
            logger.error(f"Error in technical analysis: {e}")
    
    def _show_options_strategies(self):
        """Show options strategies"""
        st.header("🎯 Options Strategies")
        
        symbol = st.session_state.selected_symbol
        time_period = st.session_state.analysis_params['time_period']
        
        try:
            # Fetch data
            with st.spinner(f"Analyzing {INDIAN_MARKET_SYMBOLS[symbol].name} options..."):
                data = self.data_fetcher.fetch_index_data(symbol, time_period)
                options_data = self.data_fetcher.fetch_options_chain(symbol)
                
                if data.empty:
                    st.error(f"No data available for {INDIAN_MARKET_SYMBOLS[symbol].name}")
                    return
            
            # Perform technical analysis for signal
            analysis = self.technical_analyzer.analyze_index(data, symbol)
            if 'error' in analysis:
                st.error(f"Analysis error: {analysis['error']}")
                return
            
            # Analyze options chain
            current_price = data['Close'].iloc[-1]
            options_analysis = self.options_engine.analyze_options_chain(options_data, current_price, symbol)
            
            # Display options metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"₹{current_price:,.0f}",
                    f"Lot Size: {get_lot_size(symbol)}"
                )
            
            with col2:
                if options_analysis.get('expirations'):
                    exp_dates = list(options_analysis['expirations'].keys())
                    nearest_exp = min(exp_dates, key=lambda x: options_analysis['expirations'][x]['days_to_expiry'])
                    days_to_exp = options_analysis['expirations'][nearest_exp]['days_to_expiry']
                    st.metric(
                        "Nearest Expiry",
                        nearest_exp,
                        f"{days_to_exp} days"
                    )
            
            with col3:
                # IV Analysis
                if options_analysis.get('expirations'):
                    exp_data = list(options_analysis['expirations'].values())[0]
                    atm_analysis = exp_data.get('atm_analysis', {})
                    avg_iv = (atm_analysis.get('call_iv', 0) + atm_analysis.get('put_iv', 0)) / 2
                    st.metric(
                        "Average IV",
                        f"{avg_iv:.1%}",
                        "Implied Volatility"
                    )
            
            with col4:
                # Put-Call Ratio
                if options_analysis.get('expirations'):
                    exp_data = list(options_analysis['expirations'].values())[0]
                    pcr = exp_data.get('put_call_ratio', {})
                    volume_pcr = pcr.get('volume_pcr', 0)
                    st.metric(
                        "Put-Call Ratio",
                        f"{volume_pcr:.2f}",
                        pcr.get('sentiment', 'Neutral')
                    )
            
            # Options strategy recommendation
            st.subheader("🎯 Strategy Recommendation")
            
            # Get technical signal
            overall_signal = analysis['signals'].get('overall', {})
            signal = overall_signal.get('signal', 'NEUTRAL')
            market_regime = analysis['market_regime'].get('trend', 'Unknown')
            
            # Recommend strategy
            recommended_strategy = self.options_engine.recommend_strategy(
                signal, options_analysis, current_price, symbol, market_regime
            )
            
            # Display strategy details
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Strategy:** {recommended_strategy.name}")
                st.write(f"**Description:** {recommended_strategy.description}")
                st.write(f"**Max Profit:** ₹{recommended_strategy.max_profit:,.0f}")
                st.write(f"**Max Loss:** ₹{recommended_strategy.max_loss:,.0f}")
                st.write(f"**Probability of Profit:** {recommended_strategy.probability_of_profit:.1%}")
            
            with col2:
                st.write(f"**Breakeven Points:** {recommended_strategy.breakeven_points}")
                st.write(f"**Risk-Reward Ratio:** {recommended_strategy.risk_reward_ratio:.2f}")
                st.write(f"**Margin Required:** ₹{recommended_strategy.margin_required:,.0f}")
                st.write(f"**Lot Size:** {recommended_strategy.lot_size}")
            
            # Strategy payoff diagram
            if recommended_strategy.legs:
                st.subheader("📊 Strategy Payoff Diagram")
                
                strategy_data = {
                    'name': recommended_strategy.name,
                    'legs': recommended_strategy.legs,
                    'breakeven_points': recommended_strategy.breakeven_points,
                    'current_price': current_price,
                    'min_price': current_price * 0.9,
                    'max_price': current_price * 1.1
                }
                
                payoff_chart = self.visualizer.create_indian_options_strategy_chart(strategy_data)
                st.plotly_chart(payoff_chart, use_container_width=True)
            
            # Options chain analysis
            if st.session_state.analysis_params['show_iv_analysis']:
                st.subheader("📊 Options Chain Analysis")
                
                if options_analysis.get('expirations'):
                    # Create IV skew chart
                    exp_data = list(options_analysis['expirations'].values())[0]
                    iv_skew = exp_data.get('iv_skew', {})
                    
                    if iv_skew:
                        st.write(f"**IV Skew Direction:** {iv_skew.get('skew_direction', 'Unknown')}")
                        st.write(f"**Put Skew:** {iv_skew.get('put_skew', 0):.2f}")
                        st.write(f"**Call Skew:** {iv_skew.get('call_skew', 0):.2f}")
                        st.write(f"**Overall Skew:** {iv_skew.get('overall_skew', 0):.2f}")
            
            # Strategy execution
            st.subheader("⚡ Execute Strategy")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📈 Add to Portfolio", type="primary"):
                    # Add strategy to portfolio
                    for leg in recommended_strategy.legs:
                        success = st.session_state.portfolio_simulator.add_position(
                            symbol=symbol,
                            instrument_type='option',
                            quantity=leg.get('quantity', 1),
                            entry_price=leg.get('premium', 0),
                            strategy=recommended_strategy.name,
                            expiry=leg.get('expiry'),
                            strike=leg.get('strike'),
                            option_type=leg.get('option_type')
                        )
                    
                    if success:
                        st.success("Strategy added to portfolio!")
                    else:
                        st.error("Failed to add strategy to portfolio")
            
            with col2:
                if st.button("📊 Analyze More"):
                    st.info("Detailed analysis would be performed here")
            
            with col3:
                if st.button("💾 Save Strategy"):
                    st.info("Strategy saved to favorites")
        
        except Exception as e:
            st.error(f"Error in options strategies: {e}")
            logger.error(f"Error in options strategies: {e}")
    
    def _show_portfolio_management(self):
        """Show portfolio management"""
        st.header("💼 Portfolio Management")
        
        # Live timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        st.markdown(f"**Last Updated:** {current_time}")
        
        try:
            # Portfolio summary
            portfolio_summary = st.session_state.portfolio_simulator.get_portfolio_summary()
            
            if not portfolio_summary:
                st.warning("No portfolio data available")
                return
            
            # Display key metrics
            metrics = portfolio_summary.get('portfolio_metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Value",
                    f"₹{metrics.get('total_value', 0):,.0f}",
                    f"{metrics.get('total_pnl_percent', 0):+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Total P&L",
                    f"₹{metrics.get('total_pnl', 0):,.0f}",
                    f"Realized: ₹{metrics.get('realized_pnl', 0):,.0f}"
                )
            
            with col3:
                st.metric(
                    "Margin Used",
                    f"₹{metrics.get('margin_used', 0):,.0f}",
                    f"Available: ₹{metrics.get('margin_available', 0):,.0f}"
                )
            
            with col4:
                st.metric(
                    "Win Rate",
                    f"{metrics.get('win_rate', 0):.1%}",
                    f"Trades: {metrics.get('total_trades', 0)}"
                )
            
            # Portfolio performance chart
            st.subheader("📊 Portfolio Performance")
            
            portfolio_data = {
                'portfolio_history': portfolio_summary.get('portfolio_history', []),
                'strategy_performance': portfolio_summary.get('strategy_performance', {}),
                'risk_metrics': {
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0)
                }
            }
            
            portfolio_chart = self.visualizer.create_indian_portfolio_performance_chart(portfolio_data)
            st.plotly_chart(portfolio_chart, use_container_width=True)
            
            # Position management
            st.subheader("📋 Position Management")
            
            positions = portfolio_summary.get('positions', {})
            open_positions = positions.get('open', 0)
            closed_positions = positions.get('closed', 0)
            
            st.write(f"**Open Positions:** {open_positions}")
            st.write(f"**Closed Positions:** {closed_positions}")
            
            # Risk management
            st.subheader("⚠️ Risk Management")
            
            risk_check = st.session_state.portfolio_simulator.check_risk_limits()
            
            if risk_check.get('within_limits'):
                st.success("✅ Portfolio is within risk limits")
            else:
                st.warning("⚠️ Portfolio exceeds some risk limits")
            
            risk_alerts = risk_check.get('risk_alerts', [])
            for alert in risk_alerts:
                severity_color = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(alert.get('severity', 'low'), '⚪')
                
                st.write(f"{severity_color} {alert.get('message', 'Unknown alert')}")
            
            # Portfolio actions
            st.subheader("⚡ Portfolio Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Update Prices"):
                    # Update position prices
                    symbol = st.session_state.selected_symbol
                    data = self.data_fetcher.fetch_index_data(symbol, "1d")
                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        price_data = {symbol: current_price}
                        st.session_state.portfolio_simulator.update_position_prices(price_data)
                        st.success("Prices updated!")
            
            with col2:
                if st.button("💾 Export Portfolio"):
                    # Export portfolio data
                    filename = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    success = st.session_state.portfolio_simulator.export_portfolio_data(filename)
                    if success:
                        st.success(f"Portfolio exported to {filename}")
                    else:
                        st.error("Failed to export portfolio")
            
            with col3:
                if st.button("📈 Generate Report"):
                    st.info("Portfolio report would be generated here")
        
        except Exception as e:
            st.error(f"Error in portfolio management: {e}")
            logger.error(f"Error in portfolio management: {e}")
    
    def _show_reports(self):
        """Show reports and analytics"""
        st.header("📋 Reports & Analytics")
        
        try:
            # Generate comprehensive report
            st.subheader("📊 Comprehensive Market Report")
            
            symbol = st.session_state.selected_symbol
            time_period = st.session_state.analysis_params['time_period']
            
            # Fetch all data
            with st.spinner("Generating comprehensive report..."):
                data = self.data_fetcher.fetch_index_data(symbol, time_period)
                options_data = self.data_fetcher.fetch_options_chain(symbol)
                news_data = self.data_fetcher.fetch_news(symbol)
                
                if data.empty:
                    st.error(f"No data available for {INDIAN_MARKET_SYMBOLS[symbol].name}")
                    return
            
            # Perform analysis
            technical_analysis = self.technical_analyzer.analyze_index(data, symbol)
            options_analysis = self.options_engine.analyze_options_chain(options_data, data['Close'].iloc[-1], symbol)
            portfolio_summary = st.session_state.portfolio_simulator.get_portfolio_summary()
            
            # Create report sections
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Technical Analysis Summary")
                if 'error' not in technical_analysis:
                    signals = technical_analysis.get('signals', {})
                    overall_signal = signals.get('overall', {})
                    
                    st.write(f"**Overall Signal:** {overall_signal.get('signal', 'NEUTRAL')}")
                    st.write(f"**Signal Strength:** {overall_signal.get('strength', 0):.1%}")
                    st.write(f"**Reasoning:** {overall_signal.get('reasoning', 'No reasoning')}")
                    
                    market_regime = technical_analysis.get('market_regime', {})
                    st.write(f"**Market Trend:** {market_regime.get('trend', 'Unknown')}")
                    st.write(f"**Volatility:** {market_regime.get('volatility', 'Unknown')}")
                    st.write(f"**Momentum:** {market_regime.get('momentum', 'Unknown')}")
            
            with col2:
                st.subheader("🎯 Options Analysis Summary")
                if options_analysis.get('expirations'):
                    exp_data = list(options_analysis['expirations'].values())[0]
                    atm_analysis = exp_data.get('atm_analysis', {})
                    pcr = exp_data.get('put_call_ratio', {})
                    
                    st.write(f"**Current Price:** ₹{data['Close'].iloc[-1]:,.0f}")
                    st.write(f"**Average IV:** {(atm_analysis.get('call_iv', 0) + atm_analysis.get('put_iv', 0)) / 2:.1%}")
                    st.write(f"**Put-Call Ratio:** {pcr.get('volume_pcr', 0):.2f}")
                    st.write(f"**Market Sentiment:** {pcr.get('sentiment', 'Neutral')}")
            
            # Portfolio performance
            st.subheader("💼 Portfolio Performance Summary")
            if portfolio_summary:
                metrics = portfolio_summary.get('portfolio_metrics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{metrics.get('total_pnl_percent', 0):+.2f}%")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
                
                with col4:
                    st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
            
            # News summary
            if news_data:
                st.subheader("📰 Recent News")
                for i, news in enumerate(news_data[:5]):
                    with st.expander(f"News {i+1}: {news.get('title', 'No title')[:50]}..."):
                        st.write(f"**Publisher:** {news.get('publisher', 'Unknown')}")
                        st.write(f"**Published:** {news.get('published', 'Unknown')}")
                        st.write(f"**Summary:** {news.get('summary', 'No summary available')}")
                        if news.get('url'):
                            st.write(f"**Link:** [Read more]({news['url']})")
            
            # Export report
            st.subheader("📤 Export Report")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export PDF"):
                    st.info("PDF report would be generated here")
            
            with col2:
                if st.button("📊 Export Excel"):
                    st.info("Excel report would be generated here")
            
            with col3:
                if st.button("📧 Email Report"):
                    st.info("Email report would be sent here")
        
        except Exception as e:
            st.error(f"Error generating reports: {e}")
            logger.error(f"Error in reports: {e}")

def main():
    """Main function to run the application"""
    try:
        app = IndianTradingApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
