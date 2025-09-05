#!/usr/bin/env python3
"""
Indian Market Trading Analysis Platform - Main Application

Structured main application interface for Indian market trading analysis including
Nifty 50, Bank Nifty, and Sensex with comprehensive features for
technical analysis, options strategies, and portfolio management.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import plotly.graph_objects as go
import time
import sys
import os
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import base64

# Import helper for robust module loading
from import_helper import import_config, import_module

# Import our structured modules using the helper
IndianMarketDataFetcher = import_module('data.indian_market_data', 'IndianMarketDataFetcher')
INDIAN_MARKET_SYMBOLS = import_module('data.indian_market_data', 'INDIAN_MARKET_SYMBOLS')
IndianMarketAnalyzer = import_module('analysis.indian_technical_analysis', 'IndianMarketAnalyzer')
IndianBacktestingEngine = import_module('analysis.indian_backtesting', 'IndianBacktestingEngine')
BacktestConfig = import_module('analysis.indian_backtesting', 'BacktestConfig')
LivePredictionEngine = import_module('analysis.live_prediction_engine', 'LivePredictionEngine')
PredictionResult = import_module('analysis.live_prediction_engine', 'PredictionResult')
MarketEntrySignal = import_module('analysis.live_prediction_engine', 'MarketEntrySignal')
IndianOptionsStrategyEngine = import_module('options.indian_options_engine', 'IndianOptionsStrategyEngine')
OptionsStrategy = import_module('options.indian_options_engine', 'OptionsStrategy')
OptionsContract = import_module('options.indian_options_engine', 'OptionsContract')
IndianPortfolioSimulator = import_module('portfolio.indian_portfolio_simulator', 'IndianPortfolioSimulator')
Position = import_module('portfolio.indian_portfolio_simulator', 'Position')
PortfolioMetrics = import_module('portfolio.indian_portfolio_simulator', 'PortfolioMetrics')
IndianMarketVisualizer = import_module('visualization.indian_visualization', 'IndianMarketVisualizer')
get_config = import_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IndianTradingApp:
    """Main application class for Indian market trading analysis"""
    
    def __init__(self):
        """Initialize the application components"""
        self.config = get_config()
        self.data_fetcher = IndianMarketDataFetcher(
            alpha_vantage_api_key=self.config.ALPHA_VANTAGE_API_KEY,
            quandl_api_key=self.config.QUANDL_API_KEY
        )
        self.technical_analyzer = IndianMarketAnalyzer()
        self.options_engine = IndianOptionsStrategyEngine()
        self.visualizer = IndianMarketVisualizer()
        self.prediction_engine = LivePredictionEngine()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'portfolio_simulator' not in st.session_state:
            st.session_state.portfolio_simulator = IndianPortfolioSimulator(self.config.DEFAULT_INITIAL_CAPITAL)
        
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = 'NIFTY_50'
        
        if 'analysis_params' not in st.session_state:
            st.session_state.analysis_params = {
                'time_period': '1mo',
                'indicators': ['RSI', 'EMA', 'MACD', 'Bollinger_Bands'],
                'show_volume': True,
                'show_support_resistance': True
            }
    
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
        
        # Setup auto-refresh timer
        self._setup_auto_refresh_timer()
        
        # Main title
        st.title("🇮🇳 Indian Market Trading Analysis Platform")
        st.markdown("Comprehensive analysis for Nifty 50, Bank Nifty, and Sensex with advanced options strategies and portfolio management.")
        
        # Auto-refresh status at the top
        if st.session_state.get('auto_refresh', False):
            refresh_interval = st.session_state.get('refresh_interval', 5)
            refresh_count = st.session_state.get('refresh_counter', 0)
            last_refresh = st.session_state.get('last_refresh', datetime.now())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"🟢 Auto-Refresh: ON (Every {refresh_interval}s)")
            with col2:
                st.info(f"🔄 Refresh Count: {refresh_count}")
            with col3:
                st.info(f"⏰ Last: {last_refresh.strftime('%H:%M:%S')}")
        
        # Sidebar
        self._create_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Market Overview", 
            "📈 Technical Analysis", 
            "🎯 Options Strategies", 
            "💼 Portfolio Management", 
            "🔮 Live Predictions",
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
            self._show_live_predictions()
        
        with tab6:
            self._show_reports()
        
        # Enable continuous refresh at the end
        self._enable_continuous_refresh()
        
        # Add JavaScript-based auto-refresh as backup
        if st.session_state.get('auto_refresh', False):
            refresh_interval = st.session_state.get('refresh_interval', 5)
            st.markdown(f"""
            <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {refresh_interval * 1000});
            </script>
            """, unsafe_allow_html=True)
    
    def _setup_live_refresh(self):
        """Setup live refresh functionality"""
        # Add refresh controls to sidebar
        st.sidebar.header("🔄 Live Refresh")
        
        # Initialize session state for refresh
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'refresh_counter' not in st.session_state:
            st.session_state.refresh_counter = 0
        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox(
            "Auto Refresh", 
            value=st.session_state.auto_refresh,
            key="auto_refresh_checkbox"
        )
        
        # Update session state
        st.session_state.auto_refresh = auto_refresh
        
        # Refresh interval selector
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval",
            options=[1, 2, 5, 10, 30, 60],
            index=[1, 2, 5, 10, 30, 60].index(st.session_state.refresh_interval),
            format_func=lambda x: f"{x} seconds",
            key="refresh_interval_select"
        )
        
        # Update session state
        st.session_state.refresh_interval = refresh_interval
        
        # Manual refresh button
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh Now", type="primary", key="manual_refresh"):
                st.session_state.last_refresh = datetime.now()
                st.session_state.refresh_counter += 1
                # Clear cache to force data refresh
                st.session_state.data_cache = {}
                st.rerun()
        
        with col2:
            if st.button("⏸️ Pause", key="pause_refresh"):
                st.session_state.auto_refresh = False
                st.rerun()
        
        # Auto-refresh logic with proper timing
        if auto_refresh:
            # Create a placeholder for the countdown
            countdown_placeholder = st.sidebar.empty()
            
            # Get current time
            current_time = datetime.now()
            
            # Calculate time since last refresh
            time_since_refresh = (current_time - st.session_state.last_refresh).total_seconds()
            
            # Update countdown display
            remaining_time = max(0, refresh_interval - time_since_refresh)
            countdown_placeholder.metric(
                "Next Refresh", 
                f"{remaining_time:.0f}s",
                delta=f"Count: {st.session_state.refresh_counter}"
            )
            
            # Auto-refresh when time is up
            if remaining_time <= 0:
                # Update refresh time and counter
                st.session_state.last_refresh = current_time
                st.session_state.refresh_counter += 1
                # Clear cache to force data refresh
                st.session_state.data_cache = {}
                st.rerun()
            
            # Use JavaScript-based auto-refresh for continuous updates
            if remaining_time > 0:
                # Add JavaScript to auto-refresh the page
                st.markdown(f"""
                <script>
                setTimeout(function() {{
                    window.location.reload();
                }}, {int(remaining_time * 1000)});
                </script>
                """, unsafe_allow_html=True)
        
        # Add live status indicator
        st.sidebar.markdown("---")
        status_color = "🟢" if auto_refresh else "🔴"
        status_text = "Live" if auto_refresh else "Paused"
        st.sidebar.markdown(f"**Status:** {status_color} {status_text}")
        
        # Show last update time
        last_update = st.session_state.last_refresh.strftime("%H:%M:%S")
        st.sidebar.markdown(f"**Last Update:** {last_update}")
        
        # Show refresh count
        st.sidebar.markdown(f"**Refresh Count:** {st.session_state.refresh_counter}")
        
        # Market status
        try:
            market_status = self.data_fetcher.get_market_status()
            market_icon = "🟢" if market_status.get('is_market_open', False) else "🔴"
            market_text = "Open" if market_status.get('is_market_open', False) else "Closed"
            st.sidebar.markdown(f"**Market:** {market_icon} {market_text}")
        except:
            st.sidebar.markdown("**Market:** ⚪ Unknown")
        
        # Add refresh status to main page
        if auto_refresh:
            # Create a status bar at the top of the main content
            status_container = st.container()
            with status_container:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.success(f"🟢 Live Auto-Refresh Active (Every {refresh_interval}s)")
                with col2:
                    st.info(f"Count: {st.session_state.refresh_counter}")
                with col3:
                    st.info(f"Last: {last_update}")
        
        # Add refresh info
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Refresh Info:**")
        st.sidebar.markdown(f"• Auto-refresh: {'ON' if auto_refresh else 'OFF'}")
        st.sidebar.markdown(f"• Interval: {refresh_interval}s")
        st.sidebar.markdown(f"• Total refreshes: {st.session_state.refresh_counter}")
    
    def _setup_auto_refresh_timer(self):
        """Setup automatic refresh timer using Streamlit's built-in capabilities"""
        try:
            # Check if auto-refresh is enabled
            if st.session_state.get('auto_refresh', False):
                # Create a placeholder for the timer
                timer_placeholder = st.empty()
                
                # Get current time and calculate next refresh
                current_time = datetime.now()
                last_refresh = st.session_state.get('last_refresh', current_time)
                refresh_interval = st.session_state.get('refresh_interval', 5)
                
                # Calculate time since last refresh
                time_since_refresh = (current_time - last_refresh).total_seconds()
                remaining_time = max(0, refresh_interval - time_since_refresh)
                
                # Update timer display
                if remaining_time > 0:
                    timer_placeholder.info(f"⏱️ Next refresh in {remaining_time:.0f} seconds")
                    
                    # Use time.sleep and st.rerun for automatic refresh
                    import time
                    time.sleep(1)  # Wait 1 second
                    st.rerun()  # This will cause the app to refresh
                    
                else:
                    timer_placeholder.success("🔄 Refreshing data...")
                    
                    # Update refresh time and counter
                    st.session_state.last_refresh = current_time
                    st.session_state.refresh_counter = st.session_state.get('refresh_counter', 0) + 1
                    
                    # Clear cache to force data refresh
                    st.session_state.data_cache = {}
                    
                    # Rerun the app
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"Error in auto-refresh timer: {e}")
        
        # Auto-refresh is now handled by the timer method above
    
    def _enable_continuous_refresh(self):
        """Enable continuous refresh using Streamlit's built-in capabilities"""
        try:
            if st.session_state.get('auto_refresh', False):
                # Get refresh settings
                refresh_interval = st.session_state.get('refresh_interval', 5)
                last_refresh = st.session_state.get('last_refresh', datetime.now())
                current_time = datetime.now()
                
                # Calculate time since last refresh
                time_since_refresh = (current_time - last_refresh).total_seconds()
                
                if time_since_refresh >= refresh_interval:
                    # Time to refresh
                    st.session_state.last_refresh = current_time
                    st.session_state.refresh_counter = st.session_state.get('refresh_counter', 0) + 1
                    
                    # Clear cache to force data refresh
                    st.session_state.data_cache = {}
                    
                    # Rerun the app
                    st.rerun()
                else:
                    # Use time.sleep and st.rerun for continuous refresh
                    import time
                    time.sleep(1)  # Wait 1 second
                    st.rerun()  # This will cause the app to refresh
                    
        except Exception as e:
            logger.error(f"Error in continuous refresh: {e}")
    
    def _should_refresh_data(self, data_key: str) -> bool:
        """Check if data should be refreshed based on cache and timing"""
        try:
            # Check if data is in cache
            if data_key not in st.session_state.data_cache:
                return True
            
            # Check if cache is expired (older than refresh interval)
            cache_time = st.session_state.data_cache[data_key].get('timestamp', datetime.min)
            current_time = datetime.now()
            time_diff = (current_time - cache_time).total_seconds()
            
            return time_diff >= st.session_state.refresh_interval
            
        except Exception as e:
            logger.error(f"Error checking refresh status: {e}")
            return True
    
    def _cache_data(self, data_key: str, data: Any) -> None:
        """Cache data with timestamp"""
        try:
            st.session_state.data_cache[data_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def _get_cached_data(self, data_key: str) -> Any:
        """Get cached data if available and not expired"""
        try:
            if data_key in st.session_state.data_cache:
                return st.session_state.data_cache[data_key]['data']
            return None
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None
    
    def _create_sidebar(self):
        """Create sidebar with controls"""
        st.sidebar.header("🎛️ Analysis Controls")
        
        # Symbol selection
        symbol_options = {symbol: info.name for symbol, info in INDIAN_MARKET_SYMBOLS.items()}
        selected_symbol = st.sidebar.selectbox(
            "Select Index",
            options=list(symbol_options.keys()),
            format_func=lambda x: symbol_options[x],
            index=list(symbol_options.keys()).index(st.session_state.selected_symbol)
        )
        
        if selected_symbol != st.session_state.selected_symbol:
            st.session_state.selected_symbol = selected_symbol
        
        # Time period selection
        time_period = st.sidebar.selectbox(
            "Time Period",
            options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=2  # Default to 1mo
        )
        st.session_state.analysis_params['time_period'] = time_period
        
        # Technical indicators
        st.sidebar.subheader("📊 Technical Indicators")
        show_rsi = st.sidebar.checkbox("RSI", value=True)
        show_ema = st.sidebar.checkbox("EMA", value=True)
        show_macd = st.sidebar.checkbox("MACD", value=True)
        show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
        show_vwap_channel = st.sidebar.checkbox("VWAP Price Channel", value=True)
        show_volume = st.sidebar.checkbox("Volume", value=True)
        show_support_resistance = st.sidebar.checkbox("Support/Resistance", value=True)
        
        # Update analysis params
        st.session_state.analysis_params.update({
            'indicators': ['RSI' if show_rsi else None, 'EMA' if show_ema else None, 
                          'MACD' if show_macd else None, 'Bollinger_Bands' if show_bollinger else None,
                          'VWAP_Price_Channel' if show_vwap_channel else None],
            'show_volume': show_volume,
            'show_support_resistance': show_support_resistance,
            'show_vwap_channel': show_vwap_channel
        })
        
        # Options analysis settings
        st.sidebar.subheader("🎯 Options Analysis")
        show_pcr = st.sidebar.checkbox("Put-Call Ratio", value=True)
        show_strategy_recommendations = st.sidebar.checkbox("Strategy Recommendations", value=True)
        
        # Portfolio settings
        st.sidebar.subheader("💼 Portfolio Settings")
        initial_capital = st.sidebar.number_input(
            "Initial Capital (₹)",
            min_value=100000.0,
            max_value=10000000.0,
            value=float(self.config.DEFAULT_INITIAL_CAPITAL),
            step=100000.0
        )
        
        # Update portfolio simulator if capital changed
        if initial_capital != st.session_state.portfolio_simulator.initial_capital:
            st.session_state.portfolio_simulator = IndianPortfolioSimulator(initial_capital)
        
        # Market status
        st.sidebar.subheader("📈 Market Status")
        try:
            market_status = self.data_fetcher.get_market_status()
            
            # Display current session status
            session_status = market_status.get('session_status', 'Unknown')
            current_session = market_status.get('current_session', 'Unknown')
            
            if market_status.get('is_market_open', False):
                st.sidebar.success(f"🟢 {session_status}")
            elif market_status.get('is_pre_market', False):
                st.sidebar.warning(f"🟡 {session_status}")
            elif market_status.get('is_post_market', False):
                st.sidebar.info(f"🔵 {session_status}")
            else:
                st.sidebar.info(f"🔴 {session_status}")
            
            # Show current time
            current_time_ist = market_status.get('current_time_ist', 'Unknown')
            st.sidebar.caption(f"⏰ {current_time_ist}")
            
            # Show next event
            next_event = market_status.get('next_event', '')
            next_event_time = market_status.get('next_event_time', '')
            if next_event and next_event_time:
                st.sidebar.caption(f"⏳ {next_event}: {next_event_time}")
            
            # Show trading hours in expander
            with st.sidebar.expander("📅 Trading Hours"):
                trading_hours = market_status.get('trading_hours', {})
                for session, hours in trading_hours.items():
                    st.write(f"**{session.replace('_', ' ').title()}:** {hours}")
                
                # Show market info
                market_info = market_status.get('market_info', {})
                if market_info:
                    st.write("**Exchange:**", market_info.get('exchange', 'NSE'))
                    st.write("**Timezone:**", market_info.get('timezone', 'IST'))
                    st.write("**Trading Days:**", market_info.get('trading_days', 'Mon-Fri'))
        
        except Exception as e:
            st.sidebar.error(f"Error fetching market status: {e}")
        
        # API Status
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔑 API Status")
        self._show_api_status()
    
    def _show_api_status(self):
        """Show API key configuration status"""
        try:
            config = get_config()
        except Exception as e:
            st.error(f"Configuration error: {e}")
            return
        
        # Check API key status
        api_status = {
            'News API': config.NEWS_API_KEY != "your_news_api_key_here",
            'Alpha Vantage': config.ALPHA_VANTAGE_API_KEY != "your_alpha_vantage_api_key_here",
            'Quandl': config.QUANDL_API_KEY != "your_quandl_api_key_here",
            'Finnhub': config.FINNHUB_API_KEY != "your_finnhub_api_key_here",
            'Polygon': config.POLYGON_API_KEY != "your_polygon_api_key_here",
            'Twitter': config.TWITTER_API_KEY != "your_twitter_api_key_here"
        }
        
        # Display status
        for api_name, is_configured in api_status.items():
            if is_configured:
                st.sidebar.success(f"✅ {api_name}")
            else:
                st.sidebar.warning(f"⚠️ {api_name}")
        
        # Show configuration help
        with st.sidebar.expander("🔧 Configure APIs"):
            st.markdown("""
            **To configure API keys:**
            1. Edit `src/config/config.py`
            2. Replace placeholder values
            3. Restart the application
            
            **See:** `API_KEYS_SETUP.md` for detailed guide
            """)
    
    def _show_market_overview(self):
        """Show market overview dashboard"""
        st.header("📊 Indian Market Overview")
        
        # Live timestamp and real-time status
        try:
            market_status = self.data_fetcher.get_market_status()
            current_time_ist = market_status.get('current_time_ist', datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"))
            session_status = market_status.get('session_status', 'Unknown')
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Last Updated:** {current_time_ist}")
            with col2:
                if market_status.get('is_market_open', False):
                    st.success(f"🟢 {session_status}")
                elif market_status.get('is_pre_market', False):
                    st.warning(f"🟡 {session_status}")
                elif market_status.get('is_post_market', False):
                    st.info(f"🔵 {session_status}")
                else:
                    st.info(f"🔴 {session_status}")
        except:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
            st.markdown(f"**Last Updated:** {current_time}")
        
        try:
            # Check if we need to refresh data
            data_key = f"market_overview_{st.session_state.selected_symbol}"
            
            if self._should_refresh_data(data_key):
                # Fetch market overview data
                with st.spinner("Fetching market data..."):
                    overview_data = self.data_fetcher.fetch_market_overview()
                    sector_data = self.data_fetcher.fetch_sector_performance()
                
                # Cache the data
                self._cache_data(data_key, {
                    'overview_data': overview_data,
                    'sector_data': sector_data
                })
            else:
                # Use cached data
                cached_data = self._get_cached_data(data_key)
                overview_data = cached_data['overview_data']
                sector_data = cached_data['sector_data']
                st.info("📊 Using cached data - click 'Refresh Now' for latest data")
            
            if not overview_data:
                st.error("Unable to fetch market data. Please try again later.")
                return
            
            # Display key metrics with timestamps
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'NIFTY_50' in overview_data:
                    nifty_data = overview_data['NIFTY_50']
                    st.metric(
                        "Nifty 50",
                        f"₹{nifty_data['current_price']:,.0f}",
                        f"{nifty_data['change_percent']:+.2f}%"
                    )
                    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            
            with col2:
                if 'BANK_NIFTY' in overview_data:
                    bank_nifty_data = overview_data['BANK_NIFTY']
                    st.metric(
                        "Bank Nifty",
                        f"₹{bank_nifty_data['current_price']:,.0f}",
                        f"{bank_nifty_data['change_percent']:+.2f}%"
                    )
                    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            
            with col3:
                if 'SENSEX' in overview_data:
                    sensex_data = overview_data['SENSEX']
                    st.metric(
                        "Sensex",
                        f"₹{sensex_data['current_price']:,.0f}",
                        f"{sensex_data['change_percent']:+.2f}%"
                    )
                    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            
            with col4:
                # Market volatility
                try:
                    volatility_data = self.data_fetcher.fetch_historical_volatility('NIFTY_50')
                    if volatility_data:
                        st.metric(
                            "Nifty 50 Volatility",
                            f"{volatility_data.get('annualized_volatility', 0)*100:.1f}%",
                            f"30d: {volatility_data.get('current_30d_volatility', 0)*100:.1f}%"
                        )
                        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
                except:
                    st.metric("Market Volatility", "N/A", "N/A")
                    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            
            # Real-time data section
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("📡 Real-Time Data")
            with col2:
                if st.button("🔄 Refresh RT", key="refresh_realtime_btn"):
                    # Clear real-time data cache
                    realtime_key = f"realtime_{st.session_state.selected_symbol}"
                    if realtime_key in st.session_state.data_cache:
                        del st.session_state.data_cache[realtime_key]
                    st.rerun()
            
            # Fetch and display real-time data
            try:
                realtime_key = f"realtime_{st.session_state.selected_symbol}"
                
                # Check if we should refresh real-time data
                if self._should_refresh_data(realtime_key):
                    with st.spinner("Fetching real-time data..."):
                        realtime_data = self.data_fetcher.fetch_realtime_data(st.session_state.selected_symbol)
                        self._cache_data(realtime_key, realtime_data)
                else:
                    realtime_data = self._get_cached_data(realtime_key)
                    if realtime_data is None:
                        # Fallback if cache is empty
                        with st.spinner("Fetching real-time data..."):
                            realtime_data = self.data_fetcher.fetch_realtime_data(st.session_state.selected_symbol)
                            self._cache_data(realtime_key, realtime_data)
                
                if realtime_data:
                    # Display real-time metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Current Price",
                            f"₹{realtime_data.get('current_price', 0):,.2f}",
                            f"{realtime_data.get('change', 0):+.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Change %",
                            f"{realtime_data.get('change_percent', 0):+.2f}%",
                            f"vs {realtime_data.get('previous_close', 0):,.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Intraday High",
                            f"₹{realtime_data.get('intraday_high', 0):,.2f}",
                            f"Low: ₹{realtime_data.get('intraday_low', 0):,.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Volume",
                            f"{realtime_data.get('volume', 0):,}",
                            f"Session: {realtime_data.get('trading_session', 'Unknown')}"
                        )
                    
                    # Show data source and last updated
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"📊 Data Source: {realtime_data.get('data_source', 'Unknown')}")
                    with col2:
                        st.caption(f"🕒 Last Updated: {realtime_data.get('last_updated', 'Unknown')}")
                    
                    # Show real-time status
                    if realtime_data.get('is_realtime', False):
                        st.success("🟢 Live Real-Time Data")
                    else:
                        st.info("📊 Historical/Demo Data")
                        
            except Exception as e:
                st.error(f"Error fetching real-time data: {e}")
                st.info("Real-time data may not be available during market hours or due to API limitations.")
            
            # Market overview chart
            st.subheader("📈 Market Performance")
            try:
                chart_data = self.visualizer.create_market_overview_chart(overview_data)
                if chart_data:
                    st.plotly_chart(chart_data, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating market chart: {e}")
            
            # Sector performance
            if sector_data:
                st.subheader("🏭 Sector Performance")
                sector_df = pd.DataFrame([
                    {
                        'Sector': sector,
                        'Change %': data.get('change_percent', 0),
                        'Current Price': data.get('current_price', 0)
                    }
                    for sector, data in sector_data.items()
                ])
                
                if not sector_df.empty:
                    st.dataframe(sector_df, use_container_width=True)
            
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
            
            # Display analysis results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"₹{analysis['current_price']:,.0f}")
                st.metric("Overall Signal", analysis['overall_signal'])
                st.metric("Signal Strength", f"{analysis['signal_strength']:.1f}%")
            
            with col2:
                st.metric("Market Trend", analysis['market_trend'])
                st.metric("Volatility", analysis['volatility_level'])
                st.metric("RSI", f"{analysis['rsi']:.1f}")
            
            with col3:
                st.metric("EMA 12", f"₹{analysis['ema_12']:,.0f}")
                st.metric("EMA 26", f"₹{analysis['ema_26']:,.0f}")
                try:
                    vwap_value = analysis['indicators']['vwap_middle'].iloc[-1]
                    if pd.isna(vwap_value):
                        vwap_value = analysis['current_price']
                    st.metric("VWAP", f"₹{vwap_value:,.0f}")
                except:
                    st.metric("VWAP", f"₹{analysis['current_price']:,.0f}")
            
            # VPC Indicator Display (like in the reference chart)
            if 'VWAP_Price_Channel' in st.session_state.analysis_params['indicators']:
                try:
                    vwap_upper = analysis['indicators']['vwap_upper'].iloc[-1]
                    vwap_lower = analysis['indicators']['vwap_lower'].iloc[-1]
                    vwap_middle = analysis['indicators']['vwap_middle'].iloc[-1]
                    current_price = analysis['current_price']
                    
                    # Check for NaN values
                    if pd.isna(vwap_upper) or pd.isna(vwap_lower) or pd.isna(vwap_middle):
                        st.warning("⚠️ VWAP Price Channel data is not available. This might be due to insufficient volume data.")
                        # Use fallback values
                        vwap_upper = current_price * 1.02
                        vwap_lower = current_price * 0.98
                        vwap_middle = current_price
                    
                    st.markdown("---")
                    st.markdown("### 📊 VPC Indicator")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("VPC", f"₹{vwap_middle:.2f}", f"₹{current_price:.2f}")
                    
                    with col2:
                        st.metric("Upper", f"₹{vwap_upper:.2f}")
                    
                    with col3:
                        st.metric("Lower", f"₹{vwap_lower:.2f}")
                    
                    with col4:
                        channel_width = ((vwap_upper - vwap_lower) / vwap_middle) * 100
                        st.metric("Width", f"{channel_width:.1f}%")
                        
                except Exception as e:
                    st.error(f"Error displaying VPC indicator: {e}")
                    logger.error(f"Error displaying VPC indicator: {e}")
            
            # Technical indicators chart
            st.subheader("📊 Technical Indicators Chart")
            
            # Add chart type selector
            chart_type = st.radio(
                "Select Chart Type:",
                ["Simple Test Chart", "Full Technical Analysis", "VWAP Price Channel Focus", "Channel Comparison: Donchian vs VWAP"],
                horizontal=True
            )
            
            try:
                if chart_type == "Simple Test Chart":
                    # Show simple test chart
                    chart = self.visualizer.create_simple_test_chart(data, symbol)
                elif chart_type == "VWAP Price Channel Focus":
                    # Show dedicated VWAP Price Channel chart
                    chart = self.visualizer.create_vwap_price_channel_chart(
                        data, analysis['indicators'], symbol
                    )
                elif chart_type == "Channel Comparison: Donchian vs VWAP":
                    # Show channel comparison chart
                    chart = self.visualizer.create_channel_comparison_chart(
                        data, analysis['indicators'], symbol
                    )
                else:
                    # Show full technical analysis chart
                    chart = self.visualizer.create_indian_technical_analysis_chart(
                        data, analysis['indicators'], symbol
                    )
                
                if chart and len(chart.data) > 0:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.error("Chart creation failed or returned empty chart")
                    st.info("Try selecting 'Simple Test Chart' to verify basic functionality")
            except Exception as e:
                st.error(f"Error creating technical chart: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.info("This might be due to insufficient data or network issues. Try refreshing the page.")
            
            # Detailed analysis
            st.subheader("🔍 Detailed Analysis")
            
            # RSI Analysis
            if 'RSI' in st.session_state.analysis_params['indicators']:
                rsi_value = analysis['rsi']
                if rsi_value > 70:
                    st.warning(f"RSI is {rsi_value:.1f} - Overbought condition")
                elif rsi_value < 30:
                    st.info(f"RSI is {rsi_value:.1f} - Oversold condition")
                else:
                    st.success(f"RSI is {rsi_value:.1f} - Neutral condition")
            
            # EMA Analysis
            if 'EMA' in st.session_state.analysis_params['indicators']:
                ema_12 = analysis['ema_12']
                ema_26 = analysis['ema_26']
                current_price = analysis['current_price']
                
                if ema_12 > ema_26 and current_price > ema_12:
                    st.success("EMA indicates bullish trend")
                elif ema_12 < ema_26 and current_price < ema_12:
                    st.error("EMA indicates bearish trend")
                else:
                    st.info("EMA indicates neutral trend")
            
            # VWAP Price Channel Analysis
            if 'VWAP_Price_Channel' in st.session_state.analysis_params['indicators']:
                vwap_upper = analysis['indicators']['vwap_upper'].iloc[-1]
                vwap_lower = analysis['indicators']['vwap_lower'].iloc[-1]
                vwap_middle = analysis['indicators']['vwap_middle'].iloc[-1]
                current_price = analysis['current_price']
                
                # Calculate distance from VWAP bands
                distance_to_upper = ((current_price - vwap_upper) / vwap_upper) * 100
                distance_to_lower = ((vwap_lower - current_price) / vwap_lower) * 100
                distance_to_vwap = ((current_price - vwap_middle) / vwap_middle) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("VWAP Upper", f"₹{vwap_upper:,.0f}", f"{distance_to_upper:+.2f}%")
                
                with col2:
                    st.metric("VWAP", f"₹{vwap_middle:,.0f}", f"{distance_to_vwap:+.2f}%")
                
                with col3:
                    st.metric("VWAP Lower", f"₹{vwap_lower:,.0f}", f"{distance_to_lower:+.2f}%")
                
                # VWAP Channel Analysis
                if current_price <= vwap_lower:
                    st.warning("Price at VWAP lower band - potential support level")
                elif current_price >= vwap_upper:
                    st.warning("Price at VWAP upper band - potential resistance level")
                elif current_price > vwap_middle:
                    st.success("Price above VWAP - bullish bias")
                elif current_price < vwap_middle:
                    st.error("Price below VWAP - bearish bias")
                else:
                    st.info("Price near VWAP - neutral")
            
            # Donchian Channel Analysis
            if 'donchian_upper' in analysis['indicators'] and 'donchian_lower' in analysis['indicators']:
                donchian_upper = analysis['indicators']['donchian_upper'].iloc[-1]
                donchian_lower = analysis['indicators']['donchian_lower'].iloc[-1]
                donchian_middle = analysis['indicators']['donchian_middle'].iloc[-1]
                current_price = analysis['current_price']
                
                st.markdown("---")
                st.markdown("### 📈 Traditional Donchian Channel Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Donchian Upper", f"₹{donchian_upper:,.0f}")
                
                with col2:
                    st.metric("Donchian Middle", f"₹{donchian_middle:,.0f}")
                
                with col3:
                    st.metric("Donchian Lower", f"₹{donchian_lower:,.0f}")
                
                # Donchian Channel Analysis
                if current_price >= donchian_upper:
                    st.success("Price at Donchian upper band - potential breakout")
                elif current_price <= donchian_lower:
                    st.warning("Price at Donchian lower band - potential breakdown")
                elif current_price > donchian_middle:
                    st.info("Price above Donchian middle - bullish bias")
                elif current_price < donchian_middle:
                    st.error("Price below Donchian middle - bearish bias")
                else:
                    st.info("Price near Donchian middle - neutral")
            
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
                technical_analysis = self.technical_analyzer.analyze_index(data, symbol)
                current_price = data['Close'].iloc[-1]
                
                # Analyze options
                options_analysis = self.options_engine.analyze_options_chain(options_data, current_price, symbol)
                
                # Get strategy recommendation
                strategy = self.options_engine.recommend_strategy(
                    technical_analysis['overall_signal'],
                    options_analysis,
                    current_price,
                    symbol,
                    'normal'
                )
            
            # Display current price and lot size
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"₹{current_price:,.0f}")
            with col2:
                lot_size = self.options_engine.get_lot_size(symbol)
                st.metric("Lot Size", lot_size)
            
            # Strategy recommendation
            st.subheader("🎯 Strategy Recommendation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Strategy:** {strategy.name}")
                st.write(f"**Description:** {strategy.description}")
                st.write(f"**Max Profit:** ₹{strategy.max_profit:,.0f}")
                st.write(f"**Max Loss:** ₹{strategy.max_loss:,.0f}")
                st.write(f"**Probability of Profit:** {strategy.probability_of_profit*100:.1f}%")
            
            with col2:
                # Format breakeven points properly
                if strategy.breakeven_points:
                    be_points = [f"₹{float(point):,.2f}" for point in strategy.breakeven_points]
                    st.write(f"**Breakeven Points:** {', '.join(be_points)}")
                else:
                    st.write(f"**Breakeven Points:** None")
                st.write(f"**Risk-Reward Ratio:** {strategy.risk_reward_ratio:.2f}")
                st.write(f"**Margin Required:** ₹{strategy.margin_required:,.0f}")
                st.write(f"**Lot Size:** {strategy.lot_size}")
            
            # Strategy payoff chart
            st.subheader("📊 Strategy Payoff Diagram")
            try:
                # Create strategy data for chart
                strategy_data = {
                    'name': strategy.name,
                    'description': strategy.description,
                    'max_profit': strategy.max_profit,
                    'max_loss': strategy.max_loss,
                    'breakeven_points': strategy.breakeven_points,
                    'risk_reward_ratio': strategy.risk_reward_ratio,
                    'probability_of_profit': strategy.probability_of_profit,
                    'current_price': current_price,
                    'legs': getattr(strategy, 'legs', [])  # Get strategy legs if available
                }
                
                # Create and display chart
                chart = self.visualizer.create_indian_options_strategy_chart(strategy_data)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("Unable to generate strategy chart")
            except Exception as e:
                st.error(f"Error creating strategy chart: {e}")
                logger.error(f"Error creating strategy chart: {e}")
            
            # Strategy action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Add to Portfolio", type="primary"):
                    # Add strategy to portfolio (simplified)
                    st.success("Strategy added to portfolio!")
            
            with col2:
                if st.button("Analyze More"):
                    st.info("Detailed analysis would be shown here")
            
            with col3:
                if st.button("Save Strategy"):
                    st.success("Strategy saved!")
            
            # Options chain analysis
            if options_data and 'expirations' in options_data:
                st.subheader("📊 Options Chain Analysis")
                
                # Show available expirations
                expirations = list(options_data['expirations'].keys())
                if expirations:
                    selected_expiry = st.selectbox("Select Expiration", expirations)
                    
                    if selected_expiry in options_data['expirations']:
                        expiry_data = options_data['expirations'][selected_expiry]
                        
                        # Display ATM analysis
                        if 'atm_analysis' in expiry_data:
                            atm = expiry_data['atm_analysis']
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("ATM Strike", f"₹{atm.get('strike', 0):,.0f}")
                            with col2:
                                st.metric("Call IV", f"{atm.get('call_iv', 0)*100:.1f}%")
                            with col3:
                                st.metric("Put IV", f"{atm.get('put_iv', 0)*100:.1f}%")
            
            # Options volatility chart
            if options_data and 'expirations' in options_data:
                st.subheader("📈 Options Volatility Analysis")
                try:
                    # Create IV chart data
                    iv_chart_data = self._create_iv_chart_data(options_data, current_price)
                    if iv_chart_data:
                        chart = self.visualizer.create_indian_market_heatmap(iv_chart_data)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.info("IV chart data available but chart generation failed")
                    else:
                        st.info("No IV data available for chart")
                except Exception as e:
                    st.warning(f"Could not create IV chart: {e}")
            
        except Exception as e:
            st.error(f"Error in options strategies: {e}")
            logger.error(f"Error in options strategies: {e}")
    
    def _show_portfolio_management(self):
        """Show portfolio management with live paper trading"""
        st.header("💼 Portfolio Management")
        
        # Live timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        st.caption(f"Last updated: {current_time}")
        
        # Auto-refresh controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader("📊 Live Paper Trading")
        with col2:
            if st.button("🔄 Refresh Positions", key="refresh_portfolio"):
                st.rerun()
        with col3:
            auto_refresh_portfolio = st.checkbox("Auto Refresh (10s)", value=True, key="auto_refresh_portfolio")
            
            # Auto-refresh logic
            if auto_refresh_portfolio:
                import time
                time.sleep(10)  # Wait 10 seconds
                st.rerun()
        
        try:
            # Get live market data and update positions
            with st.spinner("🔄 Updating positions with live market data..."):
                # Get live market prices
                live_prices = st.session_state.portfolio_simulator.get_live_market_data(self.data_fetcher)
                
                # Calculate real-time metrics
                real_time_metrics = st.session_state.portfolio_simulator.calculate_real_time_metrics(live_prices)
                
                # Update positions with live data
                live_data = st.session_state.portfolio_simulator.update_all_positions_with_live_data(self.data_fetcher)
            
            # Portfolio summary
            portfolio_summary = st.session_state.portfolio_simulator.get_portfolio_summary()
            
            if not portfolio_summary:
                st.warning("No portfolio data available")
                return
            
            # Real-time portfolio metrics
            st.subheader("📊 Real-Time Portfolio Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = real_time_metrics.get('total_value', portfolio_summary.get('total_value', 0))
                st.metric(
                    "Total Portfolio Value",
                    f"₹{total_value:,.0f}",
                    delta=f"₹{real_time_metrics.get('unrealized_pnl', 0):,.0f}"
                )
            
            with col2:
                total_pnl = real_time_metrics.get('total_pnl', 0)
                total_pnl_percent = real_time_metrics.get('total_pnl_percent', 0)
                st.metric(
                    "Total P&L",
                    f"₹{total_pnl:,.0f}",
                    delta=f"{total_pnl_percent:+.2f}%"
                )
            
            with col3:
                margin_used = real_time_metrics.get('margin_used', 0)
                available_capital = real_time_metrics.get('available_capital', 0)
                st.metric(
                    "Margin Used",
                    f"₹{margin_used:,.0f}",
                    delta=f"₹{available_capital:,.0f} available"
                )
            
            with col4:
                position_count = real_time_metrics.get('position_count', 0)
                st.metric(
                    "Open Positions",
                    position_count
                )
            
            # Portfolio performance indicator
            total_pnl = real_time_metrics.get('total_pnl', 0)
            total_pnl_percent = real_time_metrics.get('total_pnl_percent', 0)
            
            if total_pnl > 0:
                st.success(f"🎉 Portfolio is up ₹{total_pnl:,.2f} ({total_pnl_percent:+.2f}%)")
            elif total_pnl < 0:
                st.error(f"📉 Portfolio is down ₹{abs(total_pnl):,.2f} ({total_pnl_percent:+.2f}%)")
            else:
                st.info("📊 Portfolio is at break-even")
            
            # Live market prices
            if real_time_metrics.get('live_prices'):
                st.subheader("📈 Live Market Prices")
                live_prices = real_time_metrics['live_prices']
                price_cols = st.columns(len(live_prices))
                
                for i, (symbol, price) in enumerate(live_prices.items()):
                    with price_cols[i]:
                        st.metric(
                            symbol,
                            f"₹{price:,.2f}",
                            delta=f"Live"
                        )
            
            # Current positions with live P&L
            st.subheader("📊 Current Positions")
            
            positions = live_data.get('positions', [])
            open_positions = [p for p in positions if p.get('status') == 'open']
            
            if open_positions:
                # Display positions in expandable cards
                for position in open_positions:
                    with st.expander(f"📊 {position['symbol']} - {position['strategy']} ({position['quantity']} lots)", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Entry Price:** ₹{position['entry_price']:,.2f}")
                            st.write(f"**Current Price:** ₹{position['current_price']:,.2f}")
                            st.write(f"**Quantity:** {position['quantity']} lots")
                        
                        with col2:
                            pnl_color = "green" if position['unrealized_pnl'] >= 0 else "red"
                            st.markdown(f"**Unrealized P&L:** <span style='color: {pnl_color}'>₹{position['unrealized_pnl']:,.2f}</span>", unsafe_allow_html=True)
                            st.markdown(f"**P&L %:** <span style='color: {pnl_color}'>{position['unrealized_pnl_percent']:+.2f}%</span>", unsafe_allow_html=True)
                            st.write(f"**Lot Size:** {position['lot_size']}")
                        
                        with col3:
                            if st.button(f"Close Position", key=f"close_{position['position_id']}"):
                                result = st.session_state.portfolio_simulator.close_position(position['position_id'])
                                if result['success']:
                                    st.success(f"Position closed! Realized P&L: ₹{result['realized_pnl']:,.2f}")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to close position: {result['error']}")
            else:
                st.info("No open positions")
            
            # Dynamic Capital Management
            st.subheader("💰 Capital Management")
            st.info("📊 **Margin Structure:** Using NIFTY 50 margin requirements (12%) for all instruments")
            
            # Real-time capital display with live updates
            # Get fresh portfolio summary to ensure we have the latest capital values
            portfolio_summary = st.session_state.portfolio_simulator.get_portfolio_summary()
            if portfolio_summary:
                capital_util = portfolio_summary.get('capital_utilization', {})
                # Get live prices first
                live_prices = st.session_state.portfolio_simulator.get_live_market_data(self.data_fetcher)
                real_time_metrics = st.session_state.portfolio_simulator.calculate_real_time_metrics(live_prices)
                
                col_cap1, col_cap2, col_cap3, col_cap4 = st.columns(4)
                
                with col_cap1:
                    # Use the actual current capital from the simulator, not cached values
                    total_capital = st.session_state.portfolio_simulator.initial_capital
                    st.metric("💰 Total Capital", f"₹{total_capital:,.2f}")
                
                with col_cap2:
                    available_capital = real_time_metrics.get('available_capital', 0)
                    st.metric("💵 Available Capital", f"₹{available_capital:,.2f}")
                
                with col_cap3:
                    margin_used = real_time_metrics.get('margin_used', 0)
                    st.metric("📊 Margin Used", f"₹{margin_used:,.2f}")
                
                with col_cap4:
                    utilization_percent = (margin_used / total_capital * 100) if total_capital > 0 else 0
                    st.metric("📈 Capital Utilization", f"{utilization_percent:.1f}%")
                
                # Add auto-refresh for real-time updates
                col_refresh1, col_refresh2 = st.columns(2)
                with col_refresh1:
                    if st.button("🔄 Refresh Portfolio", help="Update portfolio with latest market data"):
                        st.rerun()
                with col_refresh2:
                    if st.button("💰 Update Capital", help="Refresh capital calculations"):
                        st.rerun()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("➕ Add Capital", help="Add more capital to your portfolio"):
                    st.session_state.show_add_capital = True
            
            with col2:
                if st.button("🔄 Reset Capital", help="Reset portfolio with new capital amount"):
                    st.session_state.show_reset_capital = True
            
            with col3:
                if st.button("📊 Capital Report", help="View detailed capital utilization"):
                    st.session_state.show_capital_report = True
            
            # Quick capital boost for testing
            st.write("**Quick Capital Boost (for testing):**")
            col_boost1, col_boost2, col_boost3, col_boost4 = st.columns(4)
            
            with col_boost1:
                if st.button("+₹10K", help="Add ₹10,000", key="boost_10k"):
                    success = st.session_state.portfolio_simulator.add_capital(10000)
                    if success:
                        st.success("✅ Added ₹10,000!")
                        st.balloons()
                    st.rerun()
            
            with col_boost2:
                if st.button("+₹25K", help="Add ₹25,000", key="boost_25k"):
                    success = st.session_state.portfolio_simulator.add_capital(25000)
                    if success:
                        st.success("✅ Added ₹25,000!")
                        st.balloons()
                    st.rerun()
            
            with col_boost3:
                if st.button("+₹50K", help="Add ₹50,000", key="boost_50k"):
                    success = st.session_state.portfolio_simulator.add_capital(50000)
                    if success:
                        st.success("✅ Added ₹50,000!")
                        st.balloons()
                    st.rerun()
            
            with col_boost4:
                if st.button("+₹1L", help="Add ₹1,00,000", key="boost_1l"):
                    success = st.session_state.portfolio_simulator.add_capital(100000)
                    if success:
                        st.success("✅ Added ₹1,00,000!")
                        st.balloons()
                    st.rerun()
            
            # Removed custom amount input as requested
            
            # Add Capital Form
            if st.session_state.get('show_add_capital', False):
                with st.form("add_capital_form"):
                    st.write("**Add Capital to Portfolio**")
                    add_amount = st.number_input("Amount to Add (₹)", min_value=1000, value=10000, step=1000)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.form_submit_button("Add Capital"):
                            success = st.session_state.portfolio_simulator.add_capital(add_amount)
                            if success:
                                st.success(f"✅ Added ₹{add_amount:,.2f} to portfolio!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to add capital")
                    
                    with col_b:
                        if st.form_submit_button("Cancel"):
                            st.session_state.show_add_capital = False
                            st.rerun()
            
            # Reset Capital Form
            if st.session_state.get('show_reset_capital', False):
                with st.form("reset_capital_form"):
                    st.write("**Reset Portfolio Capital**")
                    st.warning("⚠️ This will close all positions and reset your portfolio!")
                    new_capital = st.number_input("New Capital Amount (₹)", min_value=10000, value=100000, step=10000)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.form_submit_button("Reset Portfolio"):
                            success = st.session_state.portfolio_simulator.reset_capital(new_capital)
                            if success:
                                st.success(f"✅ Portfolio reset with ₹{new_capital:,.2f}!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to reset portfolio")
                    
                    with col_b:
                        if st.form_submit_button("Cancel"):
                            st.session_state.show_reset_capital = False
                            st.rerun()
            
            # Capital Report
            if st.session_state.get('show_capital_report', False):
                st.subheader("📊 Capital Utilization Report")
                portfolio_summary = st.session_state.portfolio_simulator.get_portfolio_summary()
                # Get live prices for real-time metrics
                live_prices = st.session_state.portfolio_simulator.get_live_market_data(self.data_fetcher)
                real_time_metrics = st.session_state.portfolio_simulator.calculate_real_time_metrics(live_prices)
                
                if portfolio_summary:
                    capital_util = portfolio_summary.get('capital_utilization', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        # Use current capital from simulator
                        current_capital = st.session_state.portfolio_simulator.initial_capital
                        st.metric("Total Capital", f"₹{current_capital:,.2f}")
                    with col2:
                        st.metric("Available Capital", f"₹{real_time_metrics.get('available_capital', 0):,.2f}")
                    with col3:
                        st.metric("Margin Used", f"₹{real_time_metrics.get('margin_used', 0):,.2f}")
                    with col4:
                        utilization = (real_time_metrics.get('margin_used', 0) / current_capital * 100)
                        st.metric("Utilization", f"{utilization:.1f}%")
                    
                    # Additional portfolio metrics
                    st.write("**📈 Portfolio Performance:**")
                    col_perf1, col_perf2, col_perf3 = st.columns(3)
                    with col_perf1:
                        st.metric("Total Value", f"₹{real_time_metrics.get('total_value', 0):,.2f}")
                    with col_perf2:
                        st.metric("Total P&L", f"₹{real_time_metrics.get('total_pnl', 0):,.2f}")
                    with col_perf3:
                        st.metric("P&L %", f"{real_time_metrics.get('total_pnl_percent', 0):+.2f}%")
                    
                    # Capital transaction history
                    st.write("**📋 Recent Capital Transactions:**")
                    if hasattr(st.session_state.portfolio_simulator, 'capital_transactions') and st.session_state.portfolio_simulator.capital_transactions:
                        for transaction in st.session_state.portfolio_simulator.capital_transactions[-5:]:  # Show last 5
                            st.write(f"• {transaction['timestamp']}: {transaction['type']} ₹{transaction['amount']:,.2f}")
                    else:
                        st.write("• No transaction history available")
                    
                    if st.button("Close Report"):
                        st.session_state.show_capital_report = False
                        st.rerun()
            
            # Add new position
            st.subheader("➕ Add New Position")
            
            # Show current portfolio status
            portfolio_summary = st.session_state.portfolio_simulator.get_portfolio_summary()
            if portfolio_summary:
                # Get real-time metrics for accurate display
                live_prices = st.session_state.portfolio_simulator.get_live_market_data(self.data_fetcher)
                real_time_metrics = st.session_state.portfolio_simulator.calculate_real_time_metrics(live_prices)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    available_capital = real_time_metrics.get('available_capital', 0)
                    st.metric("Available Capital", f"₹{available_capital:,.2f}")
                with col2:
                    margin_used = real_time_metrics.get('margin_used', 0)
                    st.metric("Margin Used", f"₹{margin_used:,.2f}")
                with col3:
                    unrealized_pnl = real_time_metrics.get('unrealized_pnl', 0)
                    unrealized_pnl_percent = real_time_metrics.get('unrealized_pnl_percent', 0)
                    st.metric("Unrealized P&L", f"₹{unrealized_pnl:,.2f}", 
                             delta=f"{unrealized_pnl_percent:+.2f}%")
            
            with st.form("add_position"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    symbol = st.selectbox("Symbol", list(INDIAN_MARKET_SYMBOLS.keys()))
                
                with col2:
                    quantity = st.number_input("Quantity (Lots)", min_value=1, value=1, help="Number of lots to trade")
                
                with col3:
                    # Get current market price for the selected symbol
                    try:
                        current_data = self.data_fetcher.fetch_index_data(symbol, "1d", "1m")
                        if not current_data.empty:
                            current_price = current_data['Close'].iloc[-1]
                            entry_price = st.number_input("Entry Price", min_value=0.01, value=float(current_price), step=0.01)
                        else:
                            entry_price = st.number_input("Entry Price", min_value=0.01, value=100.0, step=0.01)
                    except:
                        entry_price = st.number_input("Entry Price", min_value=0.01, value=100.0, step=0.01)
                
                with col4:
                    position_type = st.selectbox("Type", ["Long", "Short"])
                
                # Show estimated margin requirement
                if symbol and quantity and entry_price:
                    try:
                        lot_size = st.session_state.portfolio_simulator.lot_sizes.get(symbol, 50)
                        notional_value = quantity * lot_size * entry_price
                        # Use NIFTY 50 margin requirements (12%)
                        margin_rate = st.session_state.portfolio_simulator.margin_requirements.get(symbol, 0.12)
                        estimated_margin = notional_value * margin_rate
                        
                        # Get current available capital
                        portfolio_summary = st.session_state.portfolio_simulator.get_portfolio_summary()
                        live_prices = st.session_state.portfolio_simulator.get_live_market_data(self.data_fetcher)
                        real_time_metrics = st.session_state.portfolio_simulator.calculate_real_time_metrics(live_prices)
                        available_capital = real_time_metrics.get('available_capital', 0)
                        
                        # Color code based on capital availability
                        if estimated_margin <= available_capital:
                            st.success(f"✅ **Estimated Margin Required:** ₹{estimated_margin:,.2f} (Lot Size: {lot_size}, Notional: ₹{notional_value:,.2f})")
                        elif estimated_margin <= available_capital * 1.5:
                            st.warning(f"⚠️ **Estimated Margin Required:** ₹{estimated_margin:,.2f} (Lot Size: {lot_size}, Notional: ₹{notional_value:,.2f}) - Low capital warning")
                        else:
                            st.error(f"❌ **Estimated Margin Required:** ₹{estimated_margin:,.2f} (Lot Size: {lot_size}, Notional: ₹{notional_value:,.2f}) - Insufficient capital")
                        
                        st.info(f"💡 **Available Capital:** ₹{available_capital:,.2f} | **NIFTY 50 Margin Rate:** {margin_rate*100:.1f}%")
                    except Exception as e:
                        st.warning(f"Could not calculate margin requirement: {e}")
                
                if st.form_submit_button("Add Position"):
                    try:
                        # Validate inputs
                        if quantity <= 0:
                            st.error("Quantity must be greater than 0")
                        elif entry_price <= 0:
                            st.error("Entry price must be greater than 0")
                        else:
                            # Add position and check result
                            success = st.session_state.portfolio_simulator.add_position(
                                symbol=symbol,
                                instrument_type="index",
                                quantity=quantity,
                                entry_price=entry_price,
                                strategy=position_type
                            )
                            
                            if success:
                                st.success(f"✅ Position added successfully! {quantity} lots of {symbol} at ₹{entry_price:,.2f}")
                                st.rerun()
                            else:
                                # Get current capital status for better error message
                                portfolio_summary = st.session_state.portfolio_simulator.get_portfolio_summary()
                                live_prices = st.session_state.portfolio_simulator.get_live_market_data(self.data_fetcher)
                                real_time_metrics = st.session_state.portfolio_simulator.calculate_real_time_metrics(live_prices)
                                available_capital = real_time_metrics.get('available_capital', 0)
                                
                                st.error(f"❌ Failed to add position. Available capital: ₹{available_capital:,.2f}")
                                st.info("💡 Try adding more capital using the 'Add Capital' button above, or reduce the position size.")
                    except Exception as e:
                        st.error(f"❌ Error adding position: {e}")
                        logger.error(f"Error adding position: {e}")
            
            # Portfolio performance chart
            st.subheader("📈 Portfolio Performance")
            try:
                # Get portfolio history (simplified)
                portfolio_history = st.session_state.portfolio_simulator.get_portfolio_history()
                if portfolio_history:
                    chart = self.visualizer.create_indian_portfolio_performance_chart(portfolio_history)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating portfolio chart: {e}")
            
            # Auto-refresh functionality
            if auto_refresh_portfolio:
                st.markdown("---")
                st.info("🔄 Auto-refresh enabled. Portfolio will update every 10 seconds.")
                # JavaScript auto-refresh
                st.markdown("""
                <script>
                setTimeout(function(){
                    window.location.reload();
                }, 10000);
                </script>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error in portfolio management: {e}")
            logger.error(f"Error in portfolio management: {e}")
    
    def _show_live_predictions(self):
        """Show live predictions for market entry"""
        st.header("🔮 Live Nifty 50 Predictions")
        
        # Live timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        st.caption(f"Last updated: {current_time}")
        
        # Prediction controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("🎯 Market Entry Predictions")
        
        with col2:
            if st.button("🔄 Refresh Predictions", key="refresh_predictions"):
                st.rerun()
        
        with col3:
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=True, key="auto_refresh_predictions")
        
        try:
            # Fetch recent data for training and prediction
            with st.spinner("Analyzing market data for predictions..."):
                # Get recent data (last 30 days for training)
                data = self.data_fetcher.fetch_index_data("NIFTY_50", "30d", "5m")
                
                if data.empty:
                    st.error("No data available for predictions")
                    return
                
                # Train the prediction model
                if not self.prediction_engine.is_trained:
                    training_success = self.prediction_engine.train_models(data)
                    if training_success:
                        st.success("✅ Prediction model trained successfully")
                    else:
                        st.warning("⚠️ Using statistical prediction methods")
                
                # Generate predictions
                prediction = self.prediction_engine.predict_prices(data)
                entry_signal = self.prediction_engine.get_entry_signal(prediction)
            
            # Display prediction results
            st.subheader("📊 Price Predictions")
            
            # Current price
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"₹{prediction.current_price:,.2f}",
                    delta=None
                )
            
            with col2:
                change_1m = prediction.predicted_price_1m - prediction.current_price
                st.metric(
                    "1m Prediction",
                    f"₹{prediction.predicted_price_1m:,.2f}",
                    delta=f"{change_1m:+.2f}",
                    delta_color="normal"
                )
            
            with col3:
                change_2m = prediction.predicted_price_2m - prediction.current_price
                st.metric(
                    "2m Prediction",
                    f"₹{prediction.predicted_price_2m:,.2f}",
                    delta=f"{change_2m:+.2f}",
                    delta_color="normal"
                )
            
            with col4:
                change_5m = prediction.predicted_price_5m - prediction.current_price
                st.metric(
                    "5m Prediction",
                    f"₹{prediction.predicted_price_5m:,.2f}",
                    delta=f"{change_5m:+.2f}",
                    delta_color="normal"
                )
            
            with col5:
                change_10m = prediction.predicted_price_10m - prediction.current_price
                st.metric(
                    "10m Prediction",
                    f"₹{prediction.predicted_price_10m:,.2f}",
                    delta=f"{change_10m:+.2f}",
                    delta_color="normal"
                )
            
            # Confidence metrics
            st.subheader("🎯 Prediction Confidence")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "1m Confidence",
                    f"{prediction.confidence_1m:.1%}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "2m Confidence",
                    f"{prediction.confidence_2m:.1%}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "5m Confidence",
                    f"{prediction.confidence_5m:.1%}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "10m Confidence",
                    f"{prediction.confidence_10m:.1%}",
                    delta=None
                )
            
            # Prediction charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Price prediction chart
                prediction_data = {
                    'current_price': prediction.current_price,
                    'predicted_price_1m': prediction.predicted_price_1m,
                    'predicted_price_2m': prediction.predicted_price_2m,
                    'predicted_price_5m': prediction.predicted_price_5m,
                    'predicted_price_10m': prediction.predicted_price_10m,
                    'confidence_1m': prediction.confidence_1m,
                    'confidence_2m': prediction.confidence_2m,
                    'confidence_5m': prediction.confidence_5m,
                    'confidence_10m': prediction.confidence_10m,
                    'timestamp': prediction.timestamp
                }
                
                prediction_chart = self.visualizer.create_live_prediction_chart(prediction_data)
                st.plotly_chart(prediction_chart, use_container_width=True)
            
            with col2:
                # Entry signal chart
                signal_data = {
                    'signal': entry_signal.signal,
                    'confidence': entry_signal.confidence,
                    'target_price': entry_signal.target_price,
                    'stop_loss': entry_signal.stop_loss,
                    'current_price': prediction.current_price,
                    'risk_reward_ratio': entry_signal.risk_reward_ratio
                }
                
                signal_chart = self.visualizer.create_entry_signal_chart(signal_data)
                st.plotly_chart(signal_chart, use_container_width=True)
            
            # Entry signals and recommendations
            st.subheader("🎯 Market Entry Signals")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Signal details
                signal_color = "green" if entry_signal.signal == "BUY" else "red" if entry_signal.signal == "SELL" else "orange"
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {signal_color}20; border-left: 5px solid {signal_color};">
                    <h3 style="color: {signal_color}; margin-top: 0;">{entry_signal.signal} Signal</h3>
                    <p><strong>Confidence:</strong> {entry_signal.confidence:.1%}</p>
                    <p><strong>Target Price:</strong> ₹{entry_signal.target_price:,.2f}</p>
                    <p><strong>Stop Loss:</strong> ₹{entry_signal.stop_loss:,.2f}</p>
                    <p><strong>Risk-Reward Ratio:</strong> {entry_signal.risk_reward_ratio:.2f}</p>
                    <p><strong>Reasoning:</strong> {entry_signal.reasoning}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Enhanced Direction Analysis with Price Predictions and Market Trends
                st.subheader("📊 Direction Analysis & Market Trends")
                
                # Helper function to get color for status
                def get_status_color(status, status_type="direction"):
                    if status_type == "direction":
                        return "🟢" if status == "UP" else "🔴" if status == "DOWN" else "🟡"
                    elif status_type == "trend":
                        return "🟢" if status == "BULLISH" else "🔴" if status == "BEARISH" else "🟡"
                    elif status_type == "strength":
                        return "🟢" if status == "STRONG" else "🟡" if status == "MODERATE" else "🔴"
                    elif status_type == "risk":
                        return "🟢" if status == "LOW" else "🟡" if status == "MEDIUM" else "🔴"
                    return "⚪"
                
                # 1-Minute Analysis
                with st.expander("🎯 1-Minute Analysis", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Price Prediction", f"₹{prediction.predicted_price_1m:,.2f}", f"{prediction.price_change_1m:+.2f}%")
                        st.write(f"**Direction:** {get_status_color(prediction.direction_1m)} {prediction.direction_1m}")
                    with col_b:
                        st.metric("Confidence", f"{prediction.confidence_1m:.1%}")
                        st.write(f"**Trend:** {get_status_color(prediction.trend_1m, 'trend')} {prediction.trend_1m}")
                    st.write(f"**Trend Strength:** {get_status_color(prediction.trend_strength_1m, 'strength')} {prediction.trend_strength_1m}")
                
                # 2-Minute Analysis
                with st.expander("🎯 2-Minute Analysis", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Price Prediction", f"₹{prediction.predicted_price_2m:,.2f}", f"{prediction.price_change_2m:+.2f}%")
                        st.write(f"**Direction:** {get_status_color(prediction.direction_2m)} {prediction.direction_2m}")
                    with col_b:
                        st.metric("Confidence", f"{prediction.confidence_2m:.1%}")
                        st.write(f"**Trend:** {get_status_color(prediction.trend_2m, 'trend')} {prediction.trend_2m}")
                    st.write(f"**Trend Strength:** {get_status_color(prediction.trend_strength_2m, 'strength')} {prediction.trend_strength_2m}")
                
                # 5-Minute Analysis
                with st.expander("🎯 5-Minute Analysis", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Price Prediction", f"₹{prediction.predicted_price_5m:,.2f}", f"{prediction.price_change_5m:+.2f}%")
                        st.write(f"**Direction:** {get_status_color(prediction.direction_5m)} {prediction.direction_5m}")
                    with col_b:
                        st.metric("Confidence", f"{prediction.confidence_5m:.1%}")
                        st.write(f"**Trend:** {get_status_color(prediction.trend_5m, 'trend')} {prediction.trend_5m}")
                    st.write(f"**Trend Strength:** {get_status_color(prediction.trend_strength_5m, 'strength')} {prediction.trend_strength_5m}")
                
                # 10-Minute Analysis
                with st.expander("🎯 10-Minute Analysis", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Price Prediction", f"₹{prediction.predicted_price_10m:,.2f}", f"{prediction.price_change_10m:+.2f}%")
                        st.write(f"**Direction:** {get_status_color(prediction.direction_10m)} {prediction.direction_10m}")
                    with col_b:
                        st.metric("Confidence", f"{prediction.confidence_10m:.1%}")
                        st.write(f"**Trend:** {get_status_color(prediction.trend_10m, 'trend')} {prediction.trend_10m}")
                    st.write(f"**Trend Strength:** {get_status_color(prediction.trend_strength_10m, 'strength')} {prediction.trend_strength_10m}")
                
                # Overall Risk Level
                st.info(f"**Overall Risk Level:** {get_status_color(prediction.risk_level, 'risk')} {prediction.risk_level}")
            
            # Trading recommendations
            st.subheader("💡 Trading Recommendations")
            
            if entry_signal.signal == "BUY":
                st.success(f"""
                **🟢 BUY RECOMMENDATION**
                
                - **Entry Price:** ₹{prediction.current_price:,.2f}
                - **Target:** ₹{entry_signal.target_price:,.2f} ({(entry_signal.target_price/prediction.current_price - 1)*100:.2f}% gain)
                - **Stop Loss:** ₹{entry_signal.stop_loss:,.2f} ({(entry_signal.stop_loss/prediction.current_price - 1)*100:.2f}% loss)
                - **Risk-Reward:** {entry_signal.risk_reward_ratio:.2f}:1
                - **Confidence:** {entry_signal.confidence:.1%}
                
                **Strategy:** Consider entering long position with tight stop loss.
                """)
            elif entry_signal.signal == "SELL":
                st.error(f"""
                **🔴 SELL RECOMMENDATION**
                
                - **Entry Price:** ₹{prediction.current_price:,.2f}
                - **Target:** ₹{entry_signal.target_price:,.2f} ({(entry_signal.target_price/prediction.current_price - 1)*100:.2f}% gain)
                - **Stop Loss:** ₹{entry_signal.stop_loss:,.2f} ({(entry_signal.stop_loss/prediction.current_price - 1)*100:.2f}% loss)
                - **Risk-Reward:** {entry_signal.risk_reward_ratio:.2f}:1
                - **Confidence:** {entry_signal.confidence:.1%}
                
                **Strategy:** Consider entering short position or exiting long positions.
                """)
            else:
                st.warning(f"""
                **🟡 HOLD RECOMMENDATION**
                
                - **Current Price:** ₹{prediction.current_price:,.2f}
                - **Confidence:** {entry_signal.confidence:.1%}
                - **Risk Level:** {prediction.risk_level}
                
                **Strategy:** Market conditions are unclear. Wait for better entry opportunity.
                """)
            
            # Model performance
            st.subheader("📈 Model Performance")
            accuracy = self.prediction_engine.get_prediction_accuracy()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("1m Accuracy", f"{accuracy.get('accuracy_1m', 0):.1%}")
            with col2:
                st.metric("2m Accuracy", f"{accuracy.get('accuracy_2m', 0):.1%}")
            with col3:
                st.metric("5m Accuracy", f"{accuracy['accuracy_5m']:.1%}")
            with col4:
                st.metric("10m Accuracy", f"{accuracy['accuracy_10m']:.1%}")
            with col5:
                st.metric("Total Predictions", accuracy['total_predictions'])
            
            # Auto-refresh functionality
            if auto_refresh:
                st.markdown("---")
                st.info("🔄 Auto-refresh enabled. Predictions will update every 30 seconds.")
                # JavaScript auto-refresh
                st.markdown("""
                <script>
                setTimeout(function(){
                    window.location.reload();
                }, 30000);
                </script>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            logger.error(f"Error in live predictions: {e}")

    def _show_reports(self):
        """Show comprehensive reports with export and email functionality"""
        st.header("📋 Comprehensive Reports & Analytics")
        
        # Report configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Market Analysis", "Portfolio Report", "Options Analysis", "Technical Analysis", "Complete Report"]
            )
        
        with col2:
            format_type = st.selectbox(
                "Export Format",
                ["JSON", "Excel", "CSV", "PDF"]
            )
        
        with col3:
            include_charts = st.checkbox("Include Charts", value=True)
        
        # Email configuration
        st.subheader("📧 Email Configuration (Optional)")
        email_col1, email_col2 = st.columns(2)
        
        with email_col1:
            recipient_email = st.text_input("Recipient Email", placeholder="example@email.com")
        
        with email_col2:
            sender_email = st.text_input("Sender Email", placeholder="your@email.com")
            sender_password = st.text_input("Sender Password", type="password", placeholder="App Password")
        
        # Generate report button
        if st.button("🚀 Generate Comprehensive Report", type="primary"):
            try:
                symbol = st.session_state.selected_symbol
                time_period = st.session_state.analysis_params['time_period']
                
                with st.spinner("Generating comprehensive report..."):
                    # Fetch all data
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
                
                # Display comprehensive report
                st.subheader("📊 Comprehensive Market Report")
                st.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
                st.write(f"**Symbol:** {INDIAN_MARKET_SYMBOLS[symbol].name}")
                st.write(f"**Time Period:** {time_period}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Technical Analysis Summary")
                    st.write(f"**Current Price:** ₹{technical_analysis['current_price']:,.0f}")
                    st.write(f"**Overall Signal:** {technical_analysis['overall_signal']}")
                    st.write(f"**Market Trend:** {technical_analysis['market_trend']}")
                    st.write(f"**Volatility:** {technical_analysis['volatility_level']}")
                    st.write(f"**RSI:** {technical_analysis['rsi']:.1f}")
                    st.write(f"**EMA 12:** ₹{technical_analysis['ema_12']:,.0f}")
                    st.write(f"**EMA 26:** ₹{technical_analysis['ema_26']:,.0f}")
                
                with col2:
                    st.subheader("🎯 Options Analysis Summary")
                    if options_analysis:
                        st.write(f"**ATM Strike:** ₹{options_analysis.get('atm_strike', 0):,.0f}")
                        st.write(f"**IV Percentile:** {options_analysis.get('iv_percentile', 0):.1f}%")
                        st.write(f"**Put-Call Ratio:** {options_analysis.get('put_call_ratio', 0):.2f}")
                        st.write(f"**Total OI:** {options_analysis.get('total_oi', 0):,}")
                    else:
                        st.write("No options data available")
                
                # Portfolio summary
                st.subheader("💼 Portfolio Summary")
                if portfolio_summary:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Value", f"₹{portfolio_summary.get('total_value', 0):,.0f}")
                    with col2:
                        st.metric("Total P&L", f"₹{portfolio_summary.get('total_pnl', 0):,.0f}", f"{portfolio_summary.get('total_pnl_percent', 0):+.2f}%")
                    with col3:
                        positions = portfolio_summary.get('positions', {})
                        st.metric("Open Positions", positions.get('open', 0))
                else:
                    st.write("No portfolio data available")
                
                # Export options
                st.subheader("📤 Export Options")
                export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                
                # Create report data
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'symbol_name': INDIAN_MARKET_SYMBOLS[symbol].name,
                    'time_period': time_period,
                    'report_type': report_type,
                    'technical_analysis': technical_analysis,
                    'options_analysis': options_analysis,
                    'portfolio_summary': portfolio_summary,
                    'market_data': data.to_dict() if not data.empty else {}
                }
                
                with export_col1:
                    if st.button("💾 Save JSON"):
                        try:
                            report_filename = f"outputs/reports/report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            os.makedirs(os.path.dirname(report_filename), exist_ok=True)
                            with open(report_filename, 'w') as f:
                                json.dump(report_data, f, indent=2, default=str)
                            st.success(f"✅ JSON saved to {report_filename}")
                        except Exception as e:
                            st.error(f"Error saving JSON: {e}")
                
                with export_col2:
                    if st.button("📊 Export Excel"):
                        try:
                            excel_data = self._create_excel_report(report_data, data)
                            excel_filename = f"outputs/reports/report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                            os.makedirs(os.path.dirname(excel_filename), exist_ok=True)
                            excel_data.to_excel(excel_filename, index=False)
                            st.success(f"✅ Excel saved to {excel_filename}")
                            
                            # Provide download link
                            with open(excel_filename, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Excel",
                                    data=f.read(),
                                    file_name=f"market_report_{symbol}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        except Exception as e:
                            st.error(f"Error creating Excel: {e}")
                
                with export_col3:
                    if st.button("📄 Export CSV"):
                        try:
                            csv_data = self._create_csv_report(report_data, data)
                            csv_filename = f"outputs/reports/report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
                            csv_data.to_csv(csv_filename, index=False)
                            st.success(f"✅ CSV saved to {csv_filename}")
                            
                            # Provide download link
                            with open(csv_filename, 'rb') as f:
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=f.read(),
                                    file_name=f"market_report_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv"
                                )
                        except Exception as e:
                            st.error(f"Error creating CSV: {e}")
                
                with export_col4:
                    if st.button("📧 Email Report"):
                        if recipient_email and sender_email and sender_password:
                            try:
                                self._send_email_report(report_data, recipient_email, sender_email, sender_password, format_type)
                                st.success("✅ Email sent successfully!")
                            except Exception as e:
                                st.error(f"Error sending email: {e}")
                        else:
                            st.warning("Please provide email credentials to send report")
                
            except Exception as e:
                st.error(f"Error generating reports: {e}")
                logger.error(f"Error in reports: {e}")
    
    def _create_excel_report(self, report_data: Dict, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create Excel report with multiple sheets"""
        try:
            # Create summary data
            summary_data = {
                'Metric': [
                    'Symbol', 'Symbol Name', 'Timestamp', 'Time Period',
                    'Current Price', 'Overall Signal', 'Market Trend', 'Volatility Level',
                    'RSI', 'EMA 12', 'EMA 26', 'MACD Signal',
                    'Total Value', 'Total P&L', 'Total P&L %', 'Open Positions'
                ],
                'Value': [
                    report_data['symbol'],
                    report_data['symbol_name'],
                    report_data['timestamp'],
                    report_data['time_period'],
                    f"₹{report_data['technical_analysis']['current_price']:,.0f}",
                    report_data['technical_analysis']['overall_signal'],
                    report_data['technical_analysis']['market_trend'],
                    report_data['technical_analysis']['volatility_level'],
                    f"{report_data['technical_analysis']['rsi']:.1f}",
                    f"₹{report_data['technical_analysis']['ema_12']:,.0f}",
                    f"₹{report_data['technical_analysis']['ema_26']:,.0f}",
                    report_data['technical_analysis']['macd_signal'],
                    f"₹{report_data['portfolio_summary']['total_value']:,.0f}",
                    f"₹{report_data['portfolio_summary']['total_pnl']:,.0f}",
                    f"{report_data['portfolio_summary']['total_pnl_percent']:+.2f}%",
                    report_data['portfolio_summary']['positions']['open']
                ]
            }
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logger.error(f"Error creating Excel report: {e}")
            return pd.DataFrame()
    
    def _create_csv_report(self, report_data: Dict, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create CSV report"""
        try:
            # Create summary data
            summary_data = {
                'Metric': [
                    'Symbol', 'Symbol Name', 'Timestamp', 'Time Period',
                    'Current Price', 'Overall Signal', 'Market Trend', 'Volatility Level',
                    'RSI', 'EMA 12', 'EMA 26', 'MACD Signal',
                    'Total Value', 'Total P&L', 'Total P&L %', 'Open Positions'
                ],
                'Value': [
                    report_data['symbol'],
                    report_data['symbol_name'],
                    report_data['timestamp'],
                    report_data['time_period'],
                    f"₹{report_data['technical_analysis']['current_price']:,.0f}",
                    report_data['technical_analysis']['overall_signal'],
                    report_data['technical_analysis']['market_trend'],
                    report_data['technical_analysis']['volatility_level'],
                    f"{report_data['technical_analysis']['rsi']:.1f}",
                    f"₹{report_data['technical_analysis']['ema_12']:,.0f}",
                    f"₹{report_data['technical_analysis']['ema_26']:,.0f}",
                    report_data['technical_analysis']['macd_signal'],
                    f"₹{report_data['portfolio_summary']['total_value']:,.0f}",
                    f"₹{report_data['portfolio_summary']['total_pnl']:,.0f}",
                    f"{report_data['portfolio_summary']['total_pnl_percent']:+.2f}%",
                    report_data['portfolio_summary']['positions']['open']
                ]
            }
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logger.error(f"Error creating CSV report: {e}")
            return pd.DataFrame()
    
    def _send_email_report(self, report_data: Dict, recipient: str, sender: str, password: str, format_type: str):
        """Send email report"""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = f"Indian Market Report - {report_data['symbol_name']} - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Email body
            body = f"""
            <html>
            <body>
                <h2>📊 Indian Market Trading Analysis Report</h2>
                <p><strong>Symbol:</strong> {report_data['symbol_name']} ({report_data['symbol']})</p>
                <p><strong>Generated:</strong> {report_data['timestamp']}</p>
                <p><strong>Time Period:</strong> {report_data['time_period']}</p>
                
                <h3>📈 Technical Analysis Summary</h3>
                <ul>
                    <li><strong>Current Price:</strong> ₹{report_data['technical_analysis']['current_price']:,.0f}</li>
                    <li><strong>Overall Signal:</strong> {report_data['technical_analysis']['overall_signal']}</li>
                    <li><strong>Market Trend:</strong> {report_data['technical_analysis']['market_trend']}</li>
                    <li><strong>Volatility:</strong> {report_data['technical_analysis']['volatility_level']}</li>
                    <li><strong>RSI:</strong> {report_data['technical_analysis']['rsi']:.1f}</li>
                </ul>
                
                <h3>💼 Portfolio Summary</h3>
                <ul>
                    <li><strong>Total Value:</strong> ₹{report_data['portfolio_summary']['total_value']:,.0f}</li>
                    <li><strong>Total P&L:</strong> ₹{report_data['portfolio_summary']['total_pnl']:,.0f} ({report_data['portfolio_summary']['total_pnl_percent']:+.2f}%)</li>
                    <li><strong>Open Positions:</strong> {report_data['portfolio_summary']['positions']['open']}</li>
                </ul>
                
                <p><em>This report was generated by the Indian Market Trading Analysis Platform.</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender, password)
            text = msg.as_string()
            server.sendmail(sender, recipient, text)
            server.quit()
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise e
    
    def _create_iv_chart_data(self, options_data: Dict, current_price: float) -> Dict[str, Any]:
        """Create IV chart data for options analysis"""
        try:
            if not options_data or 'expirations' not in options_data:
                return None
            
            # Get the nearest expiration
            expirations = list(options_data['expirations'].keys())
            if not expirations:
                return None
            
            # Use the first available expiration
            nearest_expiry = expirations[0]
            expiry_data = options_data['expirations'][nearest_expiry]
            
            # Create IV data structure
            iv_data = {
                'title': f'Implied Volatility Analysis - {nearest_expiry}',
                'current_price': current_price,
                'expiry': nearest_expiry,
                'strikes': [],
                'call_iv': [],
                'put_iv': [],
                'call_oi': [],
                'put_oi': []
            }
            
            # Extract data from options chain
            if 'strikes' in expiry_data:
                for strike_data in expiry_data['strikes']:
                    strike = strike_data.get('strike', 0)
                    call_iv = strike_data.get('call_iv', 0)
                    put_iv = strike_data.get('put_iv', 0)
                    call_oi = strike_data.get('call_oi', 0)
                    put_oi = strike_data.get('put_oi', 0)
                    
                    iv_data['strikes'].append(strike)
                    iv_data['call_iv'].append(call_iv * 100)  # Convert to percentage
                    iv_data['put_iv'].append(put_iv * 100)
                    iv_data['call_oi'].append(call_oi)
                    iv_data['put_oi'].append(put_oi)
            
            return iv_data if iv_data['strikes'] else None
            
        except Exception as e:
            logger.error(f"Error creating IV chart data: {e}")
            return None

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
