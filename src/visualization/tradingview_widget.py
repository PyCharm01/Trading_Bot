"""
TradingView Widget Integration for Indian Trading Bot
"""

import streamlit as st
from typing import Dict, Any, Optional
import json

class TradingViewWidget:
    """TradingView widget integration for Streamlit"""
    
    def __init__(self):
        self.symbol_mapping = {
            'NIFTY_50': 'NSE:NIFTY',
            'BANK_NIFTY': 'NSE:NIFTYBANK', 
            'SENSEX': 'BSE:SENSEX',
            'NIFTY_IT': 'NSE:NIFTYIT',
            'NIFTY_AUTO': 'NSE:NIFTYAUTO',
            'NIFTY_PHARMA': 'NSE:NIFTYPHARMA'
        }
    
    def create_tradingview_chart(self, symbol: str, height: int = 600, theme: str = "dark") -> str:
        """
        Create TradingView widget HTML for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY_50')
            height: Chart height in pixels
            theme: Chart theme ('dark' or 'light')
        
        Returns:
            HTML string for TradingView widget
        """
        tradingview_symbol = self.symbol_mapping.get(symbol, f'NSE:{symbol}')
        
        widget_config = {
            "symbol": tradingview_symbol,
            "interval": "1",
            "timezone": "Asia/Kolkata",
            "theme": theme,
            "style": "1",
            "locale": "in",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": False,
            "hide_top_toolbar": False,
            "hide_legend": False,
            "save_image": False,
            "container_id": f"tradingview_{symbol.lower()}"
        }
        
        html_template = f"""
        <div class="tradingview-widget-container">
            <div id="tradingview_{symbol.lower()}" style="height: {height}px;"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://in.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span class="blue-text">Track all markets on TradingView</span>
                </a>
            </div>
        </div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({json.dumps(widget_config)});
        </script>
        """
        
        return html_template
    
    def create_multi_symbol_chart(self, symbols: list, height: int = 400) -> str:
        """
        Create TradingView widget with multiple symbols
        
        Args:
            symbols: List of symbols to display
            height: Chart height per symbol
        
        Returns:
            HTML string for multi-symbol TradingView widget
        """
        tradingview_symbols = [self.symbol_mapping.get(s, f'NSE:{s}') for s in symbols]
        
        widget_config = {
            "symbols": [{"s": "NSE:NIFTY", "d": "NIFTY 50"}],
            "showSymbolLogo": True,
            "colorTheme": "dark",
            "isTransparent": False,
            "displayMode": "adaptive",
            "locale": "in"
        }
        
        # Update symbols
        widget_config["symbols"] = [
            {"s": symbol, "d": symbol.split(':')[1] if ':' in symbol else symbol}
            for symbol in tradingview_symbols
        ]
        
        html_template = f"""
        <div class="tradingview-widget-container">
            <div id="tradingview_multi" style="height: {height}px;"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://in.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span class="blue-text">Track all markets on TradingView</span>
                </a>
            </div>
        </div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({json.dumps(widget_config)});
        </script>
        """
        
        return html_template
    
    def create_advanced_chart(self, symbol: str, indicators: list = None, height: int = 600) -> str:
        """
        Create advanced TradingView chart with custom indicators
        
        Args:
            symbol: Trading symbol
            indicators: List of indicators to add
            height: Chart height
        
        Returns:
            HTML string for advanced TradingView widget
        """
        tradingview_symbol = self.symbol_mapping.get(symbol, f'NSE:{symbol}')
        
        # Default indicators
        if indicators is None:
            indicators = ["RSI", "MACD", "Volume"]
        
        widget_config = {
            "autosize": True,
            "symbol": tradingview_symbol,
            "interval": "1",
            "timezone": "Asia/Kolkata",
            "theme": "dark",
            "style": "1",
            "locale": "in",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": False,
            "hide_top_toolbar": False,
            "hide_legend": False,
            "save_image": False,
            "container_id": f"tradingview_advanced_{symbol.lower()}",
            "studies": [
                "RSI@tv-basicstudies",
                "MACD@tv-basicstudies", 
                "Volume@tv-basicstudies"
            ],
            "show_popup_button": True,
            "popup_width": "1000",
            "popup_height": "650"
        }
        
        html_template = f"""
        <div class="tradingview-widget-container">
            <div id="tradingview_advanced_{symbol.lower()}" style="height: {height}px;"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://in.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span class="blue-text">Track all markets on TradingView</span>
                </a>
            </div>
        </div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({json.dumps(widget_config)});
        </script>
        """
        
        return html_template

def show_tradingview_chart(symbol: str, chart_type: str = "basic", height: int = 600):
    """
    Streamlit function to display TradingView chart
    
    Args:
        symbol: Trading symbol
        chart_type: Type of chart ('basic', 'advanced', 'multi')
        height: Chart height
    """
    widget = TradingViewWidget()
    
    if chart_type == "basic":
        html = widget.create_tradingview_chart(symbol, height)
    elif chart_type == "advanced":
        html = widget.create_advanced_chart(symbol, height=height)
    elif chart_type == "multi":
        html = widget.create_multi_symbol_chart([symbol], height)
    else:
        html = widget.create_tradingview_chart(symbol, height)
    
    st.components.v1.html(html, height=height + 50)

def show_tradingview_multi_chart(symbols: list, height: int = 400):
    """
    Streamlit function to display multiple TradingView charts
    
    Args:
        symbols: List of trading symbols
        height: Chart height per symbol
    """
    widget = TradingViewWidget()
    html = widget.create_multi_symbol_chart(symbols, height)
    st.components.v1.html(html, height=height + 50)
