#!/usr/bin/env python3
"""
Indian Market Data Fetcher for Nifty 50, Bank Nifty, and Sensex

This module provides comprehensive data fetching capabilities for Indian market indices
including real-time data, historical data, options chains, and market news.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
from dataclasses import dataclass
import pytz
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

@dataclass
class IndianMarketSymbol:
    """Indian market symbol configuration"""
    symbol: str
    name: str
    yahoo_symbol: str
    nse_symbol: str
    bse_symbol: str
    sector: str
    market_cap: str
    description: str

# Indian Market Symbols Configuration
INDIAN_MARKET_SYMBOLS = {
    'NIFTY_50': IndianMarketSymbol(
        symbol='^NSEI',
        name='Nifty 50',
        yahoo_symbol='^NSEI',
        nse_symbol='NIFTY 50',
        bse_symbol='NIFTY 50',
        sector='Index',
        market_cap='Large Cap',
        description='Nifty 50 is the flagship index of NSE representing 50 large-cap stocks'
    ),
    'BANK_NIFTY': IndianMarketSymbol(
        symbol='^NSEBANK',
        name='Bank Nifty',
        yahoo_symbol='^NSEBANK',
        nse_symbol='NIFTY BANK',
        bse_symbol='NIFTY BANK',
        sector='Banking',
        market_cap='Large Cap',
        description='Bank Nifty represents the performance of 12 major banking stocks'
    ),
    'SENSEX': IndianMarketSymbol(
        symbol='^BSESN',
        name='Sensex',
        yahoo_symbol='^BSESN',
        nse_symbol='SENSEX',
        bse_symbol='SENSEX',
        sector='Index',
        market_cap='Large Cap',
        description='Sensex is the flagship index of BSE representing 30 large-cap stocks'
    ),
    'NIFTY_IT': IndianMarketSymbol(
        symbol='^CNXIT',
        name='Nifty IT',
        yahoo_symbol='^CNXIT',
        nse_symbol='NIFTY IT',
        bse_symbol='NIFTY IT',
        sector='Information Technology',
        market_cap='Large Cap',
        description='Nifty IT represents the performance of IT sector stocks'
    ),
    'NIFTY_AUTO': IndianMarketSymbol(
        symbol='^CNXAUTO',
        name='Nifty Auto',
        yahoo_symbol='^CNXAUTO',
        nse_symbol='NIFTY AUTO',
        bse_symbol='NIFTY AUTO',
        sector='Automobile',
        market_cap='Large Cap',
        description='Nifty Auto represents the performance of automobile sector stocks'
    ),
    'NIFTY_PHARMA': IndianMarketSymbol(
        symbol='^CNXPHARMA',
        name='Nifty Pharma',
        yahoo_symbol='^CNXPHARMA',
        nse_symbol='NIFTY PHARMA',
        bse_symbol='NIFTY PHARMA',
        sector='Pharmaceuticals',
        market_cap='Large Cap',
        description='Nifty Pharma represents the performance of pharmaceutical sector stocks'
    )
}

class IndianMarketDataFetcher:
    """Comprehensive data fetcher for Indian market indices"""
    
    def __init__(self, alpha_vantage_api_key: Optional[str] = None, quandl_api_key: Optional[str] = None):
        # Use provided API keys or fall back to config
        try:
            # Try multiple import strategies
            try:
                from ..config.config import get_config
                config = get_config()
            except ImportError:
                try:
                    from config.config import get_config
                    config = get_config()
                except ImportError:
                    try:
                        from src.config.config import get_config
                        config = get_config()
                    except ImportError:
                        # Direct file import as final fallback
                        import os
                        import importlib.util
                        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.py')
                        if os.path.exists(config_path):
                            spec = importlib.util.spec_from_file_location("config", config_path)
                            config_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(config_module)
                            config = config_module.get_config()
                        else:
                            # Use default values if config not found
                            class DefaultConfig:
                                ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key_here"
                                QUANDL_API_KEY = "your_quandl_api_key_here"
                                NEWS_API_KEY = "your_news_api_key_here"
                                FINNHUB_API_KEY = "your_finnhub_api_key_here"
                                POLYGON_API_KEY = "your_polygon_api_key_here"
                            config = DefaultConfig()
            
            self.alpha_vantage_api_key = alpha_vantage_api_key or config.ALPHA_VANTAGE_API_KEY
            self.quandl_api_key = quandl_api_key or config.QUANDL_API_KEY
            self.news_api_key = config.NEWS_API_KEY
            self.finnhub_api_key = config.FINNHUB_API_KEY
            self.polygon_api_key = config.POLYGON_API_KEY
            
        except Exception as e:
            # Fallback to default values if all import strategies fail
            logger.warning(f"Could not load config: {e}. Using default values.")
            self.alpha_vantage_api_key = alpha_vantage_api_key or "your_alpha_vantage_api_key_here"
            self.quandl_api_key = quandl_api_key or "your_quandl_api_key_here"
            self.news_api_key = "your_news_api_key_here"
            self.finnhub_api_key = "your_finnhub_api_key_here"
            self.polygon_api_key = "your_polygon_api_key_here"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_available_symbols(self) -> Dict[str, IndianMarketSymbol]:
        """Get all available Indian market symbols"""
        return INDIAN_MARKET_SYMBOLS
    
    def fetch_alpha_vantage_data(self, symbol: str, function: str = "TIME_SERIES_DAILY") -> Dict[str, Any]:
        """Fetch data from Alpha Vantage API"""
        if not self.alpha_vantage_api_key:
            logger.warning("Alpha Vantage API key not provided")
            return {}
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.alpha_vantage_api_key,
                'outputsize': 'compact'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {}
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage API note: {data['Note']}")
                return {}
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            return {}
    
    def fetch_quandl_data(self, dataset: str, symbol: str) -> Dict[str, Any]:
        """Fetch data from Quandl API"""
        if not self.quandl_api_key:
            logger.warning("Quandl API key not provided")
            return {}
        
        try:
            url = f"https://www.quandl.com/api/v3/datasets/{dataset}/{symbol}.json"
            params = {
                'api_key': self.quandl_api_key,
                'limit': 100
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Quandl data: {e}")
            return {}
    
    def fetch_news_data(self, symbol: str = "NIFTY", limit: int = 10) -> Dict[str, Any]:
        """Fetch news data using News API"""
        if not self.news_api_key or self.news_api_key == "your_news_api_key_here":
            logger.warning("News API key not configured")
            return self._generate_mock_news_data(symbol, limit)
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} OR NSE OR BSE OR Indian stock market",
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'ok':
                return {
                    'articles': data.get('articles', []),
                    'total_results': data.get('totalResults', 0),
                    'source': 'News API'
                }
            else:
                logger.warning(f"News API error: {data.get('message', 'Unknown error')}")
                return self._generate_mock_news_data(symbol, limit)
                
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return self._generate_mock_news_data(symbol, limit)
    
    def fetch_finnhub_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Finnhub API"""
        if not self.finnhub_api_key or self.finnhub_api_key == "your_finnhub_api_key_here":
            logger.warning("Finnhub API key not configured")
            return {}
        
        try:
            # Convert Indian symbols to Finnhub format
            finnhub_symbol = self._convert_to_finnhub_symbol(symbol)
            
            url = "https://finnhub.io/api/v1/quote"
            params = {
                'symbol': finnhub_symbol,
                'token': self.finnhub_api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data and 'c' in data:  # 'c' is current price
                return {
                    'current_price': data.get('c', 0),
                    'change': data.get('d', 0),
                    'change_percent': data.get('dp', 0),
                    'high': data.get('h', 0),
                    'low': data.get('l', 0),
                    'open': data.get('o', 0),
                    'previous_close': data.get('pc', 0),
                    'source': 'Finnhub'
                }
            else:
                logger.warning(f"No data received from Finnhub for {symbol}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching Finnhub data: {e}")
            return {}
    
    def _convert_to_finnhub_symbol(self, symbol: str) -> str:
        """Convert Indian market symbols to Finnhub format"""
        symbol_mapping = {
            'NIFTY_50': '^NSEI',
            'BANK_NIFTY': '^NSEBANK',
            'SENSEX': '^BSESN',
            'NIFTY_IT': '^CNXIT',
            'NIFTY_AUTO': '^CNXAUTO',
            'NIFTY_PHARMA': '^CNXPHARMA'
        }
        return symbol_mapping.get(symbol, symbol)
    
    def _generate_mock_news_data(self, symbol: str, limit: int) -> Dict[str, Any]:
        """Generate mock news data when API is not available"""
        mock_articles = []
        for i in range(limit):
            mock_articles.append({
                'title': f"{symbol} Market Update - {i+1}",
                'description': f"Latest market analysis and trends for {symbol} index",
                'url': f"https://example.com/news/{i+1}",
                'publishedAt': datetime.now().isoformat(),
                'source': {'name': 'Mock News Source'}
            })
        
        return {
            'articles': mock_articles,
            'total_results': limit,
            'source': 'Mock Data'
        }
    
    def fetch_index_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data for Indian market indices using live data sources"""
        try:
            if symbol not in INDIAN_MARKET_SYMBOLS:
                raise ValueError(f"Symbol {symbol} not found in Indian market symbols")
            
            # Try multiple data sources in order of preference
            data = self._fetch_from_multiple_sources(symbol, period, interval)
            
            if not data.empty:
                # Convert to IST timezone
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                data.index = data.index.tz_convert('Asia/Kolkata')
                
                # Add Indian market specific columns
                data['Symbol'] = symbol
                data['Index_Name'] = INDIAN_MARKET_SYMBOLS[symbol].name
                
                logger.info(f"Successfully fetched {len(data)} LIVE records for {symbol}")
                return data
            else:
                raise ValueError(f"No live data found for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            logger.warning(f"Falling back to mock data for {symbol} - CONFIGURE LIVE DATA SOURCES")
            return self._generate_mock_data(symbol, period, interval)
    
    def _fetch_from_multiple_sources(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Try multiple data sources to get live market data"""
        yahoo_symbol = INDIAN_MARKET_SYMBOLS[symbol].yahoo_symbol
        
        # Source 1: Yahoo Finance (Primary)
        try:
            logger.info(f"Attempting to fetch {symbol} data from Yahoo Finance...")
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                logger.info(f"Successfully fetched {len(data)} records from Yahoo Finance")
                return data
            else:
                logger.warning("Yahoo Finance returned empty data")
                
        except Exception as e:
            logger.warning(f"Yahoo Finance failed: {e}")
        
        # Source 2: Alpha Vantage (if API key available)
        if self.alpha_vantage_api_key and self.alpha_vantage_api_key != "your_alpha_vantage_api_key_here":
            try:
                logger.info(f"Attempting to fetch {symbol} data from Alpha Vantage...")
                data = self._fetch_alpha_vantage_ohlcv(symbol, period, interval)
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} records from Alpha Vantage")
                    return data
            except Exception as e:
                logger.warning(f"Alpha Vantage failed: {e}")
        
        # Source 3: Finnhub (if API key available)
        if self.finnhub_api_key and self.finnhub_api_key != "your_finnhub_api_key_here":
            try:
                logger.info(f"Attempting to fetch {symbol} data from Finnhub...")
                data = self._fetch_finnhub_ohlcv(symbol, period, interval)
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} records from Finnhub")
                    return data
            except Exception as e:
                logger.warning(f"Finnhub failed: {e}")
        
        # Source 4: Polygon (if API key available)
        if self.polygon_api_key and self.polygon_api_key != "your_polygon_api_key_here":
            try:
                logger.info(f"Attempting to fetch {symbol} data from Polygon...")
                data = self._fetch_polygon_ohlcv(symbol, period, interval)
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} records from Polygon")
                    return data
            except Exception as e:
                logger.warning(f"Polygon failed: {e}")
        
        logger.error("All live data sources failed")
        return pd.DataFrame()
    
    def _fetch_alpha_vantage_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch OHLCV data from Alpha Vantage"""
        try:
            # Map intervals to Alpha Vantage format
            interval_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '60min', '1d': 'daily', '1wk': 'weekly', '1mo': 'monthly'
            }
            
            av_interval = interval_map.get(interval, 'daily')
            
            # Map periods
            period_map = {
                '1d': 'compact', '5d': 'compact', '1mo': 'compact',
                '3mo': 'full', '6mo': 'full', '1y': 'full', '2y': 'full'
            }
            
            outputsize = period_map.get(period, 'compact')
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY' if av_interval != 'daily' else 'TIME_SERIES_DAILY',
                'symbol': INDIAN_MARKET_SYMBOLS[symbol].yahoo_symbol,
                'interval': av_interval,
                'apikey': self.alpha_vantage_api_key,
                'outputsize': outputsize
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data or 'Note' in data:
                return pd.DataFrame()
            
            # Parse Alpha Vantage response
            time_series_key = 'Time Series (Daily)' if av_interval == 'daily' else f'Time Series ({av_interval})'
            time_series = data.get(time_series_key, {})
            
            if not time_series:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data, index=pd.to_datetime(list(time_series.keys())))
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            return pd.DataFrame()
    
    def _fetch_finnhub_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch OHLCV data from Finnhub"""
        try:
            # Finnhub uses different symbol format
            finnhub_symbol = self._convert_to_finnhub_symbol(symbol)
            
            # Calculate timestamp range
            end_time = int(datetime.now().timestamp())
            period_days = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
            days = period_days.get(period, 30)
            start_time = end_time - (days * 24 * 60 * 60)
            
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                'symbol': finnhub_symbol,
                'resolution': interval,
                'from': start_time,
                'to': end_time,
                'token': self.finnhub_api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') != 'ok':
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            }, index=pd.to_datetime(data['t'], unit='s'))
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub data: {e}")
            return pd.DataFrame()
    
    def _fetch_polygon_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch OHLCV data from Polygon"""
        try:
            # Polygon uses different symbol format
            polygon_symbol = INDIAN_MARKET_SYMBOLS[symbol].yahoo_symbol.replace('^', 'I:')
            
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            period_days = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
            days = period_days.get(period, 30)
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/1/{interval}/{start_date}/{end_date}"
            params = {'apikey': self.polygon_api_key}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'OK' or not data.get('results'):
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for result in data['results']:
                df_data.append({
                    'Open': result['o'],
                    'High': result['h'],
                    'Low': result['l'],
                    'Close': result['c'],
                    'Volume': result['v']
                })
            
            df = pd.DataFrame(df_data, index=pd.to_datetime([r['t'] for r in data['results']], unit='ms'))
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Polygon data: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_indices(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple Indian market indices"""
        results = {}
        
        for symbol in symbols:
            if symbol in INDIAN_MARKET_SYMBOLS:
                data = self.fetch_index_data(symbol, period)
                if not data.empty:
                    results[symbol] = data
                else:
                    logger.warning(f"No data retrieved for {symbol}")
            else:
                logger.warning(f"Unknown symbol: {symbol}")
        
        return results
    
    def fetch_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> Dict:
        """Fetch options chain data for Indian market indices"""
        try:
            if symbol not in INDIAN_MARKET_SYMBOLS:
                raise ValueError(f"Symbol {symbol} not found in Indian market symbols")
            
            yahoo_symbol = INDIAN_MARKET_SYMBOLS[symbol].yahoo_symbol
            ticker = yf.Ticker(yahoo_symbol)
            
            # Get available expiration dates
            try:
                exp_dates = ticker.options
                if not exp_dates:
                    logger.warning(f"No options data available for {symbol}")
                    return {}
            except Exception as e:
                logger.warning(f"Could not fetch options dates for {symbol}: {e}")
                return {}
            
            # Filter expiration dates
            if expiration_date:
                exp_dates = [exp for exp in exp_dates if exp == expiration_date]
            else:
                exp_dates = exp_dates[:3]  # Get next 3 expiration dates
            
            options_data = {}
            for exp_date in exp_dates:
                try:
                    opt_chain = ticker.option_chain(exp_date)
                    
                    # Process calls data
                    calls = opt_chain.calls.copy()
                    if not calls.empty:
                        calls['expiration'] = exp_date
                        calls['option_type'] = 'call'
                        calls['underlying'] = symbol
                    
                    # Process puts data
                    puts = opt_chain.puts.copy()
                    if not puts.empty:
                        puts['expiration'] = exp_date
                        puts['option_type'] = 'put'
                        puts['underlying'] = symbol
                    
                    options_data[exp_date] = {
                        'calls': calls,
                        'puts': puts,
                        'expiration_date': exp_date,
                        'days_to_expiry': (pd.to_datetime(exp_date) - datetime.now()).days
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not fetch options for {exp_date}: {e}")
                    continue
            
            logger.info(f"Successfully fetched options data for {symbol} with {len(options_data)} expiration dates")
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return {}
    
    def fetch_market_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Fetch recent news for Indian market indices"""
        try:
            if symbol not in INDIAN_MARKET_SYMBOLS:
                raise ValueError(f"Symbol {symbol} not found in Indian market symbols")
            
            yahoo_symbol = INDIAN_MARKET_SYMBOLS[symbol].yahoo_symbol
            ticker = yf.Ticker(yahoo_symbol)
            
            news = ticker.news
            if news:
                # Filter and format news
                formatted_news = []
                for article in news[:limit]:
                    formatted_news.append({
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'publisher': article.get('publisher', ''),
                        'published': article.get('providerPublishTime', ''),
                        'url': article.get('link', ''),
                        'related_symbols': article.get('relatedTickers', [])
                    })
                
                logger.info(f"Successfully fetched {len(formatted_news)} news articles for {symbol}")
                return formatted_news
            else:
                logger.warning(f"No news data available for {symbol}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def fetch_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Alias for fetch_market_news for backward compatibility"""
        return self.fetch_market_news(symbol, limit)
    
    def fetch_market_overview(self) -> Dict[str, Any]:
        """Fetch comprehensive market overview for all major Indian indices"""
        try:
            overview = {}
            
            # Fetch data for all major indices
            major_indices = ['NIFTY_50', 'BANK_NIFTY', 'SENSEX']
            
            for symbol in major_indices:
                try:
                    # Get current data
                    data = self.fetch_index_data(symbol, period="5d")
                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change = current_price - prev_close
                        change_percent = (change / prev_close) * 100
                        
                        # Get volume data
                        volume = data['Volume'].iloc[-1]
                        avg_volume = data['Volume'].mean()
                        
                        # Get high/low for the day
                        day_high = data['High'].iloc[-1]
                        day_low = data['Low'].iloc[-1]
                        
                        overview[symbol] = {
                            'name': INDIAN_MARKET_SYMBOLS[symbol].name,
                            'current_price': float(current_price),
                            'change': float(change),
                            'change_percent': float(change_percent),
                            'volume': int(volume),
                            'avg_volume': int(avg_volume),
                            'day_high': float(day_high),
                            'day_low': float(day_low),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                except Exception as e:
                    logger.error(f"Error fetching overview for {symbol}: {e}")
                    continue
            
            return overview
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return {}
    
    def fetch_sector_performance(self) -> Dict[str, Any]:
        """Fetch performance data for different sectors"""
        try:
            sector_indices = {
                'IT': 'NIFTY_IT',
                'AUTO': 'NIFTY_AUTO',
                'PHARMA': 'NIFTY_PHARMA',
                'BANKING': 'BANK_NIFTY'
            }
            
            sector_data = {}
            
            for sector, symbol in sector_indices.items():
                try:
                    data = self.fetch_index_data(symbol, period="5d")
                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change_percent = ((current_price - prev_close) / prev_close) * 100
                        
                        sector_data[sector] = {
                            'name': INDIAN_MARKET_SYMBOLS[symbol].name,
                            'current_price': float(current_price),
                            'change_percent': float(change_percent),
                            'sector': INDIAN_MARKET_SYMBOLS[symbol].sector
                        }
                        
                except Exception as e:
                    logger.error(f"Error fetching sector data for {sector}: {e}")
                    continue
            
            return sector_data
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {e}")
            return {}
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status (open/closed) and trading hours with proper timezone handling"""
        try:
            # Get current time in IST
            ist = ZoneInfo('Asia/Kolkata')
            now_utc = datetime.now(pytz.UTC)
            now_ist = now_utc.astimezone(ist)
            
            # NSE trading hours (IST)
            today = now_ist.date()
            market_open = datetime.combine(today, datetime.min.time().replace(hour=9, minute=15))
            market_close = datetime.combine(today, datetime.min.time().replace(hour=15, minute=30))
            pre_market_open = datetime.combine(today, datetime.min.time().replace(hour=9, minute=0))
            post_market_close = datetime.combine(today, datetime.min.time().replace(hour=16, minute=0))
            
            # Convert to IST timezone
            market_open = market_open.replace(tzinfo=ist)
            market_close = market_close.replace(tzinfo=ist)
            pre_market_open = pre_market_open.replace(tzinfo=ist)
            post_market_close = post_market_close.replace(tzinfo=ist)
            
            # Check if it's a weekday
            is_weekday = now_ist.weekday() < 5
            
            # Determine market status
            is_pre_market = is_weekday and pre_market_open <= now_ist < market_open
            is_market_open = is_weekday and market_open <= now_ist <= market_close
            is_post_market = is_weekday and market_close < now_ist <= post_market_close
            is_market_closed = not is_weekday or now_ist < pre_market_open or now_ist > post_market_close
            
            # Determine current session
            if is_pre_market:
                current_session = "Pre-Market"
                session_status = "Pre-Market Open"
            elif is_market_open:
                current_session = "Normal Trading"
                session_status = "Market Open"
            elif is_post_market:
                current_session = "Post-Market"
                session_status = "Post-Market Open"
            else:
                current_session = "Closed"
                session_status = "Market Closed"
            
            # Calculate time to next event
            if is_market_open:
                time_to_close = market_close - now_ist
                next_event = "Market closes in"
                next_event_time = time_to_close
            elif is_pre_market:
                time_to_open = market_open - now_ist
                next_event = "Normal trading starts in"
                next_event_time = time_to_open
            elif is_post_market:
                time_to_close = post_market_close - now_ist
                next_event = "Post-market closes in"
                next_event_time = time_to_close
            else:
                # Market closed, find next opening
                if now_ist < pre_market_open:
                    # Before pre-market today
                    time_to_open = pre_market_open - now_ist
                    next_event = "Pre-market opens in"
                    next_event_time = time_to_open
                else:
                    # After post-market, next opening is tomorrow
                    next_day = today + timedelta(days=1)
                    # Skip weekends
                    while next_day.weekday() >= 5:
                        next_day += timedelta(days=1)
                    
                    next_pre_market = datetime.combine(next_day, datetime.min.time().replace(hour=9, minute=0)).replace(tzinfo=ist)
                    time_to_open = next_pre_market - now_ist
                    next_event = "Pre-market opens in"
                    next_event_time = time_to_open
            
            # Format time duration
            total_seconds = int(next_event_time.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            if hours > 0:
                time_str = f"{hours}h {minutes}m"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            
            return {
                'is_market_open': is_market_open,
                'is_pre_market': is_pre_market,
                'is_post_market': is_post_market,
                'is_market_closed': is_market_closed,
                'is_weekday': is_weekday,
                'current_session': current_session,
                'session_status': session_status,
                'current_time_ist': now_ist.strftime('%Y-%m-%d %H:%M:%S IST'),
                'current_time_utc': now_utc.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'market_open_time': market_open.strftime('%H:%M IST'),
                'market_close_time': market_close.strftime('%H:%M IST'),
                'pre_market_open': pre_market_open.strftime('%H:%M IST'),
                'post_market_close': post_market_close.strftime('%H:%M IST'),
                'next_event': next_event,
                'next_event_time': time_str,
                'trading_hours': {
                    'pre_market': '09:00 - 09:15 IST',
                    'normal_trading': '09:15 - 15:30 IST',
                    'post_market': '15:30 - 16:00 IST'
                },
                'market_info': {
                    'exchange': 'NSE (National Stock Exchange)',
                    'timezone': 'Asia/Kolkata (IST)',
                    'trading_days': 'Monday to Friday',
                    'holidays': 'National holidays and exchange-specific holidays'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'is_market_open': False,
                'is_market_closed': True,
                'session_status': 'Error',
                'error': str(e),
                'current_time_ist': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trading_hours': {
                    'pre_market': '09:00 - 09:15 IST',
                    'normal_trading': '09:15 - 15:30 IST',
                    'post_market': '15:30 - 16:00 IST'
                }
            }
    
    def fetch_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time data for Indian market indices"""
        try:
            if symbol not in INDIAN_MARKET_SYMBOLS:
                raise ValueError(f"Symbol {symbol} not found in Indian market symbols")
            
            yahoo_symbol = INDIAN_MARKET_SYMBOLS[symbol].yahoo_symbol
            ticker = yf.Ticker(yahoo_symbol)
            
            # Get real-time info
            info = ticker.info
            
            # Get latest data (1 minute interval for last 2 days to get most recent)
            data = ticker.history(period="2d", interval="1m")
            
            if data.empty:
                raise ValueError(f"No real-time data found for {symbol}")
            
            # Convert to IST timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert('Asia/Kolkata')
            
            # Get latest values
            latest = data.iloc[-1]
            
            # Calculate real-time metrics
            current_price = latest['Close']
            prev_close = data.iloc[-2]['Close'] if len(data) > 1 else current_price
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
            
            # Get market status
            market_status = self.get_market_status()
            
            # Calculate intraday high/low
            today_data = data[data.index.date == data.index[-1].date()]
            if not today_data.empty:
                intraday_high = today_data['High'].max()
                intraday_low = today_data['Low'].min()
                volume = today_data['Volume'].sum()
            else:
                intraday_high = current_price
                intraday_low = current_price
                volume = latest['Volume']
            
            return {
                'symbol': symbol,
                'name': INDIAN_MARKET_SYMBOLS[symbol].name,
                'current_price': round(current_price, 2),
                'previous_close': round(prev_close, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'intraday_high': round(intraday_high, 2),
                'intraday_low': round(intraday_low, 2),
                'volume': int(volume),
                'last_updated': data.index[-1].strftime('%Y-%m-%d %H:%M:%S IST'),
                'market_status': market_status,
                'data_source': 'Yahoo Finance',
                'is_realtime': market_status.get('is_market_open', False),
                'trading_session': market_status.get('current_session', 'Unknown')
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            # Return mock real-time data
            return self._generate_mock_realtime_data(symbol)
    
    def _generate_mock_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock real-time data for demonstration"""
        try:
            # Get some historical data to base mock data on
            data = self.fetch_index_data(symbol, period="5d")
            if not data.empty:
                latest = data.iloc[-1]
                base_price = latest['Close']
                
                # Simulate small price movement
                import random
                change_percent = random.uniform(-2, 2)  # ±2% change
                change = base_price * (change_percent / 100)
                current_price = base_price + change
                
                market_status = self.get_market_status()
                
                return {
                    'symbol': symbol,
                    'name': INDIAN_MARKET_SYMBOLS[symbol].name,
                    'current_price': round(current_price, 2),
                    'previous_close': round(base_price, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 2),
                    'intraday_high': round(current_price * 1.01, 2),
                    'intraday_low': round(current_price * 0.99, 2),
                    'volume': random.randint(1000000, 5000000),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
                    'market_status': market_status,
                    'data_source': 'Mock Data (Demo)',
                    'is_realtime': False,
                    'trading_session': 'Mock Session'
                }
            else:
                raise ValueError("No data available for mock generation")
                
        except Exception as e:
            logger.error(f"Error generating mock real-time data: {e}")
            return {
                'symbol': symbol,
                'name': symbol,
                'current_price': 0.0,
                'previous_close': 0.0,
                'change': 0.0,
                'change_percent': 0.0,
                'intraday_high': 0.0,
                'intraday_low': 0.0,
                'volume': 0,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
                'market_status': self.get_market_status(),
                'data_source': 'Error',
                'is_realtime': False,
                'trading_session': 'Error',
                'error': str(e)
            }
    
    def fetch_historical_volatility(self, symbol: str, period: str = "1y") -> Dict[str, float]:
        """Calculate historical volatility for Indian market indices"""
        try:
            data = self.fetch_index_data(symbol, period)
            if data.empty:
                return {}
            
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            
            # Calculate volatility metrics
            daily_vol = returns.std()
            annualized_vol = daily_vol * np.sqrt(252)  # 252 trading days in a year
            
            # Calculate rolling volatility (30-day)
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            current_rolling_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
            
            # Calculate volatility percentiles
            vol_percentile_30d = (rolling_vol.iloc[-30:].rank(pct=True).iloc[-1] * 100) if len(rolling_vol) >= 30 else 50
            vol_percentile_1y = (rolling_vol.rank(pct=True).iloc[-1] * 100) if not rolling_vol.empty else 50
            
            return {
                'daily_volatility': float(daily_vol),
                'annualized_volatility': float(annualized_vol),
                'current_30d_volatility': float(current_rolling_vol),
                'vol_percentile_30d': float(vol_percentile_30d),
                'vol_percentile_1y': float(vol_percentile_1y),
                'period': period,
                'data_points': len(returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return {}
    
    def _generate_mock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Generate mock data for demonstration purposes when real data is not available"""
        logger.warning(f"Using mock data for {symbol} - this should be replaced with live data sources")
        try:
            # Base prices for different indices (updated with more recent values)
            base_prices = {
                'NIFTY_50': 24000.0,
                'BANK_NIFTY': 52000.0,
                'SENSEX': 80000.0,
                'NIFTY_IT': 42000.0,
                'NIFTY_AUTO': 22000.0,
                'NIFTY_PHARMA': 18000.0
            }
            
            base_price = base_prices.get(symbol, 25000.0)
            
            # Calculate number of days based on period
            period_days = {
                '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
                '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'ytd': 365
            }
            days = period_days.get(period, 365)
            
            # Generate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Filter for weekdays only (trading days)
            trading_days = [d for d in date_range if d.weekday() < 5]
            
            # Generate mock OHLCV data
            # Use current time as seed for more dynamic results
            np.random.seed(int(datetime.now().timestamp()) % 10000)
            n_days = len(trading_days)
            
            # Generate price movements with some trend and volatility
            # Add some market hours simulation (more volatility during trading hours)
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 15:  # Trading hours
                volatility = 0.02  # Higher volatility during market hours
                drift = 0.001  # Slightly higher drift
            else:
                volatility = 0.01  # Lower volatility outside market hours
                drift = 0.0002  # Lower drift
            
            returns = np.random.normal(drift, volatility, n_days)
            prices = [base_price]
            
            for i in range(1, n_days):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, base_price * 0.5))  # Prevent prices from going too low
            
            # Generate OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(trading_days, prices)):
                # Generate realistic OHLC from close price
                volatility = 0.01  # 1% intraday volatility
                high = price * (1 + np.random.uniform(0, volatility))
                low = price * (1 - np.random.uniform(0, volatility))
                open_price = prices[i-1] if i > 0 else price
                close = price
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': volume
                })
            
            # Create DataFrame
            df = pd.DataFrame(data, index=trading_days)
            df.index.name = 'Date'
            
            # Add Indian market specific columns
            df['Symbol'] = symbol
            df['Index_Name'] = INDIAN_MARKET_SYMBOLS[symbol].name
            
            logger.warning(f"Generated {len(df)} MOCK records for {symbol} - USE LIVE DATA IN PRODUCTION")
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock data for {symbol}: {e}")
            return pd.DataFrame()

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data fetcher
    fetcher = IndianMarketDataFetcher()
    
    # Test fetching Nifty 50 data
    print("Testing Indian Market Data Fetcher...")
    
    # Fetch Nifty 50 data
    nifty_data = fetcher.fetch_index_data('NIFTY_50', period='1mo')
    print(f"Nifty 50 data shape: {nifty_data.shape}")
    print(f"Latest Nifty 50 price: {nifty_data['Close'].iloc[-1] if not nifty_data.empty else 'No data'}")
    
    # Fetch market overview
    overview = fetcher.fetch_market_overview()
    print(f"Market overview: {len(overview)} indices")
    
    # Check market status
    status = fetcher.get_market_status()
    print(f"Market status: {status}")
    
    # Fetch sector performance
    sectors = fetcher.fetch_sector_performance()
    print(f"Sector performance: {len(sectors)} sectors")
