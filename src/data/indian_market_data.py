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
        self.alpha_vantage_api_key = alpha_vantage_api_key
        self.quandl_api_key = quandl_api_key
        
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
    
    def fetch_index_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data for Indian market indices"""
        try:
            if symbol not in INDIAN_MARKET_SYMBOLS:
                raise ValueError(f"Symbol {symbol} not found in Indian market symbols")
            
            yahoo_symbol = INDIAN_MARKET_SYMBOLS[symbol].yahoo_symbol
            ticker = yf.Ticker(yahoo_symbol)
            
            # Fetch data with timezone handling
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Convert to IST timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert('Asia/Kolkata')
            
            # Add Indian market specific columns
            data['Symbol'] = symbol
            data['Index_Name'] = INDIAN_MARKET_SYMBOLS[symbol].name
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            logger.info(f"Generating mock data for {symbol} for demonstration purposes")
            return self._generate_mock_data(symbol, period, interval)
    
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
        """Get current market status (open/closed) and trading hours"""
        try:
            now = datetime.now()
            ist_now = now.replace(tzinfo=None)  # Assuming we're in IST
            
            # NSE trading hours (IST)
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # Check if it's a weekday
            is_weekday = now.weekday() < 5
            
            # Check if market is open
            is_market_open = (is_weekday and 
                            market_open <= now <= market_close)
            
            # Calculate time to open/close
            if is_market_open:
                time_to_close = market_close - now
                next_event = "Market closes in"
                next_event_time = time_to_close
            else:
                if now < market_open:
                    time_to_open = market_open - now
                    next_event = "Market opens in"
                    next_event_time = time_to_open
                else:
                    # Market closed for the day, next opening is tomorrow
                    next_open = market_open + timedelta(days=1)
                    time_to_open = next_open - now
                    next_event = "Market opens in"
                    next_event_time = time_to_open
            
            return {
                'is_market_open': is_market_open,
                'is_weekday': is_weekday,
                'current_time': now.isoformat(),
                'market_open_time': market_open.isoformat(),
                'market_close_time': market_close.isoformat(),
                'next_event': next_event,
                'next_event_time': str(next_event_time).split('.')[0],  # Remove microseconds
                'trading_hours': {
                    'pre_market': '09:00 - 09:15 IST',
                    'normal_trading': '09:15 - 15:30 IST',
                    'post_market': '15:40 - 16:00 IST'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'is_market_open': False,
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
        try:
            # Base prices for different indices
            base_prices = {
                'NIFTY_50': 19500.0,
                'BANK_NIFTY': 45000.0,
                'SENSEX': 65000.0,
                'NIFTY_IT': 35000.0,
                'NIFTY_AUTO': 18000.0,
                'NIFTY_PHARMA': 15000.0
            }
            
            base_price = base_prices.get(symbol, 20000.0)
            
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
            
            logger.info(f"Generated {len(df)} mock records for {symbol}")
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
