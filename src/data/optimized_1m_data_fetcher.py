#!/usr/bin/env python3
"""
Optimized 1-Minute Data Fetcher for Indian Markets

This module provides optimized data fetching for 1-minute intervals with 1-month period,
specifically designed for high-frequency trading and ML model training.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
import time
import os
import json
from dataclasses import dataclass
import pytz
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics for 1-minute data"""
    total_records: int
    missing_records: int
    duplicate_records: int
    quality_score: float
    data_completeness: float
    time_gaps: List[Tuple[datetime, datetime]]

class Optimized1MDataFetcher:
    """Optimized data fetcher for 1-minute intervals with 1-month period"""
    
    def __init__(self, cache_duration: int = 300):  # 5 minutes cache
        self.cache_duration = cache_duration
        self.data_cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Indian market symbols optimized for 1-minute data
        self.symbols = {
            'NIFTY_50': '^NSEI',
            'BANK_NIFTY': '^NSEBANK', 
            'SENSEX': '^BSESN',
            'NIFTY_IT': '^CNXIT',
            'NIFTY_AUTO': '^CNXAUTO',
            'NIFTY_PHARMA': '^CNXPHARMA'
        }
        
        # Market hours in IST
        self.market_open = 9.25  # 9:15 AM
        self.market_close = 15.5  # 3:30 PM
        self.pre_market_open = 9.0  # 9:00 AM
        self.post_market_close = 16.0  # 4:00 PM
    
    def fetch_1m_data_optimized(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch optimized 1-minute data for specified number of days
        
        Args:
            symbol: Market symbol (e.g., 'NIFTY_50')
            days: Number of days to fetch (default 30 for 1 month)
            
        Returns:
            DataFrame with 1-minute OHLCV data
        """
        try:
            if symbol not in self.symbols:
                raise ValueError(f"Symbol {symbol} not supported")
            
            yahoo_symbol = self.symbols[symbol]
            cache_key = f"{symbol}_1m_{days}d"
            
            # Check cache first
            if cache_key in self.data_cache:
                cached_data, timestamp = self.data_cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    logger.info(f"Using cached 1m data for {symbol}")
                    return cached_data
            
            # Limit days to avoid API issues
            if days > 30:
                logger.warning(f"Limiting days from {days} to 30 to avoid API issues")
                days = 30
            
            # Fetch data in chunks to avoid API limits
            all_data = []
            chunk_days = min(7, days)  # Fetch max 7 days per chunk
            
            for i in range(0, days, chunk_days):
                chunk_start = datetime.now() - timedelta(days=days-i)
                chunk_end = min(chunk_start + timedelta(days=chunk_days), datetime.now())
                
                logger.info(f"Fetching 1m data chunk {i//chunk_days + 1} for {symbol}: {chunk_start.date()} to {chunk_end.date()}")
                
                chunk_data = self._fetch_yahoo_1m_chunk(yahoo_symbol, chunk_start, chunk_end)
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                
                # Rate limiting
                time.sleep(0.5)
            
            if not all_data:
                logger.error(f"No 1m data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Combine all chunks
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()
            
            # Remove duplicates and clean data
            combined_data = self._clean_1m_data(combined_data)
            
            # Filter for market hours only
            combined_data = self._filter_market_hours(combined_data)
            
            # Cache the result
            self.data_cache[cache_key] = (combined_data, time.time())
            
            logger.info(f"Successfully fetched {len(combined_data)} 1-minute records for {symbol}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error fetching 1m data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_yahoo_1m_chunk(self, yahoo_symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch 1-minute data chunk from Yahoo Finance"""
        try:
            ticker = yf.Ticker(yahoo_symbol)
            
            # Calculate period for yfinance
            days_diff = (end_date - start_date).days
            if days_diff <= 7:
                period = "7d"
            elif days_diff <= 30:
                period = "30d"
            else:
                period = "60d"
            
            # Fetch data with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = ticker.history(
                        period=period,
                        interval="1m",
                        auto_adjust=True,
                        prepost=True
                    )
                    
                    if not data.empty:
                        # Filter for the specific date range
                        data = data[(data.index >= start_date) & (data.index <= end_date)]
                        
                        if not data.empty:
                            # Convert to IST
                            if data.index.tz is None:
                                data.index = data.index.tz_localize('UTC')
                            data.index = data.index.tz_convert('Asia/Kolkata')
                            
                            return data
                    
                except Exception as e:
                    logger.warning(f"Yahoo Finance attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    continue
            
            # If all attempts failed, try with different parameters
            try:
                logger.info("Trying fallback method with different parameters...")
                data = ticker.history(
                    period=period,
                    interval="1m",
                    auto_adjust=False,
                    prepost=False
                )
                
                if not data.empty:
                    # Filter for the specific date range
                    data = data[(data.index >= start_date) & (data.index <= end_date)]
                    
                    if not data.empty:
                        # Convert to IST
                        if data.index.tz is None:
                            data.index = data.index.tz_localize('UTC')
                        data.index = data.index.tz_convert('Asia/Kolkata')
                        
                        logger.info(f"Fallback method successful: {len(data)} records")
                        return data
                
            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {fallback_error}")
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in Yahoo 1m chunk fetch: {e}")
            return pd.DataFrame()
    
    def _clean_1m_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate 1-minute data"""
        try:
            if data.empty:
                return data
            
            # Remove duplicates
            data = data[~data.index.duplicated(keep='first')]
            
            # Remove zero volume data (likely bad data)
            data = data[data['Volume'] > 0]
            
            # Remove extreme outliers (prices that are >50% different from previous)
            data['price_change'] = data['Close'].pct_change()
            data = data[abs(data['price_change']) < 0.5]  # Remove >50% price changes
            data = data.drop('price_change', axis=1)
            
            # Fill small gaps with forward fill (max 5 minutes)
            data = data.asfreq('1min', method='ffill', limit=5)
            
            # Remove any remaining NaN values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning 1m data: {e}")
            return data
    
    def _filter_market_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to include only market hours"""
        try:
            if data.empty:
                return data
            
            # Convert to IST if not already
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert('Asia/Kolkata')
            
            # Filter for market hours (9:15 AM to 3:30 PM IST)
            market_data = []
            
            for date in data.index.date:
                date_data = data[data.index.date == date]
                
                # Market hours: 9:15 AM to 3:30 PM IST
                market_start = date_data.index[0].replace(hour=9, minute=15, second=0, microsecond=0)
                market_end = date_data.index[0].replace(hour=15, minute=30, second=0, microsecond=0)
                
                # Filter for market hours
                market_day_data = date_data[
                    (date_data.index >= market_start) & 
                    (date_data.index <= market_end)
                ]
                
                if not market_day_data.empty:
                    market_data.append(market_day_data)
            
            if market_data:
                return pd.concat(market_data, ignore_index=False)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error filtering market hours: {e}")
            return data
    
    def get_data_quality_metrics(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Calculate data quality metrics for 1-minute data"""
        try:
            if data.empty:
                return DataQualityMetrics(0, 0, 0, 0.0, 0.0, [])
            
            total_records = len(data)
            
            # Check for missing records
            expected_minutes = self._calculate_expected_minutes(data)
            missing_records = max(0, expected_minutes - total_records)
            
            # Check for duplicates
            duplicate_records = len(data) - len(data.drop_duplicates())
            
            # Calculate time gaps
            time_gaps = self._find_time_gaps(data)
            
            # Calculate quality score (0-1)
            completeness = 1 - (missing_records / expected_minutes) if expected_minutes > 0 else 0
            quality_score = completeness * (1 - (duplicate_records / total_records)) if total_records > 0 else 0
            
            return DataQualityMetrics(
                total_records=total_records,
                missing_records=missing_records,
                duplicate_records=duplicate_records,
                quality_score=quality_score,
                data_completeness=completeness,
                time_gaps=time_gaps
            )
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {e}")
            return DataQualityMetrics(0, 0, 0, 0.0, 0.0, [])
    
    def _calculate_expected_minutes(self, data: pd.DataFrame) -> int:
        """Calculate expected number of 1-minute records"""
        try:
            if data.empty:
                return 0
            
            # Get unique trading days
            trading_days = data.index.date
            unique_days = len(set(trading_days))
            
            # Market hours: 9:15 AM to 3:30 PM = 6 hours 15 minutes = 375 minutes
            market_minutes_per_day = 375
            
            return unique_days * market_minutes_per_day
            
        except Exception as e:
            logger.error(f"Error calculating expected minutes: {e}")
            return 0
    
    def _find_time_gaps(self, data: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """Find time gaps in 1-minute data"""
        try:
            if len(data) < 2:
                return []
            
            gaps = []
            for i in range(1, len(data)):
                time_diff = data.index[i] - data.index[i-1]
                if time_diff > timedelta(minutes=2):  # Gap > 2 minutes
                    gaps.append((data.index[i-1], data.index[i]))
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error finding time gaps: {e}")
            return []
    
    def fetch_multiple_symbols_1m(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """Fetch 1-minute data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            logger.info(f"Fetching 1m data for {symbol}...")
            data = self.fetch_1m_data_optimized(symbol, days)
            if not data.empty:
                results[symbol] = data
                logger.info(f"✅ {symbol}: {len(data)} records")
            else:
                logger.warning(f"❌ {symbol}: No data")
        
        return results
    
    def get_latest_1m_data(self, symbol: str, minutes: int = 60) -> pd.DataFrame:
        """Get latest N minutes of 1-minute data"""
        try:
            # Fetch last 2 days to ensure we get enough data
            data = self.fetch_1m_data_optimized(symbol, days=2)
            
            if data.empty:
                return pd.DataFrame()
            
            # Get latest N minutes
            latest_data = data.tail(minutes)
            
            return latest_data
            
        except Exception as e:
            logger.error(f"Error getting latest 1m data: {e}")
            return pd.DataFrame()
    
    def save_1m_data(self, data: pd.DataFrame, symbol: str, format: str = 'parquet') -> str:
        """Save 1-minute data to file"""
        try:
            os.makedirs('outputs/data/1m', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'parquet':
                filename = f"outputs/data/1m/{symbol}_1m_{timestamp}.parquet"
                data.to_parquet(filename)
            elif format == 'csv':
                filename = f"outputs/data/1m/{symbol}_1m_{timestamp}.csv"
                data.to_csv(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved 1m data to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving 1m data: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimized fetcher
    fetcher = Optimized1MDataFetcher()
    
    # Test fetching 1-minute data
    print("Testing Optimized 1-Minute Data Fetcher...")
    
    # Fetch Nifty 50 1-minute data for 30 days
    nifty_1m = fetcher.fetch_1m_data_optimized('NIFTY_50', days=30)
    
    if not nifty_1m.empty:
        print(f"✅ Nifty 50 1m data: {len(nifty_1m)} records")
        print(f"   Date range: {nifty_1m.index[0]} to {nifty_1m.index[-1]}")
        print(f"   Latest price: ₹{nifty_1m['Close'].iloc[-1]:,.2f}")
        
        # Get quality metrics
        quality = fetcher.get_data_quality_metrics(nifty_1m)
        print(f"   Data quality score: {quality.quality_score:.2%}")
        print(f"   Data completeness: {quality.data_completeness:.2%}")
        print(f"   Missing records: {quality.missing_records}")
        
        # Save data
        filename = fetcher.save_1m_data(nifty_1m, 'NIFTY_50')
        print(f"   Data saved to: {filename}")
    else:
        print("❌ No 1-minute data retrieved")
    
    # Test multiple symbols
    print("\nTesting multiple symbols...")
    symbols = ['NIFTY_50', 'BANK_NIFTY']
    multi_data = fetcher.fetch_multiple_symbols_1m(symbols, days=7)
    
    for symbol, data in multi_data.items():
        print(f"   {symbol}: {len(data)} records")
