"""
Configuration settings for MCP-Powered Stock Analysis & Options Signal App

Centralized configuration for all application parameters, API keys, and settings.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Application configuration settings"""
    
    # Data sources
    DEFAULT_PERIOD: str = "1y"
    DEFAULT_TICKER: str = "NIFTY_50"  # Changed to Indian market default
    MAX_NEWS_ARTICLES: int = 10
    
    # Indian Market Settings
    DEFAULT_MARKET: str = "NSE"  # NSE or BSE
    TIMEZONE: str = "Asia/Kolkata"
    TRADING_HOURS: Dict[str, str] = None
    MARKET_HOLIDAYS: List[str] = None
    
    # Technical indicators
    RSI_PERIOD: int = 14
    EMA_FAST: int = 12
    EMA_SLOW: int = 26
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0
    ATR_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    STOCH_K_PERIOD: int = 14
    STOCH_D_PERIOD: int = 3
    
    # ML Model settings
    ML_TRAIN_TEST_SPLIT: float = 0.2
    ML_RANDOM_STATE: int = 42
    ML_N_ESTIMATORS: int = 100
    ML_LEARNING_RATE: float = 0.1
    ML_MAX_DEPTH: int = 6
    ML_MIN_SAMPLES: int = 50
    
    # Options settings
    DEFAULT_RISK_FREE_RATE: float = 0.02
    DEFAULT_VOLATILITY: float = 0.3
    OPTIONS_CONTRACTS_PER_POSITION: int = 100
    MAX_OPTIONS_EXPIRATIONS: int = 3
    
    # Trading signals
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    EMA_SIGNAL_THRESHOLD: float = 0.02
    ML_CONFIDENCE_THRESHOLD: float = 0.4
    SENTIMENT_WEIGHT: float = 0.3
    
    # IV Percentiles
    HIGH_IV_THRESHOLD: float = 70.0
    LOW_IV_THRESHOLD: float = 30.0
    
    # Portfolio settings
    DEFAULT_INITIAL_CAPITAL: float = 100000.0
    MAX_POSITIONS: int = 10
    DEFAULT_POSITION_SIZE: float = 0.1  # 10% of portfolio
    
    # Backtesting
    DEFAULT_BACKTEST_PERIOD: str = "2y"
    MIN_TRADING_DAYS: int = 5
    MAX_HOLDING_DAYS: int = 30
    
    # Visualization
    CHART_HEIGHT: int = 600
    CANDLESTICK_HEIGHT: int = 800
    DASHBOARD_HEIGHT: int = 1200
    
    # Colors
    BULLISH_COLOR: str = "#00C851"
    BEARISH_COLOR: str = "#FF4444"
    NEUTRAL_COLOR: str = "#FF8800"
    BACKGROUND_COLOR: str = "#F8F9FA"
    GRID_COLOR: str = "#E9ECEF"
    
    # API Keys (Direct configuration - no environment variables needed)
    NEWS_API_KEY: str = "your_news_api_key_here"
    TWITTER_API_KEY: str = "your_twitter_api_key_here"
    TWITTER_API_SECRET: str = "your_twitter_api_secret_here"
    TWITTER_ACCESS_TOKEN: str = "your_twitter_access_token_here"
    TWITTER_ACCESS_SECRET: str = "your_twitter_access_secret_here"
    ALPHA_VANTAGE_API_KEY: str = "UZ3VFKHQQEEGQMCZ"
    QUANDL_API_KEY: str = "3iDzmsGkQEQmEznLxa-z"
    
    # Additional API Keys for Indian Markets
    NSE_API_KEY: str = "your_nse_api_key_here"
    BSE_API_KEY: str = "your_bse_api_key_here"
    ZERODHA_API_KEY: str = "your_zerodha_api_key_here"
    UPSTOX_API_KEY: str = "your_upstox_api_key_here"
    FINNHUB_API_KEY: str = "d2tbtf9r01qkuv3k7mu0d2tbtf9r01qkuv3k7mug"
    POLYGON_API_KEY: str = "TwMlzxH1FcQE9jzqvLbspp0uDZoU6L0Q"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Initialize configuration settings"""
        # API Keys are now configured directly in the class definition above
        # Only override with environment variables if they exist (for backward compatibility)
        if os.getenv('NEWS_API_KEY'):
            self.NEWS_API_KEY = os.getenv('NEWS_API_KEY')
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            self.ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
        if os.getenv('QUANDL_API_KEY'):
            self.QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
        if os.getenv('TWITTER_API_KEY'):
            self.TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
        if os.getenv('TWITTER_API_SECRET'):
            self.TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
        if os.getenv('TWITTER_ACCESS_TOKEN'):
            self.TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
        if os.getenv('TWITTER_ACCESS_SECRET'):
            self.TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')
        
        # Initialize Indian market settings
        if self.TRADING_HOURS is None:
            self.TRADING_HOURS = {
                'pre_market': '09:00 - 09:15 IST',
                'normal_trading': '09:15 - 15:30 IST',
                'post_market': '15:40 - 16:00 IST'
            }
        
        if self.MARKET_HOLIDAYS is None:
            self.MARKET_HOLIDAYS = [
                '2024-01-26',  # Republic Day
                '2024-03-08',  # Holi
                '2024-03-29',  # Good Friday
                '2024-04-11',  # Eid ul Fitr
                '2024-04-17',  # Ram Navami
                '2024-05-01',  # Maharashtra Day
                '2024-06-17',  # Eid ul Adha
                '2024-08-15',  # Independence Day
                '2024-08-26',  # Janmashtami
                '2024-10-02',  # Gandhi Jayanti
                '2024-10-12',  # Dussehra
                '2024-11-01',  # Diwali
                '2024-11-15',  # Guru Nanak Jayanti
                '2024-12-25'   # Christmas
            ]
        
        # Override with environment variables if present
        self.DEFAULT_RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', self.DEFAULT_RISK_FREE_RATE))
        self.DEFAULT_VOLATILITY = float(os.getenv('DEFAULT_VOLATILITY', self.DEFAULT_VOLATILITY))
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', self.LOG_LEVEL)

# Global configuration instance
config = AppConfig()

# Indian Market Strategy configurations
INDIAN_MARKET_STRATEGIES = {
    'nifty_50_strategies': {
        'bull_call_spread': {
            'name': 'Nifty 50 Bull Call Spread',
            'description': 'Buy lower strike call, sell higher strike call on Nifty 50',
            'max_profit': 'Limited to difference between strikes minus net premium',
            'max_loss': 'Limited to net premium paid',
            'breakeven': 'Lower strike + net premium',
            'best_for': 'Moderately bullish outlook on Nifty 50 with high IV',
            'lot_size': 50,
            'margin_required': 'Approximately 1.5x of premium'
        },
        'bear_put_spread': {
            'name': 'Nifty 50 Bear Put Spread',
            'description': 'Buy higher strike put, sell lower strike put on Nifty 50',
            'max_profit': 'Limited to difference between strikes minus net premium',
            'max_loss': 'Limited to net premium paid',
            'breakeven': 'Higher strike - net premium',
            'best_for': 'Moderately bearish outlook on Nifty 50 with high IV',
            'lot_size': 50,
            'margin_required': 'Approximately 1.5x of premium'
        },
        'iron_condor': {
            'name': 'Nifty 50 Iron Condor',
            'description': 'Sell call spread + sell put spread on Nifty 50',
            'max_profit': 'Net premium received',
            'max_loss': 'Difference between strikes minus net premium',
            'breakeven': 'Two breakeven points',
            'best_for': 'Neutral outlook on Nifty 50 with high IV',
            'lot_size': 50,
            'margin_required': 'Approximately 2x of premium'
        }
    },
    'bank_nifty_strategies': {
        'bull_call_spread': {
            'name': 'Bank Nifty Bull Call Spread',
            'description': 'Buy lower strike call, sell higher strike call on Bank Nifty',
            'max_profit': 'Limited to difference between strikes minus net premium',
            'max_loss': 'Limited to net premium paid',
            'breakeven': 'Lower strike + net premium',
            'best_for': 'Moderately bullish outlook on banking sector with high IV',
            'lot_size': 25,
            'margin_required': 'Approximately 1.5x of premium'
        },
        'bear_put_spread': {
            'name': 'Bank Nifty Bear Put Spread',
            'description': 'Buy higher strike put, sell lower strike put on Bank Nifty',
            'max_profit': 'Limited to difference between strikes minus net premium',
            'max_loss': 'Limited to net premium paid',
            'breakeven': 'Higher strike - net premium',
            'best_for': 'Moderately bearish outlook on banking sector with high IV',
            'lot_size': 25,
            'margin_required': 'Approximately 1.5x of premium'
        },
        'straddle': {
            'name': 'Bank Nifty Straddle',
            'description': 'Buy call and put at same strike on Bank Nifty',
            'max_profit': 'Unlimited in either direction',
            'max_loss': 'Total premium paid',
            'breakeven': 'Strike ± total premium',
            'best_for': 'High volatility expected in banking sector',
            'lot_size': 25,
            'margin_required': 'Total premium paid'
        }
    }
}

# Strategy configurations
STRATEGY_CONFIGS = {
    'bull_call_spread': {
        'name': 'Bull Call Spread',
        'description': 'Buy lower strike call, sell higher strike call',
        'max_profit': 'Limited to difference between strikes minus net premium',
        'max_loss': 'Limited to net premium paid',
        'breakeven': 'Lower strike + net premium',
        'best_for': 'Moderately bullish outlook with high IV'
    },
    'bear_put_spread': {
        'name': 'Bear Put Spread',
        'description': 'Buy higher strike put, sell lower strike put',
        'max_profit': 'Limited to difference between strikes minus net premium',
        'max_loss': 'Limited to net premium paid',
        'breakeven': 'Higher strike - net premium',
        'best_for': 'Moderately bearish outlook with high IV'
    },
    'iron_condor': {
        'name': 'Iron Condor',
        'description': 'Sell call spread + sell put spread',
        'max_profit': 'Net premium received',
        'max_loss': 'Difference between strikes minus net premium',
        'breakeven': 'Two breakeven points',
        'best_for': 'Neutral outlook with high IV'
    },
    'long_straddle': {
        'name': 'Long Straddle',
        'description': 'Buy call and put at same strike',
        'max_profit': 'Unlimited in either direction',
        'max_loss': 'Total premium paid',
        'breakeven': 'Strike ± total premium',
        'best_for': 'High volatility expected, direction uncertain'
    },
    'long_strangle': {
        'name': 'Long Strangle',
        'description': 'Buy OTM call and OTM put',
        'max_profit': 'Unlimited in either direction',
        'max_loss': 'Total premium paid',
        'breakeven': 'Two breakeven points',
        'best_for': 'High volatility expected, direction uncertain'
    },
    'calendar_spread': {
        'name': 'Calendar Spread',
        'description': 'Sell short-term option, buy long-term option',
        'max_profit': 'Net premium received',
        'max_loss': 'Net premium paid',
        'breakeven': 'Varies with time decay',
        'best_for': 'Neutral outlook, time decay advantage'
    }
}

# Risk management rules
RISK_MANAGEMENT_RULES = {
    'max_portfolio_risk': 0.02,  # 2% of portfolio per trade
    'max_daily_loss': 0.05,      # 5% daily loss limit
    'max_drawdown': 0.15,        # 15% maximum drawdown
    'position_sizing': {
        'conservative': 0.05,    # 5% per position
        'moderate': 0.10,        # 10% per position
        'aggressive': 0.20       # 20% per position
    },
    'stop_loss_percentages': {
        'conservative': 0.10,    # 10% stop loss
        'moderate': 0.15,        # 15% stop loss
        'aggressive': 0.25       # 25% stop loss
    }
}

# Market regimes
MARKET_REGIMES = {
    'high_volatility': {
        'iv_threshold': 70,
        'strategies': ['iron_condor', 'calendar_spread', 'straddle_selling'],
        'description': 'High IV environment favors selling strategies'
    },
    'low_volatility': {
        'iv_threshold': 30,
        'strategies': ['long_straddle', 'long_strangle', 'directional_plays'],
        'description': 'Low IV environment favors buying strategies'
    },
    'trending_market': {
        'momentum_threshold': 0.02,
        'strategies': ['directional_spreads', 'momentum_plays'],
        'description': 'Strong directional movement favors directional strategies'
    },
    'ranging_market': {
        'volatility_threshold': 0.01,
        'strategies': ['iron_condor', 'butterfly_spreads', 'calendar_spreads'],
        'description': 'Sideways market favors range-bound strategies'
    }
}

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config

def update_config(**kwargs) -> None:
    """Update configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

def get_strategy_config(strategy_name: str) -> Dict[str, Any]:
    """Get strategy configuration by name"""
    return STRATEGY_CONFIGS.get(strategy_name, {})

def get_risk_rules() -> Dict[str, Any]:
    """Get risk management rules"""
    return RISK_MANAGEMENT_RULES

def get_market_regime(iv_percentile: float, momentum: float) -> str:
    """Determine market regime based on IV and momentum"""
    if iv_percentile > MARKET_REGIMES['high_volatility']['iv_threshold']:
        return 'high_volatility'
    elif iv_percentile < MARKET_REGIMES['low_volatility']['iv_threshold']:
        return 'low_volatility'
    elif abs(momentum) > MARKET_REGIMES['trending_market']['momentum_threshold']:
        return 'trending_market'
    else:
        return 'ranging_market'

def get_indian_market_strategies(index: str) -> Dict[str, Any]:
    """Get Indian market strategies for specific index"""
    if index == 'NIFTY_50':
        return INDIAN_MARKET_STRATEGIES.get('nifty_50_strategies', {})
    elif index == 'BANK_NIFTY':
        return INDIAN_MARKET_STRATEGIES.get('bank_nifty_strategies', {})
    else:
        return {}

def get_lot_size(index: str) -> int:
    """Get lot size for Indian market indices"""
    lot_sizes = {
        'NIFTY_50': 50,
        'BANK_NIFTY': 25,
        'SENSEX': 10,
        'NIFTY_IT': 25,
        'NIFTY_AUTO': 25,
        'NIFTY_PHARMA': 25
    }
    return lot_sizes.get(index, 50)

def get_margin_requirements(index: str, strategy: str) -> float:
    """Get margin requirements for Indian market strategies"""
    strategies = get_indian_market_strategies(index)
    if strategy in strategies:
        return strategies[strategy].get('margin_required', 'Varies')
    return 'Varies'

def is_market_holiday(date: str) -> bool:
    """Check if given date is a market holiday"""
    return date in config.MARKET_HOLIDAYS

def get_next_trading_day(date: str) -> str:
    """Get next trading day (excluding holidays and weekends)"""
    from datetime import datetime, timedelta
    
    current_date = datetime.strptime(date, '%Y-%m-%d')
    next_date = current_date + timedelta(days=1)
    
    # Skip weekends and holidays
    while next_date.weekday() >= 5 or is_market_holiday(next_date.strftime('%Y-%m-%d')):
        next_date += timedelta(days=1)
    
    return next_date.strftime('%Y-%m-%d')
