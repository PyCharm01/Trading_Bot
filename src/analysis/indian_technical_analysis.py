#!/usr/bin/env python3
"""
Indian Market Technical Analysis

This module provides comprehensive technical analysis specifically designed for
Indian market indices including Nifty 50, Bank Nifty, and Sensex with
market-specific indicators and analysis methods.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import talib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    """Technical analysis signal with strength and reasoning"""
    signal: str  # BUY, SELL, NEUTRAL
    strength: float  # 0.0 to 1.0
    reasoning: str
    indicators: Dict[str, Any]
    timestamp: datetime

class IndianTechnicalIndicators:
    """Advanced technical indicators specifically for Indian market indices"""
    
    def __init__(self):
        self.indicators_cache = {}
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(0)
    
    def calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price with robust error handling"""
        try:
            # Handle missing or zero volume data
            volume = volume.fillna(0)
            volume = volume.replace(0, 1)  # Replace zeros with 1 to avoid division by zero
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate VWAP with cumulative sums
            cumulative_volume = volume.cumsum()
            cumulative_typical_price_volume = (typical_price * volume).cumsum()
            
            # Calculate VWAP, handling division by zero
            vwap = cumulative_typical_price_volume / cumulative_volume
            
            # Fill any remaining NaN values with the close price
            vwap = vwap.fillna(close)
            
            return vwap
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            # Fallback to simple moving average of close price
            return close.rolling(window=20).mean().fillna(close)
    
    def calculate_vwap_price_channel(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, 
                                   period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate VWAP Price Channel with upper and lower bands - Enhanced version with robust error handling"""
        try:
            # Calculate VWAP
            vwap = self.calculate_vwap(high, low, close, volume)
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate standard deviation of price from VWAP with smoothing
            price_deviation = (typical_price - vwap).rolling(window=period).std()
            
            # Fill NaN values in price deviation
            price_deviation = price_deviation.fillna(price_deviation.mean())
            if price_deviation.isna().all():
                # Fallback: use ATR-based deviation
                atr = self.calculate_atr(high, low, close, period)
                price_deviation = atr.fillna(atr.mean())
            
            # Apply exponential smoothing to create smoother curves
            price_deviation_smooth = price_deviation.ewm(span=period//2).mean()
            price_deviation_smooth = price_deviation_smooth.fillna(price_deviation)
            
            # Calculate upper and lower bands with adaptive multiplier
            # Use higher multiplier for more volatile periods
            volatility_factor = price_deviation_smooth / price_deviation_smooth.rolling(window=period).mean()
            volatility_factor = volatility_factor.fillna(1.0)  # Default to 1.0 if NaN
            adaptive_multiplier = std_dev * (0.8 + 0.4 * volatility_factor.clip(0.5, 2.0))
            
            vwap_upper = vwap + (price_deviation_smooth * adaptive_multiplier)
            vwap_lower = vwap - (price_deviation_smooth * adaptive_multiplier)
            
            # Apply additional smoothing to create "corner cutting" effect
            vwap_upper_smooth = vwap_upper.ewm(span=period//3).mean()
            vwap_lower_smooth = vwap_lower.ewm(span=period//3).mean()
            vwap_smooth = vwap.ewm(span=period//4).mean()
            
            # Fill any remaining NaN values
            vwap_upper_smooth = vwap_upper_smooth.fillna(vwap_upper).fillna(close * 1.02)
            vwap_lower_smooth = vwap_lower_smooth.fillna(vwap_lower).fillna(close * 0.98)
            vwap_smooth = vwap_smooth.fillna(vwap).fillna(close)
            
            return vwap_upper_smooth, vwap_smooth, vwap_lower_smooth
            
        except Exception as e:
            logger.error(f"Error calculating VWAP Price Channel: {e}")
            # Fallback to simple Bollinger Bands
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            # Fill NaN values
            upper = upper.fillna(close * 1.02)
            lower = lower.fillna(close * 0.98)
            sma = sma.fillna(close)
            
            return upper, sma, lower
    
    def calculate_donchian_channel(self, high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Traditional Donchian Channel with upper, middle, and lower bands"""
        # Donchian Channel: Upper = highest high, Lower = lowest low, Middle = average
        donchian_upper = high.rolling(window=period).max()
        donchian_lower = low.rolling(window=period).min()
        donchian_middle = (donchian_upper + donchian_lower) / 2
        
        return donchian_upper, donchian_middle, donchian_lower
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Average Directional Index"""
        try:
            # Use TA-Lib if available
            adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
            plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=period)
            minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=period)
            
            return pd.Series(adx, index=close.index), pd.Series(plus_di, index=close.index), pd.Series(minus_di, index=close.index)
        except:
            # Fallback calculation
            return self._calculate_adx_fallback(high, low, close, period)
    
    def _calculate_adx_fallback(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Fallback ADX calculation without TA-Lib"""
        # Simplified ADX calculation
        tr = self.calculate_atr(high, low, close, 1)
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di

class IndianMarketAnalyzer:
    """Comprehensive technical analysis for Indian market indices"""
    
    def __init__(self):
        self.indicators = IndianTechnicalIndicators()
        self.analysis_cache = {}
    
    def analyze_index(self, data: pd.DataFrame, index_name: str) -> Dict[str, Any]:
        """Perform comprehensive technical analysis for Indian market index"""
        try:
            if data.empty:
                return {"error": "No data provided"}
            
            # Calculate all technical indicators
            indicators = self._calculate_all_indicators(data)
            
            # Generate trading signals
            signals = self._generate_trading_signals(data, indicators, index_name)
            
            # Calculate market regime
            market_regime = self._determine_market_regime(indicators)
            
            # Calculate support and resistance levels
            support_resistance = self._calculate_support_resistance(data)
            
            # Calculate volatility analysis
            volatility_analysis = self._analyze_volatility(data, indicators)
            
            # Calculate momentum analysis
            momentum_analysis = self._analyze_momentum(indicators)
            
            # Compile comprehensive analysis
            # Extract key indicators for easy access (get last values if Series)
            rsi_value = float(indicators.get('rsi', 50.0).iloc[-1] if hasattr(indicators.get('rsi', 50.0), 'iloc') else indicators.get('rsi', 50.0))
            ema_12_value = float(indicators.get('ema_12', 0.0).iloc[-1] if hasattr(indicators.get('ema_12', 0.0), 'iloc') else indicators.get('ema_12', 0.0))
            ema_26_value = float(indicators.get('ema_26', 0.0).iloc[-1] if hasattr(indicators.get('ema_26', 0.0), 'iloc') else indicators.get('ema_26', 0.0))
            macd_signal_value = indicators.get('macd_signal', 'NEUTRAL')
            if hasattr(macd_signal_value, 'iloc'):
                macd_signal_value = macd_signal_value.iloc[-1]
            
            # Determine market trend based on EMAs
            if ema_12_value > ema_26_value:
                market_trend = 'BULLISH'
            elif ema_12_value < ema_26_value:
                market_trend = 'BEARISH'
            else:
                market_trend = 'NEUTRAL'
            
            # Determine volatility level
            volatility_level = volatility_analysis.get('regime', 'NORMAL')
            
            analysis = {
                "index_name": index_name,
                "timestamp": datetime.now().isoformat(),
                "current_price": float(data['Close'].iloc[-1]),
                "indicators": indicators,
                "signals": signals,
                "overall_signal": signals.get('overall', {}).get('signal', 'NEUTRAL'),
                "signal_strength": signals.get('overall', {}).get('strength', 0.3) * 100,
                "market_trend": market_trend,
                "volatility_level": volatility_level,
                "rsi": rsi_value,
                "ema_12": ema_12_value,
                "ema_26": ema_26_value,
                "macd_signal": macd_signal_value,
                "market_regime": market_regime,
                "support_resistance": support_resistance,
                "volatility_analysis": volatility_analysis,
                "momentum_analysis": momentum_analysis,
                "summary": self._generate_analysis_summary(signals, market_regime, volatility_analysis)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {index_name}: {e}")
            return {"error": str(e)}
    
    def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # Price-based indicators
        indicators['ema_12'] = self.indicators.calculate_ema(data['Close'], 12)
        indicators['ema_26'] = self.indicators.calculate_ema(data['Close'], 26)
        indicators['ema_50'] = self.indicators.calculate_ema(data['Close'], 50)
        indicators['sma_20'] = self.indicators.calculate_sma(data['Close'], 20)
        indicators['sma_50'] = self.indicators.calculate_sma(data['Close'], 50)
        
        # Momentum indicators
        indicators['rsi'] = self.indicators.calculate_rsi(data['Close'])
        indicators['stoch_k'], indicators['stoch_d'] = self.indicators.calculate_stochastic(
            data['High'], data['Low'], data['Close']
        )
        indicators['williams_r'] = self.indicators.calculate_williams_r(
            data['High'], data['Low'], data['Close']
        )
        indicators['cci'] = self.indicators.calculate_cci(
            data['High'], data['Low'], data['Close']
        )
        
        # Trend indicators
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.indicators.calculate_macd(data['Close'])
        indicators['adx'], indicators['plus_di'], indicators['minus_di'] = self.indicators.calculate_adx(
            data['High'], data['Low'], data['Close']
        )
        
        # Volatility indicators
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self.indicators.calculate_bollinger_bands(data['Close'])
        indicators['atr'] = self.indicators.calculate_atr(data['High'], data['Low'], data['Close'])
        
        # Volume indicators
        indicators['vwap'] = self.indicators.calculate_vwap(data['High'], data['Low'], data['Close'], data['Volume'])
        indicators['obv'] = self.indicators.calculate_obv(data['Close'], data['Volume'])
        
        # VWAP Price Channel
        indicators['vwap_upper'], indicators['vwap_middle'], indicators['vwap_lower'] = self.indicators.calculate_vwap_price_channel(
            data['High'], data['Low'], data['Close'], data['Volume']
        )
        
        # Traditional Donchian Channel
        indicators['donchian_upper'], indicators['donchian_middle'], indicators['donchian_lower'] = self.indicators.calculate_donchian_channel(
            data['High'], data['Low']
        )
        
        return indicators
    
    def _generate_trading_signals(self, data: pd.DataFrame, indicators: Dict[str, Any], index_name: str) -> Dict[str, Any]:
        """Generate comprehensive trading signals"""
        signals = {}
        current_price = data['Close'].iloc[-1]
        
        # RSI signals
        rsi_current = indicators['rsi'].iloc[-1]
        if rsi_current < 30:
            signals['rsi'] = {'signal': 'BUY', 'strength': 0.8, 'reasoning': 'RSI oversold'}
        elif rsi_current > 70:
            signals['rsi'] = {'signal': 'SELL', 'strength': 0.8, 'reasoning': 'RSI overbought'}
        else:
            signals['rsi'] = {'signal': 'NEUTRAL', 'strength': 0.3, 'reasoning': 'RSI neutral'}
        
        # MACD signals
        macd_current = indicators['macd'].iloc[-1]
        macd_signal_current = indicators['macd_signal'].iloc[-1]
        macd_hist_current = indicators['macd_hist'].iloc[-1]
        
        if macd_current > macd_signal_current and macd_hist_current > 0:
            signals['macd'] = {'signal': 'BUY', 'strength': 0.7, 'reasoning': 'MACD bullish crossover'}
        elif macd_current < macd_signal_current and macd_hist_current < 0:
            signals['macd'] = {'signal': 'SELL', 'strength': 0.7, 'reasoning': 'MACD bearish crossover'}
        else:
            signals['macd'] = {'signal': 'NEUTRAL', 'strength': 0.3, 'reasoning': 'MACD neutral'}
        
        # Moving average signals
        ema_12_current = indicators['ema_12'].iloc[-1]
        ema_26_current = indicators['ema_26'].iloc[-1]
        
        if current_price > ema_12_current > ema_26_current:
            signals['ma'] = {'signal': 'BUY', 'strength': 0.6, 'reasoning': 'Price above EMAs, bullish trend'}
        elif current_price < ema_12_current < ema_26_current:
            signals['ma'] = {'signal': 'SELL', 'strength': 0.6, 'reasoning': 'Price below EMAs, bearish trend'}
        else:
            signals['ma'] = {'signal': 'NEUTRAL', 'strength': 0.3, 'reasoning': 'Mixed MA signals'}
        
        # Bollinger Bands signals
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        
        if current_price <= bb_lower:
            signals['bb'] = {'signal': 'BUY', 'strength': 0.7, 'reasoning': 'Price at lower Bollinger Band'}
        elif current_price >= bb_upper:
            signals['bb'] = {'signal': 'SELL', 'strength': 0.7, 'reasoning': 'Price at upper Bollinger Band'}
        else:
            signals['bb'] = {'signal': 'NEUTRAL', 'strength': 0.3, 'reasoning': 'Price within Bollinger Bands'}
        
        # Stochastic signals
        stoch_k_current = indicators['stoch_k'].iloc[-1]
        stoch_d_current = indicators['stoch_d'].iloc[-1]
        
        if stoch_k_current < 20 and stoch_d_current < 20:
            signals['stoch'] = {'signal': 'BUY', 'strength': 0.6, 'reasoning': 'Stochastic oversold'}
        elif stoch_k_current > 80 and stoch_d_current > 80:
            signals['stoch'] = {'signal': 'SELL', 'strength': 0.6, 'reasoning': 'Stochastic overbought'}
        else:
            signals['stoch'] = {'signal': 'NEUTRAL', 'strength': 0.3, 'reasoning': 'Stochastic neutral'}
        
        # VWAP Price Channel signals
        vwap_upper = indicators['vwap_upper'].iloc[-1]
        vwap_lower = indicators['vwap_lower'].iloc[-1]
        vwap_middle = indicators['vwap_middle'].iloc[-1]
        
        if current_price <= vwap_lower:
            signals['vwap_channel'] = {'signal': 'BUY', 'strength': 0.7, 'reasoning': 'Price at VWAP lower band - potential support'}
        elif current_price >= vwap_upper:
            signals['vwap_channel'] = {'signal': 'SELL', 'strength': 0.7, 'reasoning': 'Price at VWAP upper band - potential resistance'}
        elif current_price > vwap_middle:
            signals['vwap_channel'] = {'signal': 'BUY', 'strength': 0.5, 'reasoning': 'Price above VWAP - bullish bias'}
        elif current_price < vwap_middle:
            signals['vwap_channel'] = {'signal': 'SELL', 'strength': 0.5, 'reasoning': 'Price below VWAP - bearish bias'}
        else:
            signals['vwap_channel'] = {'signal': 'NEUTRAL', 'strength': 0.3, 'reasoning': 'Price near VWAP - neutral'}
        
        # Overall signal aggregation
        buy_signals = sum(1 for s in signals.values() if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in signals.values() if s['signal'] == 'SELL')
        
        if buy_signals > sell_signals:
            overall_signal = 'BUY'
            overall_strength = min(buy_signals / len(signals), 1.0)
        elif sell_signals > buy_signals:
            overall_signal = 'SELL'
            overall_strength = min(sell_signals / len(signals), 1.0)
        else:
            overall_signal = 'NEUTRAL'
            overall_strength = 0.3
        
        signals['overall'] = {
            'signal': overall_signal,
            'strength': overall_strength,
            'reasoning': f'{buy_signals} buy signals, {sell_signals} sell signals'
        }
        
        return signals
    
    def _determine_market_regime(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Determine current market regime"""
        regime = {}
        
        # Volatility regime
        atr_current = indicators['atr'].iloc[-1]
        atr_avg = indicators['atr'].mean()
        
        if atr_current > atr_avg * 1.5:
            regime['volatility'] = 'High'
        elif atr_current < atr_avg * 0.5:
            regime['volatility'] = 'Low'
        else:
            regime['volatility'] = 'Normal'
        
        # Trend regime
        ema_12 = indicators['ema_12'].iloc[-1]
        ema_26 = indicators['ema_26'].iloc[-1]
        ema_50 = indicators['ema_50'].iloc[-1]
        
        if ema_12 > ema_26 > ema_50:
            regime['trend'] = 'Strong Uptrend'
        elif ema_12 < ema_26 < ema_50:
            regime['trend'] = 'Strong Downtrend'
        elif ema_12 > ema_26:
            regime['trend'] = 'Weak Uptrend'
        elif ema_12 < ema_26:
            regime['trend'] = 'Weak Downtrend'
        else:
            regime['trend'] = 'Sideways'
        
        # Momentum regime
        rsi_current = indicators['rsi'].iloc[-1]
        if rsi_current > 70:
            regime['momentum'] = 'Overbought'
        elif rsi_current < 30:
            regime['momentum'] = 'Oversold'
        else:
            regime['momentum'] = 'Neutral'
        
        return regime
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate key support and resistance levels"""
        # Simple pivot-based support and resistance
        high_20 = data['High'].rolling(window=20).max().iloc[-1]
        low_20 = data['Low'].rolling(window=20).min().iloc[-1]
        high_50 = data['High'].rolling(window=50).max().iloc[-1]
        low_50 = data['Low'].rolling(window=50).min().iloc[-1]
        
        current_price = data['Close'].iloc[-1]
        
        return {
            'resistance_1': float(high_20),
            'resistance_2': float(high_50),
            'support_1': float(low_20),
            'support_2': float(low_50),
            'current_price': float(current_price),
            'distance_to_resistance_1': float((high_20 - current_price) / current_price * 100),
            'distance_to_support_1': float((current_price - low_20) / current_price * 100)
        }
    
    def _analyze_volatility(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility patterns"""
        # Calculate volatility metrics
        returns = data['Close'].pct_change().dropna()
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Bollinger Band width
        bb_width = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
        current_bb_width = bb_width.iloc[-1]
        avg_bb_width = bb_width.mean()
        
        # ATR analysis
        atr_current = indicators['atr'].iloc[-1]
        atr_avg = indicators['atr'].mean()
        
        return {
            'daily_volatility': float(daily_vol),
            'annualized_volatility': float(annualized_vol),
            'current_atr': float(atr_current),
            'atr_percentile': float((atr_current / atr_avg - 1) * 100),
            'bb_width_current': float(current_bb_width),
            'bb_width_percentile': float((current_bb_width / avg_bb_width - 1) * 100),
            'volatility_regime': 'High' if current_bb_width > avg_bb_width * 1.2 else 'Low' if current_bb_width < avg_bb_width * 0.8 else 'Normal'
        }
    
    def _analyze_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        rsi_current = indicators['rsi'].iloc[-1]
        macd_current = indicators['macd'].iloc[-1]
        macd_signal_current = indicators['macd_signal'].iloc[-1]
        stoch_k_current = indicators['stoch_k'].iloc[-1]
        stoch_d_current = indicators['stoch_d'].iloc[-1]
        
        # Momentum score
        momentum_score = 0
        if rsi_current > 50:
            momentum_score += 1
        if macd_current > macd_signal_current:
            momentum_score += 1
        if stoch_k_current > stoch_d_current:
            momentum_score += 1
        
        momentum_strength = momentum_score / 3
        
        return {
            'rsi': float(rsi_current),
            'macd': float(macd_current),
            'macd_signal': float(macd_signal_current),
            'stochastic_k': float(stoch_k_current),
            'stochastic_d': float(stoch_d_current),
            'momentum_score': float(momentum_strength),
            'momentum_direction': 'Bullish' if momentum_strength > 0.6 else 'Bearish' if momentum_strength < 0.4 else 'Neutral'
        }
    
    def _generate_analysis_summary(self, signals: Dict[str, Any], market_regime: Dict[str, Any], volatility_analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis summary"""
        overall_signal = signals.get('overall', {})
        signal_text = overall_signal.get('signal', 'NEUTRAL')
        strength = overall_signal.get('strength', 0.3)
        
        trend = market_regime.get('trend', 'Unknown')
        volatility = volatility_analysis.get('volatility_regime', 'Normal')
        
        summary = f"Overall Signal: {signal_text} (Strength: {strength:.1%})\n"
        summary += f"Market Trend: {trend}\n"
        summary += f"Volatility: {volatility}\n"
        
        # Add specific recommendations
        if signal_text == 'BUY' and strength > 0.6:
            summary += "Recommendation: Consider long positions with proper risk management"
        elif signal_text == 'SELL' and strength > 0.6:
            summary += "Recommendation: Consider short positions or exit long positions"
        else:
            summary += "Recommendation: Wait for clearer signals or use range-bound strategies"
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create analyzer
    analyzer = IndianMarketAnalyzer()
    
    # Test with sample data
    print("Testing Indian Market Technical Analysis...")
    
    # This would typically be called with real market data
    print("Indian Market Technical Analysis module loaded successfully")
