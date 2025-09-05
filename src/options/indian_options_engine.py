#!/usr/bin/env python3
"""
Indian Options Strategy Engine

This module provides comprehensive options strategy analysis and recommendations
specifically designed for Indian market derivatives including Nifty 50, Bank Nifty,
and Sensex options with Indian market-specific parameters and regulations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy.stats import norm
import math

logger = logging.getLogger(__name__)

@dataclass
class OptionsContract:
    """Options contract details"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    premium: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    lot_size: int

@dataclass
class OptionsStrategy:
    """Options strategy configuration"""
    name: str
    description: str
    legs: List[Dict[str, Any]]
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    risk_reward_ratio: float
    margin_required: float
    lot_size: int

class BlackScholesCalculator:
    """Black-Scholes options pricing calculator for Indian markets"""
    
    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option price using Black-Scholes model"""
        try:
            # Convert time to years (Indian markets have different expiry cycles)
            if T <= 0:
                return 0.0
            
            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Calculate option price
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating option price: {e}")
            return 0.0
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            if T <= 0:
                return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:  # put
                delta = norm.cdf(d1) - 1
            
            # Gamma
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            if option_type.lower() == 'call':
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            
            # Vega
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Convert to daily theta
                'vega': vega / 100     # Convert to 1% volatility change
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

class IndianOptionsStrategyEngine:
    """Comprehensive options strategy engine for Indian markets"""
    
    def __init__(self):
        self.bs_calculator = BlackScholesCalculator()
        self.risk_free_rate = 0.06  # Indian 10-year bond yield approximation
        self.lot_sizes = {
            'NIFTY_50': 50,
            'BANK_NIFTY': 25,
            'SENSEX': 10,
            'NIFTY_IT': 25,
            'NIFTY_AUTO': 25,
            'NIFTY_PHARMA': 25
        }
    
    def analyze_options_chain(self, options_data: Dict, current_price: float, index_name: str) -> Dict[str, Any]:
        """Analyze options chain for Indian market indices"""
        try:
            analysis = {
                'index_name': index_name,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'expirations': {}
            }
            
            for exp_date, data in options_data.items():
                calls = data.get('calls', pd.DataFrame())
                puts = data.get('puts', pd.DataFrame())
                
                if calls.empty or puts.empty:
                    continue
                
                # Calculate days to expiry
                expiry_date = pd.to_datetime(exp_date)
                days_to_expiry = (expiry_date - datetime.now()).days
                
                if days_to_expiry <= 0:
                    continue
                
                # Analyze ATM options
                atm_analysis = self._analyze_atm_options(calls, puts, current_price, days_to_expiry)
                
                # Calculate IV skew
                iv_skew = self._calculate_iv_skew(calls, puts, current_price)
                
                # Calculate put-call ratio
                pcr = self._calculate_put_call_ratio(calls, puts)
                
                # Analyze volume and OI
                volume_oi_analysis = self._analyze_volume_oi(calls, puts)
                
                analysis['expirations'][exp_date] = {
                    'days_to_expiry': days_to_expiry,
                    'atm_analysis': atm_analysis,
                    'iv_skew': iv_skew,
                    'put_call_ratio': pcr,
                    'volume_oi_analysis': volume_oi_analysis,
                    'total_calls_volume': int(calls['volume'].sum()) if 'volume' in calls.columns else 0,
                    'total_puts_volume': int(puts['volume'].sum()) if 'volume' in puts.columns else 0,
                    'total_calls_oi': int(calls['openInterest'].sum()) if 'openInterest' in calls.columns else 0,
                    'total_puts_oi': int(puts['openInterest'].sum()) if 'openInterest' in puts.columns else 0
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing options chain: {e}")
            return {}
    
    def _analyze_atm_options(self, calls: pd.DataFrame, puts: pd.DataFrame, current_price: float, days_to_expiry: float) -> Dict[str, Any]:
        """Analyze At-The-Money options"""
        try:
            # Find ATM strike
            atm_calls = calls[abs(calls['strike'] - current_price) == abs(calls['strike'] - current_price).min()]
            atm_puts = puts[abs(puts['strike'] - current_price) == abs(puts['strike'] - current_price).min()]
            
            if atm_calls.empty or atm_puts.empty:
                return {}
            
            call_premium = atm_calls['lastPrice'].iloc[0] if 'lastPrice' in atm_calls.columns else 0
            put_premium = atm_puts['lastPrice'].iloc[0] if 'lastPrice' in atm_puts.columns else 0
            
            # Calculate implied volatility
            call_iv = atm_calls['impliedVolatility'].iloc[0] if 'impliedVolatility' in atm_calls.columns else 0
            put_iv = atm_puts['impliedVolatility'].iloc[0] if 'impliedVolatility' in atm_puts.columns else 0
            
            # Calculate time value
            time_value = (call_premium + put_premium) / 2
            
            # Calculate Greeks
            time_to_expiry = days_to_expiry / 365
            call_greeks = self.bs_calculator.calculate_greeks(
                current_price, atm_calls['strike'].iloc[0], time_to_expiry, 
                self.risk_free_rate, call_iv, 'call'
            )
            put_greeks = self.bs_calculator.calculate_greeks(
                current_price, atm_puts['strike'].iloc[0], time_to_expiry, 
                self.risk_free_rate, put_iv, 'put'
            )
            
            return {
                'strike': float(atm_calls['strike'].iloc[0]),
                'call_premium': float(call_premium),
                'put_premium': float(put_premium),
                'call_iv': float(call_iv),
                'put_iv': float(put_iv),
                'time_value': float(time_value),
                'call_greeks': call_greeks,
                'put_greeks': put_greeks,
                'iv_difference': float(call_iv - put_iv)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ATM options: {e}")
            return {}
    
    def _calculate_iv_skew(self, calls: pd.DataFrame, puts: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate implied volatility skew"""
        try:
            # Calculate IV for different strikes
            strikes = []
            call_ivs = []
            put_ivs = []
            
            for _, row in calls.iterrows():
                if 'impliedVolatility' in row and 'strike' in row:
                    strikes.append(row['strike'])
                    call_ivs.append(row['impliedVolatility'])
            
            for _, row in puts.iterrows():
                if 'impliedVolatility' in row and 'strike' in row:
                    if row['strike'] in strikes:
                        idx = strikes.index(row['strike'])
                        put_ivs.append(row['impliedVolatility'])
                    else:
                        strikes.append(row['strike'])
                        put_ivs.append(row['impliedVolatility'])
            
            if not strikes:
                return {}
            
            # Calculate skew metrics
            atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - current_price))
            
            # Put skew (lower strikes vs ATM)
            put_skew = 0
            if len(put_ivs) > atm_idx and atm_idx > 0:
                put_skew = put_ivs[atm_idx] - put_ivs[0] if put_ivs[0] > 0 else 0
            
            # Call skew (higher strikes vs ATM)
            call_skew = 0
            if len(call_ivs) > atm_idx and atm_idx < len(call_ivs) - 1:
                call_skew = call_ivs[-1] - call_ivs[atm_idx] if call_ivs[-1] > 0 else 0
            
            return {
                'put_skew': float(put_skew),
                'call_skew': float(call_skew),
                'overall_skew': float(put_skew - call_skew),
                'skew_direction': 'Put skew' if put_skew > call_skew else 'Call skew' if call_skew > put_skew else 'Neutral'
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV skew: {e}")
            return {}
    
    def _calculate_put_call_ratio(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Dict[str, float]:
        """Calculate put-call ratio"""
        try:
            # Volume PCR
            call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            volume_pcr = put_volume / call_volume if call_volume > 0 else 0
            
            # Open Interest PCR
            call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
            put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
            oi_pcr = put_oi / call_oi if call_oi > 0 else 0
            
            return {
                'volume_pcr': float(volume_pcr),
                'oi_pcr': float(oi_pcr),
                'sentiment': 'Bearish' if volume_pcr > 1.2 else 'Bullish' if volume_pcr < 0.8 else 'Neutral'
            }
            
        except Exception as e:
            logger.error(f"Error calculating PCR: {e}")
            return {}
    
    def _analyze_volume_oi(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume and open interest patterns"""
        try:
            # Find highest volume strikes
            if 'volume' in calls.columns and 'strike' in calls.columns:
                max_volume_call = calls.loc[calls['volume'].idxmax()]
                max_volume_call_strike = max_volume_call['strike']
                max_volume_call_volume = max_volume_call['volume']
            else:
                max_volume_call_strike = 0
                max_volume_call_volume = 0
            
            if 'volume' in puts.columns and 'strike' in puts.columns:
                max_volume_put = puts.loc[puts['volume'].idxmax()]
                max_volume_put_strike = max_volume_put['strike']
                max_volume_put_volume = max_volume_put['volume']
            else:
                max_volume_put_strike = 0
                max_volume_put_volume = 0
            
            # Find highest OI strikes
            if 'openInterest' in calls.columns and 'strike' in calls.columns:
                max_oi_call = calls.loc[calls['openInterest'].idxmax()]
                max_oi_call_strike = max_oi_call['strike']
                max_oi_call_oi = max_oi_call['openInterest']
            else:
                max_oi_call_strike = 0
                max_oi_call_oi = 0
            
            if 'openInterest' in puts.columns and 'strike' in puts.columns:
                max_oi_put = puts.loc[puts['openInterest'].idxmax()]
                max_oi_put_strike = max_oi_put['strike']
                max_oi_put_oi = max_oi_put['openInterest']
            else:
                max_oi_put_strike = 0
                max_oi_put_oi = 0
            
            return {
                'max_volume_call': {
                    'strike': float(max_volume_call_strike),
                    'volume': int(max_volume_call_volume)
                },
                'max_volume_put': {
                    'strike': float(max_volume_put_strike),
                    'volume': int(max_volume_put_volume)
                },
                'max_oi_call': {
                    'strike': float(max_oi_call_strike),
                    'open_interest': int(max_oi_call_oi)
                },
                'max_oi_put': {
                    'strike': float(max_oi_put_strike),
                    'open_interest': int(max_oi_put_oi)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume/OI: {e}")
            return {}
    
    def recommend_strategy(self, technical_signal: str, options_analysis: Dict, current_price: float, 
                          index_name: str, market_regime: str) -> OptionsStrategy:
        """Recommend options strategy based on technical analysis and market conditions"""
        try:
            # Get lot size for the index
            lot_size = self.lot_sizes.get(index_name, 50)
            
            # If no options data available, generate mock options analysis
            if not options_analysis or not options_analysis.get('expirations'):
                logger.info(f"No options data available for {index_name}, generating mock strategy")
                return self._create_mock_strategy(technical_signal, current_price, index_name, lot_size)
            
            # Analyze market conditions
            iv_percentile = self._calculate_iv_percentile(options_analysis)
            is_high_iv = iv_percentile > 70
            is_low_iv = iv_percentile < 30
            
            # Get best expiration
            best_expiry = self._get_best_expiry(options_analysis)
            
            if not best_expiry:
                return self._create_mock_strategy(technical_signal, current_price, index_name, lot_size)
            
            # Strategy selection logic
            if technical_signal == 'BUY':
                if is_high_iv:
                    return self._create_bull_call_spread(index_name, current_price, best_expiry, lot_size)
                else:
                    return self._create_long_call(index_name, current_price, best_expiry, lot_size)
            
            elif technical_signal == 'SELL':
                if is_high_iv:
                    return self._create_bear_put_spread(index_name, current_price, best_expiry, lot_size)
                else:
                    return self._create_long_put(index_name, current_price, best_expiry, lot_size)
            
            else:  # NEUTRAL
                if is_high_iv:
                    return self._create_iron_condor(index_name, current_price, best_expiry, lot_size)
                else:
                    return self._create_long_straddle(index_name, current_price, best_expiry, lot_size)
            
        except Exception as e:
            logger.error(f"Error recommending strategy: {e}")
            return self._create_mock_strategy(technical_signal, current_price, index_name, lot_size)
    
    def _calculate_iv_percentile(self, options_analysis: Dict) -> float:
        """Calculate IV percentile from options analysis"""
        try:
            # Simplified IV percentile calculation
            # In practice, this would use historical IV data
            all_ivs = []
            
            for exp_date, data in options_analysis.get('expirations', {}).items():
                atm_analysis = data.get('atm_analysis', {})
                if atm_analysis:
                    call_iv = atm_analysis.get('call_iv', 0)
                    put_iv = atm_analysis.get('put_iv', 0)
                    if call_iv > 0 and put_iv > 0:
                        all_ivs.append((call_iv + put_iv) / 2)
            
            if not all_ivs:
                return 50.0  # Default to median
            
            avg_iv = np.mean(all_ivs)
            # Simplified percentile calculation (would need historical data for accuracy)
            return min(avg_iv * 100, 100)
            
        except Exception as e:
            logger.error(f"Error calculating IV percentile: {e}")
            return 50.0
    
    def _get_best_expiry(self, options_analysis: Dict) -> Optional[str]:
        """Get the best expiration date for strategy"""
        try:
            best_expiry = None
            best_score = 0
            
            for exp_date, data in options_analysis.get('expirations', {}).items():
                days_to_expiry = data.get('days_to_expiry', 0)
                
                # Prefer 15-30 days to expiry for most strategies
                if 15 <= days_to_expiry <= 30:
                    score = 100 - abs(days_to_expiry - 22)  # 22 days is optimal
                elif 7 <= days_to_expiry < 15:
                    score = 50  # Short-term strategies
                elif 30 < days_to_expiry <= 60:
                    score = 30  # Long-term strategies
                else:
                    score = 10  # Too short or too long
                
                if score > best_score:
                    best_score = score
                    best_expiry = exp_date
            
            return best_expiry
            
        except Exception as e:
            logger.error(f"Error getting best expiry: {e}")
            return None
    
    def _create_bull_call_spread(self, index_name: str, current_price: float, expiry: str, lot_size: int) -> OptionsStrategy:
        """Create bull call spread strategy"""
        # Find appropriate strikes
        strike_spacing = 100 if index_name == 'NIFTY_50' else 200 if index_name == 'BANK_NIFTY' else 50
        
        buy_strike = current_price - (current_price % strike_spacing)
        sell_strike = buy_strike + strike_spacing
        
        # Estimate premiums (simplified)
        buy_premium = 50  # Estimated
        sell_premium = 30  # Estimated
        net_debit = buy_premium - sell_premium
        
        max_profit = (sell_strike - buy_strike) - net_debit
        max_loss = net_debit
        breakeven = buy_strike + net_debit
        
        return OptionsStrategy(
            name=f"{index_name} Bull Call Spread",
            description=f"Buy {buy_strike} call, sell {sell_strike} call",
            legs=[
                {'action': 'BUY', 'strike': buy_strike, 'option_type': 'call', 'premium': buy_premium},
                {'action': 'SELL', 'strike': sell_strike, 'option_type': 'call', 'premium': sell_premium}
            ],
            max_profit=max_profit * lot_size,
            max_loss=max_loss * lot_size,
            breakeven_points=[breakeven],
            probability_of_profit=0.65,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            margin_required=max_loss * lot_size * 1.5,
            lot_size=lot_size
        )
    
    def _create_bear_put_spread(self, index_name: str, current_price: float, expiry: str, lot_size: int) -> OptionsStrategy:
        """Create bear put spread strategy"""
        strike_spacing = 100 if index_name == 'NIFTY_50' else 200 if index_name == 'BANK_NIFTY' else 50
        
        buy_strike = current_price + (current_price % strike_spacing)
        sell_strike = buy_strike - strike_spacing
        
        buy_premium = 50  # Estimated
        sell_premium = 30  # Estimated
        net_debit = buy_premium - sell_premium
        
        max_profit = (buy_strike - sell_strike) - net_debit
        max_loss = net_debit
        breakeven = buy_strike - net_debit
        
        return OptionsStrategy(
            name=f"{index_name} Bear Put Spread",
            description=f"Buy {buy_strike} put, sell {sell_strike} put",
            legs=[
                {'action': 'BUY', 'strike': buy_strike, 'option_type': 'put', 'premium': buy_premium},
                {'action': 'SELL', 'strike': sell_strike, 'option_type': 'put', 'premium': sell_premium}
            ],
            max_profit=max_profit * lot_size,
            max_loss=max_loss * lot_size,
            breakeven_points=[breakeven],
            probability_of_profit=0.65,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            margin_required=max_loss * lot_size * 1.5,
            lot_size=lot_size
        )
    
    def _create_iron_condor(self, index_name: str, current_price: float, expiry: str, lot_size: int) -> OptionsStrategy:
        """Create iron condor strategy"""
        strike_spacing = 100 if index_name == 'NIFTY_50' else 200 if index_name == 'BANK_NIFTY' else 50
        
        # Create iron condor around current price
        put_sell_strike = current_price - strike_spacing
        put_buy_strike = put_sell_strike - strike_spacing
        call_sell_strike = current_price + strike_spacing
        call_buy_strike = call_sell_strike + strike_spacing
        
        # Estimate premiums
        put_sell_premium = 40
        put_buy_premium = 20
        call_sell_premium = 40
        call_buy_premium = 20
        
        net_credit = (put_sell_premium + call_sell_premium) - (put_buy_premium + call_buy_premium)
        max_profit = net_credit
        max_loss = (strike_spacing * 2) - net_credit
        
        return OptionsStrategy(
            name=f"{index_name} Iron Condor",
            description=f"Sell {put_sell_strike} put, buy {put_buy_strike} put, sell {call_sell_strike} call, buy {call_buy_strike} call",
            legs=[
                {'action': 'SELL', 'strike': put_sell_strike, 'option_type': 'put', 'premium': put_sell_premium},
                {'action': 'BUY', 'strike': put_buy_strike, 'option_type': 'put', 'premium': put_buy_premium},
                {'action': 'SELL', 'strike': call_sell_strike, 'option_type': 'call', 'premium': call_sell_premium},
                {'action': 'BUY', 'strike': call_buy_strike, 'option_type': 'call', 'premium': call_buy_premium}
            ],
            max_profit=max_profit * lot_size,
            max_loss=max_loss * lot_size,
            breakeven_points=[put_sell_strike - net_credit, call_sell_strike + net_credit],
            probability_of_profit=0.70,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            margin_required=max_loss * lot_size * 2,
            lot_size=lot_size
        )
    
    def _create_long_call(self, index_name: str, current_price: float, expiry: str, lot_size: int) -> OptionsStrategy:
        """Create long call strategy"""
        strike = current_price
        premium = 80  # Estimated
        
        return OptionsStrategy(
            name=f"{index_name} Long Call",
            description=f"Buy {strike} call",
            legs=[{'action': 'BUY', 'strike': strike, 'option_type': 'call', 'premium': premium}],
            max_profit=float('inf'),
            max_loss=premium * lot_size,
            breakeven_points=[strike + premium],
            probability_of_profit=0.40,
            risk_reward_ratio=3.0,
            margin_required=premium * lot_size,
            lot_size=lot_size
        )
    
    def _create_long_put(self, index_name: str, current_price: float, expiry: str, lot_size: int) -> OptionsStrategy:
        """Create long put strategy"""
        strike = current_price
        premium = 80  # Estimated
        
        return OptionsStrategy(
            name=f"{index_name} Long Put",
            description=f"Buy {strike} put",
            legs=[{'action': 'BUY', 'strike': strike, 'option_type': 'put', 'premium': premium}],
            max_profit=strike * lot_size,
            max_loss=premium * lot_size,
            breakeven_points=[strike - premium],
            probability_of_profit=0.40,
            risk_reward_ratio=3.0,
            margin_required=premium * lot_size,
            lot_size=lot_size
        )
    
    def _create_long_straddle(self, index_name: str, current_price: float, expiry: str, lot_size: int) -> OptionsStrategy:
        """Create long straddle strategy"""
        strike = current_price
        call_premium = 80
        put_premium = 80
        total_premium = call_premium + put_premium
        
        return OptionsStrategy(
            name=f"{index_name} Long Straddle",
            description=f"Buy {strike} call and put",
            legs=[
                {'action': 'BUY', 'strike': strike, 'option_type': 'call', 'premium': call_premium},
                {'action': 'BUY', 'strike': strike, 'option_type': 'put', 'premium': put_premium}
            ],
            max_profit=float('inf'),
            max_loss=total_premium * lot_size,
            breakeven_points=[strike - total_premium, strike + total_premium],
            probability_of_profit=0.30,
            risk_reward_ratio=2.0,
            margin_required=total_premium * lot_size,
            lot_size=lot_size
        )
    
    def _create_mock_strategy(self, technical_signal: str, current_price: float, index_name: str, lot_size: int) -> OptionsStrategy:
        """Create realistic mock strategy when no options data is available"""
        import random
        
        # Generate realistic strategy based on technical signal
        if technical_signal == 'BUY':
            # Bullish strategy
            strike1 = current_price * 0.98  # 2% OTM
            strike2 = current_price * 1.02  # 2% ITM
            premium1 = current_price * 0.015  # 1.5% premium
            premium2 = current_price * 0.025  # 2.5% premium
            
            max_profit = (strike2 - strike1 - premium1 + premium2) * lot_size
            max_loss = (premium1 - premium2) * lot_size
            breakeven = strike1 + premium1 - premium2
            
            return OptionsStrategy(
                name=f"{index_name} Bull Call Spread",
                description="Buy lower strike call, sell higher strike call - bullish strategy",
                legs=[
                    {"type": "call", "action": "buy", "strike": strike1, "premium": premium1},
                    {"type": "call", "action": "sell", "strike": strike2, "premium": premium2}
                ],
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                probability_of_profit=0.65,
                risk_reward_ratio=max_profit / abs(max_loss) if max_loss != 0 else 0,
                margin_required=abs(max_loss) * 1.2,
                lot_size=lot_size
            )
            
        elif technical_signal == 'SELL':
            # Bearish strategy
            strike1 = current_price * 1.02  # 2% OTM
            strike2 = current_price * 0.98  # 2% ITM
            premium1 = current_price * 0.015  # 1.5% premium
            premium2 = current_price * 0.025  # 2.5% premium
            
            max_profit = (strike1 - strike2 - premium1 + premium2) * lot_size
            max_loss = (premium1 - premium2) * lot_size
            breakeven = strike1 - premium1 + premium2
            
            return OptionsStrategy(
                name=f"{index_name} Bear Put Spread",
                description="Buy higher strike put, sell lower strike put - bearish strategy",
                legs=[
                    {"type": "put", "action": "buy", "strike": strike1, "premium": premium1},
                    {"type": "put", "action": "sell", "strike": strike2, "premium": premium2}
                ],
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                probability_of_profit=0.65,
                risk_reward_ratio=max_profit / abs(max_loss) if max_loss != 0 else 0,
                margin_required=abs(max_loss) * 1.2,
                lot_size=lot_size
            )
            
        else:  # NEUTRAL
            # Neutral strategy - Iron Condor
            atm_strike = round(current_price / 50) * 50  # Round to nearest 50
            call_strike1 = atm_strike + 100
            call_strike2 = atm_strike + 200
            put_strike1 = atm_strike - 100
            put_strike2 = atm_strike - 200
            
            call_premium1 = current_price * 0.008
            call_premium2 = current_price * 0.004
            put_premium1 = current_price * 0.008
            put_premium2 = current_price * 0.004
            
            net_credit = (call_premium2 + put_premium2 - call_premium1 - put_premium1) * lot_size
            max_loss = (100 - net_credit / lot_size) * lot_size
            
            return OptionsStrategy(
                name=f"{index_name} Iron Condor",
                description="Sell call spread + sell put spread - neutral strategy for range-bound market",
                legs=[
                    {"type": "call", "action": "sell", "strike": call_strike1, "premium": call_premium1},
                    {"type": "call", "action": "buy", "strike": call_strike2, "premium": call_premium2},
                    {"type": "put", "action": "sell", "strike": put_strike1, "premium": put_premium1},
                    {"type": "put", "action": "buy", "strike": put_strike2, "premium": put_premium2}
                ],
                max_profit=net_credit,
                max_loss=max_loss,
                breakeven_points=[call_strike1 - net_credit/lot_size, put_strike1 + net_credit/lot_size],
                probability_of_profit=0.70,
                risk_reward_ratio=net_credit / abs(max_loss) if max_loss != 0 else 0,
                margin_required=abs(max_loss) * 0.8,
                lot_size=lot_size
            )

    def _create_neutral_strategy(self, index_name: str, lot_size: int) -> OptionsStrategy:
        """Create neutral strategy when no clear signal"""
        return OptionsStrategy(
            name=f"{index_name} Neutral Strategy",
            description="No clear signal - wait for better opportunity",
            legs=[],
            max_profit=0,
            max_loss=0,
            breakeven_points=[],
            probability_of_profit=0.50,
            risk_reward_ratio=0,
            margin_required=0,
            lot_size=lot_size
        )

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create options engine
    engine = IndianOptionsStrategyEngine()
    
    print("Testing Indian Options Strategy Engine...")
    print("Indian Options Strategy Engine module loaded successfully")
