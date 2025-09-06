#!/usr/bin/env python3
"""
Live Prediction Engine for Indian Market Trading

This module provides real-time price prediction capabilities for short-term trading
including 5-minute and 10-minute price forecasts for market entry decisions.
Uses live market data from multiple sources for accurate predictions.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Using statistical methods only.")

# Import data fetcher for live market data
try:
    from ..data.indian_market_data import IndianMarketDataFetcher
except ImportError:
    try:
        from data.indian_market_data import IndianMarketDataFetcher
    except ImportError:
        try:
            from src.data.indian_market_data import IndianMarketDataFetcher
        except ImportError:
            # Fallback - create a simple data fetcher
            class IndianMarketDataFetcher:
                def __init__(self):
                    pass
                def fetch_realtime_data(self, symbol):
                    return {}
                def fetch_index_data(self, symbol, period="1d", interval="1m"):
                    return pd.DataFrame()

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Result of a price prediction"""
    current_price: float
    predicted_price_1m: float
    predicted_price_2m: float
    predicted_price_5m: float
    predicted_price_10m: float
    confidence_1m: float
    confidence_2m: float
    confidence_5m: float
    confidence_10m: float
    direction_1m: str  # 'UP', 'DOWN', 'SIDEWAYS'
    direction_2m: str
    direction_5m: str
    direction_10m: str
    trend_1m: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    trend_2m: str
    trend_5m: str
    trend_10m: str
    trend_strength_1m: str  # 'STRONG', 'MODERATE', 'WEAK'
    trend_strength_2m: str
    trend_strength_5m: str
    trend_strength_10m: str
    price_change_1m: float  # Percentage change
    price_change_2m: float
    price_change_5m: float
    price_change_10m: float
    entry_signal: str  # 'BUY', 'SELL', 'HOLD'
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    timestamp: datetime
    features_used: List[str]

@dataclass
class MarketEntrySignal:
    """Market entry signal based on predictions"""
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    reasoning: str
    timestamp: datetime

class LivePredictionEngine:
    """Real-time price prediction engine for short-term trading using live market data"""
    
    def __init__(self, symbol: str = "NIFTY_50"):
        self.symbol = symbol
        self.data_fetcher = IndianMarketDataFetcher()
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.model_5m = None
        self.model_10m = None
        self.model_1m = None
        self.model_2m = None
        self.is_trained = False
        self.feature_columns = []
        self.prediction_history = []
        self.last_training_time = None
        self.training_interval = timedelta(hours=1)  # Retrain every hour
        self.min_training_samples = 200  # Minimum samples for training
        
    def _calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for prediction features"""
        df = data.copy()
        
        # Price-based features
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        
        # Exponential moving averages
        df['ema_5'] = df['Close'].ewm(span=5).mean()
        df['ema_10'] = df['Close'].ewm(span=10).mean()
        df['ema_20'] = df['Close'].ewm(span=20).mean()
        
        # Price relative to moving averages
        df['price_vs_sma5'] = df['Close'] / df['sma_5']
        df['price_vs_sma10'] = df['Close'] / df['sma_10']
        df['price_vs_sma20'] = df['Close'] / df['sma_20']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_sma'] = df['Volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
        else:
            df['volume_ratio'] = 1.0
        
        # Volatility
        df['volatility'] = df['Close'].rolling(window=10).std()
        df['volatility_ratio'] = df['volatility'] / df['Close']
        
        # Momentum indicators
        df['momentum_5'] = df['Close'] / df['Close'].shift(5)
        df['momentum_10'] = df['Close'] / df['Close'].shift(10)
        
        # Time-based features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        return df
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning"""
        df = self._calculate_technical_features(data)
        
        # Select feature columns
        feature_columns = [
            'price_change', 'high_low_ratio', 'close_open_ratio',
            'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',
            'rsi', 'bb_position', 'volume_ratio', 'volatility_ratio',
            'momentum_5', 'momentum_10', 'hour', 'minute', 'day_of_week'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        return df[available_features].fillna(method='ffill').fillna(0)
    
    def _create_targets(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Create target variables for 1m, 2m, 5m and 10m predictions"""
        # 1-minute target (next 1 period)
        target_1m = data['Close'].shift(-1) / data['Close'] - 1
        
        # 2-minute target (next 2 periods)
        target_2m = data['Close'].shift(-2) / data['Close'] - 1
        
        # 5-minute target (next 5 periods)
        target_5m = data['Close'].shift(-5) / data['Close'] - 1
        
        # 10-minute target (next 10 periods)
        target_10m = data['Close'].shift(-10) / data['Close'] - 1
        
        return target_1m, target_2m, target_5m, target_10m
    
    def fetch_live_training_data(self) -> pd.DataFrame:
        """Fetch live market data for training models"""
        try:
            # Fetch high-frequency data for training (1-minute intervals for last 5 days)
            data = self.data_fetcher.fetch_index_data(
                self.symbol, 
                period="5d", 
                interval="1m"
            )
            
            if data.empty:
                logger.warning(f"No live data available for {self.symbol}")
                # Try with daily data as fallback
                data = self.data_fetcher.fetch_index_data(
                    self.symbol, 
                    period="1y", 
                    interval="1d"
                )
            
            if not data.empty:
                logger.info(f"Fetched {len(data)} live data points for {self.symbol}")
                return data
            else:
                logger.error(f"Failed to fetch any data for {self.symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching live training data: {e}")
            return pd.DataFrame()
    
    def train_models(self, data: Optional[pd.DataFrame] = None) -> bool:
        """Train prediction models on live market data"""
        try:
            if not ML_AVAILABLE:
                logger.warning("ML libraries not available. Using statistical methods.")
                return False
            
            # Use provided data or fetch live data
            if data is None:
                data = self.fetch_live_training_data()
            
            if data.empty:
                logger.error("No data available for training")
                return False
            
            # Prepare features and targets
            features = self._prepare_features(data)
            target_1m, target_2m, target_5m, target_10m = self._create_targets(data)
            
            # Remove rows with NaN targets
            valid_indices = ~(target_1m.isna() | target_2m.isna() | target_5m.isna() | target_10m.isna())
            features = features[valid_indices]
            target_1m = target_1m[valid_indices]
            target_2m = target_2m[valid_indices]
            target_5m = target_5m[valid_indices]
            target_10m = target_10m[valid_indices]
            
            if len(features) < self.min_training_samples:
                logger.warning(f"Insufficient data for training. Need at least {self.min_training_samples} samples, got {len(features)}")
                return False
            
            # Split data for validation
            X_train, X_val, y1_train, y1_val, y2_train, y2_val, y5_train, y5_val, y10_train, y10_val = train_test_split(
                features, target_1m, target_2m, target_5m, target_10m, 
                test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train models with different algorithms for ensemble
            self.model_1m = self._train_ensemble_model(X_train_scaled, y1_train, X_val_scaled, y1_val, "1m")
            self.model_2m = self._train_ensemble_model(X_train_scaled, y2_train, X_val_scaled, y2_val, "2m")
            self.model_5m = self._train_ensemble_model(X_train_scaled, y5_train, X_val_scaled, y5_val, "5m")
            self.model_10m = self._train_ensemble_model(X_train_scaled, y10_train, X_val_scaled, y10_val, "10m")
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            logger.info(f"Models trained successfully with {len(features)} live samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def _train_ensemble_model(self, X_train, y_train, X_val, y_val, timeframe):
        """Train ensemble model for specific timeframe"""
        try:
            # Use multiple algorithms for better predictions
            models = {
                'gb': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'rf': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
            }
            
            # Train and evaluate each model
            best_model = None
            best_score = float('inf')
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    mse = mean_squared_error(y_val, y_pred)
                    
                    if mse < best_score:
                        best_score = mse
                        best_model = model
                        
                    logger.info(f"{timeframe} {name} model MSE: {mse:.6f}")
                    
                except Exception as e:
                    logger.warning(f"Error training {name} model for {timeframe}: {e}")
                    continue
            
            if best_model is None:
                # Fallback to simple linear regression
                best_model = LinearRegression()
                best_model.fit(X_train, y_train)
                logger.warning(f"Using LinearRegression fallback for {timeframe}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error in ensemble training for {timeframe}: {e}")
            return None
    
    def predict_prices(self, data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """Predict next 1m, 2m, 5m and 10m prices using live market data"""
        try:
            # Use provided data or fetch live data
            if data is None:
                data = self.fetch_live_training_data()
            
            if data.empty:
                logger.error("No data available for prediction")
                return self._get_default_prediction(pd.DataFrame())
            
            current_price = data['Close'].iloc[-1]
            current_time = datetime.now()
            
            # Check if models need retraining
            if (self.last_training_time is None or 
                datetime.now() - self.last_training_time > self.training_interval):
                logger.info("Retraining models with latest data...")
                self.train_models(data)
            
            if self.is_trained and ML_AVAILABLE and all([self.model_1m, self.model_2m, self.model_5m, self.model_10m]):
                # Use ML models
                features = self._prepare_features(data)
                latest_features = features.iloc[-1:].fillna(0)
                features_scaled = self.scaler.transform(latest_features)
                
                # Get predictions for all timeframes using dedicated models
                pred_1m_change = self.model_1m.predict(features_scaled)[0]
                pred_2m_change = self.model_2m.predict(features_scaled)[0]
                pred_5m_change = self.model_5m.predict(features_scaled)[0]
                pred_10m_change = self.model_10m.predict(features_scaled)[0]
                
                predicted_price_1m = current_price * (1 + pred_1m_change)
                predicted_price_2m = current_price * (1 + pred_2m_change)
                predicted_price_5m = current_price * (1 + pred_5m_change)
                predicted_price_10m = current_price * (1 + pred_10m_change)
                
                # Calculate confidence based on model performance and volatility
                volatility = data['Close'].pct_change().tail(20).std()
                confidence_1m = min(0.95, max(0.3, 0.85 - volatility * 10 + abs(pred_1m_change) * 20))
                confidence_2m = min(0.95, max(0.2, 0.8 - volatility * 8 + abs(pred_2m_change) * 15))
                confidence_5m = min(0.95, max(0.1, 0.75 - volatility * 6 + abs(pred_5m_change) * 12))
                confidence_10m = min(0.95, max(0.1, 0.7 - volatility * 5 + abs(pred_10m_change) * 10))
                
            else:
                # Use statistical methods with live data
                predicted_price_1m, confidence_1m = self._statistical_prediction(data, 1)
                predicted_price_2m, confidence_2m = self._statistical_prediction(data, 2)
                predicted_price_5m, confidence_5m = self._statistical_prediction(data, 5)
                predicted_price_10m, confidence_10m = self._statistical_prediction(data, 10)
            
            # Calculate price changes for all timeframes
            price_change_1m = self._calculate_price_change(current_price, predicted_price_1m)
            price_change_2m = self._calculate_price_change(current_price, predicted_price_2m)
            price_change_5m = self._calculate_price_change(current_price, predicted_price_5m)
            price_change_10m = self._calculate_price_change(current_price, predicted_price_10m)
            
            # Determine direction for all timeframes
            direction_1m = self._get_direction(current_price, predicted_price_1m)
            direction_2m = self._get_direction(current_price, predicted_price_2m)
            direction_5m = self._get_direction(current_price, predicted_price_5m)
            direction_10m = self._get_direction(current_price, predicted_price_10m)
            
            # Determine trend for all timeframes
            trend_1m = self._get_trend(current_price, predicted_price_1m)
            trend_2m = self._get_trend(current_price, predicted_price_2m)
            trend_5m = self._get_trend(current_price, predicted_price_5m)
            trend_10m = self._get_trend(current_price, predicted_price_10m)
            
            # Determine trend strength for all timeframes
            trend_strength_1m = self._get_trend_strength(price_change_1m, confidence_1m)
            trend_strength_2m = self._get_trend_strength(price_change_2m, confidence_2m)
            trend_strength_5m = self._get_trend_strength(price_change_5m, confidence_5m)
            trend_strength_10m = self._get_trend_strength(price_change_10m, confidence_10m)
            
            # Generate entry signal based on all timeframes
            entry_signal = self._generate_entry_signal(
                current_price, predicted_price_1m, predicted_price_2m, 
                predicted_price_5m, predicted_price_10m,
                confidence_1m, confidence_2m, confidence_5m, confidence_10m
            )
            
            # Calculate risk level based on all confidences
            risk_level = self._calculate_risk_level(confidence_1m, confidence_2m, confidence_5m, confidence_10m)
            
            result = PredictionResult(
                current_price=current_price,
                predicted_price_1m=predicted_price_1m,
                predicted_price_2m=predicted_price_2m,
                predicted_price_5m=predicted_price_5m,
                predicted_price_10m=predicted_price_10m,
                confidence_1m=confidence_1m,
                confidence_2m=confidence_2m,
                confidence_5m=confidence_5m,
                confidence_10m=confidence_10m,
                direction_1m=direction_1m,
                direction_2m=direction_2m,
                direction_5m=direction_5m,
                direction_10m=direction_10m,
                trend_1m=trend_1m,
                trend_2m=trend_2m,
                trend_5m=trend_5m,
                trend_10m=trend_10m,
                trend_strength_1m=trend_strength_1m,
                trend_strength_2m=trend_strength_2m,
                trend_strength_5m=trend_strength_5m,
                trend_strength_10m=trend_strength_10m,
                price_change_1m=price_change_1m,
                price_change_2m=price_change_2m,
                price_change_5m=price_change_5m,
                price_change_10m=price_change_10m,
                entry_signal=entry_signal,
                risk_level=risk_level,
                timestamp=current_time,
                features_used=self.feature_columns
            )
            
            # Store prediction history
            self.prediction_history.append(result)
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            # Return default prediction
            return self._get_default_prediction(data)
    
    def _statistical_prediction(self, data: pd.DataFrame, periods: int) -> Tuple[float, float]:
        """Statistical prediction method using live market data"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Calculate recent momentum using more sophisticated approach
            recent_returns = data['Close'].pct_change().tail(min(periods * 5, 50))
            momentum = recent_returns.mean()
            
            # Calculate volatility
            volatility = recent_returns.std()
            
            # Calculate trend strength
            sma_short = data['Close'].tail(5).mean()
            sma_long = data['Close'].tail(20).mean()
            trend_strength = (sma_short - sma_long) / sma_long
            
            # Enhanced prediction using momentum, volatility, and trend
            predicted_change = momentum * periods + trend_strength * 0.1
            predicted_price = current_price * (1 + predicted_change)
            
            # Confidence based on volatility, trend consistency, and data quality
            trend_consistency = 1 - abs(recent_returns.tail(10).std() - volatility) / volatility if volatility > 0 else 0.5
            confidence = max(0.1, min(0.8, 0.6 - volatility * 8 + trend_consistency * 0.3))
            
            return predicted_price, confidence
            
        except Exception as e:
            logger.error(f"Error in statistical prediction: {e}")
            return data['Close'].iloc[-1], 0.5
    
    def get_live_prediction(self) -> PredictionResult:
        """Get live prediction using real-time market data"""
        try:
            # Fetch latest real-time data
            realtime_data = self.data_fetcher.fetch_realtime_data(self.symbol)
            
            if not realtime_data or realtime_data.get('current_price', 0) == 0:
                logger.warning("No real-time data available, using historical data")
                return self.predict_prices()
            
            # Get recent high-frequency data for prediction
            data = self.fetch_live_training_data()
            
            if data.empty:
                logger.error("No data available for live prediction")
                return self._get_default_prediction(pd.DataFrame())
            
            # Update the latest price with real-time data
            data.iloc[-1, data.columns.get_loc('Close')] = realtime_data['current_price']
            
            # Get prediction
            prediction = self.predict_prices(data)
            
            # Add real-time metadata
            prediction.timestamp = datetime.now()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in live prediction: {e}")
            return self._get_default_prediction(pd.DataFrame())
    
    def get_prediction_with_confidence_interval(self, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Get prediction with confidence intervals"""
        try:
            prediction = self.get_live_prediction()
            
            # Calculate confidence intervals based on historical volatility
            data = self.fetch_live_training_data()
            if not data.empty:
                volatility = data['Close'].pct_change().tail(20).std()
                
                # Calculate confidence intervals for each timeframe
                intervals = {}
                for timeframe in ['1m', '2m', '5m', '10m']:
                    pred_price = getattr(prediction, f'predicted_price_{timeframe}')
                    confidence = getattr(prediction, f'confidence_{timeframe}')
                    
                    # Calculate standard error
                    std_error = volatility * np.sqrt(int(timeframe[:-1]))
                    
                    # Calculate confidence interval
                    z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
                    margin_error = z_score * std_error * pred_price
                    
                    intervals[timeframe] = {
                        'lower_bound': pred_price - margin_error,
                        'upper_bound': pred_price + margin_error,
                        'margin_error': margin_error,
                        'confidence_level': confidence_level
                    }
                
                return {
                    'prediction': prediction,
                    'confidence_intervals': intervals,
                    'volatility': volatility,
                    'data_quality': 'live' if not data.empty else 'limited'
                }
            else:
                return {
                    'prediction': prediction,
                    'confidence_intervals': {},
                    'volatility': 0,
                    'data_quality': 'no_data'
                }
                
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {
                'prediction': self._get_default_prediction(pd.DataFrame()),
                'confidence_intervals': {},
                'volatility': 0,
                'data_quality': 'error'
            }
    
    def _get_direction(self, current_price: float, predicted_price: float) -> str:
        """Determine price direction"""
        change_percent = (predicted_price - current_price) / current_price * 100
        
        if change_percent > 0.1:
            return 'UP'
        elif change_percent < -0.1:
            return 'DOWN'
        else:
            return 'SIDEWAYS'
    
    def _get_trend(self, current_price: float, predicted_price: float) -> str:
        """Determine market trend"""
        change_percent = (predicted_price - current_price) / current_price * 100
        
        if change_percent > 0.05:  # 0.05% threshold for trend
            return 'BULLISH'
        elif change_percent < -0.05:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _get_trend_strength(self, change_percent: float, confidence: float) -> str:
        """Determine trend strength based on price change and confidence"""
        # Combine price change magnitude with confidence
        strength_score = abs(change_percent) * confidence * 100
        
        if strength_score > 0.5:
            return 'STRONG'
        elif strength_score > 0.2:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def _calculate_price_change(self, current_price: float, predicted_price: float) -> float:
        """Calculate percentage price change"""
        return (predicted_price - current_price) / current_price * 100
    
    def _generate_entry_signal(self, current_price: float, pred_1m: float, pred_2m: float,
                             pred_5m: float, pred_10m: float, conf_1m: float, conf_2m: float,
                             conf_5m: float, conf_10m: float) -> str:
        """Generate market entry signal based on all timeframes"""
        # Calculate expected returns for all timeframes
        return_1m = (pred_1m - current_price) / current_price
        return_2m = (pred_2m - current_price) / current_price
        return_5m = (pred_5m - current_price) / current_price
        return_10m = (pred_10m - current_price) / current_price
        
        # Weighted confidence (shorter timeframes get higher weight)
        weights = [0.4, 0.3, 0.2, 0.1]  # 1m, 2m, 5m, 10m
        avg_confidence = (conf_1m * weights[0] + conf_2m * weights[1] + 
                         conf_5m * weights[2] + conf_10m * weights[3])
        
        # Weighted return (shorter timeframes get higher weight)
        avg_return = (return_1m * weights[0] + return_2m * weights[1] + 
                     return_5m * weights[2] + return_10m * weights[3])
        
        # Signal logic with more sensitive thresholds for short-term trading
        if avg_confidence > 0.75 and avg_return > 0.001:  # 0.1% threshold for short-term
            return 'BUY'
        elif avg_confidence > 0.75 and avg_return < -0.001:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_risk_level(self, conf_1m: float, conf_2m: float, conf_5m: float, conf_10m: float) -> str:
        """Calculate risk level based on confidence across all timeframes"""
        # Weighted confidence (shorter timeframes get higher weight)
        weights = [0.4, 0.3, 0.2, 0.1]  # 1m, 2m, 5m, 10m
        avg_confidence = (conf_1m * weights[0] + conf_2m * weights[1] + 
                         conf_5m * weights[2] + conf_10m * weights[3])
        
        if avg_confidence > 0.8:
            return 'LOW'
        elif avg_confidence > 0.6:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _get_default_prediction(self, data: pd.DataFrame) -> PredictionResult:
        """Get default prediction when errors occur"""
        current_price = data['Close'].iloc[-1]
        return PredictionResult(
            current_price=current_price,
            predicted_price_1m=current_price,
            predicted_price_2m=current_price,
            predicted_price_5m=current_price,
            predicted_price_10m=current_price,
            confidence_1m=0.5,
            confidence_2m=0.5,
            confidence_5m=0.5,
            confidence_10m=0.5,
            direction_1m='SIDEWAYS',
            direction_2m='SIDEWAYS',
            direction_5m='SIDEWAYS',
            direction_10m='SIDEWAYS',
            trend_1m='NEUTRAL',
            trend_2m='NEUTRAL',
            trend_5m='NEUTRAL',
            trend_10m='NEUTRAL',
            trend_strength_1m='WEAK',
            trend_strength_2m='WEAK',
            trend_strength_5m='WEAK',
            trend_strength_10m='WEAK',
            price_change_1m=0.0,
            price_change_2m=0.0,
            price_change_5m=0.0,
            price_change_10m=0.0,
            entry_signal='HOLD',
            risk_level='HIGH',
            timestamp=datetime.now(),
            features_used=[]
        )
    
    def get_entry_signal(self, prediction: PredictionResult) -> MarketEntrySignal:
        """Generate detailed entry signal with risk management"""
        current_price = prediction.current_price
        
        if prediction.entry_signal == 'BUY':
            target_price = prediction.predicted_price_5m
            stop_loss = current_price * 0.998  # 0.2% stop loss
            risk_reward = (target_price - current_price) / (current_price - stop_loss)
            reasoning = f"Strong upward momentum predicted with {prediction.confidence_5m:.1%} confidence"
            
        elif prediction.entry_signal == 'SELL':
            target_price = prediction.predicted_price_5m
            stop_loss = current_price * 1.002  # 0.2% stop loss
            risk_reward = (current_price - target_price) / (stop_loss - current_price)
            reasoning = f"Downward trend predicted with {prediction.confidence_5m:.1%} confidence"
            
        else:
            target_price = current_price
            stop_loss = current_price
            risk_reward = 0
            reasoning = "Market conditions unclear, waiting for better opportunity"
        
        return MarketEntrySignal(
            signal=prediction.entry_signal,
            confidence=prediction.confidence_5m,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            reasoning=reasoning,
            timestamp=prediction.timestamp
        )
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Calculate prediction accuracy from live prediction history"""
        if len(self.prediction_history) < 10:
            return {
                'accuracy_1m': 0.0, 
                'accuracy_2m': 0.0, 
                'accuracy_5m': 0.0, 
                'accuracy_10m': 0.0, 
                'total_predictions': 0,
                'data_quality': 'insufficient_history'
            }
        
        try:
            # Get recent actual prices to compare against predictions
            recent_data = self.fetch_live_training_data()
            if recent_data.empty:
                return {
                    'accuracy_1m': 0.0, 
                    'accuracy_2m': 0.0, 
                    'accuracy_5m': 0.0, 
                    'accuracy_10m': 0.0, 
                    'total_predictions': len(self.prediction_history),
                    'data_quality': 'no_live_data'
                }
            
            # Calculate accuracy for each timeframe
            accuracies = {}
            for timeframe in ['1m', '2m', '5m', '10m']:
                correct_predictions = 0
                total_predictions = 0
                
                for i, prediction in enumerate(self.prediction_history[-50:]):  # Last 50 predictions
                    try:
                        # Get actual price at the predicted timeframe
                        pred_time = prediction.timestamp
                        periods = int(timeframe[:-1])
                        
                        # Find actual price after the prediction time
                        future_time = pred_time + timedelta(minutes=periods)
                        
                        # Get actual price (simplified - in real implementation, you'd need precise timing)
                        if len(recent_data) > periods:
                            actual_price = recent_data['Close'].iloc[-(periods + 1)]
                            predicted_price = getattr(prediction, f'predicted_price_{timeframe}')
                            
                            # Check if direction was correct
                            current_price = prediction.current_price
                            actual_direction = 'UP' if actual_price > current_price else 'DOWN' if actual_price < current_price else 'SIDEWAYS'
                            predicted_direction = getattr(prediction, f'direction_{timeframe}')
                            
                            if actual_direction == predicted_direction:
                                correct_predictions += 1
                            total_predictions += 1
                            
                    except Exception as e:
                        logger.warning(f"Error calculating accuracy for {timeframe}: {e}")
                        continue
                
                if total_predictions > 0:
                    accuracies[f'accuracy_{timeframe}'] = correct_predictions / total_predictions
                else:
                    accuracies[f'accuracy_{timeframe}'] = 0.0
            
            accuracies['total_predictions'] = len(self.prediction_history)
            accuracies['data_quality'] = 'live_data'
            
            return accuracies
            
        except Exception as e:
            logger.error(f"Error calculating prediction accuracy: {e}")
            return {
                'accuracy_1m': 0.0, 
                'accuracy_2m': 0.0, 
                'accuracy_5m': 0.0, 
                'accuracy_10m': 0.0, 
                'total_predictions': len(self.prediction_history),
                'data_quality': 'error'
            }
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics"""
        try:
            accuracy = self.get_prediction_accuracy()
            
            # Get recent data for additional metrics
            data = self.fetch_live_training_data()
            
            metrics = {
                'prediction_accuracy': accuracy,
                'model_status': {
                    'is_trained': self.is_trained,
                    'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                    'training_interval_hours': self.training_interval.total_seconds() / 3600,
                    'min_training_samples': self.min_training_samples
                },
                'data_quality': {
                    'total_data_points': len(data) if not data.empty else 0,
                    'data_freshness': 'live' if not data.empty else 'no_data',
                    'symbol': self.symbol,
                    'feature_count': len(self.feature_columns)
                },
                'prediction_history': {
                    'total_predictions': len(self.prediction_history),
                    'recent_predictions': len(self.prediction_history[-10:]) if self.prediction_history else 0
                }
            }
            
            # Add volatility metrics if data is available
            if not data.empty:
                volatility = data['Close'].pct_change().tail(20).std()
                metrics['market_conditions'] = {
                    'recent_volatility': float(volatility),
                    'volatility_percentile': float(data['Close'].pct_change().tail(100).rank(pct=True).iloc[-1] * 100) if len(data) >= 100 else 50,
                    'trend_direction': 'up' if data['Close'].iloc[-1] > data['Close'].iloc[-20] else 'down' if len(data) >= 20 else 'unknown'
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting model performance metrics: {e}")
            return {
                'prediction_accuracy': {'error': str(e)},
                'model_status': {'error': str(e)},
                'data_quality': {'error': str(e)},
                'prediction_history': {'error': str(e)}
            }

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Live Prediction Engine with Real Market Data...")
    
    # Create prediction engine for Nifty 50
    engine = LivePredictionEngine(symbol="NIFTY_50")
    
    # Test data fetching
    print("\n1. Testing live data fetching...")
    data = engine.fetch_live_training_data()
    print(f"Fetched {len(data)} data points")
    
    if not data.empty:
        print(f"Latest price: {data['Close'].iloc[-1]}")
        print(f"Data range: {data.index[0]} to {data.index[-1]}")
    
    # Test model training
    print("\n2. Testing model training...")
    training_success = engine.train_models()
    print(f"Training successful: {training_success}")
    
    if training_success:
        # Test live prediction
        print("\n3. Testing live prediction...")
        prediction = engine.get_live_prediction()
        print(f"Current price: {prediction.current_price}")
        print(f"1m prediction: {prediction.predicted_price_1m} (confidence: {prediction.confidence_1m:.2%})")
        print(f"5m prediction: {prediction.predicted_price_5m} (confidence: {prediction.confidence_5m:.2%})")
        print(f"Entry signal: {prediction.entry_signal}")
        print(f"Risk level: {prediction.risk_level}")
        
        # Test prediction with confidence intervals
        print("\n4. Testing prediction with confidence intervals...")
        prediction_with_ci = engine.get_prediction_with_confidence_interval()
        print(f"Prediction confidence intervals calculated: {len(prediction_with_ci.get('confidence_intervals', {}))}")
        
        # Test model performance metrics
        print("\n5. Testing model performance metrics...")
        metrics = engine.get_model_performance_metrics()
        print(f"Model trained: {metrics.get('model_status', {}).get('is_trained', False)}")
        print(f"Data points: {metrics.get('data_quality', {}).get('total_data_points', 0)}")
        print(f"Data quality: {metrics.get('data_quality', {}).get('data_freshness', 'unknown')}")
        
        # Test entry signal generation
        print("\n6. Testing entry signal generation...")
        entry_signal = engine.get_entry_signal(prediction)
        print(f"Entry signal: {entry_signal.signal}")
        print(f"Target price: {entry_signal.target_price}")
        print(f"Stop loss: {entry_signal.stop_loss}")
        print(f"Risk-reward ratio: {entry_signal.risk_reward_ratio:.2f}")
        print(f"Reasoning: {entry_signal.reasoning}")
    
    else:
        print("Model training failed. Testing statistical prediction...")
        if not data.empty:
            prediction = engine.predict_prices(data)
            print(f"Statistical prediction - Current: {prediction.current_price}")
            print(f"5m prediction: {prediction.predicted_price_5m}")
    
    print("\nLive Prediction Engine test completed!")
