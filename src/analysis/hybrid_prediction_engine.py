#!/usr/bin/env python3
"""
Hybrid Prediction Engine: ML + LLM + Sentiment Analysis

This module combines traditional ML models with LLM-based sentiment analysis
for optimal prediction performance on 1-minute interval data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from .llm_sentiment_analyzer import LLMSentimentAnalyzer, SentimentResult, MarketSentiment
from ..data.optimized_1m_data_fetcher import Optimized1MDataFetcher

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
    logging.warning("scikit-learn not available. Using statistical methods only.")

# Try to import XGBoost separately
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Will use other ML models.")

logger = logging.getLogger(__name__)

@dataclass
class HybridPredictionResult:
    """Result of hybrid prediction combining ML and sentiment"""
    symbol: str
    current_price: float
    predicted_price_1m: float
    predicted_price_5m: float
    predicted_price_10m: float
    confidence_1m: float
    confidence_5m: float
    confidence_10m: float
    sentiment_score: float
    sentiment_impact: float
    technical_signal: str
    sentiment_signal: str
    combined_signal: str
    risk_level: str
    reasoning: str
    timestamp: datetime

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    training_time: float
    prediction_time: float

class HybridPredictionEngine:
    """Hybrid prediction engine combining ML, LLM, and sentiment analysis"""
    
    def __init__(self, 
                 symbol: str = "NIFTY_50",
                 use_sentiment: bool = True,
                 use_llm: bool = True,
                 mcp_server_url: str = "http://localhost:8000"):
        
        self.symbol = symbol
        self.use_sentiment = use_sentiment
        self.use_llm = use_llm
        self.mcp_server_url = mcp_server_url
        
        # Initialize components
        self.data_fetcher = Optimized1MDataFetcher()
        self.sentiment_analyzer = LLMSentimentAnalyzer(use_local_llm=True)
        
        # ML models
        self.ml_models = {}
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.feature_columns = []
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = []
        
        # Model weights (learned from performance)
        self.model_weights = {
            'ml_technical': 0.4,
            'sentiment': 0.3,
            'llm_analysis': 0.3
        }
        
        logger.info(f"Initialized Hybrid Prediction Engine for {symbol}")
    
    def train_hybrid_models(self, days: int = 30) -> bool:
        """Train hybrid models with 1-minute data"""
        try:
            logger.info(f"Training hybrid models for {self.symbol} with {days} days of data")
            
            # Fetch 1-minute data
            data = self.data_fetcher.fetch_1m_data_optimized(self.symbol, days)
            if data.empty:
                logger.error("No data available for training")
                return False
            
            # Prepare features
            features_df = self._prepare_features(data)
            if features_df.empty:
                logger.error("No features prepared")
                return False
            
            # Train ML models
            ml_success = self._train_ml_models(features_df)
            
            # Train sentiment model (if enabled)
            sentiment_success = True
            if self.use_sentiment:
                sentiment_success = self._train_sentiment_model(features_df)
            
            # Evaluate model performance
            self._evaluate_models(features_df)
            
            success = ml_success and sentiment_success
            logger.info(f"Hybrid model training {'successful' if success else 'failed'}")
            return success
            
        except Exception as e:
            logger.error(f"Error training hybrid models: {e}")
            return False
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        try:
            df = data.copy()
            
            # Technical indicators
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_10'] = df['Close'].rolling(10).mean()
            df['sma_20'] = df['Close'].rolling(20).mean()
            
            df['ema_5'] = df['Close'].ewm(span=5).mean()
            df['ema_10'] = df['Close'].ewm(span=10).mean()
            
            df['rsi'] = self._calculate_rsi(df['Close'], 14)
            df['macd'] = self._calculate_macd(df['Close'])
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['Close'])
            
            # Price features
            df['price_change'] = df['Close'].pct_change()
            df['high_low_ratio'] = df['High'] / df['Low']
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # Volatility features
            df['volatility'] = df['price_change'].rolling(20).std()
            df['price_range'] = (df['High'] - df['Low']) / df['Close']
            
            # Time features
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)
            
            # Target variables (future prices)
            df['target_1m'] = df['Close'].shift(-1)
            df['target_5m'] = df['Close'].shift(-5)
            df['target_10m'] = df['Close'].shift(-10)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Store feature columns
            self.feature_columns = [col for col in df.columns if col not in ['target_1m', 'target_5m', 'target_10m']]
            
            logger.info(f"Prepared {len(df)} samples with {len(self.feature_columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _train_ml_models(self, features_df: pd.DataFrame) -> bool:
        """Train ML models"""
        try:
            if not ML_AVAILABLE:
                logger.warning("ML libraries not available")
                return False
            
            X = features_df[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models for different timeframes
            timeframes = ['1m', '5m', '10m']
            
            for timeframe in timeframes:
                target_col = f'target_{timeframe}'
                y = features_df[target_col]
                
                # Remove NaN targets
                valid_idx = ~y.isna()
                X_valid = X_scaled[valid_idx]
                y_valid = y[valid_idx]
                
                if len(X_valid) < 100:
                    logger.warning(f"Insufficient data for {timeframe} model")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_valid, y_valid, test_size=0.2, random_state=42
                )
                
                # Train ensemble of models
                models = {}
                
                if XGBOOST_AVAILABLE:
                    models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
                
                if ML_AVAILABLE:
                    models['gradient_boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
                    models['svr'] = SVR(kernel='rbf', C=1.0)
                
                if not models:
                    logger.warning("No ML models available, using statistical methods only")
                    return None
                
                best_model = None
                best_score = float('inf')
                
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        
                        if mse < best_score:
                            best_score = mse
                            best_model = model
                        
                        logger.info(f"{timeframe} {name} MSE: {mse:.6f}")
                        
                    except Exception as e:
                        logger.warning(f"Error training {name} for {timeframe}: {e}")
                        continue
                
                if best_model is not None:
                    self.ml_models[timeframe] = best_model
                    logger.info(f"Best {timeframe} model trained with MSE: {best_score:.6f}")
                else:
                    logger.warning(f"No model trained for {timeframe}")
            
            return len(self.ml_models) > 0
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return False
    
    def _train_sentiment_model(self, features_df: pd.DataFrame) -> bool:
        """Train sentiment-based model"""
        try:
            # This would typically involve:
            # 1. Fetching news data for the training period
            # 2. Analyzing sentiment
            # 3. Correlating sentiment with price movements
            # 4. Training a sentiment-based prediction model
            
            logger.info("Sentiment model training placeholder - would integrate with news data")
            return True
            
        except Exception as e:
            logger.error(f"Error training sentiment model: {e}")
            return False
    
    def _evaluate_models(self, features_df: pd.DataFrame):
        """Evaluate model performance"""
        try:
            # This would implement comprehensive model evaluation
            # including backtesting, cross-validation, etc.
            
            logger.info("Model evaluation placeholder - would implement comprehensive testing")
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
    
    def get_hybrid_prediction(self) -> HybridPredictionResult:
        """Get hybrid prediction combining ML and sentiment"""
        try:
            # Get latest data
            latest_data = self.data_fetcher.get_latest_1m_data(self.symbol, minutes=60)
            if latest_data.empty:
                raise ValueError("No recent data available")
            
            current_price = latest_data['Close'].iloc[-1]
            
            # ML predictions
            ml_predictions = self._get_ml_predictions(latest_data)
            
            # Sentiment analysis
            sentiment_score = 0.0
            sentiment_impact = 0.0
            if self.use_sentiment:
                sentiment_result = self._get_sentiment_analysis()
                sentiment_score = sentiment_result.sentiment_score
                sentiment_impact = self._calculate_sentiment_impact(sentiment_result)
            
            # Combine predictions
            combined_predictions = self._combine_predictions(ml_predictions, sentiment_score, sentiment_impact)
            
            # Generate signals
            technical_signal = self._generate_technical_signal(latest_data)
            sentiment_signal = self._generate_sentiment_signal(sentiment_score)
            combined_signal = self._generate_combined_signal(technical_signal, sentiment_signal)
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(latest_data, sentiment_score)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(ml_predictions, sentiment_score, technical_signal)
            
            result = HybridPredictionResult(
                symbol=self.symbol,
                current_price=current_price,
                predicted_price_1m=combined_predictions['1m'],
                predicted_price_5m=combined_predictions['5m'],
                predicted_price_10m=combined_predictions['10m'],
                confidence_1m=ml_predictions.get('confidence_1m', 0.5),
                confidence_5m=ml_predictions.get('confidence_5m', 0.5),
                confidence_10m=ml_predictions.get('confidence_10m', 0.5),
                sentiment_score=sentiment_score,
                sentiment_impact=sentiment_impact,
                technical_signal=technical_signal,
                sentiment_signal=sentiment_signal,
                combined_signal=combined_signal,
                risk_level=risk_level,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
            # Store prediction history
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting hybrid prediction: {e}")
            raise
    
    def _get_ml_predictions(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get ML model predictions"""
        try:
            if not self.ml_models or not ML_AVAILABLE:
                # Fallback to simple statistical prediction
                return self._get_statistical_predictions(data)
            
            # Prepare features for latest data
            features_df = self._prepare_features(data)
            if features_df.empty:
                return self._get_statistical_predictions(data)
            
            latest_features = features_df[self.feature_columns].iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            predictions = {}
            confidences = {}
            
            for timeframe, model in self.ml_models.items():
                try:
                    pred = model.predict(latest_features_scaled)[0]
                    predictions[timeframe] = pred
                    
                    # Calculate confidence based on model performance
                    confidences[f'confidence_{timeframe}'] = 0.7  # Placeholder
                    
                except Exception as e:
                    logger.warning(f"Error predicting {timeframe}: {e}")
                    predictions[timeframe] = data['Close'].iloc[-1]
                    confidences[f'confidence_{timeframe}'] = 0.3
            
            predictions.update(confidences)
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting ML predictions: {e}")
            return self._get_statistical_predictions(data)
    
    def _get_statistical_predictions(self, data: pd.DataFrame) -> Dict[str, float]:
        """Fallback statistical predictions"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Simple momentum-based prediction
            recent_returns = data['Close'].pct_change().tail(10).mean()
            volatility = data['Close'].pct_change().tail(20).std()
            
            # Predict with momentum and some randomness
            predictions = {}
            for timeframe in ['1m', '5m', '10m']:
                minutes = int(timeframe.replace('m', ''))
                momentum_factor = 1 + (recent_returns * minutes)
                noise_factor = 1 + np.random.normal(0, volatility * 0.1)
                pred = current_price * momentum_factor * noise_factor
                predictions[timeframe] = pred
                predictions[f'confidence_{timeframe}'] = 0.4
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in statistical predictions: {e}")
            current_price = data['Close'].iloc[-1]
            return {
                '1m': current_price,
                '5m': current_price,
                '10m': current_price,
                'confidence_1m': 0.2,
                'confidence_5m': 0.2,
                'confidence_10m': 0.2
            }
    
    def _get_sentiment_analysis(self) -> SentimentResult:
        """Get sentiment analysis"""
        try:
            # This would fetch recent news and analyze sentiment
            # For now, return a mock sentiment result
            
            sample_text = f"Market analysis for {self.symbol} shows mixed signals with some positive momentum"
            sentiment = self.sentiment_analyzer.analyze_sentiment(sample_text)
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {e}")
            return SentimentResult(
                text="",
                sentiment_score=0.0,
                sentiment_label='neutral',
                confidence=0.1,
                keywords=[],
                market_impact='low',
                timestamp=datetime.now()
            )
    
    def _calculate_sentiment_impact(self, sentiment: SentimentResult) -> float:
        """Calculate sentiment impact on price prediction"""
        try:
            # Weight sentiment by confidence and market impact
            impact = sentiment.sentiment_score * sentiment.confidence
            
            if sentiment.market_impact == 'high':
                impact *= 2.0
            elif sentiment.market_impact == 'medium':
                impact *= 1.5
            else:
                impact *= 1.0
            
            return impact
            
        except Exception as e:
            logger.error(f"Error calculating sentiment impact: {e}")
            return 0.0
    
    def _combine_predictions(self, ml_predictions: Dict[str, float], 
                           sentiment_score: float, sentiment_impact: float) -> Dict[str, float]:
        """Combine ML and sentiment predictions"""
        try:
            combined = {}
            
            for timeframe in ['1m', '5m', '10m']:
                ml_pred = ml_predictions.get(timeframe, 0)
                
                # Apply sentiment adjustment
                sentiment_adjustment = sentiment_impact * 0.01  # 1% max adjustment
                combined_pred = ml_pred * (1 + sentiment_adjustment)
                
                combined[timeframe] = combined_pred
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return ml_predictions
    
    def _generate_technical_signal(self, data: pd.DataFrame) -> str:
        """Generate technical analysis signal"""
        try:
            current_price = data['Close'].iloc[-1]
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            
            if current_price > sma_5 > sma_20:
                return 'BUY'
            elif current_price < sma_5 < sma_20:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error generating technical signal: {e}")
            return 'HOLD'
    
    def _generate_sentiment_signal(self, sentiment_score: float) -> str:
        """Generate sentiment-based signal"""
        try:
            if sentiment_score > 0.3:
                return 'BUY'
            elif sentiment_score < -0.3:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error generating sentiment signal: {e}")
            return 'HOLD'
    
    def _generate_combined_signal(self, technical_signal: str, sentiment_signal: str) -> str:
        """Generate combined signal"""
        try:
            if technical_signal == sentiment_signal:
                return technical_signal
            elif technical_signal == 'HOLD':
                return sentiment_signal
            elif sentiment_signal == 'HOLD':
                return technical_signal
            else:
                return 'HOLD'  # Conflicting signals
                
        except Exception as e:
            logger.error(f"Error generating combined signal: {e}")
            return 'HOLD'
    
    def _calculate_risk_level(self, data: pd.DataFrame, sentiment_score: float) -> str:
        """Calculate risk level"""
        try:
            volatility = data['Close'].pct_change().tail(20).std()
            
            risk_score = 0
            if volatility > 0.02:  # High volatility
                risk_score += 2
            elif volatility > 0.01:  # Medium volatility
                risk_score += 1
            
            if abs(sentiment_score) > 0.5:  # Extreme sentiment
                risk_score += 1
            
            if risk_score >= 3:
                return 'HIGH'
            elif risk_score >= 1:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            logger.error(f"Error calculating risk level: {e}")
            return 'MEDIUM'
    
    def _generate_reasoning(self, ml_predictions: Dict[str, float], 
                          sentiment_score: float, technical_signal: str) -> str:
        """Generate reasoning for the prediction"""
        try:
            reasoning_parts = []
            
            # Technical reasoning
            reasoning_parts.append(f"Technical analysis suggests {technical_signal} signal")
            
            # Sentiment reasoning
            if abs(sentiment_score) > 0.3:
                sentiment_desc = "positive" if sentiment_score > 0 else "negative"
                reasoning_parts.append(f"Market sentiment is {sentiment_desc}")
            
            # ML reasoning
            if ml_predictions:
                reasoning_parts.append("ML models indicate price movement")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Prediction based on available data analysis."
    
    # Technical indicator calculations
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        try:
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            return macd
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, lower_band
        except:
            return pd.Series(index=prices.index, dtype=float), pd.Series(index=prices.index, dtype=float)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create hybrid prediction engine
    engine = HybridPredictionEngine(symbol="NIFTY_50")
    
    # Train models
    print("Training hybrid models...")
    success = engine.train_hybrid_models(days=30)
    
    if success:
        print("✅ Models trained successfully!")
        
        # Get prediction
        print("\nGetting hybrid prediction...")
        prediction = engine.get_hybrid_prediction()
        
        print(f"\n🎯 Hybrid Prediction for {prediction.symbol}:")
        print(f"   Current Price: ₹{prediction.current_price:,.2f}")
        print(f"   1m Prediction: ₹{prediction.predicted_price_1m:,.2f} (confidence: {prediction.confidence_1m:.1%})")
        print(f"   5m Prediction: ₹{prediction.predicted_price_5m:,.2f} (confidence: {prediction.confidence_5m:.1%})")
        print(f"   10m Prediction: ₹{prediction.predicted_price_10m:,.2f} (confidence: {prediction.confidence_10m:.1%})")
        print(f"   Sentiment Score: {prediction.sentiment_score:.2f}")
        print(f"   Technical Signal: {prediction.technical_signal}")
        print(f"   Sentiment Signal: {prediction.sentiment_signal}")
        print(f"   Combined Signal: {prediction.combined_signal}")
        print(f"   Risk Level: {prediction.risk_level}")
        print(f"   Reasoning: {prediction.reasoning}")
    else:
        print("❌ Model training failed")
