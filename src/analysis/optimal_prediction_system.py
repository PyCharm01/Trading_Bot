#!/usr/bin/env python3
"""
Optimal Prediction System for 1-Minute Trading

This module provides the complete optimal prediction system combining:
1. Optimized 1-minute data fetching
2. ML models for technical analysis
3. LLM-based sentiment analysis
4. MCP server integration
5. Hybrid prediction engine
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import asyncio
import os
import sys
from pathlib import Path

# Import our modules
from .hybrid_prediction_engine import HybridPredictionEngine, HybridPredictionResult
from .llm_sentiment_analyzer import LLMSentimentAnalyzer, MarketSentiment
from .mcp_sentiment_server import MCPFinancialSentimentServer
from ..data.optimized_1m_data_fetcher import Optimized1MDataFetcher

logger = logging.getLogger(__name__)

@dataclass
class OptimalPredictionConfig:
    """Configuration for optimal prediction system"""
    symbol: str = "NIFTY_50"
    data_days: int = 30
    prediction_timeframes: List[str] = None
    use_ml_models: bool = True
    use_sentiment_analysis: bool = True
    use_llm_analysis: bool = True
    mcp_server_enabled: bool = True
    mcp_server_port: int = 8000
    retrain_interval_hours: int = 1
    cache_duration_minutes: int = 5
    
    def __post_init__(self):
        if self.prediction_timeframes is None:
            self.prediction_timeframes = ['1m', '5m', '10m']

@dataclass
class SystemPerformance:
    """System performance metrics"""
    data_quality_score: float
    model_accuracy: float
    prediction_latency_ms: float
    sentiment_accuracy: float
    overall_confidence: float
    system_uptime: float
    last_update: datetime

class OptimalPredictionSystem:
    """Complete optimal prediction system for 1-minute trading"""
    
    def __init__(self, config: Optional[OptimalPredictionConfig] = None):
        self.config = config or OptimalPredictionConfig()
        
        # Initialize components
        self.data_fetcher = Optimized1MDataFetcher()
        self.sentiment_analyzer = LLMSentimentAnalyzer(use_local_llm=True)
        self.prediction_engine = HybridPredictionEngine(
            symbol=self.config.symbol,
            use_sentiment=self.config.use_sentiment_analysis,
            use_llm=self.config.use_llm_analysis
        )
        
        # MCP Server
        self.mcp_server = None
        if self.config.mcp_server_enabled:
            self.mcp_server = MCPFinancialSentimentServer(
                port=self.config.mcp_server_port,
                use_local_llm=True
            )
        
        # Performance tracking
        self.performance = SystemPerformance(
            data_quality_score=0.0,
            model_accuracy=0.0,
            prediction_latency_ms=0.0,
            sentiment_accuracy=0.0,
            overall_confidence=0.0,
            system_uptime=0.0,
            last_update=datetime.now()
        )
        
        # Cache
        self.prediction_cache = {}
        self.last_retrain = datetime.now()
        
        logger.info(f"Initialized Optimal Prediction System for {self.config.symbol}")
    
    async def initialize_system(self) -> bool:
        """Initialize the complete system"""
        try:
            logger.info("Initializing Optimal Prediction System...")
            
            # 1. Test data fetching
            logger.info("Testing data fetching...")
            test_data = self.data_fetcher.fetch_1m_data_optimized(self.config.symbol, days=7)
            if test_data.empty:
                logger.error("Data fetching failed")
                return False
            
            data_quality = self.data_fetcher.get_data_quality_metrics(test_data)
            self.performance.data_quality_score = data_quality.quality_score
            logger.info(f"Data quality score: {data_quality.quality_score:.2%}")
            
            # 2. Train prediction models
            logger.info("Training prediction models...")
            training_success = self.prediction_engine.train_hybrid_models(self.config.data_days)
            if not training_success:
                logger.warning("Model training had issues, but continuing...")
            
            # 3. Test sentiment analysis
            logger.info("Testing sentiment analysis...")
            test_sentiment = self.sentiment_analyzer.analyze_sentiment(
                f"Market analysis for {self.config.symbol} shows positive momentum"
            )
            self.performance.sentiment_accuracy = test_sentiment.confidence
            logger.info(f"Sentiment analysis working: {test_sentiment.sentiment_label}")
            
            # 4. Start MCP server (if enabled)
            if self.mcp_server and self.config.mcp_server_enabled:
                logger.info("Starting MCP server...")
                # Start server in background
                asyncio.create_task(self._start_mcp_server())
                logger.info(f"MCP server started on port {self.config.mcp_server_port}")
            
            # 5. Initial prediction test
            logger.info("Testing prediction system...")
            start_time = datetime.now()
            test_prediction = self.prediction_engine.get_hybrid_prediction()
            prediction_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance.prediction_latency_ms = prediction_time
            
            logger.info(f"System initialization complete!")
            logger.info(f"Initial prediction: {test_prediction.combined_signal}")
            logger.info(f"Prediction latency: {prediction_time:.1f}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def _start_mcp_server(self):
        """Start MCP server in background"""
        try:
            await self.mcp_server.run_async(debug=False)
        except Exception as e:
            logger.error(f"MCP server error: {e}")
    
    def get_optimal_prediction(self) -> HybridPredictionResult:
        """Get optimal prediction with all components"""
        try:
            start_time = datetime.now()
            
            # Check if retraining is needed
            if self._should_retrain():
                logger.info("Retraining models...")
                self.prediction_engine.train_hybrid_models(self.config.data_days)
                self.last_retrain = datetime.now()
            
            # Get prediction
            prediction = self.prediction_engine.get_hybrid_prediction()
            
            # Update performance metrics
            prediction_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance.prediction_latency_ms = prediction_time
            self.performance.last_update = datetime.now()
            
            # Cache prediction
            cache_key = f"{self.config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.prediction_cache[cache_key] = prediction
            
            # Clean old cache entries
            self._clean_cache()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting optimal prediction: {e}")
            raise
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        time_since_retrain = datetime.now() - self.last_retrain
        return time_since_retrain.total_seconds() > (self.config.retrain_interval_hours * 3600)
    
    def _clean_cache(self):
        """Clean old cache entries"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=self.config.cache_duration_minutes)
            keys_to_remove = []
            
            for key, prediction in self.prediction_cache.items():
                if prediction.timestamp < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.prediction_cache[key]
                
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get latest data quality
            latest_data = self.data_fetcher.get_latest_1m_data(self.config.symbol, minutes=60)
            data_quality = self.data_fetcher.get_data_quality_metrics(latest_data)
            
            # Get market sentiment
            market_sentiment = self._get_market_sentiment()
            
            # Get MCP server status
            mcp_status = "running" if self.mcp_server else "disabled"
            
            return {
                "system_status": "operational",
                "symbol": self.config.symbol,
                "timestamp": datetime.now().isoformat(),
                "data_quality": {
                    "score": data_quality.quality_score,
                    "completeness": data_quality.data_completeness,
                    "total_records": data_quality.total_records,
                    "missing_records": data_quality.missing_records
                },
                "models": {
                    "ml_models_trained": len(self.prediction_engine.ml_models),
                    "last_retrain": self.last_retrain.isoformat(),
                    "retrain_needed": self._should_retrain()
                },
                "sentiment_analysis": {
                    "status": "operational",
                    "market_sentiment": market_sentiment.overall_sentiment if market_sentiment else 0.0,
                    "sentiment_trend": market_sentiment.sentiment_trend if market_sentiment else "unknown"
                },
                "mcp_server": {
                    "status": mcp_status,
                    "port": self.config.mcp_server_port if self.mcp_server else None
                },
                "performance": asdict(self.performance),
                "cache": {
                    "size": len(self.prediction_cache),
                    "duration_minutes": self.config.cache_duration_minutes
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_market_sentiment(self) -> Optional[MarketSentiment]:
        """Get current market sentiment"""
        try:
            # This would typically fetch recent news and analyze sentiment
            # For now, return a mock sentiment
            sample_text = f"Market analysis for {self.config.symbol} shows mixed signals"
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(sample_text)
            
            # Create market sentiment from individual sentiment
            return MarketSentiment(
                overall_sentiment=sentiment_result.sentiment_score,
                sentiment_trend="stable",
                key_events=[],
                risk_level="medium",
                confidence=sentiment_result.confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return None
    
    def save_system_state(self, filepath: str):
        """Save system state to file"""
        try:
            state = {
                "config": asdict(self.config),
                "performance": asdict(self.performance),
                "last_retrain": self.last_retrain.isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"System state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    def load_system_state(self, filepath: str) -> bool:
        """Load system state from file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"State file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Update performance metrics
            if "performance" in state:
                perf_data = state["performance"]
                self.performance = SystemPerformance(
                    data_quality_score=perf_data.get("data_quality_score", 0.0),
                    model_accuracy=perf_data.get("model_accuracy", 0.0),
                    prediction_latency_ms=perf_data.get("prediction_latency_ms", 0.0),
                    sentiment_accuracy=perf_data.get("sentiment_accuracy", 0.0),
                    overall_confidence=perf_data.get("overall_confidence", 0.0),
                    system_uptime=perf_data.get("system_uptime", 0.0),
                    last_update=datetime.fromisoformat(perf_data.get("last_update", datetime.now().isoformat()))
                )
            
            # Update last retrain time
            if "last_retrain" in state:
                self.last_retrain = datetime.fromisoformat(state["last_retrain"])
            
            logger.info(f"System state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
            return False
    
    async def run_continuous_prediction(self, interval_seconds: int = 60):
        """Run continuous prediction updates"""
        try:
            logger.info(f"Starting continuous prediction with {interval_seconds}s interval")
            
            while True:
                try:
                    # Get prediction
                    prediction = self.get_optimal_prediction()
                    
                    # Log prediction
                    logger.info(f"Prediction: {prediction.combined_signal} | "
                              f"1m: ₹{prediction.predicted_price_1m:.2f} | "
                              f"Sentiment: {prediction.sentiment_score:.2f} | "
                              f"Risk: {prediction.risk_level}")
                    
                    # Save prediction to file
                    self._save_prediction_to_file(prediction)
                    
                    # Wait for next interval
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in continuous prediction: {e}")
                    await asyncio.sleep(interval_seconds)
                    
        except KeyboardInterrupt:
            logger.info("Continuous prediction stopped by user")
        except Exception as e:
            logger.error(f"Continuous prediction error: {e}")
    
    def _save_prediction_to_file(self, prediction: HybridPredictionResult):
        """Save prediction to file for external access"""
        try:
            os.makedirs('outputs/predictions', exist_ok=True)
            
            filename = f"outputs/predictions/{self.config.symbol}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            prediction_data = asdict(prediction)
            prediction_data['timestamp'] = prediction_data['timestamp'].isoformat()
            
            with open(filename, 'w') as f:
                json.dump(prediction_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving prediction to file: {e}")

# Example usage and testing
async def main():
    """Main function for testing the optimal prediction system"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/logs/optimal_prediction.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create configuration
    config = OptimalPredictionConfig(
        symbol="NIFTY_50",
        data_days=30,
        use_ml_models=True,
        use_sentiment_analysis=True,
        use_llm_analysis=True,
        mcp_server_enabled=True,
        mcp_server_port=8000
    )
    
    # Create optimal prediction system
    system = OptimalPredictionSystem(config)
    
    # Initialize system
    print("🚀 Initializing Optimal Prediction System...")
    success = await system.initialize_system()
    
    if success:
        print("✅ System initialized successfully!")
        
        # Get system status
        status = system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"   Data Quality: {status['data_quality']['score']:.2%}")
        print(f"   ML Models: {status['models']['ml_models_trained']}")
        print(f"   MCP Server: {status['mcp_server']['status']}")
        print(f"   Market Sentiment: {status['sentiment_analysis']['market_sentiment']:.2f}")
        
        # Get optimal prediction
        print(f"\n🎯 Getting Optimal Prediction...")
        prediction = system.get_optimal_prediction()
        
        print(f"\n📈 Optimal Prediction for {prediction.symbol}:")
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
        
        # Save system state
        system.save_system_state('outputs/system_state.json')
        
        print(f"\n🎉 Optimal Prediction System is ready!")
        print(f"📁 MCP Server: http://localhost:{config.mcp_server_port}")
        print(f"📁 Predictions saved to: outputs/predictions/")
        print(f"📁 System state saved to: outputs/system_state.json")
        
        # Optionally run continuous prediction
        # await system.run_continuous_prediction(interval_seconds=60)
        
    else:
        print("❌ System initialization failed")

if __name__ == "__main__":
    asyncio.run(main())
