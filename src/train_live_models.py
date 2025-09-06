#!/usr/bin/env python3
"""
Live Model Training Script

This script trains ML models using live data from Yahoo Finance and Upstox APIs,
saves them as pickle files, and demonstrates their usage for predictions.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

from analysis.live_prediction_engine import LivePredictionEngine
from data.indian_market_data import IndianMarketDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/model_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    print("🚀 Live Model Training Script")
    print("=" * 50)
    
    # Initialize prediction engine
    print("\n1. Initializing Live Prediction Engine...")
    engine = LivePredictionEngine(symbol="NIFTY_50")
    
    # Test data fetcher
    print("\n2. Testing Data Fetcher...")
    data_fetcher = IndianMarketDataFetcher()
    
    # Test live data fetching
    print("\n3. Testing Live Data Fetching...")
    try:
        # Test Yahoo Finance
        print("   📊 Testing Yahoo Finance...")
        yahoo_data = data_fetcher._fetch_yahoo_finance_enhanced("^NSEI", "5d", "1m")
        if not yahoo_data.empty:
            print(f"   ✅ Yahoo Finance: {len(yahoo_data)} records")
            print(f"      Latest price: ₹{yahoo_data['Close'].iloc[-1]:,.2f}")
            print(f"      Data range: {yahoo_data.index[0]} to {yahoo_data.index[-1]}")
        else:
            print("   ❌ Yahoo Finance: No data")
        
        # Test Upstox (if API keys available)
        print("   📊 Testing Upstox API...")
        if (data_fetcher.upstox_api_key != "your_upstox_api_key_here" and 
            data_fetcher.upstox_access_token != "your_upstox_access_token_here"):
            upstox_data = data_fetcher._fetch_upstox_ohlcv("NIFTY_50", "5d", "1m")
            if not upstox_data.empty:
                print(f"   ✅ Upstox: {len(upstox_data)} records")
                print(f"      Latest price: ₹{upstox_data['Close'].iloc[-1]:,.2f}")
            else:
                print("   ❌ Upstox: No data")
        else:
            print("   ⚠️ Upstox: API keys not configured")
            
    except Exception as e:
        print(f"   ❌ Data fetching error: {e}")
    
    # Train models with live data
    print("\n4. Training Models with Live Data...")
    try:
        training_success = engine.train_models_with_live_data(force_retrain=True)
        
        if training_success:
            print("   ✅ Model training completed successfully!")
            
            # Show model status
            print("\n5. Model Status:")
            print(f"   📊 Models trained: {engine.is_trained}")
            print(f"   🕒 Last training: {engine.last_training_time}")
            print(f"   📈 Feature columns: {len(engine.feature_columns)}")
            
            # Test predictions
            print("\n6. Testing Predictions...")
            prediction = engine.get_live_prediction()
            
            print(f"   📊 Current Price: ₹{prediction.current_price:,.2f}")
            print(f"   🎯 1m Prediction: ₹{prediction.predicted_price_1m:,.2f} (confidence: {prediction.confidence_1m:.1%})")
            print(f"   🎯 5m Prediction: ₹{prediction.predicted_price_5m:,.2f} (confidence: {prediction.confidence_5m:.1%})")
            print(f"   🎯 10m Prediction: ₹{prediction.predicted_price_10m:,.2f} (confidence: {prediction.confidence_10m:.1%})")
            print(f"   📈 Entry Signal: {prediction.entry_signal}")
            print(f"   ⚠️ Risk Level: {prediction.risk_level}")
            
            # Test entry signal
            print("\n7. Testing Entry Signal...")
            entry_signal = engine.get_entry_signal(prediction)
            print(f"   🎯 Signal: {entry_signal.signal}")
            print(f"   🎯 Target: ₹{entry_signal.target_price:,.2f}")
            print(f"   🛑 Stop Loss: ₹{entry_signal.stop_loss:,.2f}")
            print(f"   📊 Risk-Reward: {entry_signal.risk_reward_ratio:.2f}")
            print(f"   💭 Reasoning: {entry_signal.reasoning}")
            
            # Model performance metrics
            print("\n8. Model Performance Metrics...")
            metrics = engine.get_model_performance_metrics()
            print(f"   📊 Model Status: {metrics.get('model_status', {})}")
            print(f"   📈 Data Quality: {metrics.get('data_quality', {})}")
            print(f"   🎯 Prediction History: {len(metrics.get('prediction_history', []))} predictions")
            
            print("\n✅ Training and testing completed successfully!")
            print(f"📁 Models saved in: {engine.models_dir}")
            
        else:
            print("   ❌ Model training failed!")
            return False
            
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        logger.error(f"Training error: {e}")
        return False
    
    return True

def test_model_loading():
    """Test loading saved models"""
    print("\n🔄 Testing Model Loading...")
    
    try:
        # Create new engine instance (should load saved models)
        engine2 = LivePredictionEngine(symbol="NIFTY_50")
        
        if engine2.is_trained:
            print("   ✅ Models loaded successfully from pickle files!")
            print(f"   📊 Last training: {engine2.last_training_time}")
            
            # Test prediction with loaded models
            prediction = engine2.get_live_prediction()
            print(f"   🎯 Prediction with loaded models: ₹{prediction.predicted_price_5m:,.2f}")
            
        else:
            print("   ❌ Failed to load models from pickle files")
            
    except Exception as e:
        print(f"   ❌ Model loading error: {e}")

if __name__ == "__main__":
    try:
        # Ensure outputs directory exists
        os.makedirs('outputs/logs', exist_ok=True)
        os.makedirs('outputs/models', exist_ok=True)
        
        # Run main training
        success = main()
        
        if success:
            # Test model loading
            test_model_loading()
            
            print("\n🎉 All tests completed successfully!")
            print("\n📋 Next Steps:")
            print("   1. Configure API keys in config.py for better data access")
            print("   2. Models are saved in outputs/models/ directory")
            print("   3. Use the trained models in your trading application")
            print("   4. Models will auto-retrain every hour with fresh data")
            
        else:
            print("\n❌ Training failed. Check logs for details.")
            
    except Exception as e:
        print(f"\n❌ Script error: {e}")
        logger.error(f"Script error: {e}")
