#!/usr/bin/env python3
"""
Test script for Live Predictions functionality
"""

import sys
import os
sys.path.append('src')

from data.indian_market_data import IndianMarketDataFetcher
from analysis.live_prediction_engine import LivePredictionEngine
from visualization.indian_visualization import IndianMarketVisualizer
from config.config import get_config

def test_live_predictions():
    """Test the live prediction functionality"""
    print("🔮 Testing Live Predictions Functionality")
    print("=" * 50)
    
    try:
        # Initialize components
        config = get_config()
        data_fetcher = IndianMarketDataFetcher(
            alpha_vantage_api_key=config.ALPHA_VANTAGE_API_KEY,
            quandl_api_key=config.QUANDL_API_KEY
        )
        prediction_engine = LivePredictionEngine()
        visualizer = IndianMarketVisualizer()
        
        print("✅ Components initialized successfully")
        
        # Fetch data
        print("\n📊 Fetching Nifty 50 data...")
        data = data_fetcher.fetch_index_data("NIFTY_50", "30d", "5m")
        
        if data.empty:
            print("❌ No data available")
            return
        
        print(f"✅ Data fetched: {len(data)} records")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Latest price: ₹{data['Close'].iloc[-1]:,.2f}")
        
        # Train model
        print("\n🤖 Training prediction model...")
        training_success = prediction_engine.train_models(data)
        
        if training_success:
            print("✅ Model trained successfully")
        else:
            print("⚠️ Using statistical methods")
        
        # Generate predictions
        print("\n🔮 Generating predictions...")
        prediction = prediction_engine.predict_prices(data)
        entry_signal = prediction_engine.get_entry_signal(prediction)
        
        print("✅ Predictions generated successfully")
        
        # Display results
        print("\n📈 Prediction Results:")
        print(f"   Current Price: ₹{prediction.current_price:,.2f}")
        print(f"   5m Prediction: ₹{prediction.predicted_price_5m:,.2f}")
        print(f"   10m Prediction: ₹{prediction.predicted_price_10m:,.2f}")
        print(f"   5m Confidence: {prediction.confidence_5m:.1%}")
        print(f"   10m Confidence: {prediction.confidence_10m:.1%}")
        print(f"   5m Direction: {prediction.direction_5m}")
        print(f"   10m Direction: {prediction.direction_10m}")
        print(f"   Entry Signal: {entry_signal.signal}")
        print(f"   Risk Level: {prediction.risk_level}")
        
        # Test visualization
        print("\n📊 Testing visualization...")
        
        # Test prediction chart
        prediction_data = {
            'current_price': prediction.current_price,
            'predicted_price_5m': prediction.predicted_price_5m,
            'predicted_price_10m': prediction.predicted_price_10m,
            'confidence_5m': prediction.confidence_5m,
            'confidence_10m': prediction.confidence_10m,
            'timestamp': prediction.timestamp
        }
        
        prediction_chart = visualizer.create_live_prediction_chart(prediction_data)
        print("✅ Prediction chart created successfully")
        
        # Test signal chart
        signal_data = {
            'signal': entry_signal.signal,
            'confidence': entry_signal.confidence,
            'target_price': entry_signal.target_price,
            'stop_loss': entry_signal.stop_loss,
            'current_price': prediction.current_price,
            'risk_reward_ratio': entry_signal.risk_reward_ratio
        }
        
        signal_chart = visualizer.create_entry_signal_chart(signal_data)
        print("✅ Signal chart created successfully")
        
        # Model performance
        accuracy = prediction_engine.get_prediction_accuracy()
        print(f"\n📊 Model Performance:")
        print(f"   5m Accuracy: {accuracy['accuracy_5m']:.1%}")
        print(f"   10m Accuracy: {accuracy['accuracy_10m']:.1%}")
        print(f"   Total Predictions: {accuracy['total_predictions']}")
        
        print("\n🎯 Trading Recommendation:")
        if entry_signal.signal == "BUY":
            print("   🟢 BUY - Consider entering long position")
        elif entry_signal.signal == "SELL":
            print("   🔴 SELL - Consider entering short position")
        else:
            print("   🟡 HOLD - Wait for better opportunity")
        
        print(f"\n   Target: ₹{entry_signal.target_price:,.2f}")
        print(f"   Stop Loss: ₹{entry_signal.stop_loss:,.2f}")
        print(f"   Risk-Reward: {entry_signal.risk_reward_ratio:.2f}:1")
        print(f"   Confidence: {entry_signal.confidence:.1%}")
        
        print("\n✅ Live Predictions test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in live predictions test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_live_predictions()
