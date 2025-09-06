#!/usr/bin/env python3
"""
Setup Script for Optimal Prediction System

This script sets up the complete optimal prediction system including:
1. Optimized 1-minute data fetching
2. ML model training
3. LLM sentiment analysis
4. MCP server setup
5. Hybrid prediction engine
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.analysis.optimal_prediction_system import OptimalPredictionSystem, OptimalPredictionConfig
from src.data.optimized_1m_data_fetcher import Optimized1MDataFetcher
from src.analysis.llm_sentiment_analyzer import LLMSentimentAnalyzer
from src.analysis.mcp_sentiment_server import MCPFinancialSentimentServer

def setup_directories():
    """Create necessary directories"""
    directories = [
        'outputs',
        'outputs/data',
        'outputs/data/1m',
        'outputs/models',
        'outputs/predictions',
        'outputs/logs',
        'outputs/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_data_fetching():
    """Test optimized 1-minute data fetching"""
    print("\n🔍 Testing Optimized 1-Minute Data Fetching...")
    
    try:
        fetcher = Optimized1MDataFetcher()
        
        # Test fetching 1-minute data for 7 days
        data = fetcher.fetch_1m_data_optimized('NIFTY_50', days=7)
        
        if not data.empty:
            quality = fetcher.get_data_quality_metrics(data)
            print(f"✅ Data fetching successful!")
            print(f"   Records: {len(data)}")
            print(f"   Quality Score: {quality.quality_score:.2%}")
            print(f"   Completeness: {quality.data_completeness:.2%}")
            print(f"   Date Range: {data.index[0]} to {data.index[-1]}")
            return True
        else:
            print("❌ No data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Data fetching error: {e}")
        return False

def test_sentiment_analysis():
    """Test LLM sentiment analysis"""
    print("\n🔍 Testing LLM Sentiment Analysis...")
    
    try:
        analyzer = LLMSentimentAnalyzer(use_local_llm=True)
        
        # Test sentiment analysis
        test_text = "Nifty 50 shows strong bullish momentum with positive earnings reports"
        sentiment = analyzer.analyze_sentiment(test_text)
        
        print(f"✅ Sentiment analysis working!")
        print(f"   Text: {test_text}")
        print(f"   Sentiment: {sentiment.sentiment_label} (score: {sentiment.sentiment_score:.2f})")
        print(f"   Confidence: {sentiment.confidence:.2f}")
        print(f"   Market Impact: {sentiment.market_impact}")
        return True
        
    except Exception as e:
        print(f"❌ Sentiment analysis error: {e}")
        return False

def test_mcp_server():
    """Test MCP server setup"""
    print("\n🔍 Testing MCP Server Setup...")
    
    try:
        server = MCPFinancialSentimentServer(
            port=8000,
            use_local_llm=True
        )
        
        print(f"✅ MCP server configured!")
        print(f"   Port: 8000")
        print(f"   Local LLM: Enabled")
        print(f"   Status: Ready to start")
        return True
        
    except Exception as e:
        print(f"❌ MCP server setup error: {e}")
        return False

async def test_optimal_system():
    """Test the complete optimal prediction system"""
    print("\n🔍 Testing Optimal Prediction System...")
    
    try:
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
        
        # Create system
        system = OptimalPredictionSystem(config)
        
        # Initialize system
        success = await system.initialize_system()
        
        if success:
            print("✅ Optimal prediction system working!")
            
            # Get system status
            status = system.get_system_status()
            print(f"   Data Quality: {status['data_quality']['score']:.2%}")
            print(f"   ML Models: {status['models']['ml_models_trained']}")
            print(f"   MCP Server: {status['mcp_server']['status']}")
            
            # Get prediction
            prediction = system.get_optimal_prediction()
            print(f"   Prediction Signal: {prediction.combined_signal}")
            print(f"   Risk Level: {prediction.risk_level}")
            
            return True
        else:
            print("❌ System initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Optimal system error: {e}")
        return False

def create_config_file():
    """Create configuration file"""
    print("\n📝 Creating Configuration File...")
    
    config_content = '''# Optimal Prediction System Configuration

# Data Configuration
SYMBOL = "NIFTY_50"
DATA_DAYS = 30
PREDICTION_TIMEFRAMES = ["1m", "5m", "10m"]

# Model Configuration
USE_ML_MODELS = True
USE_SENTIMENT_ANALYSIS = True
USE_LLM_ANALYSIS = True

# MCP Server Configuration
MCP_SERVER_ENABLED = True
MCP_SERVER_PORT = 8000
MCP_SERVER_HOST = "localhost"

# Training Configuration
RETRAIN_INTERVAL_HOURS = 1
CACHE_DURATION_MINUTES = 5

# API Keys (Optional - for enhanced features)
OPENAI_API_KEY = "your_openai_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"

# Local LLM Configuration
USE_LOCAL_LLM = True
LOCAL_LLM_URL = "http://localhost:11434"
LOCAL_LLM_MODEL = "llama2"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "outputs/logs/optimal_prediction.log"
'''
    
    with open('optimal_prediction_config.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Configuration file created: optimal_prediction_config.py")

def create_startup_script():
    """Create startup script"""
    print("\n📝 Creating Startup Script...")
    
    startup_content = '''#!/usr/bin/env python3
"""
Startup Script for Optimal Prediction System
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.analysis.optimal_prediction_system import OptimalPredictionSystem, OptimalPredictionConfig

async def main():
    """Start the optimal prediction system"""
    print("🚀 Starting Optimal Prediction System...")
    
    # Load configuration
    config = OptimalPredictionConfig(
        symbol="NIFTY_50",
        data_days=30,
        use_ml_models=True,
        use_sentiment_analysis=True,
        use_llm_analysis=True,
        mcp_server_enabled=True,
        mcp_server_port=8000
    )
    
    # Create and initialize system
    system = OptimalPredictionSystem(config)
    success = await system.initialize_system()
    
    if success:
        print("✅ System started successfully!")
        print("📊 MCP Server: http://localhost:8000")
        print("📁 Predictions: outputs/predictions/")
        print("📁 Logs: outputs/logs/")
        
        # Run continuous prediction
        await system.run_continuous_prediction(interval_seconds=60)
    else:
        print("❌ System startup failed")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open('start_optimal_prediction.py', 'w') as f:
        f.write(startup_content)
    
    # Make it executable
    os.chmod('start_optimal_prediction.py', 0o755)
    
    print("✅ Startup script created: start_optimal_prediction.py")

def create_requirements_file():
    """Create requirements file"""
    print("\n📝 Creating Requirements File...")
    
    requirements_content = '''# Optimal Prediction System Requirements

# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.6.0

# Data fetching
yfinance>=0.2.0
requests>=2.28.0

# LLM and sentiment analysis
openai>=1.0.0
anthropic>=0.7.0
aiohttp>=3.8.0

# MCP Server
fastapi>=0.100.0
uvicorn>=0.20.0
pydantic>=2.0.0

# Technical analysis
ta-lib>=0.4.0

# Async support
asyncio-mqtt>=0.11.0

# Logging and monitoring
python-json-logger>=2.0.0

# Optional: Local LLM support
# ollama>=0.1.0  # Uncomment if using local LLM
'''
    
    with open('requirements_optimal.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✅ Requirements file created: requirements_optimal.txt")

def create_documentation():
    """Create documentation"""
    print("\n📝 Creating Documentation...")
    
    doc_content = '''# Optimal Prediction System Documentation

## Overview

The Optimal Prediction System combines multiple approaches for accurate 1-minute trading predictions:

1. **Optimized 1-Minute Data Fetching**: Efficient data retrieval with quality metrics
2. **ML Models**: XGBoost, Random Forest, and ensemble methods for technical analysis
3. **LLM Sentiment Analysis**: AI-powered sentiment analysis of financial news
4. **MCP Server**: Real-time sentiment analysis API server
5. **Hybrid Prediction Engine**: Combines all approaches for optimal predictions

## Features

### Data Fetching
- 1-minute interval data for 1-month period
- Multiple data sources (Yahoo Finance, Upstox, Alpha Vantage)
- Data quality metrics and validation
- Market hours filtering
- Caching for performance

### ML Models
- XGBoost for gradient boosting
- Random Forest for ensemble learning
- Support Vector Regression
- Technical indicators (RSI, MACD, Bollinger Bands)
- Feature engineering for 1-minute data

### Sentiment Analysis
- LLM-powered sentiment analysis
- Support for OpenAI, Anthropic, and local LLMs
- News sentiment correlation
- Market impact assessment
- Real-time sentiment scoring

### MCP Server
- RESTful API for sentiment analysis
- Batch processing capabilities
- Market sentiment aggregation
- Caching and performance optimization
- Health monitoring

### Hybrid Engine
- Combines ML and sentiment predictions
- Weighted ensemble approach
- Risk assessment
- Signal generation
- Performance tracking

## Installation

1. Install dependencies:
```bash
pip install -r requirements_optimal.txt
```

2. Run setup:
```bash
python setup_optimal_prediction.py
```

3. Start the system:
```bash
python start_optimal_prediction.py
```

## Configuration

Edit `optimal_prediction_config.py` to customize:
- Symbol selection
- Data period
- Model parameters
- API keys
- Server settings

## API Usage

### MCP Server Endpoints

- `GET /` - Server status
- `POST /analyze/sentiment` - Analyze single text
- `POST /analyze/news/batch` - Batch news analysis
- `POST /analyze/market/sentiment` - Market sentiment
- `GET /cache/stats` - Cache statistics
- `GET /models/status` - Model status

### Example API Call

```python
import requests

# Analyze sentiment
response = requests.post('http://localhost:8000/analyze/sentiment', json={
    'text': 'Nifty 50 shows strong bullish momentum',
    'context': 'financial market'
})

result = response.json()
print(f"Sentiment: {result['sentiment_label']}")
print(f"Score: {result['sentiment_score']}")
```

## Performance

### Expected Performance
- Data Quality: >90%
- Prediction Latency: <100ms
- Model Accuracy: >70%
- Sentiment Accuracy: >80%

### Optimization Tips
1. Use local LLM for faster sentiment analysis
2. Enable caching for repeated requests
3. Adjust retrain intervals based on market volatility
4. Monitor system performance metrics

## Troubleshooting

### Common Issues

1. **No data retrieved**: Check internet connection and API keys
2. **Model training fails**: Ensure sufficient data (minimum 200 samples)
3. **Sentiment analysis errors**: Verify LLM configuration
4. **MCP server not starting**: Check port availability

### Logs

Check logs in `outputs/logs/` for detailed error information:
- `optimal_prediction.log` - Main system logs
- `model_training.log` - Model training logs
- `sentiment_analysis.log` - Sentiment analysis logs

## Support

For issues and questions:
1. Check the logs for error details
2. Verify configuration settings
3. Test individual components
4. Review system status endpoint

## License

This system is provided as-is for educational and research purposes.
'''
    
    with open('OPTIMAL_PREDICTION_GUIDE.md', 'w') as f:
        f.write(doc_content)
    
    print("✅ Documentation created: OPTIMAL_PREDICTION_GUIDE.md")

def main():
    """Main setup function"""
    print("🚀 Setting up Optimal Prediction System")
    print("=" * 50)
    
    # Create directories
    setup_directories()
    
    # Test components
    data_ok = test_data_fetching()
    sentiment_ok = test_sentiment_analysis()
    mcp_ok = test_mcp_server()
    
    # Test complete system
    system_ok = asyncio.run(test_optimal_system())
    
    # Create configuration files
    create_config_file()
    create_startup_script()
    create_requirements_file()
    create_documentation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    print(f"   Data Fetching: {'✅' if data_ok else '❌'}")
    print(f"   Sentiment Analysis: {'✅' if sentiment_ok else '❌'}")
    print(f"   MCP Server: {'✅' if mcp_ok else '❌'}")
    print(f"   Complete System: {'✅' if system_ok else '❌'}")
    
    if all([data_ok, sentiment_ok, mcp_ok, system_ok]):
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Review configuration: optimal_prediction_config.py")
        print("   2. Start the system: python start_optimal_prediction.py")
        print("   3. Access MCP server: http://localhost:8000")
        print("   4. Check predictions: outputs/predictions/")
        print("   5. Monitor logs: outputs/logs/")
    else:
        print("\n⚠️ Setup completed with some issues")
        print("   Check the logs above for details")
        print("   Some components may need manual configuration")

if __name__ == "__main__":
    main()
