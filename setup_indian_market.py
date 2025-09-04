#!/usr/bin/env python3
"""
Indian Market Trading Platform Setup Script

This script automates the setup process for the Indian Market Trading Analysis Platform.
It handles dependency installation, configuration, and initial setup.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndianMarketSetup:
    """Setup class for Indian Market Trading Platform"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.config_file = self.project_root / "config.py"
        self.env_file = self.project_root / ".env"
        
    def run_setup(self):
        """Run the complete setup process"""
        try:
            logger.info("🚀 Starting Indian Market Trading Platform Setup")
            
            # Check Python version
            self.check_python_version()
            
            # Install dependencies
            self.install_dependencies()
            
            # Create configuration files
            self.create_config_files()
            
            # Verify installation
            self.verify_installation()
            
            # Create sample data directories
            self.create_directories()
            
            logger.info("✅ Setup completed successfully!")
            self.print_next_steps()
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            sys.exit(1)
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.info("🔍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise Exception(f"Python 3.8+ required. Current version: {version.major}.{version.minor}")
        
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("📦 Installing dependencies...")
        
        # Core dependencies
        core_deps = [
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "yfinance>=0.2.0",
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "scikit-learn>=1.3.0",
            "requests>=2.31.0",
            "python-dateutil>=2.8.0",
            "pytz>=2023.3"
        ]
        
        # Install core dependencies
        for dep in core_deps:
            logger.info(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        
        # Optional dependencies
        optional_deps = [
            "TA-Lib>=0.4.25",
            "transformers>=4.30.0",
            "torch>=2.0.0"
        ]
        
        logger.info("📦 Installing optional dependencies...")
        for dep in optional_deps:
            try:
                logger.info(f"Installing {dep}...")
                subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️ Failed to install {dep}. This is optional and the platform will work without it.")
        
        logger.info("✅ Dependencies installed successfully")
    
    def create_config_files(self):
        """Create configuration files"""
        logger.info("⚙️ Creating configuration files...")
        
        # Create .env file if it doesn't exist
        if not self.env_file.exists():
            env_content = """# Indian Market Trading Platform Configuration

# Risk Management
RISK_FREE_RATE=0.06
DEFAULT_VOLATILITY=0.25

# API Keys (Optional - for enhanced features)
# NEWS_API_KEY=your_news_api_key_here
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
# TWITTER_API_KEY=your_twitter_api_key_here
# TWITTER_API_SECRET=your_twitter_api_secret_here

# Logging
LOG_LEVEL=INFO

# Market Settings
DEFAULT_MARKET=NSE
TIMEZONE=Asia/Kolkata
"""
            self.env_file.write_text(env_content)
            logger.info("✅ Created .env configuration file")
        
        # Create data directories
        data_dirs = ["data", "logs", "reports", "backtests"]
        for dir_name in data_dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✅ Created {dir_name} directory")
    
    def verify_installation(self):
        """Verify that all components are working"""
        logger.info("🔍 Verifying installation...")
        
        # Test imports
        test_imports = [
            "pandas",
            "numpy", 
            "yfinance",
            "streamlit",
            "plotly",
            "matplotlib",
            "seaborn",
            "sklearn",
            "requests"
        ]
        
        for module in test_imports:
            try:
                __import__(module)
                logger.info(f"✅ {module} imported successfully")
            except ImportError as e:
                logger.error(f"❌ Failed to import {module}: {e}")
                raise
        
        # Test optional imports
        optional_imports = ["talib", "transformers", "torch"]
        for module in optional_imports:
            try:
                __import__(module)
                logger.info(f"✅ {module} imported successfully (optional)")
            except ImportError:
                logger.warning(f"⚠️ {module} not available (optional)")
        
        # Test our custom modules
        custom_modules = [
            "indian_market_data",
            "indian_technical_analysis", 
            "indian_options_engine",
            "indian_portfolio_simulator",
            "indian_visualization",
            "indian_trading_app",
            "indian_backtesting"
        ]
        
        for module in custom_modules:
            try:
                __import__(module)
                logger.info(f"✅ {module} imported successfully")
            except ImportError as e:
                logger.error(f"❌ Failed to import {module}: {e}")
                raise
        
        logger.info("✅ Installation verification completed")
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("📁 Creating project directories...")
        
        directories = [
            "data/historical",
            "data/options",
            "data/news",
            "logs/application",
            "logs/backtesting",
            "reports/daily",
            "reports/backtesting",
            "backtests/results",
            "backtests/configs"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {dir_path}")
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("🎉 Indian Market Trading Platform Setup Complete!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. Launch the application:")
        print("   streamlit run indian_trading_app.py")
        print("\n2. Open your browser and go to:")
        print("   http://localhost:8501")
        print("\n3. Start analyzing Indian markets:")
        print("   - Select Nifty 50, Bank Nifty, or Sensex")
        print("   - Configure your analysis parameters")
        print("   - Explore technical analysis and options strategies")
        print("\n4. Optional: Configure API keys in .env file for enhanced features")
        print("\n📚 Documentation:")
        print("   - Read INDIAN_MARKET_README.md for detailed usage")
        print("   - Check code comments for technical details")
        print("\n⚠️ Important:")
        print("   - This platform is for educational purposes only")
        print("   - Always consult a financial advisor before trading")
        print("   - Past performance does not guarantee future results")
        print("\n🚀 Happy Trading!")
        print("="*60)

def main():
    """Main function"""
    try:
        setup = IndianMarketSetup()
        setup.run_setup()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
