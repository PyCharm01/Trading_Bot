#!/usr/bin/env python3
"""
Indian Market Trading Platform - Structured Setup

Automated setup script for the Indian market trading analysis platform
with proper module structure and organized output.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IndianMarketSetup:
    """Setup class for Indian market trading platform"""
    
    def __init__(self):
        """Initialize setup"""
        self.base_dir = Path(__file__).parent.parent
        self.src_dir = self.base_dir / "src"
        self.outputs_dir = self.base_dir / "outputs"
        
    def run_setup(self):
        """Run complete setup process"""
        print("\n🚀 Starting Indian Market Trading Platform Setup")
        print("=" * 60)
        
        try:
            # Step 1: Check Python version
            self._check_python_version()
            
            # Step 2: Install dependencies
            self._install_dependencies()
            
            # Step 3: Create directory structure
            self._create_directories()
            
            # Step 4: Create configuration files
            self._create_config_files()
            
            # Step 5: Verify installation
            self._verify_installation()
            
            print("\n✅ Setup completed successfully!")
            print("\n📋 Next Steps:")
            print("1. Run demo: python src/demo.py")
            print("2. Launch app: streamlit run src/app.py")
            print("3. Check outputs in: outputs/ directory")
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            logger.error(f"Setup failed: {e}")
            sys.exit(1)
    
    def _check_python_version(self):
        """Check Python version compatibility"""
        print("\n🔍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise Exception(f"Python 3.8+ required, found {version.major}.{version.minor}")
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    
    def _install_dependencies(self):
        """Install required dependencies"""
        print("\n📦 Installing dependencies...")
        
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
        
        # Optional dependencies
        optional_deps = [
            "TA-Lib>=0.4.25",
            "transformers>=4.30.0",
            "torch>=2.0.0"
        ]
        
        # Install core dependencies
        for dep in core_deps:
            print(f"Installing {dep}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {dep}: {e}")
        
        # Install optional dependencies
        print("\n📦 Installing optional dependencies...")
        for dep in optional_deps:
            print(f"Installing {dep}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {dep}: {e}")
        
        print("✅ Dependencies installed successfully")
    
    def _create_directories(self):
        """Create directory structure"""
        print("\n📁 Creating directory structure...")
        
        # Create outputs directories
        output_dirs = [
            "outputs/logs",
            "outputs/reports", 
            "outputs/data",
            "outputs/backtests",
            "outputs/portfolios",
            "outputs/demo"
        ]
        
        for dir_path in output_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Created {dir_path}")
        
        print("✅ Directory structure created successfully")
    
    def _create_config_files(self):
        """Create configuration files"""
        print("\n⚙️ Creating configuration files...")
        
        # Create .env file
        env_content = """# Indian Market Trading Platform Configuration
# Copy this file and modify as needed

# API Keys (if using premium data sources)
# ALPHA_VANTAGE_API_KEY=your_key_here
# QUANDL_API_KEY=your_key_here

# Application Settings
LOG_LEVEL=INFO
DEFAULT_INITIAL_CAPITAL=1000000.0
DEFAULT_TIME_PERIOD=1mo
DEFAULT_TICKER=NIFTY_50

# Market Settings
TIMEZONE=Asia/Kolkata
TRADING_HOURS_START=09:15
TRADING_HOURS_END=15:30

# Risk Management
MAX_POSITION_SIZE=0.1
STOP_LOSS_PERCENTAGE=0.05
TAKE_PROFIT_PERCENTAGE=0.10

# Output Settings
SAVE_REPORTS=true
SAVE_BACKTESTS=true
REPORT_FORMAT=json
"""
        
        env_file = self.base_dir / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env configuration file")
        
        # Create requirements.txt
        requirements_content = """# Indian Market Trading Platform Requirements
# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
yfinance>=0.2.0
streamlit>=1.28.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
requests>=2.31.0
python-dateutil>=2.8.0
pytz>=2023.3

# Optional dependencies for enhanced functionality
TA-Lib>=0.4.25
transformers>=4.30.0
torch>=2.0.0
scipy>=1.6.0
"""
        
        requirements_file = self.base_dir / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        print("✅ Created requirements.txt")
        
        print("✅ Configuration files created successfully")
    
    def _verify_installation(self):
        """Verify installation by importing modules"""
        print("\n🔍 Verifying installation...")
        
        # Test imports
        test_imports = [
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("yfinance", "yfinance"),
            ("streamlit", "streamlit"),
            ("plotly", "plotly"),
            ("matplotlib", "matplotlib"),
            ("seaborn", "seaborn"),
            ("sklearn", "sklearn"),
            ("requests", "requests"),
            ("talib", "talib (optional)"),
            ("transformers", "transformers (optional)"),
            ("torch", "torch (optional)")
        ]
        
        # Add src to path for testing
        sys.path.insert(0, str(self.src_dir))
        
        for module_name, display_name in test_imports:
            try:
                __import__(module_name)
                print(f"✅ {display_name} imported successfully")
            except ImportError as e:
                if "optional" in display_name:
                    print(f"⚠️ {display_name} not available (optional)")
                else:
                    print(f"❌ {display_name} import failed: {e}")
        
        # Test our custom modules
        custom_modules = [
            ("src.data.indian_market_data", "Indian Market Data"),
            ("src.analysis.indian_technical_analysis", "Technical Analysis"),
            ("src.options.indian_options_engine", "Options Engine"),
            ("src.portfolio.indian_portfolio_simulator", "Portfolio Simulator"),
            ("src.visualization.indian_visualization", "Visualization"),
            ("src.config.config", "Configuration")
        ]
        
        print("\n🔍 Testing custom modules...")
        for module_name, display_name in custom_modules:
            try:
                __import__(module_name)
                print(f"✅ {display_name} imported successfully")
            except ImportError as e:
                print(f"❌ {display_name} import failed: {e}")
        
        print("✅ Installation verification completed")

def main():
    """Main function to run setup"""
    setup = IndianMarketSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
