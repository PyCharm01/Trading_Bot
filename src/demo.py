#!/usr/bin/env python3
"""
Indian Market Trading Platform - Structured Demo

Comprehensive demonstration of the Indian market trading analysis platform
with proper module structure and organized output.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our structured modules
from data.indian_market_data import IndianMarketDataFetcher, INDIAN_MARKET_SYMBOLS
from analysis.indian_technical_analysis import IndianMarketAnalyzer
from analysis.indian_backtesting import IndianBacktestingEngine, BacktestConfig
from options.indian_options_engine import IndianOptionsStrategyEngine
from portfolio.indian_portfolio_simulator import IndianPortfolioSimulator
from visualization.indian_visualization import IndianMarketVisualizer
from config.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IndianMarketDemo:
    """Demo class for Indian market trading platform"""
    
    def __init__(self):
        """Initialize demo components"""
        self.config = get_config()
        self.data_fetcher = IndianMarketDataFetcher()
        self.technical_analyzer = IndianMarketAnalyzer()
        self.options_engine = IndianOptionsStrategyEngine()
        self.portfolio_simulator = IndianPortfolioSimulator(self.config.DEFAULT_INITIAL_CAPITAL)
        self.visualizer = IndianMarketVisualizer()
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs/demo', exist_ok=True)
    
    def run_demo(self):
        """Run comprehensive demo"""
        print("\n🇮🇳 Indian Market Trading Platform Demo")
        print("=" * 50)
        
        try:
            # Demo 1: Market Data Fetching
            self._demo_market_data()
            
            # Demo 2: Technical Analysis
            self._demo_technical_analysis()
            
            # Demo 3: Options Analysis
            self._demo_options_analysis()
            
            # Demo 4: Portfolio Management
            self._demo_portfolio_management()
            
            # Demo 5: Backtesting
            self._demo_backtesting()
            
            print("\n🎉 Demo completed successfully!")
            print("Launch the full application with: streamlit run src/app.py")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Demo failed: {e}")
    
    def _demo_market_data(self):
        """Demo market data fetching"""
        print("\n📊 Demo 1: Market Data Fetching")
        print("-" * 30)
        
        try:
            print("Fetching market overview...")
            overview = self.data_fetcher.fetch_market_overview()
            
            if overview:
                print("✅ Market Overview:")
                for symbol, data in overview.items():
                    symbol_info = INDIAN_MARKET_SYMBOLS.get(symbol)
                    name = symbol_info.name if symbol_info else symbol
                    print(f"  {name}: ₹{data['current_price']:,.0f} ({data['change_percent']:+.2f}%)")
            
            # Fetch sector performance
            print("\nFetching sector performance...")
            sectors = self.data_fetcher.fetch_sector_performance()
            
            if sectors:
                print("✅ Sector Performance:")
                for sector, data in sectors.items():
                    symbol_info = INDIAN_MARKET_SYMBOLS.get(sector)
                    name = symbol_info.name if symbol_info else sector
                    print(f"  {name}: {data.get('change_percent', 0):+.2f}%")
            
            # Check market status
            print("\nChecking market status...")
            status = self.data_fetcher.get_market_status()
            print(f"✅ Market Status: {'Open' if status.get('is_market_open', False) else 'Closed'}")
            print(f"  Next Event: {status.get('next_event', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Market data demo failed: {e}")
    
    def _demo_technical_analysis(self):
        """Demo technical analysis"""
        print("\n📈 Demo 2: Technical Analysis")
        print("-" * 30)
        
        try:
            print("Analyzing Nifty 50...")
            data = self.data_fetcher.fetch_index_data('NIFTY_50', '1mo')
            
            if data.empty:
                print("❌ No data available for Nifty 50")
                return
            
            analysis = self.technical_analyzer.analyze_index(data, 'NIFTY_50')
            
            print("✅ Technical Analysis Results:")
            print(f"  Current Price: ₹{analysis['current_price']:,.0f}")
            print(f"  Overall Signal: {analysis['overall_signal']}")
            print(f"  Signal Strength: {analysis['signal_strength']:.1f}%")
            print(f"  Market Trend: {analysis['market_trend']}")
            print(f"  Volatility: {analysis['volatility_level']}")
            print(f"  RSI: {analysis['rsi']:.1f}")
            print(f"  EMA 12: ₹{analysis['ema_12']:,.0f}")
            print(f"  EMA 26: ₹{analysis['ema_26']:,.0f}")
            
        except Exception as e:
            print(f"❌ Technical analysis demo failed: {e}")
    
    def _demo_options_analysis(self):
        """Demo options analysis"""
        print("\n🎯 Demo 3: Options Analysis")
        print("-" * 30)
        
        try:
            print("Analyzing Nifty 50 options...")
            data = self.data_fetcher.fetch_index_data('NIFTY_50', '1mo')
            options_data = self.data_fetcher.fetch_options_chain('NIFTY_50')
            
            if data.empty:
                print("❌ No data available for Nifty 50")
                return
            
            current_price = data['Close'].iloc[-1]
            analysis = self.technical_analyzer.analyze_index(data, 'NIFTY_50')
            
            # Get strategy recommendation
            strategy = self.options_engine.recommend_strategy(
                analysis['overall_signal'],
                options_data,
                current_price,
                'NIFTY_50',
                'normal'
            )
            
            print("✅ Options Strategy Recommendation:")
            print(f"  Strategy: {strategy.name}")
            print(f"  Description: {strategy.description}")
            print(f"  Max Profit: ₹{strategy.max_profit:,.0f}")
            print(f"  Max Loss: ₹{strategy.max_loss:,.0f}")
            print(f"  Probability of Profit: {strategy.probability_of_profit*100:.1f}%")
            print(f"  Risk-Reward Ratio: {strategy.risk_reward_ratio:.2f}")
            print(f"  Margin Required: ₹{strategy.margin_required:,.0f}")
            print(f"  Lot Size: {strategy.lot_size}")
            
        except Exception as e:
            print(f"❌ Options analysis demo failed: {e}")
    
    def _demo_portfolio_management(self):
        """Demo portfolio management"""
        print("\n💼 Demo 4: Portfolio Management")
        print("-" * 30)
        
        try:
            print("Adding sample positions...")
            
            # Add sample positions
            self.portfolio_simulator.add_position('NIFTY_50', 1, 150, 'Long')
            self.portfolio_simulator.add_position('BANK_NIFTY', 1, 200, 'Long')
            
            print("✅ Sample positions added successfully")
            
            # Update position prices
            print("Updating position prices...")
            self.portfolio_simulator.update_position_prices({
                'NIFTY_50': 19500,
                'BANK_NIFTY': 45000
            })
            
            # Generate portfolio summary
            print("Generating portfolio summary...")
            summary = self.portfolio_simulator.get_portfolio_summary()
            
            print("✅ Portfolio Summary:")
            print(f"  Total Value: ₹{summary['total_value']:,.0f}")
            print(f"  Total P&L: ₹{summary['total_pnl']:,.0f}")
            print(f"  Total P&L %: {summary['total_pnl_percent']:+.2f}%")
            print(f"  Unrealized P&L: ₹{summary['unrealized_pnl']:,.0f}")
            print(f"  Realized P&L: ₹{summary['realized_pnl']:,.0f}")
            print(f"  Margin Used: ₹{summary['margin_used']:,.0f}")
            print(f"  Available Capital: ₹{summary['available_capital']:,.0f}")
            print(f"  Open Positions: {summary['open_positions']}")
            print(f"  Closed Positions: {summary['closed_positions']}")
            
            # Check risk limits
            print("\nChecking risk limits...")
            risk_status = self.portfolio_simulator.check_risk_limits()
            print(f"✅ Portfolio is {'within' if risk_status else 'outside'} risk limits")
            
        except Exception as e:
            print(f"❌ Portfolio management demo failed: {e}")
    
    def _demo_backtesting(self):
        """Demo backtesting"""
        print("\n🔬 Demo 5: Backtesting")
        print("-" * 30)
        
        try:
            print("Running momentum strategy backtest on Nifty 50...")
            print("  Period: 2023-01-01 to 2023-12-31")
            print("  Initial Capital: ₹1,000,000")
            print("  Stop Loss: 5.0%")
            print("  Take Profit: 10.0%")
            
            # Create backtest config
            config = BacktestConfig(
                symbol='NIFTY_50',
                start_date='2023-01-01',
                end_date='2023-12-31',
                initial_capital=1000000,
                strategy='momentum_strategy',
                stop_loss=0.05,
                take_profit=0.10
            )
            
            # Run backtest
            backtest_engine = IndianBacktestingEngine()
            results = backtest_engine.run_backtest(config)
            
            if results:
                print("✅ Backtest Results:")
                print(f"  Total Return: {results.get('total_return', 0)*100:+.2f}%")
                print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")
                print(f"  Win Rate: {results.get('win_rate', 0)*100:.1f}%")
                print(f"  Total Trades: {results.get('total_trades', 0)}")
                
                # Save backtest results
                import json
                results_filename = f"outputs/backtests/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"  Results saved to: {results_filename}")
            else:
                print("❌ Backtesting demo failed: No results generated")
            
        except Exception as e:
            print(f"❌ Backtesting demo failed: {e}")

def main():
    """Main function to run the demo"""
    demo = IndianMarketDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
