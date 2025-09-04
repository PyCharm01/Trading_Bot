#!/usr/bin/env python3
"""
Simple test for portfolio functionality
"""

import sys
import os
sys.path.append('src')

from portfolio.indian_portfolio_simulator import IndianPortfolioSimulator

def simple_test():
    """Simple portfolio test"""
    print("💼 Simple Portfolio Test")
    print("=" * 30)
    
    try:
        # Initialize with higher capital
        portfolio_simulator = IndianPortfolioSimulator(initial_capital=1000000)  # 10 lakh
        
        print(f"Initial Capital: ₹{portfolio_simulator.initial_capital:,.2f}")
        print(f"Available Capital: ₹{portfolio_simulator.available_capital:,.2f}")
        
        # Test adding position
        print("\nAdding NIFTY 50 position...")
        success = portfolio_simulator.add_position(
            symbol="NIFTY_50",
            instrument_type="index",
            quantity=1,
            entry_price=19000.0,
            strategy="Long"
        )
        
        if success:
            print("✅ Position added successfully")
            
            # Get summary
            summary = portfolio_simulator.get_portfolio_summary()
            print(f"Total Value: ₹{summary['total_value']:,.2f}")
            print(f"Available Capital: ₹{summary['capital_utilization']['available_capital']:,.2f}")
            print(f"Open Positions: {summary['positions']['open']}")
            
        else:
            print("❌ Failed to add position")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
