#!/usr/bin/env python3
"""
Test script for Paper Trading functionality
"""

import sys
import os
sys.path.append('src')

from data.indian_market_data import IndianMarketDataFetcher
from portfolio.indian_portfolio_simulator import IndianPortfolioSimulator
from config.config import get_config

def test_paper_trading():
    """Test the paper trading functionality"""
    print("💼 Testing Paper Trading Functionality")
    print("=" * 50)
    
    try:
        # Initialize components
        config = get_config()
        data_fetcher = IndianMarketDataFetcher(
            alpha_vantage_api_key=config.ALPHA_VANTAGE_API_KEY,
            quandl_api_key=config.QUANDL_API_KEY
        )
        portfolio_simulator = IndianPortfolioSimulator(initial_capital=1000000)  # 10 lakh
        
        print("✅ Components initialized successfully")
        
        # Test 1: Add a position
        print("\n📊 Test 1: Adding a position...")
        success = portfolio_simulator.add_position(
            symbol="NIFTY_50",
            instrument_type="index",
            quantity=1,
            entry_price=19000.0,
            strategy="Long"
        )
        
        if success:
            print("✅ Position added successfully")
        else:
            print("❌ Failed to add position")
            return
        
        # Test 2: Get portfolio summary
        print("\n📈 Test 2: Getting portfolio summary...")
        summary = portfolio_simulator.get_portfolio_summary()
        print(f"   Total Value: ₹{summary['total_value']:,.2f}")
        print(f"   Total P&L: ₹{summary['total_pnl']:,.2f}")
        print(f"   Available Capital: ₹{summary['capital_utilization']['available_capital']:,.2f}")
        
        # Test 3: Update positions with live data
        print("\n🔄 Test 3: Updating positions with live data...")
        live_data = portfolio_simulator.update_all_positions_with_live_data(data_fetcher)
        
        if live_data and 'positions' in live_data:
            print("✅ Live data update successful")
            print(f"   Open Positions: {live_data['open_positions_count']}")
            print(f"   Total Unrealized P&L: ₹{live_data['total_unrealized_pnl']:,.2f}")
            print(f"   Total Value: ₹{live_data['total_value']:,.2f}")
            
            # Display position details
            for position in live_data['positions']:
                if position['status'] == 'open':
                    print(f"\n   Position: {position['symbol']} {position['position_type']}")
                    print(f"   Entry Price: ₹{position['entry_price']:,.2f}")
                    print(f"   Current Price: ₹{position['current_price']:,.2f}")
                    print(f"   Unrealized P&L: ₹{position['unrealized_pnl']:,.2f}")
                    print(f"   P&L %: {position['unrealized_pnl_percent']:+.2f}%")
        else:
            print("❌ Failed to update with live data")
        
        # Test 4: Add another position
        print("\n📊 Test 4: Adding another position...")
        success = portfolio_simulator.add_position(
            symbol="BANK_NIFTY",
            instrument_type="index",
            quantity=1,
            entry_price=45000.0,
            strategy="Short"
        )
        
        if success:
            print("✅ Second position added successfully")
        else:
            print("❌ Failed to add second position")
        
        # Test 5: Update again with both positions
        print("\n🔄 Test 5: Updating with both positions...")
        live_data = portfolio_simulator.update_all_positions_with_live_data(data_fetcher)
        
        if live_data and 'positions' in live_data:
            print("✅ Live data update successful")
            print(f"   Open Positions: {live_data['open_positions_count']}")
            print(f"   Total Unrealized P&L: ₹{live_data['total_unrealized_pnl']:,.2f}")
            
            # Display all positions
            for position in live_data['positions']:
                if position['status'] == 'open':
                    pnl_color = "🟢" if position['unrealized_pnl'] >= 0 else "🔴"
                    print(f"\n   {pnl_color} {position['symbol']} {position['position_type']}")
                    print(f"   Entry: ₹{position['entry_price']:,.2f} | Current: ₹{position['current_price']:,.2f}")
                    print(f"   P&L: ₹{position['unrealized_pnl']:,.2f} ({position['unrealized_pnl_percent']:+.2f}%)")
        
        # Test 6: Close a position
        print("\n❌ Test 6: Closing a position...")
        if live_data and 'positions' in live_data:
            open_positions = [p for p in live_data['positions'] if p['status'] == 'open']
            if open_positions:
                position_to_close = open_positions[0]
                result = portfolio_simulator.close_position(position_to_close['position_id'])
                
                if result['success']:
                    print(f"✅ Position closed successfully")
                    print(f"   Realized P&L: ₹{result['realized_pnl']:,.2f}")
                    print(f"   Exit Price: ₹{result['exit_price']:,.2f}")
                else:
                    print(f"❌ Failed to close position: {result['error']}")
        
        # Test 7: Final portfolio summary
        print("\n📊 Test 7: Final portfolio summary...")
        final_summary = portfolio_simulator.get_portfolio_summary()
        print(f"   Total Value: ₹{final_summary['total_value']:,.2f}")
        print(f"   Total P&L: ₹{final_summary['total_pnl']:,.2f}")
        print(f"   Available Capital: ₹{final_summary['capital_utilization']['available_capital']:,.2f}")
        print(f"   Open Positions: {final_summary['positions']['open']}")
        print(f"   Closed Positions: {final_summary['positions']['closed']}")
        
        print("\n✅ Paper trading test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in paper trading test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_paper_trading()
