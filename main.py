#!/usr/bin/env python3
"""
Indian Market Trading Platform - Main Entry Point

Main entry point for the Indian market trading analysis platform.
This script provides easy access to all platform features.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Indian Market Trading Analysis Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup          # Setup the platform
  python main.py demo           # Run demo
  python main.py app            # Launch Streamlit app
  python main.py --help         # Show this help
        """
    )
    
    parser.add_argument(
        'command',
        choices=['setup', 'demo', 'app'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        print("🚀 Running setup...")
        from setup import main as setup_main
        setup_main()
    
    elif args.command == 'demo':
        print("🎯 Running demo...")
        from demo import main as demo_main
        demo_main()
    
    elif args.command == 'app':
        print("🌐 Launching Streamlit app...")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/app.py"
        ])

if __name__ == "__main__":
    main()
