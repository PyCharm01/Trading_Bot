"""
Configuration management for the Indian market trading platform.
"""

from .config import get_config, get_indian_market_strategies, get_lot_size, get_margin_requirements

__all__ = ['get_config', 'get_indian_market_strategies', 'get_lot_size', 'get_margin_requirements']
