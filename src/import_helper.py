"""
Import Helper for Indian Trading App

This module provides robust import functionality that works regardless of how the app is run.
"""

import sys
import os
import importlib.util

def setup_paths():
    """Setup Python paths for the application"""
    # Get the directory where this file is located (src directory)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root)
    project_root = os.path.dirname(src_dir)
    
    # Add both directories to Python path if not already there
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return src_dir, project_root

def import_config():
    """Import config module with fallback strategies"""
    try:
        # Try relative import first
        from config.config import get_config
        return get_config()
    except ImportError:
        try:
            # Try absolute import
            from src.config.config import get_config
            return get_config()
        except ImportError:
            try:
                # Try direct file import
                src_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(src_dir, 'config', 'config.py')
                
                if os.path.exists(config_path):
                    spec = importlib.util.spec_from_file_location("config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    return config_module.get_config()
                else:
                    raise ImportError("Config file not found")
            except Exception as e:
                print(f"Error importing config: {e}")
                # Return a default config object
                class DefaultConfig:
                    NEWS_API_KEY = "your_news_api_key_here"
                    ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key_here"
                    QUANDL_API_KEY = "your_quandl_api_key_here"
                    TWITTER_API_KEY = "your_twitter_api_key_here"
                    FINNHUB_API_KEY = "your_finnhub_api_key_here"
                    POLYGON_API_KEY = "your_polygon_api_key_here"
                
                return DefaultConfig()

def import_module(module_name, class_name=None):
    """Import a module with fallback strategies"""
    try:
        # Try relative import first
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        if class_name:
            return getattr(module, class_name)
        return module
    except ImportError:
        try:
            # Try absolute import
            full_module_name = f"src.{module_name}"
            module = __import__(full_module_name, fromlist=[class_name] if class_name else [])
            if class_name:
                return getattr(module, class_name)
            return module
        except ImportError:
            # Try direct file import
            src_dir = os.path.dirname(os.path.abspath(__file__))
            module_path = os.path.join(src_dir, f"{module_name.replace('.', os.sep)}.py")
            
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if class_name:
                    return getattr(module, class_name)
                return module
            else:
                raise ImportError(f"Module {module_name} not found")

# Setup paths when this module is imported
setup_paths()
