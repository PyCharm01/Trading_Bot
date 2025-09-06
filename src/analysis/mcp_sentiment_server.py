#!/usr/bin/env python3
"""
MCP Server for Financial Sentiment Analysis

This module provides an MCP (Model Context Protocol) server for real-time
sentiment analysis of financial news and market data.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Import our sentiment analyzer
from .llm_sentiment_analyzer import LLMSentimentAnalyzer, SentimentResult, NewsArticle, MarketSentiment

logger = logging.getLogger(__name__)

# Pydantic models for API
class SentimentRequest(BaseModel):
    text: str
    context: str = "financial market"
    use_async: bool = False

class NewsAnalysisRequest(BaseModel):
    articles: List[Dict[str, Any]]
    batch_process: bool = True

class MarketSentimentRequest(BaseModel):
    time_window: int = 3600  # seconds
    symbols: Optional[List[str]] = None

class SentimentResponse(BaseModel):
    sentiment_score: float
    sentiment_label: str
    confidence: float
    keywords: List[str]
    market_impact: str
    timestamp: str
    processing_time_ms: float

class MarketSentimentResponse(BaseModel):
    overall_sentiment: float
    sentiment_trend: str
    key_events: List[str]
    risk_level: str
    confidence: float
    timestamp: str
    symbol_sentiments: Dict[str, float]

class MCPFinancialSentimentServer:
    """MCP Server for Financial Sentiment Analysis"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8000,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 use_local_llm: bool = True):
        
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Financial Sentiment Analysis MCP Server",
            description="Real-time sentiment analysis for financial markets",
            version="1.0.0"
        )
        
        # Initialize sentiment analyzer
        self.analyzer = LLMSentimentAnalyzer(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            use_local_llm=use_local_llm
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Cache for recent sentiment data
        self.sentiment_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Financial Sentiment Analysis MCP Server",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "analyzer_ready": True
            }
        
        @self.app.post("/analyze/sentiment", response_model=SentimentResponse)
        async def analyze_sentiment(request: SentimentRequest):
            """Analyze sentiment of a single text"""
            try:
                start_time = datetime.now()
                
                # Check cache first
                cache_key = f"sentiment_{hash(request.text)}_{request.context}"
                if cache_key in self.sentiment_cache:
                    cached_result, timestamp = self.sentiment_cache[cache_key]
                    if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                        return SentimentResponse(
                            sentiment_score=cached_result.sentiment_score,
                            sentiment_label=cached_result.sentiment_label,
                            confidence=cached_result.confidence,
                            keywords=cached_result.keywords,
                            market_impact=cached_result.market_impact,
                            timestamp=cached_result.timestamp.isoformat(),
                            processing_time_ms=0.0
                        )
                
                # Analyze sentiment
                if request.use_async:
                    result = await self.analyzer.analyze_sentiment_async(request.text, request.context)
                else:
                    result = self.analyzer.analyze_sentiment(request.text, request.context)
                
                # Cache result
                self.sentiment_cache[cache_key] = (result, datetime.now())
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return SentimentResponse(
                    sentiment_score=result.sentiment_score,
                    sentiment_label=result.sentiment_label,
                    confidence=result.confidence,
                    keywords=result.keywords,
                    market_impact=result.market_impact,
                    timestamp=result.timestamp.isoformat(),
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analyze/news/batch")
        async def analyze_news_batch(request: NewsAnalysisRequest):
            """Analyze sentiment for a batch of news articles"""
            try:
                start_time = datetime.now()
                
                # Convert articles to NewsArticle objects
                articles = []
                for article_data in request.articles:
                    article = NewsArticle(
                        title=article_data.get('title', ''),
                        content=article_data.get('content', ''),
                        source=article_data.get('source', ''),
                        published_at=datetime.fromisoformat(article_data.get('published_at', datetime.now().isoformat())),
                        url=article_data.get('url', ''),
                        symbols=article_data.get('symbols', []),
                        category=article_data.get('category', '')
                    )
                    articles.append(article)
                
                # Analyze sentiment
                if request.batch_process:
                    results = await self.analyzer.analyze_news_batch_async(articles)
                else:
                    results = self.analyzer.analyze_news_batch(articles)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Convert results to response format
                response_data = []
                for i, result in enumerate(results):
                    response_data.append({
                        "article_index": i,
                        "sentiment_score": result.sentiment_score,
                        "sentiment_label": result.sentiment_label,
                        "confidence": result.confidence,
                        "keywords": result.keywords,
                        "market_impact": result.market_impact,
                        "timestamp": result.timestamp.isoformat()
                    })
                
                return {
                    "results": response_data,
                    "total_articles": len(articles),
                    "processing_time_ms": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error in batch news analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analyze/market/sentiment", response_model=MarketSentimentResponse)
        async def analyze_market_sentiment(request: MarketSentimentRequest):
            """Analyze overall market sentiment"""
            try:
                start_time = datetime.now()
                
                # Get recent sentiment data (this would typically come from a database)
                # For now, we'll use cached data or generate sample data
                recent_sentiments = self._get_recent_sentiments(request.time_window, request.symbols)
                
                # Calculate market sentiment
                market_sentiment = self.analyzer.calculate_market_sentiment(recent_sentiments)
                
                # Calculate symbol-specific sentiments
                symbol_sentiments = self._calculate_symbol_sentiments(recent_sentiments, request.symbols)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return MarketSentimentResponse(
                    overall_sentiment=market_sentiment.overall_sentiment,
                    sentiment_trend=market_sentiment.sentiment_trend,
                    key_events=market_sentiment.key_events,
                    risk_level=market_sentiment.risk_level,
                    confidence=market_sentiment.confidence,
                    timestamp=market_sentiment.timestamp.isoformat(),
                    symbol_sentiments=symbol_sentiments
                )
                
            except Exception as e:
                logger.error(f"Error in market sentiment analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/cache/stats")
        async def get_cache_stats():
            """Get cache statistics"""
            return {
                "cache_size": len(self.sentiment_cache),
                "cache_ttl": self.cache_ttl,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.delete("/cache/clear")
        async def clear_cache():
            """Clear sentiment cache"""
            self.sentiment_cache.clear()
            return {
                "message": "Cache cleared",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/models/status")
        async def get_models_status():
            """Get status of available models"""
            return {
                "local_llm": self.analyzer.use_local_llm,
                "openai_available": bool(self.analyzer.openai_api_key),
                "anthropic_available": bool(self.analyzer.anthropic_api_key),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_recent_sentiments(self, time_window: int, symbols: Optional[List[str]] = None) -> List[SentimentResult]:
        """Get recent sentiment data (mock implementation)"""
        # In a real implementation, this would query a database
        # For now, return cached sentiments or generate sample data
        
        recent_sentiments = []
        for cache_key, (result, timestamp) in self.sentiment_cache.items():
            if (datetime.now() - timestamp).total_seconds() <= time_window:
                recent_sentiments.append(result)
        
        # If no cached data, generate sample data
        if not recent_sentiments:
            sample_texts = [
                "Market shows strong bullish momentum with positive earnings reports",
                "Volatility concerns rise as inflation data exceeds expectations",
                "Banking sector shows mixed performance with some positive developments"
            ]
            
            for text in sample_texts:
                sentiment = self.analyzer.analyze_sentiment(text)
                recent_sentiments.append(sentiment)
        
        return recent_sentiments
    
    def _calculate_symbol_sentiments(self, sentiments: List[SentimentResult], symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate symbol-specific sentiment scores"""
        symbol_sentiments = {}
        
        if symbols:
            for symbol in symbols:
                # In a real implementation, this would filter sentiments by symbol
                # For now, use overall sentiment as proxy
                if sentiments:
                    symbol_sentiments[symbol] = np.mean([s.sentiment_score for s in sentiments])
                else:
                    symbol_sentiments[symbol] = 0.0
        
        return symbol_sentiments
    
    def run(self, debug: bool = False):
        """Run the MCP server"""
        logger.info(f"Starting Financial Sentiment Analysis MCP Server on {self.host}:{self.port}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info" if not debug else "debug",
            reload=debug
        )
    
    async def run_async(self, debug: bool = False):
        """Run the MCP server asynchronously"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info" if not debug else "debug",
            reload=debug
        )
        server = uvicorn.Server(config)
        await server.serve()

# CLI interface
def main():
    """Main function for running the MCP server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial Sentiment Analysis MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--use-local-llm", action="store_true", default=True, help="Use local LLM")
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument("--anthropic-key", help="Anthropic API key")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run server
    server = MCPFinancialSentimentServer(
        host=args.host,
        port=args.port,
        openai_api_key=args.openai_key,
        anthropic_api_key=args.anthropic_key,
        use_local_llm=args.use_local_llm
    )
    
    try:
        server.run(debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
