#!/usr/bin/env python3
"""
LLM-Based Sentiment Analysis and News Processing

This module provides LLM-powered sentiment analysis for financial news and market data,
including MCP server setup for real-time sentiment analysis.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import json
import requests
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Optional async dependencies
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available. Async functionality will be limited.")

# Optional LLM dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai not available. OpenAI functionality will be disabled.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("anthropic not available. Anthropic functionality will be disabled.")

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0 to 1
    keywords: List[str]
    market_impact: str  # 'high', 'medium', 'low'
    timestamp: datetime

@dataclass
class NewsArticle:
    """Financial news article"""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    symbols: List[str]
    category: str

@dataclass
class MarketSentiment:
    """Overall market sentiment"""
    overall_sentiment: float
    sentiment_trend: str  # 'improving', 'deteriorating', 'stable'
    key_events: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    confidence: float
    timestamp: datetime

class LLMSentimentAnalyzer:
    """LLM-based sentiment analyzer for financial news and market data"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 use_local_llm: bool = False,
                 local_llm_url: str = "http://localhost:11434"):
        
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        self.use_local_llm = use_local_llm
        self.local_llm_url = local_llm_url
        
        # Financial sentiment keywords
        self.positive_keywords = [
            'bullish', 'surge', 'rally', 'gains', 'profit', 'growth', 'positive',
            'strong', 'robust', 'outperform', 'beat', 'exceed', 'rise', 'up',
            'increase', 'boost', 'momentum', 'breakthrough', 'success'
        ]
        
        self.negative_keywords = [
            'bearish', 'decline', 'fall', 'drop', 'loss', 'negative', 'weak',
            'underperform', 'miss', 'disappoint', 'concern', 'risk', 'volatility',
            'uncertainty', 'pressure', 'challenge', 'crisis', 'crash'
        ]
        
        self.market_impact_keywords = {
            'high': ['fed', 'rate', 'inflation', 'gdp', 'earnings', 'merger', 'acquisition'],
            'medium': ['sector', 'industry', 'company', 'stock', 'market'],
            'low': ['analyst', 'opinion', 'forecast', 'prediction']
        }
        
        # Initialize session for API calls
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def analyze_sentiment_async(self, text: str, context: str = "financial market") -> SentimentResult:
        """Analyze sentiment asynchronously using LLM"""
        try:
            if self.use_local_llm:
                return await self._analyze_with_local_llm(text, context)
            elif self.openai_api_key:
                return await self._analyze_with_openai(text, context)
            elif self.anthropic_api_key:
                return await self._analyze_with_anthropic(text, context)
            else:
                return self._analyze_with_keywords(text)
                
        except Exception as e:
            logger.error(f"Error in async sentiment analysis: {e}")
            return self._analyze_with_keywords(text)
    
    def analyze_sentiment(self, text: str, context: str = "financial market") -> SentimentResult:
        """Analyze sentiment synchronously"""
        try:
            if self.use_local_llm:
                return self._analyze_with_local_llm_sync(text, context)
            elif self.openai_api_key:
                return self._analyze_with_openai_sync(text, context)
            elif self.anthropic_api_key:
                return self._analyze_with_anthropic_sync(text, context)
            else:
                return self._analyze_with_keywords(text)
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._analyze_with_keywords(text)
    
    async def _analyze_with_openai(self, text: str, context: str) -> SentimentResult:
        """Analyze sentiment using OpenAI API"""
        try:
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI not available, falling back to keyword analysis")
                return self._analyze_with_keywords(text)
            
            client = openai.AsyncOpenAI(api_key=self.openai_api_key)
            
            prompt = f"""
            Analyze the sentiment of this financial news text and provide a structured response:
            
            Text: "{text}"
            Context: {context}
            
            Please provide:
            1. Sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
            2. Sentiment label (positive/negative/neutral)
            3. Confidence score (0 to 1)
            4. Key financial keywords found
            5. Market impact level (high/medium/low)
            
            Respond in JSON format:
            {{
                "sentiment_score": float,
                "sentiment_label": "string",
                "confidence": float,
                "keywords": ["string"],
                "market_impact": "string"
            }}
            """
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis expert. Analyze the sentiment of financial news and market information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            result_data = json.loads(result_text)
            
            return SentimentResult(
                text=text,
                sentiment_score=result_data.get('sentiment_score', 0.0),
                sentiment_label=result_data.get('sentiment_label', 'neutral'),
                confidence=result_data.get('confidence', 0.5),
                keywords=result_data.get('keywords', []),
                market_impact=result_data.get('market_impact', 'medium'),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error with OpenAI sentiment analysis: {e}")
            return self._analyze_with_keywords(text)
    
    def _analyze_with_openai_sync(self, text: str, context: str) -> SentimentResult:
        """Synchronous OpenAI sentiment analysis"""
        try:
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI not available, falling back to keyword analysis")
                return self._analyze_with_keywords(text)
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            prompt = f"""
            Analyze the sentiment of this financial news text and provide a structured response:
            
            Text: "{text}"
            Context: {context}
            
            Please provide:
            1. Sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
            2. Sentiment label (positive/negative/neutral)
            3. Confidence score (0 to 1)
            4. Key financial keywords found
            5. Market impact level (high/medium/low)
            
            Respond in JSON format:
            {{
                "sentiment_score": float,
                "sentiment_label": "string",
                "confidence": float,
                "keywords": ["string"],
                "market_impact": "string"
            }}
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis expert. Analyze the sentiment of financial news and market information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            result_data = json.loads(result_text)
            
            return SentimentResult(
                text=text,
                sentiment_score=result_data.get('sentiment_score', 0.0),
                sentiment_label=result_data.get('sentiment_label', 'neutral'),
                confidence=result_data.get('confidence', 0.5),
                keywords=result_data.get('keywords', []),
                market_impact=result_data.get('market_impact', 'medium'),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error with OpenAI sentiment analysis: {e}")
            return self._analyze_with_keywords(text)
    
    async def _analyze_with_anthropic(self, text: str, context: str) -> SentimentResult:
        """Analyze sentiment using Anthropic Claude API"""
        try:
            if not ANTHROPIC_AVAILABLE:
                logger.warning("Anthropic not available, falling back to keyword analysis")
                return self._analyze_with_keywords(text)
            
            client = anthropic.AsyncAnthropic(api_key=self.anthropic_api_key)
            
            prompt = f"""
            Analyze the sentiment of this financial news text and provide a structured response:
            
            Text: "{text}"
            Context: {context}
            
            Please provide:
            1. Sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
            2. Sentiment label (positive/negative/neutral)
            3. Confidence score (0 to 1)
            4. Key financial keywords found
            5. Market impact level (high/medium/low)
            
            Respond in JSON format:
            {{
                "sentiment_score": float,
                "sentiment_label": "string",
                "confidence": float,
                "keywords": ["string"],
                "market_impact": "string"
            }}
            """
            
            response = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result_text = response.content[0].text
            result_data = json.loads(result_text)
            
            return SentimentResult(
                text=text,
                sentiment_score=result_data.get('sentiment_score', 0.0),
                sentiment_label=result_data.get('sentiment_label', 'neutral'),
                confidence=result_data.get('confidence', 0.5),
                keywords=result_data.get('keywords', []),
                market_impact=result_data.get('market_impact', 'medium'),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error with Anthropic sentiment analysis: {e}")
            return self._analyze_with_keywords(text)
    
    def _analyze_with_anthropic_sync(self, text: str, context: str) -> SentimentResult:
        """Synchronous Anthropic sentiment analysis"""
        try:
            if not ANTHROPIC_AVAILABLE:
                logger.warning("Anthropic not available, falling back to keyword analysis")
                return self._analyze_with_keywords(text)
            
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            prompt = f"""
            Analyze the sentiment of this financial news text and provide a structured response:
            
            Text: "{text}"
            Context: {context}
            
            Please provide:
            1. Sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
            2. Sentiment label (positive/negative/neutral)
            3. Confidence score (0 to 1)
            4. Key financial keywords found
            5. Market impact level (high/medium/low)
            
            Respond in JSON format:
            {{
                "sentiment_score": float,
                "sentiment_label": "string",
                "confidence": float,
                "keywords": ["string"],
                "market_impact": "string"
            }}
            """
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result_text = response.content[0].text
            result_data = json.loads(result_text)
            
            return SentimentResult(
                text=text,
                sentiment_score=result_data.get('sentiment_score', 0.0),
                sentiment_label=result_data.get('sentiment_label', 'neutral'),
                confidence=result_data.get('confidence', 0.5),
                keywords=result_data.get('keywords', []),
                market_impact=result_data.get('market_impact', 'medium'),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error with Anthropic sentiment analysis: {e}")
            return self._analyze_with_keywords(text)
    
    async def _analyze_with_local_llm(self, text: str, context: str) -> SentimentResult:
        """Analyze sentiment using local LLM (Ollama)"""
        try:
            if not AIOHTTP_AVAILABLE:
                logger.warning("aiohttp not available, falling back to keyword analysis")
                return self._analyze_with_keywords(text)
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llama2",
                    "prompt": f'''
                    Analyze the sentiment of this financial news text and provide a structured response:
                    
                    Text: "{text}"
                    Context: {context}
                    
                    Please provide:
                    1. Sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
                    2. Sentiment label (positive/negative/neutral)
                    3. Confidence score (0 to 1)
                    4. Key financial keywords found
                    5. Market impact level (high/medium/low)
                    
                    Respond in JSON format:
                    {{
                        "sentiment_score": float,
                        "sentiment_label": "string",
                        "confidence": float,
                        "keywords": ["string"],
                        "market_impact": "string"
                    }}
                    '''
                    ,
                    "stream": False
                }
                
                async with session.post(f"{self.local_llm_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        result_text = result.get('response', '')
                        
                        # Try to extract JSON from response
                        try:
                            # Find JSON in the response
                            start_idx = result_text.find('{')
                            end_idx = result_text.rfind('}') + 1
                            if start_idx != -1 and end_idx != -1:
                                json_text = result_text[start_idx:end_idx]
                                result_data = json.loads(json_text)
                            else:
                                raise ValueError("No JSON found in response")
                        except:
                            # Fallback to keyword analysis
                            return self._analyze_with_keywords(text)
                        
                        return SentimentResult(
                            text=text,
                            sentiment_score=result_data.get('sentiment_score', 0.0),
                            sentiment_label=result_data.get('sentiment_label', 'neutral'),
                            confidence=result_data.get('confidence', 0.5),
                            keywords=result_data.get('keywords', []),
                            market_impact=result_data.get('market_impact', 'medium'),
                            timestamp=datetime.now()
                        )
                    else:
                        logger.error(f"Local LLM API error: {response.status}")
                        return self._analyze_with_keywords(text)
                        
        except Exception as e:
            logger.error(f"Error with local LLM sentiment analysis: {e}")
            return self._analyze_with_keywords(text)
    
    def _analyze_with_local_llm_sync(self, text: str, context: str) -> SentimentResult:
        """Synchronous local LLM sentiment analysis"""
        try:
            payload = {
                "model": "llama2",
                "prompt": f'''
                Analyze the sentiment of this financial news text and provide a structured response:
                
                Text: "{text}"
                Context: {context}
                
                Please provide:
                1. Sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
                2. Sentiment label (positive/negative/neutral)
                3. Confidence score (0 to 1)
                4. Key financial keywords found
                5. Market impact level (high/medium/low)
                
                Respond in JSON format:
                {{
                    "sentiment_score": float,
                    "sentiment_label": "string",
                    "confidence": float,
                    "keywords": ["string"],
                    "market_impact": "string"
                }}
                ''',
                "stream": False
            }
            
            response = self.session.post(f"{self.local_llm_url}/api/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                result_text = result.get('response', '')
                
                # Try to extract JSON from response
                try:
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_text = result_text[start_idx:end_idx]
                        result_data = json.loads(json_text)
                    else:
                        raise ValueError("No JSON found in response")
                except:
                    return self._analyze_with_keywords(text)
                
                return SentimentResult(
                    text=text,
                    sentiment_score=result_data.get('sentiment_score', 0.0),
                    sentiment_label=result_data.get('sentiment_label', 'neutral'),
                    confidence=result_data.get('confidence', 0.5),
                    keywords=result_data.get('keywords', []),
                    market_impact=result_data.get('market_impact', 'medium'),
                    timestamp=datetime.now()
                )
            else:
                logger.error(f"Local LLM API error: {response.status_code}")
                return self._analyze_with_keywords(text)
                
        except Exception as e:
            logger.error(f"Error with local LLM sentiment analysis: {e}")
            return self._analyze_with_keywords(text)
    
    def _analyze_with_keywords(self, text: str) -> SentimentResult:
        """Fallback keyword-based sentiment analysis"""
        try:
            text_lower = text.lower()
            
            # Count positive and negative keywords
            positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
            negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
            
            # Calculate sentiment score
            total_keywords = positive_count + negative_count
            if total_keywords == 0:
                sentiment_score = 0.0
                sentiment_label = 'neutral'
                confidence = 0.3
            else:
                sentiment_score = (positive_count - negative_count) / total_keywords
                if sentiment_score > 0.1:
                    sentiment_label = 'positive'
                elif sentiment_score < -0.1:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
                confidence = min(0.8, total_keywords * 0.1)
            
            # Find keywords
            found_keywords = []
            for keyword in self.positive_keywords + self.negative_keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
            
            # Determine market impact
            market_impact = 'low'
            for impact_level, keywords in self.market_impact_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    market_impact = impact_level
                    break
            
            return SentimentResult(
                text=text,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                confidence=confidence,
                keywords=found_keywords,
                market_impact=market_impact,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in keyword sentiment analysis: {e}")
            return SentimentResult(
                text=text,
                sentiment_score=0.0,
                sentiment_label='neutral',
                confidence=0.1,
                keywords=[],
                market_impact='low',
                timestamp=datetime.now()
            )
    
    def analyze_news_batch(self, articles: List[NewsArticle]) -> List[SentimentResult]:
        """Analyze sentiment for a batch of news articles"""
        results = []
        
        for article in articles:
            # Combine title and content for analysis
            full_text = f"{article.title}. {article.content}"
            sentiment = self.analyze_sentiment(full_text, "financial news")
            results.append(sentiment)
        
        return results
    
    async def analyze_news_batch_async(self, articles: List[NewsArticle]) -> List[SentimentResult]:
        """Analyze sentiment for a batch of news articles asynchronously"""
        tasks = []
        
        for article in articles:
            full_text = f"{article.title}. {article.content}"
            task = self.analyze_sentiment_async(full_text, "financial news")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch sentiment analysis: {result}")
                processed_results.append(SentimentResult(
                    text="",
                    sentiment_score=0.0,
                    sentiment_label='neutral',
                    confidence=0.1,
                    keywords=[],
                    market_impact='low',
                    timestamp=datetime.now()
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def calculate_market_sentiment(self, sentiment_results: List[SentimentResult]) -> MarketSentiment:
        """Calculate overall market sentiment from multiple sentiment results"""
        try:
            if not sentiment_results:
                return MarketSentiment(
                    overall_sentiment=0.0,
                    sentiment_trend='stable',
                    key_events=[],
                    risk_level='medium',
                    confidence=0.1,
                    timestamp=datetime.now()
                )
            
            # Weight sentiment by confidence and market impact
            weighted_scores = []
            for result in sentiment_results:
                weight = result.confidence
                if result.market_impact == 'high':
                    weight *= 2.0
                elif result.market_impact == 'medium':
                    weight *= 1.5
                else:
                    weight *= 1.0
                
                weighted_scores.append(result.sentiment_score * weight)
            
            # Calculate overall sentiment
            overall_sentiment = np.mean(weighted_scores) if weighted_scores else 0.0
            
            # Determine sentiment trend
            recent_results = [r for r in sentiment_results if (datetime.now() - r.timestamp).total_seconds() < 3600]  # Last hour
            if len(recent_results) >= 2:
                recent_sentiment = np.mean([r.sentiment_score for r in recent_results])
                if recent_sentiment > overall_sentiment + 0.1:
                    sentiment_trend = 'improving'
                elif recent_sentiment < overall_sentiment - 0.1:
                    sentiment_trend = 'deteriorating'
                else:
                    sentiment_trend = 'stable'
            else:
                sentiment_trend = 'stable'
            
            # Identify key events
            key_events = []
            for result in sentiment_results:
                if result.market_impact == 'high' and abs(result.sentiment_score) > 0.5:
                    key_events.append(f"{result.sentiment_label.title()} event: {result.text[:100]}...")
            
            # Determine risk level
            high_impact_negative = sum(1 for r in sentiment_results 
                                     if r.market_impact == 'high' and r.sentiment_score < -0.3)
            if high_impact_negative >= 2:
                risk_level = 'high'
            elif high_impact_negative >= 1 or overall_sentiment < -0.3:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Calculate confidence
            avg_confidence = np.mean([r.confidence for r in sentiment_results])
            confidence = min(0.9, avg_confidence)
            
            return MarketSentiment(
                overall_sentiment=overall_sentiment,
                sentiment_trend=sentiment_trend,
                key_events=key_events[:5],  # Top 5 events
                risk_level=risk_level,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return MarketSentiment(
                overall_sentiment=0.0,
                sentiment_trend='stable',
                key_events=[],
                risk_level='medium',
                confidence=0.1,
                timestamp=datetime.now()
            )

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sentiment analyzer
    analyzer = LLMSentimentAnalyzer(use_local_llm=True)  # Use local LLM
    
    # Test sentiment analysis
    test_texts = [
        "Nifty 50 surges to new all-time high on strong earnings and positive market sentiment",
        "Market faces volatility as inflation concerns and rate hike fears weigh on investor confidence",
        "Banking stocks show mixed performance with some banks reporting better than expected results"
    ]
    
    print("Testing LLM Sentiment Analysis...")
    
    for text in test_texts:
        print(f"\nText: {text}")
        sentiment = analyzer.analyze_sentiment(text)
        print(f"Sentiment: {sentiment.sentiment_label} (score: {sentiment.sentiment_score:.2f})")
        print(f"Confidence: {sentiment.confidence:.2f}")
        print(f"Market Impact: {sentiment.market_impact}")
        print(f"Keywords: {sentiment.keywords}")
    
    # Test market sentiment calculation
    print("\n" + "="*50)
    print("Testing Market Sentiment Calculation...")
    
    # Create sample news articles
    articles = [
        NewsArticle(
            title="Strong Q4 Earnings Boost Market Confidence",
            content="Major companies report better than expected earnings, driving market optimism",
            source="Financial Times",
            published_at=datetime.now(),
            url="https://example.com/news1",
            symbols=["NIFTY_50"],
            category="earnings"
        ),
        NewsArticle(
            title="Inflation Concerns Weigh on Market",
            content="Rising inflation data creates uncertainty about future rate hikes",
            source="Economic Times",
            published_at=datetime.now(),
            url="https://example.com/news2",
            symbols=["NIFTY_50"],
            category="economic"
        )
    ]
    
    # Analyze sentiment for articles
    sentiment_results = analyzer.analyze_news_batch(articles)
    
    # Calculate overall market sentiment
    market_sentiment = analyzer.calculate_market_sentiment(sentiment_results)
    
    print(f"Overall Market Sentiment: {market_sentiment.overall_sentiment:.2f}")
    print(f"Sentiment Trend: {market_sentiment.sentiment_trend}")
    print(f"Risk Level: {market_sentiment.risk_level}")
    print(f"Confidence: {market_sentiment.confidence:.2f}")
    print(f"Key Events: {market_sentiment.key_events}")

