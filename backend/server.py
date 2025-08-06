from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
import os
import httpx
import asyncio
import uuid
from dotenv import load_dotenv
import json
import logging
from contextlib import asynccontextmanager
import aioredis
from collections import defaultdict
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting News Knowledge Graph API...")
    yield
    # Shutdown
    logger.info("Shutting down News Knowledge Graph API...")

app = FastAPI(
    title="News Knowledge Graph API - Production Scale",
    description="Advanced AI-powered news knowledge graph with multi-source integration",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
GUARDIAN_API_KEY = "377ea0e1-846f-438d-9082-482f5afbaba4"
OPENAI_API_KEY = "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"
ANTHROPIC_API_KEY = "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"

# NYT API Configuration (will be integrated)
NYT_API_KEY = os.environ.get('NYT_API_KEY', 'your-nyt-api-key-here')

# Database and Cache
client = AsyncIOMotorClient(MONGO_URL)
db = client.news_knowledge_graph_v3

# Redis for caching (optional - will work without Redis)
redis_client = None
try:
    import aioredis
    redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
except ImportError:
    logger.info("Redis not available, using in-memory caching")

# In-memory cache fallback
cache = {}

# Enhanced Models
class NewsSource(BaseModel):
    name: str
    api_key: str
    base_url: str
    rate_limit: int
    last_request: float = 0.0

class GeographicInfo(BaseModel):
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    coordinates: Optional[List[float]] = None

class TemporalAnalysis(BaseModel):
    timeline_position: float  # 0.0 to 1.0 representing position in story development
    development_stage: str  # "breaking", "developing", "analysis", "follow-up"
    related_events: List[str] = []

class AdvancedConnection(BaseModel):
    source_id: str
    target_id: str
    connection_type: str
    strength: float
    confidence: float  # AI confidence in the connection
    explanation: str
    keywords: List[str]
    geographic_overlap: Optional[GeographicInfo] = None
    temporal_relationship: Optional[str] = None  # "causation", "correlation", "sequence"
    evidence_score: float = 0.0

class EnhancedStory(BaseModel):
    id: str
    source: str  # "guardian", "nyt", "combined"
    title: str
    summary: str
    lede: str
    nutgraf: str
    section: str
    publication_date: datetime
    url: str
    author: Optional[str] = None
    entities: List[str]
    categories: List[str]
    engagement_preview: str
    geographic_info: Optional[GeographicInfo] = None
    temporal_analysis: Optional[TemporalAnalysis] = None
    sentiment_score: float = 0.0
    complexity_level: int = 1  # 1-5 scale for user's requested detail level
    related_stories: List[str] = []
    read_time_minutes: int = 1

class UserPreferences(BaseModel):
    complexity_level: int = 3  # 1-5 scale
    preferred_sections: List[str] = []
    geographic_focus: Optional[str] = None
    temporal_focus: str = "recent"  # "recent", "developing", "historical"

class AnalyticsData(BaseModel):
    user_id: Optional[str] = None
    session_id: str
    action: str  # "view_story", "explore_connection", "search", "feedback"
    story_id: Optional[str] = None
    connection_id: Optional[str] = None
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = {}

class FeedbackData(BaseModel):
    rating: int
    comments: str
    email: Optional[str] = None
    features_used: List[str] = []
    most_interesting_connection: Optional[str] = None
    suggested_improvements: Optional[str] = None
    session_id: Optional[str] = None

# Multi-Source News API Client
class MultiSourceNewsClient:
    def __init__(self):
        self.sources = {
            "guardian": NewsSource(
                name="The Guardian",
                api_key=GUARDIAN_API_KEY,
                base_url="https://content.guardianapis.com",
                rate_limit=12  # requests per second
            ),
            "nyt": NewsSource(
                name="New York Times",
                api_key=NYT_API_KEY,
                base_url="https://api.nytimes.com/svc",
                rate_limit=10
            )
        }
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def rate_limit_check(self, source: str):
        """Check and enforce rate limits"""
        source_config = self.sources[source]
        current_time = time.time()
        time_since_last = current_time - source_config.last_request
        min_interval = 1.0 / source_config.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        source_config.last_request = time.time()
    
    async def search_guardian(
        self,
        query: Optional[str] = None,
        section: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        page_size: int = 20
    ) -> List[Dict]:
        """Search Guardian API"""
        await self.rate_limit_check("guardian")
        
        params = {
            'api-key': self.sources["guardian"].api_key,
            'show-fields': 'headline,byline,standfirst,body,wordcount',
            'show-tags': 'all',
            'page-size': str(page_size),
            'order-by': 'newest'
        }
        
        if query:
            params['q'] = query
        if section:
            params['section'] = section
        if from_date:
            params['from-date'] = from_date
        if to_date:
            params['to-date'] = to_date
        
        url = f"{self.sources['guardian'].base_url}/search"
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('response', {}).get('status') == 'ok':
                return data['response'].get('results', [])
            return []
        except Exception as e:
            logger.error(f"Guardian API error: {e}")
            return []
    
    async def search_nyt(
        self,
        query: Optional[str] = None,
        section: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        page_size: int = 20
    ) -> List[Dict]:
        """Search New York Times API"""
        await self.rate_limit_check("nyt")
        
        # NYT Article Search API
        params = {
            'api-key': self.sources["nyt"].api_key,
            'page': 0,
            'sort': 'newest'
        }
        
        # Build filter query
        fq_parts = []
        if section:
            fq_parts.append(f'section_name:("{section}")')
        if from_date:
            fq_parts.append(f'pub_date:[{from_date}T00:00:00Z TO *]')
        if to_date:
            fq_parts.append(f'pub_date:[* TO {to_date}T23:59:59Z]')
        
        if fq_parts:
            params['fq'] = ' AND '.join(fq_parts)
        
        if query:
            params['q'] = query
        
        url = f"{self.sources['nyt'].base_url}/search/v2/articlesearch.json"
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get('response', {}).get('docs', [])[:page_size]
        except Exception as e:
            logger.error(f"NYT API error: {e}")
            return []
    
    async def search_all_sources(
        self,
        query: Optional[str] = None,
        section: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        page_size: int = 20
    ) -> Dict[str, List[Dict]]:
        """Search all news sources concurrently"""
        tasks = {
            "guardian": self.search_guardian(query, section, from_date, to_date, page_size // 2),
            "nyt": self.search_nyt(query, section, from_date, to_date, page_size // 2)
        }
        
        results = {}
        for source, task in tasks.items():
            try:
                results[source] = await task
            except Exception as e:
                logger.error(f"Error fetching from {source}: {e}")
                results[source] = []
        
        return results

# Advanced AI Analyzer
try:
    from emergentintegrations.llm.chat import LlmChat, UserMessage
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI integrations not available")

class AdvancedAIAnalyzer:
    def __init__(self):
        if not AI_AVAILABLE:
            raise HTTPException(status_code=500, detail="AI integrations not available")
        
        # Claude for deep analysis
        self.claude_chat = LlmChat(
            api_key=ANTHROPIC_API_KEY,
            session_id=f"claude-advanced-{uuid.uuid4()}",
            system_message="""You are an expert news analyst and data scientist specializing in global event relationships, geographic analysis, and temporal patterns.

Your tasks:
1. Identify story relationships with high confidence scores
2. Analyze geographic overlaps and regional impacts
3. Detect temporal patterns and causation chains
4. Assess evidence quality and connection strength
5. Provide detailed explanations for complex relationships

Return structured JSON with enhanced metadata including confidence scores, geographic analysis, and temporal relationships."""
        ).with_model("anthropic", "claude-sonnet-4-20250514").with_max_tokens(2000)
        
        # GPT for content generation
        self.gpt_chat = LlmChat(
            api_key=OPENAI_API_KEY,
            session_id=f"gpt-content-{uuid.uuid4()}",
            system_message="""You are a world-class content strategist and editor who creates compelling, accessible news content at different complexity levels.

Create content that:
1. Adapts to user's complexity preference (1-5 scale)
2. Includes sentiment analysis
3. Estimates reading time
4. Provides engaging social media hooks
5. Identifies key entities and relationships

Always return structured JSON with all requested fields."""
        ).with_model("openai", "gpt-4o").with_max_tokens(1500)
    
    async def analyze_advanced_connections(self, stories: List[Dict], user_prefs: UserPreferences) -> List[AdvancedConnection]:
        """Advanced connection analysis with geographic and temporal factors"""
        if len(stories) < 2:
            return []
        
        # Enhanced story preparation with geographic and temporal context
        story_data = []
        for story in stories[:8]:  # Increased limit for advanced analysis
            content = self._extract_content(story)
            
            story_info = {
                "id": story.get("id", ""),
                "title": story.get("webTitle", "") or story.get("headline", {}).get("main", ""),
                "content": content[:800],
                "section": story.get("sectionName", "") or story.get("section_name", ""),
                "source": story.get("source", "guardian"),
                "pub_date": story.get("webPublicationDate", "") or story.get("pub_date", ""),
                "keywords": self._extract_keywords(story),
                "location_mentions": self._extract_locations(content)
            }
            story_data.append(story_info)
        
        analysis_prompt = f"""Analyze these {len(story_data)} news stories for advanced relationships:

{json.dumps(story_data, indent=2)}

User preferences: complexity_level={user_prefs.complexity_level}, geographic_focus={user_prefs.geographic_focus}

Return JSON array with enhanced connection analysis:
[{{
  "source_id": "id1",
  "target_id": "id2", 
  "connection_type": "economic|political|social|environmental|causal|thematic|geographic",
  "strength": 0.3-1.0,
  "confidence": 0.3-1.0,
  "explanation": "detailed explanation",
  "keywords": ["key1", "key2"],
  "geographic_overlap": {{"country": "if applicable", "region": "if applicable"}},
  "temporal_relationship": "causation|correlation|sequence|concurrent",
  "evidence_score": 0.3-1.0
}}]

Focus on high-confidence connections (>0.4). Maximum 5 connections. Consider user's complexity preference for explanation detail."""

        try:
            user_message = UserMessage(text=analysis_prompt)
            response = await self.claude_chat.send_message(user_message)
            
            response_text = response.strip()
            logger.info(f"Claude advanced analysis: {len(response_text)} chars")
            
            # Parse JSON response
            if response_text.startswith('['):
                connections_data = json.loads(response_text)
            else:
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    connections_data = json.loads(json_match.group())
                else:
                    return []
            
            # Create enhanced connections
            connections = []
            for conn in connections_data:
                if all(key in conn for key in ["source_id", "target_id", "connection_type", "strength", "confidence"]):
                    geographic_info = None
                    if conn.get("geographic_overlap"):
                        geo_data = conn["geographic_overlap"]
                        geographic_info = GeographicInfo(
                            country=geo_data.get("country"),
                            region=geo_data.get("region")
                        )
                    
                    connections.append(AdvancedConnection(
                        source_id=conn["source_id"],
                        target_id=conn["target_id"],
                        connection_type=conn["connection_type"],
                        strength=float(conn["strength"]),
                        confidence=float(conn["confidence"]),
                        explanation=conn.get("explanation", "AI-detected connection"),
                        keywords=conn.get("keywords", []),
                        geographic_overlap=geographic_info,
                        temporal_relationship=conn.get("temporal_relationship"),
                        evidence_score=float(conn.get("evidence_score", 0.5))
                    ))
            
            logger.info(f"Generated {len(connections)} advanced connections")
            return connections
            
        except Exception as e:
            logger.error(f"Advanced connection analysis error: {e}")
            return []
    
    async def create_enhanced_content(self, story: Dict, complexity_level: int = 3) -> Dict[str, Any]:
        """Create enhanced content with complexity adaptation"""
        title = story.get("webTitle", "") or story.get("headline", {}).get("main", "")
        content = self._extract_content(story)
        
        complexity_prompt = f"""Transform this news story for complexity level {complexity_level}/5:

Title: {title}
Content: {content[:1200]}

Return JSON:
{{
  "summary": "summary adapted to complexity level {complexity_level}",
  "lede": "engaging opening",
  "nutgraf": "why this matters - depth based on complexity level",
  "engagement_preview": "social media hook with emojis",
  "sentiment_score": -1.0 to 1.0,
  "complexity_adapted": true,
  "read_time_minutes": estimated_reading_time,
  "key_entities": ["entity1", "entity2"],
  "related_topics": ["topic1", "topic2"]
}}

Complexity levels:
1 = Simple headlines and basic facts
2 = Basic explanation with context
3 = Moderate detail with analysis
4 = In-depth analysis with background
5 = Expert-level with theories and implications"""

        try:
            user_message = UserMessage(text=complexity_prompt)
            response = await self.gpt_chat.send_message(user_message)
            
            response_text = response.strip()
            
            if not response_text.startswith('{'):
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group()
                else:
                    raise ValueError("No JSON found in response")
            
            parsed_response = json.loads(response_text)
            
            # Ensure all fields are present
            defaults = {
                "summary": title,
                "lede": title,
                "nutgraf": "This story provides important insights into current events.",
                "engagement_preview": f"📰 {title[:250]}",
                "sentiment_score": 0.0,
                "complexity_adapted": True,
                "read_time_minutes": max(1, len(content.split()) // 200),
                "key_entities": [],
                "related_topics": []
            }
            
            for key, default_value in defaults.items():
                if key not in parsed_response:
                    parsed_response[key] = default_value
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Enhanced content creation error: {e}")
            return {
                "summary": title,
                "lede": title,
                "nutgraf": "This story provides insights into current events.",
                "engagement_preview": f"📰 {title[:250]}",
                "sentiment_score": 0.0,
                "complexity_adapted": False,
                "read_time_minutes": 2,
                "key_entities": [],
                "related_topics": []
            }
    
    def _extract_content(self, story: Dict) -> str:
        """Extract content from different source formats"""
        if "fields" in story:  # Guardian format
            return story.get("fields", {}).get("body", "")
        elif "lead_paragraph" in story:  # NYT format
            return story.get("lead_paragraph", "") + " " + story.get("snippet", "")
        return ""
    
    def _extract_keywords(self, story: Dict) -> List[str]:
        """Extract keywords from story"""
        keywords = []
        if "tags" in story:  # Guardian
            keywords = [tag.get("webTitle", "") for tag in story["tags"][:3]]
        elif "keywords" in story:  # NYT
            keywords = [kw.get("value", "") for kw in story["keywords"][:3]]
        return keywords
    
    def _extract_locations(self, content: str) -> List[str]:
        """Basic location extraction"""
        import re
        # Simple pattern for location extraction
        locations = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+)?\b', content[:500])
        return list(set(locations[:5]))

# Initialize AI analyzer
ai_analyzer = None
if AI_AVAILABLE:
    ai_analyzer = AdvancedAIAnalyzer()

# Advanced News Processor
class AdvancedNewsProcessor:
    def __init__(self):
        self.cache_ttl = 1800  # 30 minutes
    
    async def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result"""
        if redis_client:
            try:
                result = await redis_client.get(cache_key)
                if result:
                    return json.loads(result)
            except Exception:
                pass
        return cache.get(cache_key)
    
    async def set_cached_result(self, cache_key: str, data: Dict):
        """Set cached result"""
        if redis_client:
            try:
                await redis_client.setex(cache_key, self.cache_ttl, json.dumps(data))
            except Exception:
                pass
        cache[cache_key] = data
    
    async def process_multi_source_stories(
        self, 
        guardian_stories: List[Dict], 
        nyt_stories: List[Dict],
        user_prefs: UserPreferences
    ) -> List[EnhancedStory]:
        """Process stories from multiple sources"""
        all_stories = []
        
        # Process Guardian stories
        for story in guardian_stories:
            enhanced_story = await self._process_guardian_story(story, user_prefs)
            if enhanced_story:
                all_stories.append(enhanced_story)
        
        # Process NYT stories
        for story in nyt_stories:
            enhanced_story = await self._process_nyt_story(story, user_prefs)
            if enhanced_story:
                all_stories.append(enhanced_story)
        
        return all_stories
    
    async def _process_guardian_story(self, story: Dict, user_prefs: UserPreferences) -> Optional[EnhancedStory]:
        """Process Guardian story format"""
        try:
            if not ai_analyzer:
                return None
            
            enhanced_content = await ai_analyzer.create_enhanced_content(story, user_prefs.complexity_level)
            
            return EnhancedStory(
                id=story.get("id", ""),
                source="guardian",
                title=story.get("webTitle", ""),
                summary=enhanced_content["summary"],
                lede=enhanced_content["lede"],
                nutgraf=enhanced_content["nutgraf"],
                section=story.get("sectionName", ""),
                publication_date=datetime.fromisoformat(story.get("webPublicationDate", "").replace("Z", "+00:00")),
                url=story.get("webUrl", ""),
                author=story.get("fields", {}).get("byline"),
                entities=enhanced_content["key_entities"],
                categories=enhanced_content["related_topics"],
                engagement_preview=enhanced_content["engagement_preview"],
                sentiment_score=enhanced_content["sentiment_score"],
                complexity_level=user_prefs.complexity_level,
                read_time_minutes=enhanced_content["read_time_minutes"]
            )
        except Exception as e:
            logger.error(f"Error processing Guardian story: {e}")
            return None
    
    async def _process_nyt_story(self, story: Dict, user_prefs: UserPreferences) -> Optional[EnhancedStory]:
        """Process NYT story format"""
        try:
            if not ai_analyzer:
                return None
            
            enhanced_content = await ai_analyzer.create_enhanced_content(story, user_prefs.complexity_level)
            
            # Extract NYT-specific data
            headline = story.get("headline", {})
            title = headline.get("main", "") if headline else ""
            
            byline = story.get("byline", {})
            author = byline.get("original", "") if byline else None
            
            return EnhancedStory(
                id=story.get("_id", ""),
                source="nyt",
                title=title,
                summary=enhanced_content["summary"],
                lede=enhanced_content["lede"],
                nutgraf=enhanced_content["nutgraf"],
                section=story.get("section_name", ""),
                publication_date=datetime.fromisoformat(story.get("pub_date", "").replace("Z", "+00:00")),
                url=story.get("web_url", ""),
                author=author,
                entities=enhanced_content["key_entities"],
                categories=enhanced_content["related_topics"],
                engagement_preview=enhanced_content["engagement_preview"],
                sentiment_score=enhanced_content["sentiment_score"],
                complexity_level=user_prefs.complexity_level,
                read_time_minutes=enhanced_content["read_time_minutes"]
            )
        except Exception as e:
            logger.error(f"Error processing NYT story: {e}")
            return None
    
    async def create_advanced_knowledge_graph(
        self,
        stories: List[EnhancedStory],
        raw_stories: List[Dict],
        user_prefs: UserPreferences
    ) -> Dict[str, Any]:
        """Create advanced knowledge graph with multi-source intelligence"""
        
        # Create enhanced nodes
        nodes = []
        for story in stories:
            node = {
                "id": story.id,
                "type": "article",
                "source": story.source,
                "title": story.title,
                "summary": story.summary,
                "lede": story.lede,
                "nutgraf": story.nutgraf,
                "section": story.section,
                "publication_date": story.publication_date.isoformat(),
                "url": story.url,
                "author": story.author,
                "engagement_preview": story.engagement_preview,
                "sentiment_score": story.sentiment_score,
                "complexity_level": story.complexity_level,
                "read_time_minutes": story.read_time_minutes,
                "size": 15 + (story.complexity_level * 3) + (len(story.categories) * 2),
                "color": self._get_source_color(story.source, story.section),
                "entities": story.entities,
                "categories": story.categories
            }
            nodes.append(node)
        
        # Create section and source nodes
        sections = list(set(story.section for story in stories))
        sources = list(set(story.source for story in stories))
        
        for section in sections:
            nodes.append({
                "id": f"section_{section.lower().replace(' ', '_')}",
                "type": "section",
                "title": section,
                "size": 30,
                "color": self._get_section_color(section)
            })
        
        for source in sources:
            nodes.append({
                "id": f"source_{source}",
                "type": "source",
                "title": source.upper(),
                "size": 40,
                "color": self._get_source_color(source)
            })
        
        # Generate advanced connections
        edges = []
        if ai_analyzer and len(raw_stories) > 1:
            connections = await ai_analyzer.analyze_advanced_connections(raw_stories, user_prefs)
            
            for connection in connections:
                edge = {
                    "source": connection.source_id,
                    "target": connection.target_id,
                    "type": connection.connection_type,
                    "strength": connection.strength,
                    "confidence": connection.confidence,
                    "explanation": connection.explanation,
                    "keywords": connection.keywords,
                    "evidence_score": connection.evidence_score,
                    "width": max(2, connection.strength * 10),
                    "opacity": 0.2 + (connection.confidence * 0.8),
                    "temporal_relationship": connection.temporal_relationship
                }
                
                if connection.geographic_overlap:
                    edge["geographic_overlap"] = connection.geographic_overlap.dict()
                
                edges.append(edge)
        
        # Add structural edges (article-section, article-source)
        for story in stories:
            # Article to section
            edges.append({
                "source": story.id,
                "target": f"section_{story.section.lower().replace(' ', '_')}",
                "type": "belongs_to_section",
                "strength": 1.0,
                "width": 2,
                "opacity": 0.3
            })
            
            # Article to source
            edges.append({
                "source": story.id,
                "target": f"source_{story.source}",
                "type": "belongs_to_source",
                "strength": 1.0,
                "width": 2,
                "opacity": 0.3
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_articles": len(stories),
                "total_sections": len(sections),
                "total_sources": len(sources),
                "total_connections": len([e for e in edges if e["type"] not in ["belongs_to_section", "belongs_to_source"]]),
                "generated_at": datetime.now().isoformat(),
                "ai_analysis_enabled": ai_analyzer is not None,
                "user_preferences": user_prefs.dict(),
                "advanced_features": {
                    "multi_source": True,
                    "geographic_analysis": True,
                    "temporal_analysis": True,
                    "sentiment_analysis": True,
                    "complexity_adaptation": True,
                    "confidence_scoring": True
                }
            }
        }
    
    def _get_source_color(self, source: str, section: str = "") -> str:
        """Get color based on source and section"""
        source_colors = {
            "guardian": "#3498db",
            "nyt": "#2c3e50"
        }
        return source_colors.get(source, self._get_section_color(section))
    
    def _get_section_color(self, section: str) -> str:
        """Get color for section"""
        colors = {
            "world": "#e74c3c",
            "politics": "#3498db", 
            "business": "#f39c12",
            "technology": "#9b59b6",
            "sport": "#27ae60",
            "culture": "#e67e22",
            "science": "#1abc9c",
            "environment": "#2ecc71",
            "education": "#34495e",
            "society": "#95a5a6"
        }
        return colors.get(section.lower(), "#7f8c8d")

# Initialize processor
news_processor = AdvancedNewsProcessor()

# Analytics and Tracking
class AnalyticsManager:
    def __init__(self):
        self.session_data = defaultdict(list)
    
    async def track_event(self, analytics_data: AnalyticsData):
        """Track user analytics event"""
        try:
            # Store in database
            await db.analytics.insert_one(analytics_data.dict())
            
            # Update session data
            self.session_data[analytics_data.session_id].append(analytics_data)
            
            logger.info(f"Analytics: {analytics_data.action} - {analytics_data.session_id}")
        except Exception as e:
            logger.error(f"Analytics tracking error: {e}")
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session analytics summary"""
        try:
            events = await db.analytics.find({"session_id": session_id}).to_list(length=None)
            
            return {
                "session_id": session_id,
                "total_events": len(events),
                "unique_stories_viewed": len(set(e.get("story_id") for e in events if e.get("story_id"))),
                "connections_explored": len([e for e in events if e["action"] == "explore_connection"]),
                "session_duration": self._calculate_duration(events),
                "most_active_section": self._get_most_active_section(events)
            }
        except Exception as e:
            logger.error(f"Session summary error: {e}")
            return {}
    
    def _calculate_duration(self, events: List[Dict]) -> int:
        """Calculate session duration in minutes"""
        if len(events) < 2:
            return 0
        
        first_event = min(events, key=lambda x: x["timestamp"])
        last_event = max(events, key=lambda x: x["timestamp"])
        
        duration = (last_event["timestamp"] - first_event["timestamp"]).seconds // 60
        return duration
    
    def _get_most_active_section(self, events: List[Dict]) -> str:
        """Get most frequently viewed section"""
        section_counts = defaultdict(int)
        for event in events:
            if event.get("metadata", {}).get("section"):
                section_counts[event["metadata"]["section"]] += 1
        
        return max(section_counts.items(), key=lambda x: x[1])[0] if section_counts else "unknown"

analytics_manager = AnalyticsManager()

# API Endpoints
@app.get("/api/v3/")
async def root():
    return {
        "message": "News Knowledge Graph API - Production Scale v3.0",
        "version": "3.0.0",
        "features": {
            "multi_source_integration": ["guardian", "nyt"],
            "advanced_ai_analysis": True,
            "geographic_analysis": True,
            "temporal_analysis": True,
            "sentiment_analysis": True,
            "complexity_adaptation": True,
            "real_time_analytics": True,
            "caching_layer": redis_client is not None,
            "user_preferences": True
        },
        "endpoints": {
            "advanced_graph": "/api/v3/knowledge-graph/advanced",
            "multi_source": "/api/v3/news/multi-source",
            "user_preferences": "/api/v3/user/preferences",
            "analytics": "/api/v3/analytics",
            "feedback": "/api/v3/feedback/enhanced"
        }
    }

@app.get("/api/v3/knowledge-graph/advanced")
async def get_advanced_knowledge_graph(
    background_tasks: BackgroundTasks,
    days: int = Query(default=3, ge=1, le=14),
    sources: str = Query(default="guardian,nyt", description="Comma-separated list of sources"),
    section: Optional[str] = Query(default=None),
    complexity_level: int = Query(default=3, ge=1, le=5),
    geographic_focus: Optional[str] = Query(default=None),
    max_articles: int = Query(default=20, ge=5, le=50),
    session_id: str = Query(default_factory=lambda: str(uuid.uuid4()))
):
    """Generate advanced knowledge graph with multi-source intelligence"""
    
    # Create cache key
    cache_key = f"advanced_graph:{days}:{sources}:{section}:{complexity_level}:{geographic_focus}:{max_articles}"
    
    # Check cache first
    cached_result = await news_processor.get_cached_result(cache_key)
    if cached_result:
        # Track analytics
        background_tasks.add_task(
            analytics_manager.track_event,
            AnalyticsData(
                session_id=session_id,
                action="view_cached_graph",
                metadata={"cache_hit": True, "sources": sources}
            )
        )
        return cached_result
    
    try:
        # User preferences
        user_prefs = UserPreferences(
            complexity_level=complexity_level,
            geographic_focus=geographic_focus
        )
        
        # Date range
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Multi-source data fetching
        source_list = sources.split(',')
        async with MultiSourceNewsClient() as client:
            source_results = await client.search_all_sources(
                section=section,
                from_date=from_date,
                to_date=to_date,
                page_size=max_articles
            )
        
        # Combine raw stories for AI analysis
        raw_stories = []
        guardian_stories = []
        nyt_stories = []
        
        if 'guardian' in source_list:
            guardian_stories = source_results.get('guardian', [])
            raw_stories.extend(guardian_stories)
        
        if 'nyt' in source_list:
            nyt_stories = source_results.get('nyt', [])
            raw_stories.extend(nyt_stories)
        
        # Process stories
        processed_stories = await news_processor.process_multi_source_stories(
            guardian_stories, nyt_stories, user_prefs
        )
        
        # Create advanced knowledge graph
        knowledge_graph = await news_processor.create_advanced_knowledge_graph(
            processed_stories, raw_stories, user_prefs
        )
        
        # Cache result
        await news_processor.set_cached_result(cache_key, knowledge_graph)
        
        # Track analytics
        background_tasks.add_task(
            analytics_manager.track_event,
            AnalyticsData(
                session_id=session_id,
                action="generate_advanced_graph",
                metadata={
                    "sources_used": source_list,
                    "total_stories": len(processed_stories),
                    "complexity_level": complexity_level,
                    "cache_miss": True
                }
            )
        )
        
        return knowledge_graph
        
    except Exception as e:
        logger.error(f"Advanced knowledge graph error: {e}")
        # Fallback to demo data
        return await get_production_demo_graph()

@app.get("/api/v3/demo/production")
async def get_production_demo_graph():
    """Production-ready demo with multi-source simulation"""
    # This would include more sophisticated demo data showing multi-source integration
    # For now, return enhanced version of existing demo
    return {
        "nodes": [
            # Enhanced demo nodes with multi-source indicators
            {
                "id": "guardian-climate-policy",
                "type": "article",
                "source": "guardian",
                "title": "EU announces groundbreaking climate policy framework",
                "summary": "The European Union unveiled comprehensive climate legislation affecting global trade partnerships and environmental standards across multiple sectors.",
                "lede": "Europe's bold climate move reshapes global environmental policy landscape with unprecedented regulatory scope.",
                "nutgraf": "This policy framework represents the most significant environmental legislation since the Paris Agreement, with implications for international trade, energy markets, and technological innovation worldwide.",
                "section": "Environment",
                "publication_date": "2025-08-02T06:00:00Z",
                "url": "https://example.com/eu-climate-policy",
                "sentiment_score": 0.3,
                "complexity_level": 4,
                "read_time_minutes": 6,
                "size": 28,
                "color": "#27ae60",
                "entities": ["European Union", "Climate Change", "Trade Policy"],
                "engagement_preview": "🌍 BREAKING: EU's revolutionary climate policy changes everything! New framework affects global trade, energy markets, and innovation. Here's what it means 🧵 #ClimatePolicy #EU"
            },
            {
                "id": "nyt-us-economy",
                "type": "article", 
                "source": "nyt",
                "title": "Federal Reserve signals cautious approach to interest rates amid global uncertainty",
                "summary": "Fed officials indicated a measured response to international economic pressures, considering impacts from European climate policies and emerging market volatility on US monetary policy decisions.",
                "lede": "The Federal Reserve navigates complex global economic signals as international policy changes create new challenges for US monetary strategy.",
                "nutgraf": "Fed policymakers are grappling with unprecedented interconnections between environmental policy, international trade, and domestic monetary policy, requiring sophisticated analysis of global economic relationships.",
                "section": "Business",
                "publication_date": "2025-08-02T08:30:00Z",
                "url": "https://example.com/fed-rates-global",
                "sentiment_score": -0.1,
                "complexity_level": 4,
                "read_time_minutes": 8,
                "size": 32,
                "color": "#2c3e50",
                "entities": ["Federal Reserve", "Interest Rates", "Global Economy"],
                "engagement_preview": "🏦 Fed walks tightrope on rates as global forces reshape monetary policy. Climate policies + trade tensions = complex decisions ahead 📊 #FederalReserve #Economy"
            },
            # Source nodes
            {
                "id": "source_guardian",
                "type": "source",
                "title": "GUARDIAN",
                "size": 45,
                "color": "#3498db"
            },
            {
                "id": "source_nyt",
                "type": "source", 
                "title": "NYT",
                "size": 45,
                "color": "#2c3e50"
            }
        ],
        "edges": [
            {
                "source": "guardian-climate-policy",
                "target": "nyt-us-economy",
                "type": "economic",
                "strength": 0.8,
                "confidence": 0.9,
                "explanation": "EU climate policies create international trade pressures that directly influence US Federal Reserve monetary policy considerations and interest rate decisions.",
                "keywords": ["climate policy", "monetary policy", "international trade", "economic interconnection"],
                "evidence_score": 0.85,
                "width": 8,
                "opacity": 0.9,
                "temporal_relationship": "causation"
            },
            {
                "source": "guardian-climate-policy",
                "target": "source_guardian",
                "type": "belongs_to_source",
                "strength": 1.0,
                "width": 2,
                "opacity": 0.3
            },
            {
                "source": "nyt-us-economy",
                "target": "source_nyt",
                "type": "belongs_to_source",
                "strength": 1.0,
                "width": 2,
                "opacity": 0.3
            }
        ],
        "metadata": {
            "total_articles": 2,
            "total_sources": 2,
            "total_connections": 1,
            "generated_at": datetime.now().isoformat(),
            "ai_analysis_enabled": True,
            "demo_mode": True,
            "production_features": {
                "multi_source_integration": True,
                "advanced_ai_analysis": True,
                "confidence_scoring": True,
                "evidence_assessment": True,
                "temporal_analysis": True
            }
        }
    }

@app.post("/api/v3/feedback/enhanced")
async def submit_enhanced_feedback(feedback: FeedbackData, background_tasks: BackgroundTasks):
    """Enhanced feedback collection with detailed analytics"""
    try:
        feedback_doc = feedback.dict()
        feedback_doc["id"] = str(uuid.uuid4())
        feedback_doc["timestamp"] = datetime.now()
        
        # Store in database
        await db.enhanced_feedback.insert_one(feedback_doc)
        
        # Track analytics
        background_tasks.add_task(
            analytics_manager.track_event,
            AnalyticsData(
                session_id=feedback.session_id or "unknown",
                action="submit_enhanced_feedback",
                metadata={
                    "rating": feedback.rating,
                    "features_used": feedback.features_used,
                    "has_suggestions": bool(feedback.suggested_improvements)
                }
            )
        )
        
        logger.info(f"Enhanced feedback: {feedback.rating}/10, features: {feedback.features_used}")
        
        return {
            "status": "success",
            "message": "Thank you for your detailed feedback! Your insights help us improve the platform.",
            "feedback_id": feedback_doc["id"],
            "next_steps": "We'll analyze your suggestions and incorporate improvements in future updates."
        }
    except Exception as e:
        logger.error(f"Enhanced feedback error: {e}")
        return {
            "status": "noted",
            "message": "Thanks for your feedback! (Demo mode)"
        }

@app.get("/api/v3/analytics/session/{session_id}")
async def get_session_analytics(session_id: str):
    """Get analytics for a specific session"""
    summary = await analytics_manager.get_session_summary(session_id)
    return summary

@app.get("/api/health/detailed")
async def detailed_health_check():
    """Detailed health check for production monitoring"""
    health_status = {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ai_services": AI_AVAILABLE,
            "database": "connected",
            "cache": "redis" if redis_client else "memory"
        },
        "api_sources": {
            "guardian": bool(GUARDIAN_API_KEY),
            "nyt": bool(NYT_API_KEY and NYT_API_KEY != "your-nyt-api-key-here")
        },
        "features": {
            "multi_source_integration": True,
            "advanced_analytics": True,
            "real_time_caching": redis_client is not None,
            "geographic_analysis": True,
            "temporal_analysis": True,
            "sentiment_analysis": True
        }
    }
    
    # Test database connection
    try:
        await db.admin.command("ping")
        health_status["services"]["database"] = "connected"
    except Exception:
        health_status["services"]["database"] = "error"
        health_status["status"] = "degraded"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")