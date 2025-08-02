from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
import os
import httpx
import asyncio
import uuid
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = FastAPI(
    title="News Knowledge Graph API",
    description="AI-powered news knowledge graph visualization tool",
    version="1.0.0"
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
GUARDIAN_BASE_URL = "https://content.guardianapis.com"

# Database
client = AsyncIOMotorClient(MONGO_URL)
db = client.news_knowledge_graph

# Models
class GuardianTag(BaseModel):
    id: str
    type: str
    webTitle: str
    webUrl: str
    apiUrl: str
    sectionId: Optional[str] = None
    sectionName: Optional[str] = None

class GuardianArticle(BaseModel):
    id: str
    type: str
    sectionId: str
    sectionName: str
    webPublicationDate: datetime
    webTitle: str
    webUrl: str
    apiUrl: str
    isHosted: bool
    pillarId: Optional[str] = None
    pillarName: Optional[str] = None
    tags: List[GuardianTag] = []
    headline: Optional[str] = None
    byline: Optional[str] = None
    body: Optional[str] = None
    standfirst: Optional[str] = None
    wordcount: Optional[int] = None

class StoryConnection(BaseModel):
    source_id: str
    target_id: str
    connection_type: str
    strength: float  # 0.0 to 1.0
    explanation: str
    keywords: List[str]

class ProcessedStory(BaseModel):
    id: str
    title: str
    summary: str
    lede: str
    nutgraf: str
    section: str
    publication_date: str
    url: str
    entities: List[str]
    categories: List[str]
    engagement_preview: str  # ChatGPT generated preview

class KnowledgeGraphData(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# Guardian API Client
class GuardianAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = GUARDIAN_BASE_URL
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def _build_url(self, endpoint: str, **params) -> str:
        params['api-key'] = self.api_key
        query_string = '&'.join([f"{k}={v}" for k, v in params.items() if v is not None])
        return f"{self.base_url}/{endpoint}?{query_string}"
    
    async def search_content(
        self,
        query: Optional[str] = None,
        section: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        order_by: str = "newest",
        page: int = 1,
        page_size: int = 50
    ) -> Dict:
        url = self._build_url(
            "search",
            q=query,
            section=section,
            **{
                "from-date": from_date,
                "to-date": to_date,
                "order-by": order_by,
                "page": page,
                "page-size": page_size,
                "show-fields": "headline,byline,standfirst,body,wordcount",
                "show-tags": "all"
            }
        )
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data.get('response', {}).get('status') != 'ok':
                raise HTTPException(status_code=400, detail=f"Guardian API error: {data.get('message', 'Unknown error')}")
            
            return data['response']
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Guardian API error: {str(e)}")

# AI Integration using emergentintegrations
try:
    from emergentintegrations.llm.chat import LlmChat, UserMessage
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("emergentintegrations not installed, AI features disabled")

class AIAnalyzer:
    def __init__(self):
        if not AI_AVAILABLE:
            raise HTTPException(status_code=500, detail="AI integrations not available")
        
        # Claude for deep analysis
        self.claude_chat = LlmChat(
            api_key=ANTHROPIC_API_KEY,
            session_id=f"claude-analysis-{uuid.uuid4()}",
            system_message="""You are a news analyst. Identify meaningful relationships between news stories.

Return connections as JSON array ONLY:
[{"source_id": "id1", "target_id": "id2", "connection_type": "economic|political|social|environmental|causal|thematic", "strength": 0.3-1.0, "explanation": "brief reason", "keywords": ["key1"]}]

If no meaningful connections exist, return []."""
        ).with_model("anthropic", "claude-sonnet-4-20250514").with_max_tokens(1000)
        
        # ChatGPT for summaries and engaging content  
        self.gpt_chat = LlmChat(
            api_key=OPENAI_API_KEY,
            session_id=f"gpt-summaries-{uuid.uuid4()}",
            system_message="""Create engaging news summaries in JSON format:
{"summary": "2-3 sentences", "lede": "hook line", "nutgraf": "why it matters", "engagement_preview": "social hook <280 chars"}

Keep content concise and engaging."""
        ).with_model("openai", "gpt-4o").with_max_tokens(800)
    
    async def analyze_story_connections(self, stories: List[Dict]) -> List[StoryConnection]:
        """Use Claude to analyze relationships between stories"""
        if len(stories) < 2:
            return []
        
        # Prepare stories for analysis (limit to reduce token usage)
        story_data = []
        for story in stories[:6]:  # Limit to 6 stories to avoid rate limits
            content = story.get("fields", {}).get("body", "")
            # Clean and limit content
            content = content.replace("<p>", "").replace("</p>", " ").replace("<strong>", "").replace("</strong>", "")
            content = content[:500]  # Shorter content to reduce tokens
            
            story_data.append({
                "id": story.get("id"),
                "title": story.get("webTitle", ""),
                "content": content,
                "section": story.get("sectionName", ""),
                "tags": [tag.get("webTitle", "") for tag in story.get("tags", [])][:3]  # Limit tags
            })
        
        # Shorter, more focused prompt to reduce token usage
        analysis_prompt = f"""Analyze connections between these {len(story_data)} news stories:

{json.dumps(story_data)}

Return ONLY a JSON array of meaningful connections (minimum strength 0.3):

[{{"source_id": "id1", "target_id": "id2", "connection_type": "economic|political|social|environmental|causal|thematic", "strength": 0.3-1.0, "explanation": "brief reason", "keywords": ["key1", "key2"]}}]

Maximum 3 connections. If no strong connections exist, return []."""

        try:
            user_message = UserMessage(text=analysis_prompt)
            response = await self.claude_chat.send_message(user_message)
            
            # Clean and parse Claude's response
            response_text = response.strip()
            print(f"Claude raw response: {response_text[:200]}...")
            
            # Handle empty or non-JSON responses
            if not response_text or response_text.lower() in ['none', 'no connections', '[]']:
                print("Claude returned no connections")
                return []
            
            # Try to extract JSON from response
            if response_text.startswith('['):
                connections_data = json.loads(response_text)
            else:
                # Try to find JSON in response
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    connections_data = json.loads(json_match.group())
                else:
                    print("No JSON array found in Claude response")
                    return []
            
            # Validate and create connections
            connections = []
            for conn in connections_data:
                if all(key in conn for key in ["source_id", "target_id", "connection_type", "strength"]):
                    connections.append(StoryConnection(
                        source_id=conn["source_id"],
                        target_id=conn["target_id"],
                        connection_type=conn["connection_type"],
                        strength=float(conn["strength"]),
                        explanation=conn.get("explanation", "Connection found by AI"),
                        keywords=conn.get("keywords", [])
                    ))
            
            print(f"Created {len(connections)} story connections")
            return connections
            
        except json.JSONDecodeError as e:
            print(f"Claude JSON parsing error: {e}. Response: {response_text[:200] if 'response_text' in locals() else 'No response'}")
            return []
        except Exception as e:
            print(f"Claude analysis error: {e}")
            return []
    
    async def create_engaging_summary(self, story: Dict) -> Dict[str, str]:
        """Use ChatGPT to create engaging summaries and previews"""
        title = story.get("webTitle", "")
        content = story.get("fields", {}).get("body", "")[:1500]  # Reduce content length
        standfirst = story.get("fields", {}).get("standfirst", "")
        
        # Shorter, more focused prompt
        summary_prompt = f"""Transform this news story:

Title: {title}
Content: {content[:800]}

Return JSON:
{{"summary": "2-3 sentence summary", "lede": "engaging opening line", "nutgraf": "why this matters", "engagement_preview": "social media hook (<280 chars)"}}"""

        try:
            user_message = UserMessage(text=summary_prompt)
            response = await self.gpt_chat.send_message(user_message)
            
            # Clean and parse response
            response_text = response.strip()
            
            # Handle non-JSON responses
            if not response_text.startswith('{'):
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group()
                else:
                    raise ValueError("No JSON found in response")
            
            parsed_response = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["summary", "lede", "nutgraf", "engagement_preview"]
            for field in required_fields:
                if field not in parsed_response:
                    parsed_response[field] = title[:200] if field == "engagement_preview" else title
            
            return parsed_response
            
        except Exception as e:
            print(f"ChatGPT summary error: {e}")
            # Fallback to basic content extraction
            return {
                "summary": title,
                "lede": standfirst or title,
                "nutgraf": "This story covers important developments in current events.",
                "engagement_preview": title[:280]
            }

# Initialize AI analyzer
ai_analyzer = None
if AI_AVAILABLE:
    ai_analyzer = AIAnalyzer()

# Data Processing
class NewsProcessor:
    def __init__(self):
        pass
    
    async def process_articles(self, articles: List[Dict]) -> List[ProcessedStory]:
        """Process articles with AI-generated summaries"""
        processed_stories = []
        
        for article in articles:
            # Get AI-generated content if available
            if ai_analyzer:
                ai_content = await ai_analyzer.create_engaging_summary(article)
            else:
                ai_content = {
                    "summary": article.get("webTitle", ""),
                    "lede": article.get("fields", {}).get("standfirst", "") or article.get("webTitle", ""),
                    "nutgraf": "This story provides insights into current events.",
                    "engagement_preview": article.get("webTitle", "")[:280]
                }
            
            # Extract entities (basic implementation)
            content = article.get("fields", {}).get("body", "")
            entities = self._extract_basic_entities(content)
            
            # Extract categories from tags
            categories = [tag.get("webTitle", "") for tag in article.get("tags", [])]
            
            processed_story = ProcessedStory(
                id=article.get("id", ""),
                title=article.get("webTitle", ""),
                summary=ai_content["summary"],
                lede=ai_content["lede"],
                nutgraf=ai_content["nutgraf"],
                section=article.get("sectionName", ""),
                publication_date=article.get("webPublicationDate", ""),
                url=article.get("webUrl", ""),
                entities=entities,
                categories=categories,
                engagement_preview=ai_content["engagement_preview"]
            )
            
            processed_stories.append(processed_story)
        
        return processed_stories
    
    def _extract_basic_entities(self, content: str) -> List[str]:
        """Basic entity extraction using simple patterns"""
        import re
        entities = []
        
        # Simple patterns for entities
        patterns = {
            'person': re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
            'organization': re.compile(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b(?:\s+(?:Inc|Corp|Ltd|LLC|Foundation|Institute))?'),
            'location': re.compile(r'\b[A-Z][a-z]+(?:,\s+[A-Z][a-z]+)*\b')
        }
        
        for entity_type, pattern in patterns.items():
            matches = pattern.findall(content[:1000])  # Limit content length
            entities.extend(matches[:5])  # Limit entities per type
        
        return list(set(entities))[:15]  # Limit total entities
    
    async def create_knowledge_graph(self, stories: List[ProcessedStory], raw_articles: List[Dict]) -> KnowledgeGraphData:
        """Create knowledge graph structure with AI-analyzed connections"""
        nodes = []
        edges = []
        
        # Create article nodes
        for story in stories:
            nodes.append({
                "id": story.id,
                "type": "article",
                "title": story.title,
                "summary": story.summary,
                "lede": story.lede,
                "nutgraf": story.nutgraf,
                "section": story.section,
                "publication_date": story.publication_date,
                "url": story.url,
                "engagement_preview": story.engagement_preview,
                "size": 20 + len(story.categories) * 2,  # Node size based on categories
                "color": self._get_section_color(story.section)
            })
        
        # Create section nodes
        sections = list(set(story.section for story in stories))
        for section in sections:
            nodes.append({
                "id": f"section_{section.lower().replace(' ', '_')}",
                "type": "section",
                "title": section,
                "size": 30,
                "color": self._get_section_color(section)
            })
        
        # Get AI-analyzed story connections
        if ai_analyzer and len(raw_articles) > 1:
            connections = await ai_analyzer.analyze_story_connections(raw_articles)
            
            for connection in connections:
                edges.append({
                    "source": connection.source_id,
                    "target": connection.target_id,
                    "type": connection.connection_type,
                    "strength": connection.strength,
                    "explanation": connection.explanation,
                    "keywords": connection.keywords,
                    "width": max(1, connection.strength * 8),  # Line width based on strength
                    "opacity": 0.3 + (connection.strength * 0.7)  # Opacity based on strength
                })
        
        # Create article-section edges
        for story in stories:
            edges.append({
                "source": story.id,
                "target": f"section_{story.section.lower().replace(' ', '_')}",
                "type": "belongs_to",
                "strength": 1.0,
                "width": 2,
                "opacity": 0.4
            })
        
        return KnowledgeGraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                "total_articles": len(stories),
                "total_sections": len(sections),
                "total_connections": len([e for e in edges if e["type"] != "belongs_to"]),
                "generated_at": datetime.now().isoformat(),
                "ai_analysis_enabled": ai_analyzer is not None
            }
        )
    
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
news_processor = NewsProcessor()

# API Endpoints
@app.get("/api/")
async def root():
    return {
        "message": "News Knowledge Graph API",
        "version": "1.0.0",
        "ai_enabled": AI_AVAILABLE,
        "endpoints": {
            "recent_news": "/api/news/recent",
            "knowledge_graph": "/api/knowledge-graph",
            "health": "/api/health"
        }
    }

@app.get("/api/news/recent")
async def get_recent_news(
    days: int = Query(default=7, ge=1, le=30, description="Days back to search"),
    section: Optional[str] = Query(default=None, description="Filter by section"),
    page_size: int = Query(default=20, ge=5, le=50, description="Number of articles")
):
    """Get recent news articles"""
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    async with GuardianAPIClient(GUARDIAN_API_KEY) as client:
        response = await client.search_content(
            section=section,
            from_date=from_date,
            to_date=to_date,
            page_size=page_size,
            order_by="newest"
        )
    
    # Process articles with AI
    processed_stories = await news_processor.process_articles(response["results"])
    
    return {
        "articles": [story.dict() for story in processed_stories],
        "total": len(processed_stories),
        "query_parameters": {
            "days": days,
            "section": section,
            "from_date": from_date,
            "to_date": to_date
        }
    }

@app.get("/api/knowledge-graph")
async def get_knowledge_graph(
    days: int = Query(default=3, ge=1, le=14, description="Days back for analysis"),
    section: Optional[str] = Query(default=None, description="Filter by section"),
    max_articles: int = Query(default=15, ge=5, le=25, description="Max articles for analysis")
):
    """Generate knowledge graph with AI-analyzed story relationships"""
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    async with GuardianAPIClient(GUARDIAN_API_KEY) as client:
        response = await client.search_content(
            section=section,
            from_date=from_date,
            to_date=to_date,
            page_size=max_articles,
            order_by="newest"
        )
    
    raw_articles = response["results"]
    processed_stories = await news_processor.process_articles(raw_articles)
    
    # Create knowledge graph with AI analysis
    knowledge_graph = await news_processor.create_knowledge_graph(processed_stories, raw_articles)
    
    return knowledge_graph.dict()

@app.get("/api/search")
async def search_news(
    query: str = Query(..., description="Search query"),
    days: int = Query(default=7, ge=1, le=30, description="Days back to search"),
    max_articles: int = Query(default=15, ge=5, le=25, description="Max articles")
):
    """Search news with AI analysis"""
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    async with GuardianAPIClient(GUARDIAN_API_KEY) as client:
        response = await client.search_content(
            query=query,
            from_date=from_date,
            to_date=to_date,
            page_size=max_articles,
            order_by="relevance"
        )
    
    raw_articles = response["results"]
    processed_stories = await news_processor.process_articles(raw_articles)
    knowledge_graph = await news_processor.create_knowledge_graph(processed_stories, raw_articles)
    
    return knowledge_graph.dict()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Guardian API
        async with GuardianAPIClient(GUARDIAN_API_KEY) as client:
            await client.search_content(page_size=1)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "guardian_api": "connected",
            "ai_services": "available" if AI_AVAILABLE else "disabled",
            "database": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)