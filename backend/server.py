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

@app.get("/api/demo-graph")
async def get_demo_knowledge_graph():
    """Demo knowledge graph with pre-analyzed connections - no API calls needed"""
    
    # Demo data with rich AI-style content
    demo_nodes = [
        {
            "id": "trump-labor-firing",
            "type": "article",
            "title": "Trump fires labor statistics chief amid job data controversy",
            "summary": "President Trump dismissed the head of labor statistics, claiming she manipulated job numbers to damage his administration. The move has sparked concerns about the politicization of economic data and its potential impact on market confidence.",
            "lede": "Trump's latest firing sends shockwaves through the economics community as data integrity comes under question.",
            "nutgraf": "This unprecedented dismissal of a key statistical official raises serious questions about the independence of economic data collection, which is crucial for policy-making and market stability worldwide.",
            "section": "Politics", 
            "publication_date": "2025-08-02T01:00:00Z",
            "url": "https://example.com/trump-labor-firing",
            "engagement_preview": "🚨 Trump fires labor chief over job numbers! Claims data was rigged to hurt his presidency. What does this mean for economic transparency? #TrumpFiring #EconomicData",
            "size": 25,
            "color": "#3498db"
        },
        {
            "id": "fed-interest-rates",
            "type": "article", 
            "title": "Federal Reserve under pressure as Trump demands rate cuts",
            "summary": "Trump intensifies his criticism of Fed Chair Jerome Powell, calling for aggressive interest rate cuts and threatening to override his decisions. The unprecedented political pressure on the Fed raises concerns about monetary policy independence.",
            "lede": "The Fed faces its greatest political challenge in decades as Trump escalates his attack on monetary policy independence.",
            "nutgraf": "Trump's direct confrontation with the Federal Reserve represents a fundamental threat to the institution's autonomy, which has been a cornerstone of stable monetary policy since the 1950s.",
            "section": "Business",
            "publication_date": "2025-08-02T02:00:00Z", 
            "url": "https://example.com/fed-pressure",
            "engagement_preview": "🏦 BREAKING: Trump vs. The Fed intensifies! Demands rate cuts, threatens Powell's authority. Could this destabilize markets? #TrumpVsFed #InterestRates",
            "size": 30,
            "color": "#f39c12"
        },
        {
            "id": "gaza-diplomacy",
            "type": "article",
            "title": "Australia's foreign minister criticizes Israel over Gaza humanitarian crisis", 
            "summary": "Foreign Minister Penny Wong held a closed-door meeting with Israel's ambassador, expressing strong concerns about Gaza's humanitarian situation and calling for increased aid access. The diplomatic intervention reflects growing international pressure on Israel.",
            "lede": "Australia joins the growing chorus of nations demanding humanitarian access to Gaza as the crisis deepens.",
            "nutgraf": "Wong's diplomatic intervention signals a shift in Australia's Middle East policy, potentially affecting regional alliances and international humanitarian efforts in Gaza.",
            "section": "World",
            "publication_date": "2025-08-02T03:00:00Z",
            "url": "https://example.com/gaza-diplomacy", 
            "engagement_preview": "🇦🇺 Australia takes stand on Gaza! Foreign Minister Wong confronts Israeli ambassador in private meeting. Diplomatic tensions rising? #GazaCrisis #Diplomacy",
            "size": 22,
            "color": "#e74c3c"
        },
        {
            "id": "climate-summit",
            "type": "article",
            "title": "Global climate summit addresses economic impacts of environmental policy",
            "summary": "World leaders gather to discuss how climate policies are reshaping global economics, with particular focus on how environmental regulations affect market stability and international trade relationships.",
            "lede": "Climate policy meets economic reality as world leaders seek sustainable solutions to environmental challenges.",  
            "nutgraf": "The intersection of environmental policy and economic stability has become a critical issue for global governance, affecting everything from energy markets to international trade agreements.",
            "section": "Environment",
            "publication_date": "2025-08-02T04:00:00Z",
            "url": "https://example.com/climate-summit",
            "engagement_preview": "🌍 Climate summit tackles the trillion-dollar question: How do we save the planet without crashing the economy? #ClimatePolicy #GlobalEconomy",
            "size": 20,
            "color": "#27ae60"
        },
        {
            "id": "tech-regulation",
            "type": "article", 
            "title": "New tech regulations spark debate over innovation vs. oversight",
            "summary": "Government proposals for stricter tech regulation have divided opinion, with supporters citing data privacy concerns while critics warn of potential impacts on innovation and economic competitiveness in the global market.",
            "lede": "The battle over tech regulation intensifies as governments worldwide grapple with digital oversight challenges.",
            "nutgraf": "These regulatory proposals represent a critical moment in defining the relationship between technological innovation and democratic governance in the digital age.",
            "section": "Technology",
            "publication_date": "2025-08-02T05:00:00Z", 
            "url": "https://example.com/tech-regulation",
            "engagement_preview": "⚖️ Tech giants face new regulations! Innovation vs. oversight battle heats up. Will this stifle the next breakthrough or protect our privacy? #TechRegulation",
            "size": 18,
            "color": "#9b59b6"
        },
        # Section nodes
        {
            "id": "section_politics",
            "type": "section",
            "title": "Politics",
            "size": 35,
            "color": "#3498db"
        },
        {
            "id": "section_business", 
            "type": "section",
            "title": "Business",
            "size": 35,
            "color": "#f39c12"
        },
        {
            "id": "section_world",
            "type": "section", 
            "title": "World",
            "size": 35,
            "color": "#e74c3c"
        },
        {
            "id": "section_environment",
            "type": "section",
            "title": "Environment", 
            "size": 35,
            "color": "#27ae60"
        },
        {
            "id": "section_technology",
            "type": "section",
            "title": "Technology",
            "size": 35, 
            "color": "#9b59b6"
        }
    ]
    
    demo_edges = [
        # Story connections with varying strengths
        {
            "source": "trump-labor-firing",
            "target": "fed-interest-rates", 
            "type": "political",
            "strength": 0.9,
            "explanation": "Both stories demonstrate Trump's direct interference with independent federal institutions",
            "keywords": ["Trump", "federal institutions", "political pressure", "independence"],
            "width": 7.2,
            "opacity": 0.93
        },
        {
            "source": "fed-interest-rates",
            "target": "climate-summit",
            "type": "economic", 
            "strength": 0.6,
            "explanation": "Interest rates and environmental policies both significantly impact economic stability and investment decisions",
            "keywords": ["economic policy", "market stability", "investment", "government intervention"],
            "width": 4.8,
            "opacity": 0.72
        },
        {
            "source": "tech-regulation",
            "target": "fed-interest-rates",
            "type": "economic",
            "strength": 0.5,
            "explanation": "Both regulatory pressures affect market confidence and innovation in their respective sectors", 
            "keywords": ["regulation", "market impact", "innovation", "oversight"],
            "width": 4.0,
            "opacity": 0.65
        },
        {
            "source": "gaza-diplomacy",
            "target": "climate-summit", 
            "type": "thematic",
            "strength": 0.4,
            "explanation": "Both involve international diplomatic coordination on global challenges requiring multilateral solutions",
            "keywords": ["international diplomacy", "global cooperation", "multilateral solutions"],
            "width": 3.2,
            "opacity": 0.58
        },
        {
            "source": "trump-labor-firing",
            "target": "tech-regulation",
            "type": "causal",
            "strength": 0.7,
            "explanation": "Political attacks on institutions create precedent for government interference in regulatory processes",
            "keywords": ["institutional independence", "government interference", "regulatory autonomy"],
            "width": 5.6, 
            "opacity": 0.79
        },
        # Section connections
        {
            "source": "trump-labor-firing",
            "target": "section_politics",
            "type": "belongs_to",
            "strength": 1.0,
            "width": 2,
            "opacity": 0.4
        },
        {
            "source": "fed-interest-rates", 
            "target": "section_business",
            "type": "belongs_to",
            "strength": 1.0,
            "width": 2,
            "opacity": 0.4
        },
        {
            "source": "gaza-diplomacy",
            "target": "section_world",
            "type": "belongs_to", 
            "strength": 1.0,
            "width": 2,
            "opacity": 0.4
        },
        {
            "source": "climate-summit",
            "target": "section_environment",
            "type": "belongs_to",
            "strength": 1.0, 
            "width": 2,
            "opacity": 0.4
        },
        {
            "source": "tech-regulation",
            "target": "section_technology", 
            "type": "belongs_to",
            "strength": 1.0,
            "width": 2,
            "opacity": 0.4
        }
    ]
    
    return {
        "nodes": demo_nodes,
        "edges": demo_edges,
        "metadata": {
            "total_articles": 5,
            "total_sections": 5, 
            "total_connections": 5,
            "generated_at": datetime.now().isoformat(),
            "ai_analysis_enabled": True,
            "demo_mode": True,
            "note": "This demo showcases AI-powered story relationship analysis with realistic news scenarios"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)