# New York Times API Integration Guide

## 🗞️ **Complete Integration Guide for NYT API**

This guide explains how to integrate The New York Times API with your existing Guardian API setup to create a comprehensive multi-source news intelligence platform.

---

## 📋 **Prerequisites**

### 1. **Get NYT API Access**
- Visit: https://developer.nytimes.com/
- Click "Get Started" 
- Create an account or sign in
- Go to "My Apps" in your dashboard
- Click "New App"
- Fill out the application:
  - **App Name**: "News Knowledge Graph"  
  - **App Description**: "AI-powered news analysis and visualization platform"
  - **APIs**: Select "Article Search API" (primary) and optionally "Top Stories API"

### 2. **API Key Setup**
- After approval (usually instant), copy your API key
- Add to your `.env` file:
```bash
NYT_API_KEY=your-nyt-api-key-here
```

---

## 🔧 **Technical Integration**

### 1. **NYT API Structure Overview**

The NYT API differs from The Guardian API in several key ways:

| Aspect | Guardian API | NYT API |
|--------|--------------|---------|
| **Base URL** | `https://content.guardianapis.com` | `https://api.nytimes.com/svc` |
| **Main Endpoint** | `/search` | `/search/v2/articlesearch.json` |
| **Rate Limit** | 12 requests/second | 10 requests/minute (500/day free) |
| **Date Format** | `YYYY-MM-DD` | `YYYY-MM-DD` |
| **Auth Method** | `api-key` parameter | `api-key` parameter |

### 2. **NYT API Response Structure**

```json
{
  "status": "OK",
  "copyright": "Copyright (c) 2023 The New York Times Company...",
  "response": {
    "docs": [
      {
        "_id": "unique_article_id",
        "web_url": "https://www.nytimes.com/...",
        "snippet": "Brief excerpt of the article...",
        "lead_paragraph": "First paragraph of the article",
        "headline": {
          "main": "Main headline text",
          "kicker": "Optional kicker",
          "content_kicker": null
        },
        "pub_date": "2023-08-02T10:30:15+0000",
        "section_name": "World",
        "byline": {
          "original": "By Author Name",
          "person": [{"firstname": "First", "lastname": "Last"}]
        },
        "keywords": [
          {
            "name": "subject",
            "value": "Politics and Government",
            "rank": 1
          }
        ]
      }
    ],
    "meta": {
      "hits": 1234,
      "time": 45,
      "offset": 0
    }
  }
}
```

### 3. **Updated Backend Integration**

The backend code has already been enhanced to support multi-source integration. Here's how it works:

#### **MultiSourceNewsClient Class**
```python
class MultiSourceNewsClient:
    async def search_nyt(self, query=None, section=None, from_date=None, to_date=None, page_size=20):
        """Search New York Times API"""
        await self.rate_limit_check("nyt")
        
        params = {
            'api-key': self.sources["nyt"].api_key,
            'page': 0,
            'sort': 'newest'
        }
        
        # Build filter query for NYT
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
        
        response = await self.client.get(url, params=params)
        return response.json().get('response', {}).get('docs', [])[:page_size]
```

#### **Data Processing for NYT**
```python
async def _process_nyt_story(self, story: Dict, user_prefs: UserPreferences):
    """Process NYT story format"""
    headline = story.get("headline", {})
    title = headline.get("main", "") if headline else ""
    
    byline = story.get("byline", {})
    author = byline.get("original", "") if byline else None
    
    enhanced_content = await ai_analyzer.create_enhanced_content(story, user_prefs.complexity_level)
    
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
```

---

## 🚀 **API Usage Examples**

### 1. **Basic Article Search**
```python
# Search for articles about climate change
params = {
    'api-key': 'your-nyt-api-key',
    'q': 'climate change',
    'sort': 'newest',
    'page': 0
}

response = requests.get(
    'https://api.nytimes.com/svc/search/v2/articlesearch.json',
    params=params
)
```

### 2. **Advanced Filtering**
```python
# Search with date range and section filter
params = {
    'api-key': 'your-nyt-api-key',
    'q': 'artificial intelligence',
    'fq': 'section_name:("Technology") AND pub_date:[2023-08-01T00:00:00Z TO 2023-08-02T23:59:59Z]',
    'sort': 'newest',
    'page': 0
}
```

### 3. **Section-Specific Search**
```python
# Get business news from last week
params = {
    'api-key': 'your-nyt-api-key',
    'fq': 'section_name:("Business") AND pub_date:[2023-07-26T00:00:00Z TO *]',
    'sort': 'newest',
    'page': 0
}
```

---

## ⚙️ **Configuration & Rate Limiting**

### 1. **Rate Limits**
- **Free Tier**: 500 requests per day, 10 requests per minute  
- **Paid Tier**: 4,000 requests per day, 10 requests per minute

### 2. **Rate Limiting Implementation**
The backend automatically handles rate limiting:

```python
async def rate_limit_check(self, source: str):
    """Check and enforce rate limits"""
    source_config = self.sources[source]
    current_time = time.time()
    time_since_last = current_time - source_config.last_request
    min_interval = 1.0 / source_config.rate_limit
    
    if time_since_last < min_interval:
        await asyncio.sleep(min_interval - time_since_last)
    
    source_config.last_request = time.time()
```

### 3. **Error Handling**
```python
try:
    response = await self.client.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get('response', {}).get('docs', [])[:page_size]
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        logger.error("NYT API rate limit exceeded")
        await asyncio.sleep(60)  # Wait 1 minute
        return []
    raise e
except Exception as e:
    logger.error(f"NYT API error: {e}")
    return []
```

---

## 🎯 **Best Practices**

### 1. **Query Optimization**
- Use specific `fq` (filter query) parameters instead of broad searches
- Combine date ranges with section filters for better performance
- Use `sort` parameter to get most relevant results first

### 2. **Data Efficiency**
```python
# Good: Specific, filtered query
params = {
    'q': 'climate policy',
    'fq': 'section_name:("Climate") AND pub_date:[2023-08-01T00:00:00Z TO *]',
    'fl': 'web_url,headline,pub_date,section_name,byline,snippet'  # Limit fields
}

# Avoid: Broad, unfiltered queries
params = {
    'q': '*',  # Gets everything - inefficient
    'sort': 'newest'
}
```

### 3. **Caching Strategy**
```python
# Implement caching for expensive operations
cache_key = f"nyt_search:{query}:{section}:{from_date}:{to_date}"
cached_result = await redis_client.get(cache_key)

if cached_result:
    return json.loads(cached_result)

# If not cached, make API call and cache result
result = await fetch_from_nyt_api(params)
await redis_client.setex(cache_key, 3600, json.dumps(result))  # Cache for 1 hour
```

---

## 🔧 **Testing Your Integration**

### 1. **Test NYT API Connection**
```bash
curl "https://api.nytimes.com/svc/search/v2/articlesearch.json?q=test&api-key=YOUR_API_KEY"
```

### 2. **Test Multi-Source Endpoint**
```bash
curl "http://localhost:8001/api/v3/knowledge-graph/advanced?sources=guardian,nyt&days=3&complexity_level=3"
```

### 3. **Verify Data Processing**
Check that both Guardian and NYT articles appear in the knowledge graph with proper source attribution.

---

## 🚨 **Common Issues & Solutions**

### 1. **Authentication Errors**
**Problem**: `401 Unauthorized`
**Solution**: 
- Verify API key is correct
- Check that API key is added to `.env` file
- Ensure environment variables are loaded properly

### 2. **Rate Limit Exceeded**
**Problem**: `429 Too Many Requests`
**Solution**:
- Implement exponential backoff
- Reduce request frequency
- Consider upgrading to paid tier

### 3. **Empty Results**
**Problem**: No articles returned
**Solution**:
- Check date range (NYT API might not have articles for very recent times)
- Verify section names match NYT's format
- Use broader search terms

### 4. **Data Format Issues**
**Problem**: Missing fields in processed data
**Solution**:
- Add null checks for all NYT-specific fields
- Implement fallbacks for missing data
- Use the unified data processing pipeline

---

## 📈 **Advanced Features**

### 1. **Geographic Analysis**
NYT articles include geographic metadata that can enhance your knowledge graph:

```python
def extract_nyt_locations(story):
    """Extract location data from NYT article"""
    locations = []
    for keyword in story.get('keywords', []):
        if keyword.get('name') == 'glocations':
            locations.append({
                'name': keyword.get('value'),
                'rank': keyword.get('rank', 0)
            })
    return locations
```

### 2. **Content Classification**
```python
def get_nyt_content_type(story):
    """Determine content type from NYT metadata"""
    document_type = story.get('document_type', 'article')
    news_desk = story.get('news_desk', '')
    
    if 'opinion' in news_desk.lower():
        return 'opinion'
    elif document_type == 'multimedia':
        return 'multimedia'
    else:
        return 'news'
```

### 3. **Author Analysis**
```python
def extract_nyt_authors(story):
    """Extract detailed author information"""
    byline = story.get('byline', {})
    authors = []
    
    if 'person' in byline:
        for person in byline['person']:
            authors.append({
                'name': f"{person.get('firstname', '')} {person.get('lastname', '')}".strip(),
                'role': person.get('role', 'author')
            })
    
    return authors
```

---

## 🎉 **Next Steps**

Once your NYT integration is working:

1. **Monitor Usage**: Track your API usage in the NYT Developer portal
2. **Optimize Queries**: Use analytics to identify most effective search patterns
3. **Expand Sources**: Consider adding other news APIs (Reuters, BBC, AP News)
4. **Enhance AI Analysis**: Use the richer dataset for better relationship detection
5. **Geographic Features**: Implement location-based story clustering
6. **Temporal Analysis**: Create timeline visualizations across sources

---

## 📞 **Support Resources**

- **NYT API Documentation**: https://developer.nytimes.com/docs/articlesearch-product/1/overview
- **API Explorer**: https://developer.nytimes.com/docs/articlesearch-product/1/routes/articlesearch.json/get
- **Rate Limits**: https://developer.nytimes.com/faq
- **Support**: https://developer.nytimes.com/support

---

Your multi-source news intelligence platform is now ready to provide comprehensive, cross-referenced analysis from both The Guardian and The New York Times! 🚀