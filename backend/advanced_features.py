"""
Advanced Features Module for News Knowledge Graph
Production-ready extensions with real-time capabilities
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class TrendingTopic:
    topic: str
    score: float
    articles: List[str]
    growth_rate: float
    geographic_distribution: Dict[str, int]
    sentiment_trend: List[float]

@dataclass
class InfluenceMetrics:
    centrality_score: float
    reach_potential: int
    connection_density: float
    topic_authority: Dict[str, float]

class RealTimeAnalytics:
    """Real-time analytics and trend detection system"""
    
    def __init__(self):
        self.trending_cache = {}
        self.story_vectors = {}
        self.topic_history = defaultdict(list)
        self.influence_graph = nx.Graph()
    
    async def detect_trending_topics(self, stories: List[Dict], time_window_hours: int = 24) -> List[TrendingTopic]:
        """Detect trending topics using advanced NLP and temporal analysis"""
        
        # Extract text content for analysis
        documents = []
        story_metadata = []
        
        for story in stories:
            content = self._extract_full_content(story)
            if content:
                documents.append(content)
                story_metadata.append({
                    'id': story.get('id'),
                    'timestamp': story.get('webPublicationDate') or story.get('pub_date'),
                    'section': story.get('sectionName') or story.get('section_name', ''),
                    'url': story.get('webUrl') or story.get('web_url', '')
                })
        
        if len(documents) < 3:
            return []
        
        # TF-IDF vectorization for topic extraction
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate topic scores
            topic_scores = tfidf_matrix.sum(axis=0).A1
            
            trending_topics = []
            
            # Get top topics
            top_indices = np.argsort(topic_scores)[-10:][::-1]
            
            for idx in top_indices:
                topic = feature_names[idx]
                score = topic_scores[idx]
                
                # Find articles containing this topic
                topic_articles = []
                for doc_idx, doc in enumerate(documents):
                    if topic.lower() in doc.lower():
                        topic_articles.append(story_metadata[doc_idx]['id'])
                
                # Calculate growth rate (simplified)
                growth_rate = self._calculate_topic_growth(topic, len(topic_articles))
                
                # Geographic distribution (placeholder)
                geo_dist = self._analyze_geographic_distribution(topic_articles, story_metadata)
                
                # Sentiment trend (simplified)
                sentiment_trend = [0.0, 0.1, 0.2]  # Placeholder for actual sentiment analysis
                
                trending_topic = TrendingTopic(
                    topic=topic,
                    score=float(score),
                    articles=topic_articles,
                    growth_rate=growth_rate,
                    geographic_distribution=geo_dist,
                    sentiment_trend=sentiment_trend
                )
                
                trending_topics.append(trending_topic)
            
            return trending_topics[:5]  # Top 5 trending topics
            
        except Exception as e:
            logger.error(f"Trending topics detection error: {e}")
            return []
    
    def _extract_full_content(self, story: Dict) -> str:
        """Extract full content from story for analysis"""
        content_parts = []
        
        # Guardian format
        if 'webTitle' in story:
            content_parts.append(story.get('webTitle', ''))
            if 'fields' in story:
                content_parts.append(story['fields'].get('standfirst', ''))
                content_parts.append(story['fields'].get('body', ''))
        
        # NYT format
        elif 'headline' in story:
            headline = story.get('headline', {})
            content_parts.append(headline.get('main', ''))
            content_parts.append(story.get('snippet', ''))
            content_parts.append(story.get('lead_paragraph', ''))
        
        return ' '.join(content_parts)
    
    def _calculate_topic_growth(self, topic: str, current_count: int) -> float:
        """Calculate topic growth rate"""
        history = self.topic_history[topic]
        history.append((datetime.now(), current_count))
        
        # Keep only last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        history = [(ts, count) for ts, count in history if ts > cutoff]
        self.topic_history[topic] = history
        
        if len(history) < 2:
            return 0.0
        
        # Simple growth calculation
        old_count = history[0][1]
        if old_count == 0:
            return 1.0
        
        return (current_count - old_count) / old_count
    
    def _analyze_geographic_distribution(self, article_ids: List[str], metadata: List[Dict]) -> Dict[str, int]:
        """Analyze geographic distribution of topic mentions"""
        # Simplified geographic analysis
        # In production, this would use actual geolocation data
        regions = ['US', 'Europe', 'Asia', 'Americas', 'Global']
        distribution = {region: np.random.randint(0, len(article_ids) + 1) for region in regions}
        return distribution
    
    async def calculate_story_influence(self, story_id: str, connections: List[Dict]) -> InfluenceMetrics:
        """Calculate influence metrics for a story"""
        
        # Add story and connections to influence graph
        self.influence_graph.add_node(story_id)
        
        for conn in connections:
            if conn.get('source') == story_id:
                target = conn.get('target')
                if target:
                    self.influence_graph.add_edge(story_id, target, weight=conn.get('strength', 0.5))
            elif conn.get('target') == story_id:
                source = conn.get('source')
                if source:
                    self.influence_graph.add_edge(source, story_id, weight=conn.get('strength', 0.5))
        
        # Calculate centrality metrics
        try:
            centrality_scores = nx.betweenness_centrality(self.influence_graph)
            centrality_score = centrality_scores.get(story_id, 0.0)
            
            # Calculate reach potential
            reach_potential = len(nx.single_source_shortest_path(self.influence_graph, story_id, cutoff=3))
            
            # Connection density
            neighbors = list(self.influence_graph.neighbors(story_id))
            if len(neighbors) > 1:
                subgraph = self.influence_graph.subgraph(neighbors + [story_id])
                connection_density = nx.density(subgraph)
            else:
                connection_density = 0.0
            
            # Topic authority (simplified)
            topic_authority = {
                'politics': 0.7,
                'business': 0.8,
                'technology': 0.6
            }
            
            return InfluenceMetrics(
                centrality_score=centrality_score,
                reach_potential=reach_potential,
                connection_density=connection_density,
                topic_authority=topic_authority
            )
            
        except Exception as e:
            logger.error(f"Influence calculation error: {e}")
            return InfluenceMetrics(
                centrality_score=0.0,
                reach_potential=1,
                connection_density=0.0,
                topic_authority={}
            )

class GeographicAnalyzer:
    """Geographic analysis and visualization capabilities"""
    
    def __init__(self):
        self.location_cache = {}
        self.country_coordinates = {
            'US': {'lat': 39.8283, 'lon': -98.5795},
            'UK': {'lat': 55.3781, 'lon': -3.4360},
            'Germany': {'lat': 51.1657, 'lon': 10.4515},
            'France': {'lat': 46.6034, 'lon': 1.8883},
            'China': {'lat': 35.8617, 'lon': 104.1954},
            'Japan': {'lat': 36.2048, 'lon': 138.2529},
            'Australia': {'lat': -25.2744, 'lon': 133.7751},
            'Brazil': {'lat': -14.2350, 'lon': -51.9253},
            'India': {'lat': 20.5937, 'lon': 78.9629},
            'Russia': {'lat': 61.5240, 'lon': 105.3188}
        }
    
    async def analyze_geographic_patterns(self, stories: List[Dict]) -> Dict[str, Any]:
        """Analyze geographic patterns in news stories"""
        
        geographic_data = {
            'story_locations': {},
            'regional_clusters': {},
            'cross_border_connections': [],
            'global_hotspots': []
        }
        
        # Extract locations from stories
        for story in stories:
            story_id = story.get('id', '')
            locations = self._extract_locations_from_story(story)
            
            if locations:
                geographic_data['story_locations'][story_id] = locations
        
        # Identify regional clusters
        geographic_data['regional_clusters'] = self._identify_regional_clusters(
            geographic_data['story_locations']
        )
        
        # Find cross-border connections
        geographic_data['cross_border_connections'] = self._find_cross_border_connections(
            geographic_data['story_locations']
        )
        
        # Identify global hotspots
        geographic_data['global_hotspots'] = self._identify_global_hotspots(
            geographic_data['story_locations']
        )
        
        return geographic_data
    
    def _extract_locations_from_story(self, story: Dict) -> List[Dict[str, Any]]:
        """Extract location information from story content"""
        locations = []
        
        # Guardian format
        if 'tags' in story:
            for tag in story['tags']:
                if tag.get('type') == 'keyword':
                    location = self._resolve_location(tag.get('webTitle', ''))
                    if location:
                        locations.append(location)
        
        # NYT format
        elif 'keywords' in story:
            for keyword in story['keywords']:
                if keyword.get('name') == 'glocations':
                    location = self._resolve_location(keyword.get('value', ''))
                    if location:
                        locations.append(location)
        
        return locations
    
    def _resolve_location(self, location_text: str) -> Optional[Dict[str, Any]]:
        """Resolve location text to geographic coordinates"""
        
        # Simple location resolution (in production, use geocoding service)
        for country, coords in self.country_coordinates.items():
            if country.lower() in location_text.lower() or location_text.lower() in country.lower():
                return {
                    'name': location_text,
                    'country': country,
                    'coordinates': coords
                }
        
        return None
    
    def _identify_regional_clusters(self, story_locations: Dict) -> Dict[str, List[str]]:
        """Identify regional clusters of stories"""
        clusters = defaultdict(list)
        
        for story_id, locations in story_locations.items():
            for location in locations:
                region = self._get_region(location['country'])
                clusters[region].append(story_id)
        
        return dict(clusters)
    
    def _get_region(self, country: str) -> str:
        """Map country to region"""
        region_mapping = {
            'US': 'North America',
            'UK': 'Europe',
            'Germany': 'Europe',
            'France': 'Europe',
            'China': 'Asia',
            'Japan': 'Asia',
            'Australia': 'Oceania',
            'Brazil': 'South America',
            'India': 'Asia',
            'Russia': 'Europe'
        }
        return region_mapping.get(country, 'Other')
    
    def _find_cross_border_connections(self, story_locations: Dict) -> List[Dict]:
        """Find connections between stories from different regions"""
        connections = []
        
        story_list = list(story_locations.items())
        
        for i, (story1_id, locations1) in enumerate(story_list):
            for story2_id, locations2 in story_list[i+1:]:
                
                regions1 = {self._get_region(loc['country']) for loc in locations1}
                regions2 = {self._get_region(loc['country']) for loc in locations2}
                
                if regions1 != regions2:  # Different regions
                    connections.append({
                        'source': story1_id,
                        'target': story2_id,
                        'regions': list(regions1.union(regions2)),
                        'type': 'cross_border'
                    })
        
        return connections
    
    def _identify_global_hotspots(self, story_locations: Dict) -> List[Dict]:
        """Identify geographic hotspots with high story concentration"""
        region_counts = defaultdict(int)
        region_stories = defaultdict(list)
        
        for story_id, locations in story_locations.items():
            for location in locations:
                region = self._get_region(location['country'])
                region_counts[region] += 1
                region_stories[region].append(story_id)
        
        # Sort by story count
        hotspots = []
        for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:  # Minimum threshold
                hotspots.append({
                    'region': region,
                    'story_count': count,
                    'stories': region_stories[region],
                    'intensity': min(1.0, count / 10.0)  # Normalize intensity
                })
        
        return hotspots[:5]  # Top 5 hotspots

class TemporalAnalyzer:
    """Temporal analysis and timeline generation"""
    
    def __init__(self):
        self.timeline_cache = {}
        self.event_sequences = defaultdict(list)
    
    async def create_story_timeline(self, stories: List[Dict]) -> Dict[str, Any]:
        """Create temporal timeline of story development"""
        
        timeline_data = {
            'events': [],
            'sequences': [],
            'temporal_clusters': [],
            'development_stages': {}
        }
        
        # Sort stories by publication date
        sorted_stories = sorted(stories, key=lambda x: self._get_story_timestamp(x))
        
        # Create timeline events
        for story in sorted_stories:
            timestamp = self._get_story_timestamp(story)
            event = {
                'id': story.get('id', ''),
                'timestamp': timestamp.isoformat(),
                'title': story.get('webTitle') or story.get('headline', {}).get('main', ''),
                'type': self._classify_event_type(story),
                'importance': self._calculate_event_importance(story),
                'coordinates': [timestamp.timestamp(), 0]  # x-axis: time, y-axis: importance
            }
            timeline_data['events'].append(event)
        
        # Identify event sequences
        timeline_data['sequences'] = self._identify_event_sequences(sorted_stories)
        
        # Create temporal clusters
        timeline_data['temporal_clusters'] = self._create_temporal_clusters(sorted_stories)
        
        # Analyze development stages
        timeline_data['development_stages'] = self._analyze_development_stages(sorted_stories)
        
        return timeline_data
    
    def _get_story_timestamp(self, story: Dict) -> datetime:
        """Extract timestamp from story"""
        timestamp_str = story.get('webPublicationDate') or story.get('pub_date', '')
        
        try:
            if 'Z' in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(timestamp_str)
        except:
            return datetime.now()
    
    def _classify_event_type(self, story: Dict) -> str:
        """Classify the type of news event"""
        title = (story.get('webTitle') or story.get('headline', {}).get('main', '')).lower()
        
        if any(word in title for word in ['breaking', 'urgent', 'alert']):
            return 'breaking'
        elif any(word in title for word in ['analysis', 'opinion', 'commentary']):
            return 'analysis'
        elif any(word in title for word in ['update', 'latest', 'continues']):
            return 'update'
        else:
            return 'news'
    
    def _calculate_event_importance(self, story: Dict) -> float:
        """Calculate relative importance of an event"""
        importance = 0.5  # Base importance
        
        # Boost importance based on section
        section = story.get('sectionName') or story.get('section_name', '')
        section_weights = {
            'world': 0.9,
            'politics': 0.8,
            'business': 0.7,
            'technology': 0.6,
            'sport': 0.4
        }
        importance += section_weights.get(section.lower(), 0.1)
        
        # Normalize to 0-1 range
        return min(1.0, importance)
    
    def _identify_event_sequences(self, sorted_stories: List[Dict]) -> List[Dict]:
        """Identify sequences of related events"""
        sequences = []
        
        # Group stories by topic/keywords (simplified)
        topic_groups = defaultdict(list)
        
        for story in sorted_stories:
            # Extract key terms for grouping
            title = (story.get('webTitle') or story.get('headline', {}).get('main', '')).lower()
            
            # Simple keyword extraction
            key_terms = []
            for word in title.split():
                if len(word) > 4 and word.isalpha():
                    key_terms.append(word)
            
            # Group by first significant term
            if key_terms:
                topic_groups[key_terms[0]].append(story)
        
        # Create sequences from groups with multiple stories
        for topic, stories in topic_groups.items():
            if len(stories) >= 2:
                sequence = {
                    'topic': topic,
                    'story_count': len(stories),
                    'stories': [s.get('id', '') for s in stories],
                    'duration': (
                        self._get_story_timestamp(stories[-1]) - 
                        self._get_story_timestamp(stories[0])
                    ).total_seconds() / 3600,  # Duration in hours
                    'development_pattern': self._analyze_development_pattern(stories)
                }
                sequences.append(sequence)
        
        return sequences
    
    def _analyze_development_pattern(self, stories: List[Dict]) -> str:
        """Analyze how a story develops over time"""
        if len(stories) < 2:
            return 'single_event'
        
        # Analyze title patterns
        titles = [story.get('webTitle') or story.get('headline', {}).get('main', '') for story in stories]
        
        breaking_count = sum(1 for title in titles if 'breaking' in title.lower())
        update_count = sum(1 for title in titles if any(word in title.lower() for word in ['update', 'latest']))
        
        if breaking_count > 0:
            return 'breaking_development'
        elif update_count > 0:
            return 'ongoing_coverage'
        else:
            return 'related_coverage'
    
    def _create_temporal_clusters(self, sorted_stories: List[Dict]) -> List[Dict]:
        """Create clusters of stories published around the same time"""
        clusters = []
        
        if not sorted_stories:
            return clusters
        
        current_cluster = []
        cluster_threshold = timedelta(hours=6)  # Stories within 6 hours are clustered
        
        for story in sorted_stories:
            timestamp = self._get_story_timestamp(story)
            
            if not current_cluster:
                current_cluster = [story]
            else:
                last_timestamp = self._get_story_timestamp(current_cluster[-1])
                
                if timestamp - last_timestamp <= cluster_threshold:
                    current_cluster.append(story)
                else:
                    # Finalize current cluster
                    if len(current_cluster) >= 2:
                        clusters.append({
                            'start_time': self._get_story_timestamp(current_cluster[0]).isoformat(),
                            'end_time': self._get_story_timestamp(current_cluster[-1]).isoformat(),
                            'story_count': len(current_cluster),
                            'stories': [s.get('id', '') for s in current_cluster],
                            'cluster_type': self._classify_cluster_type(current_cluster)
                        })
                    
                    # Start new cluster
                    current_cluster = [story]
        
        # Handle last cluster
        if len(current_cluster) >= 2:
            clusters.append({
                'start_time': self._get_story_timestamp(current_cluster[0]).isoformat(),
                'end_time': self._get_story_timestamp(current_cluster[-1]).isoformat(),
                'story_count': len(current_cluster),
                'stories': [s.get('id', '') for s in current_cluster],
                'cluster_type': self._classify_cluster_type(current_cluster)
            })
        
        return clusters
    
    def _classify_cluster_type(self, cluster_stories: List[Dict]) -> str:
        """Classify the type of temporal cluster"""
        sections = [story.get('sectionName') or story.get('section_name', '') for story in cluster_stories]
        unique_sections = set(sections)
        
        if len(unique_sections) == 1:
            return f"single_topic_{list(unique_sections)[0].lower()}"
        elif len(unique_sections) > len(cluster_stories) * 0.7:
            return "mixed_topics"
        else:
            return "related_topics"
    
    def _analyze_development_stages(self, sorted_stories: List[Dict]) -> Dict[str, Any]:
        """Analyze different stages of story development"""
        
        if not sorted_stories:
            return {}
        
        total_duration = (
            self._get_story_timestamp(sorted_stories[-1]) - 
            self._get_story_timestamp(sorted_stories[0])
        ).total_seconds() / 3600  # Hours
        
        stages = {
            'initial_reports': [],
            'developing_coverage': [],
            'analysis_phase': [],
            'follow_up': []
        }
        
        # Classify each story into development stages
        for i, story in enumerate(sorted_stories):
            progress = i / len(sorted_stories) if len(sorted_stories) > 1 else 0
            
            if progress < 0.25:
                stages['initial_reports'].append(story.get('id', ''))
            elif progress < 0.5:
                stages['developing_coverage'].append(story.get('id', ''))
            elif progress < 0.75:
                stages['analysis_phase'].append(story.get('id', ''))
            else:
                stages['follow_up'].append(story.get('id', ''))
        
        return {
            'stages': stages,
            'total_duration_hours': total_duration,
            'coverage_intensity': len(sorted_stories) / max(1, total_duration),
            'development_speed': 'rapid' if total_duration < 24 else 'gradual'
        }

# Initialize advanced analyzers
real_time_analytics = RealTimeAnalytics()
geographic_analyzer = GeographicAnalyzer()
temporal_analyzer = TemporalAnalyzer()