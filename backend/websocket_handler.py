"""
WebSocket Handler for Real-time Updates
Provides live updates for the knowledge graph
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
import uuid

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of subscribed topics
        self.client_preferences: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        if not client_id:
            client_id = str(uuid.uuid4())
        
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        self.client_preferences[client_id] = {}
        
        logger.info(f"WebSocket connected: {client_id}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "available_subscriptions": [
                "trending_topics",
                "new_connections",
                "story_updates",
                "breaking_news",
                "geographic_alerts"
            ]
        }, client_id)
        
        return client_id
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        if client_id in self.client_preferences:
            del self.client_preferences[client_id]
        
        logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: Dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_subscribers(self, message: Dict, topic: str):
        """Broadcast message to all clients subscribed to a topic"""
        disconnected_clients = []
        
        for client_id, topics in self.subscriptions.items():
            if topic in topics:
                try:
                    await self.send_personal_message(message, client_id)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def subscribe_client(self, client_id: str, topic: str):
        """Subscribe client to a topic"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].add(topic)
            logger.info(f"Client {client_id} subscribed to {topic}")
    
    def unsubscribe_client(self, client_id: str, topic: str):
        """Unsubscribe client from a topic"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(topic)
            logger.info(f"Client {client_id} unsubscribed from {topic}")
    
    def update_client_preferences(self, client_id: str, preferences: Dict):
        """Update client preferences for personalized updates"""
        if client_id in self.client_preferences:
            self.client_preferences[client_id].update(preferences)
            logger.info(f"Updated preferences for {client_id}: {preferences}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections"""
        topic_counts = {}
        for topics in self.subscriptions.values():
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "total_connections": len(self.active_connections),
            "total_subscriptions": sum(len(topics) for topics in self.subscriptions.values()),
            "topic_distribution": topic_counts,
            "active_clients": list(self.active_connections.keys())
        }

class RealTimeUpdateService:
    """Service for generating and broadcasting real-time updates"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.update_interval = 30  # seconds
        self.last_story_count = 0
        self.running = False
    
    async def start_update_service(self):
        """Start the real-time update service"""
        self.running = True
        logger.info("Real-time update service started")
        
        while self.running:
            try:
                await self.generate_updates()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Update service error: {e}")
                await asyncio.sleep(self.update_interval)
    
    def stop_update_service(self):
        """Stop the real-time update service"""
        self.running = False
        logger.info("Real-time update service stopped")
    
    async def generate_updates(self):
        """Generate various types of real-time updates"""
        
        # Simulate trending topics update
        await self.broadcast_trending_topics()
        
        # Simulate new connections discovery
        await self.broadcast_new_connections()
        
        # Simulate breaking news alerts
        await self.broadcast_breaking_news()
        
        # Geographic activity updates
        await self.broadcast_geographic_alerts()
    
    async def broadcast_trending_topics(self):
        """Broadcast trending topics update"""
        # Simulate trending topics data
        trending_topics = [
            {
                "topic": "climate summit",
                "score": 0.85,
                "growth_rate": 0.23,
                "article_count": 15,
                "sentiment": "neutral"
            },
            {
                "topic": "federal reserve",
                "score": 0.78,
                "growth_rate": 0.18,
                "article_count": 12,
                "sentiment": "negative"
            },
            {
                "topic": "artificial intelligence",
                "score": 0.72,
                "growth_rate": 0.31,
                "article_count": 9,
                "sentiment": "positive"
            }
        ]
        
        message = {
            "type": "trending_topics_update",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "trending_topics": trending_topics,
                "update_reason": "periodic_analysis",
                "next_update": (datetime.now().timestamp() + self.update_interval)
            }
        }
        
        await self.connection_manager.broadcast_to_subscribers(message, "trending_topics")
    
    async def broadcast_new_connections(self):
        """Broadcast newly discovered story connections"""
        # Simulate new connection discovery
        new_connections = [
            {
                "id": str(uuid.uuid4()),
                "source_story": "climate-policy-eu",
                "target_story": "us-energy-markets",
                "connection_type": "economic",
                "strength": 0.72,
                "confidence": 0.89,
                "explanation": "EU climate policies are creating ripple effects in US energy markets",
                "discovered_at": datetime.now().isoformat()
            }
        ]
        
        message = {
            "type": "new_connections_discovered",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "connections": new_connections,
                "analysis_method": "real_time_ai_analysis",
                "impact_score": 0.75
            }
        }
        
        await self.connection_manager.broadcast_to_subscribers(message, "new_connections")
    
    async def broadcast_breaking_news(self):
        """Broadcast breaking news alerts"""
        # Simulate occasional breaking news
        import random
        
        if random.random() < 0.1:  # 10% chance of breaking news
            breaking_news = {
                "id": str(uuid.uuid4()),
                "headline": "Major development in ongoing climate negotiations",
                "urgency": "high",
                "category": "world",
                "estimated_impact": "global",
                "preliminary_connections": [
                    "economic_markets",
                    "environmental_policy", 
                    "international_relations"
                ],
                "source": "guardian",
                "timestamp": datetime.now().isoformat()
            }
            
            message = {
                "type": "breaking_news_alert",
                "timestamp": datetime.now().isoformat(),
                "data": breaking_news
            }
            
            await self.connection_manager.broadcast_to_subscribers(message, "breaking_news")
    
    async def broadcast_geographic_alerts(self):
        """Broadcast geographic activity alerts"""
        # Simulate geographic hotspots
        geographic_activity = [
            {
                "region": "Europe",
                "activity_level": "high",
                "story_count": 8,
                "dominant_topics": ["climate policy", "economic integration"],
                "cross_border_connections": 3
            },
            {
                "region": "North America", 
                "activity_level": "medium",
                "story_count": 5,
                "dominant_topics": ["federal policy", "business developments"],
                "cross_border_connections": 2
            }
        ]
        
        message = {
            "type": "geographic_activity_update",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "hotspots": geographic_activity,
                "global_activity_index": 0.68,
                "most_connected_regions": ["Europe", "North America"]
            }
        }
        
        await self.connection_manager.broadcast_to_subscribers(message, "geographic_alerts")
    
    async def broadcast_story_update(self, story_data: Dict):
        """Broadcast update about a specific story"""
        message = {
            "type": "story_update",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "story": story_data,
                "update_type": "new_analysis",
                "connections_discovered": True,
                "impact_assessment": "medium"
            }
        }
        
        await self.connection_manager.broadcast_to_subscribers(message, "story_updates")
    
    async def broadcast_system_metrics(self):
        """Broadcast system performance metrics"""
        connection_stats = self.connection_manager.get_connection_stats()
        
        message = {
            "type": "system_metrics",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "connections": connection_stats,
                "analysis_queue_size": 0,
                "processing_speed": "optimal",
                "ai_confidence_average": 0.78,
                "uptime": "99.9%"
            }
        }
        
        await self.connection_manager.broadcast_to_subscribers(message, "system_metrics")

# Global connection manager instance
connection_manager = ConnectionManager()
update_service = RealTimeUpdateService(connection_manager)

async def handle_websocket_message(websocket: WebSocket, client_id: str, message: Dict):
    """Handle incoming WebSocket messages from clients"""
    
    message_type = message.get("type")
    
    if message_type == "subscribe":
        topic = message.get("topic")
        if topic:
            connection_manager.subscribe_client(client_id, topic)
            await connection_manager.send_personal_message({
                "type": "subscription_confirmed",
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            }, client_id)
    
    elif message_type == "unsubscribe":
        topic = message.get("topic")
        if topic:
            connection_manager.unsubscribe_client(client_id, topic)
            await connection_manager.send_personal_message({
                "type": "unsubscription_confirmed", 
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            }, client_id)
    
    elif message_type == "update_preferences":
        preferences = message.get("preferences", {})
        connection_manager.update_client_preferences(client_id, preferences)
        await connection_manager.send_personal_message({
            "type": "preferences_updated",
            "preferences": preferences,
            "timestamp": datetime.now().isoformat()
        }, client_id)
    
    elif message_type == "request_stats":
        stats = connection_manager.get_connection_stats()
        await connection_manager.send_personal_message({
            "type": "connection_stats",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }, client_id)
    
    elif message_type == "ping":
        await connection_manager.send_personal_message({
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        }, client_id)