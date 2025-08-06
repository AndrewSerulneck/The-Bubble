import React, { useState, useEffect, useRef } from 'react';
import './RealTimeUpdates.css';

const RealTimeUpdates = ({ onUpdate }) => {
  const [connected, setConnected] = useState(false);
  const [updates, setUpdates] = useState([]);
  const [subscriptions, setSubscriptions] = useState(new Set());
  const [metrics, setMetrics] = useState({});
  const wsRef = useRef(null);
  const [clientId, setClientId] = useState(null);

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
  const wsUrl = backendUrl.replace('http://', 'ws://').replace('https://', 'wss://');

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    try {
      wsRef.current = new WebSocket(`${wsUrl}/ws`);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
      };
      
      wsRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      };
      
      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          if (!connected) {
            connectWebSocket();
          }
        }, 5000);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnected(false);
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
    }
  };

  const handleWebSocketMessage = (message) => {
    const { type, data, timestamp } = message;

    switch (type) {
      case 'connection_established':
        setClientId(message.client_id);
        // Auto-subscribe to key topics
        subscribeToTopic('trending_topics');
        subscribeToTopic('new_connections');
        subscribeToTopic('breaking_news');
        break;

      case 'trending_topics_update':
        handleTrendingTopicsUpdate(data);
        break;

      case 'new_connections_discovered':
        handleNewConnectionsUpdate(data);
        break;

      case 'breaking_news_alert':
        handleBreakingNewsAlert(data);
        break;

      case 'geographic_activity_update':
        handleGeographicUpdate(data);
        break;

      case 'story_update':
        handleStoryUpdate(data);
        break;

      case 'system_metrics':
        setMetrics(data);
        break;

      default:
        console.log('Received message:', message);
    }

    // Add to recent updates
    addUpdate({
      id: Date.now() + Math.random(),
      type,
      data,
      timestamp: new Date(timestamp),
      processed: false
    });
  };

  const addUpdate = (update) => {
    setUpdates(prev => [update, ...prev.slice(0, 19)]); // Keep last 20 updates
  };

  const subscribeToTopic = (topic) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        topic: topic
      }));
      setSubscriptions(prev => new Set([...prev, topic]));
    }
  };

  const unsubscribeFromTopic = (topic) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'unsubscribe',
        topic: topic
      }));
      setSubscriptions(prev => {
        const newSet = new Set(prev);
        newSet.delete(topic);
        return newSet;
      });
    }
  };

  const handleTrendingTopicsUpdate = (data) => {
    console.log('Trending topics updated:', data);
    if (onUpdate) {
      onUpdate({
        type: 'trending_topics',
        data: data.trending_topics
      });
    }
  };

  const handleNewConnectionsUpdate = (data) => {
    console.log('New connections discovered:', data);
    if (onUpdate) {
      onUpdate({
        type: 'new_connections',
        data: data.connections
      });
    }
  };

  const handleBreakingNewsAlert = (data) => {
    console.log('Breaking news:', data);
    
    // Show browser notification if permitted
    if (Notification.permission === 'granted') {
      new Notification('Breaking News Alert', {
        body: data.headline,
        icon: '/favicon.ico',
        tag: 'breaking-news'
      });
    }

    if (onUpdate) {
      onUpdate({
        type: 'breaking_news',
        data: data
      });
    }
  };

  const handleGeographicUpdate = (data) => {
    console.log('Geographic activity update:', data);
    if (onUpdate) {
      onUpdate({
        type: 'geographic_activity',
        data: data
      });
    }
  };

  const handleStoryUpdate = (data) => {
    console.log('Story update:', data);
    if (onUpdate) {
      onUpdate({
        type: 'story_update',
        data: data.story
      });
    }
  };

  const requestNotificationPermission = () => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission().then(permission => {
        console.log('Notification permission:', permission);
      });
    }
  };

  const getUpdateIcon = (type) => {
    const icons = {
      'trending_topics_update': '📈',
      'new_connections_discovered': '🔗',
      'breaking_news_alert': '🚨',
      'geographic_activity_update': '🌍',
      'story_update': '📰',
      'system_metrics': '⚙️'
    };
    return icons[type] || '📊';
  };

  const getUpdateColor = (type) => {
    const colors = {
      'trending_topics_update': '#f39c12',
      'new_connections_discovered': '#3498db',
      'breaking_news_alert': '#e74c3c',
      'geographic_activity_update': '#27ae60',
      'story_update': '#9b59b6',
      'system_metrics': '#95a5a6'
    };
    return colors[type] || '#7f8c8d';
  };

  const formatUpdateTime = (timestamp) => {
    const now = new Date();
    const diff = now - timestamp;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (seconds < 60) return `${seconds}s ago`;
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return timestamp.toLocaleDateString();
  };

  const availableTopics = [
    { id: 'trending_topics', label: '📈 Trending Topics', description: 'Real-time topic trends' },
    { id: 'new_connections', label: '🔗 New Connections', description: 'Story relationship discoveries' },
    { id: 'breaking_news', label: '🚨 Breaking News', description: 'Urgent news alerts' },
    { id: 'geographic_alerts', label: '🌍 Geographic Activity', description: 'Regional news patterns' },
    { id: 'story_updates', label: '📰 Story Updates', description: 'Individual story changes' },
    { id: 'system_metrics', label: '⚙️ System Status', description: 'Platform performance' }
  ];

  return (
    <div className="real-time-updates">
      <div className="updates-header">
        <div className="connection-status">
          <div className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
            <div className="status-dot"></div>
            <span>{connected ? 'Live Updates' : 'Connecting...'}</span>
          </div>
          {connected && clientId && (
            <span className="client-id">ID: {clientId.slice(-8)}</span>
          )}
        </div>

        {connected && (
          <button 
            className="notification-permission-btn"
            onClick={requestNotificationPermission}
            style={{ 
              display: Notification.permission === 'default' ? 'block' : 'none' 
            }}
          >
            🔔 Enable Notifications
          </button>
        )}
      </div>

      <div className="subscription-controls">
        <h4>📡 Live Subscriptions</h4>
        <div className="topic-subscriptions">
          {availableTopics.map(topic => (
            <div key={topic.id} className="subscription-item">
              <label className="subscription-checkbox">
                <input
                  type="checkbox"
                  checked={subscriptions.has(topic.id)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      subscribeToTopic(topic.id);
                    } else {
                      unsubscribeFromTopic(topic.id);
                    }
                  }}
                />
                <span className="checkbox-label">
                  {topic.label}
                  <small>{topic.description}</small>
                </span>
              </label>
            </div>
          ))}
        </div>
      </div>

      {Object.keys(metrics).length > 0 && (
        <div className="metrics-display">
          <h4>📊 Live Metrics</h4>
          <div className="metrics-grid">
            {metrics.connections && (
              <div className="metric-item">
                <span className="metric-label">Active Users</span>
                <span className="metric-value">{metrics.connections.total_connections}</span>
              </div>
            )}
            <div className="metric-item">
              <span className="metric-label">AI Confidence</span>
              <span className="metric-value">{(metrics.ai_confidence_average * 100).toFixed(0)}%</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Processing</span>
              <span className="metric-value">{metrics.processing_speed}</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Uptime</span>
              <span className="metric-value">{metrics.uptime}</span>
            </div>
          </div>
        </div>
      )}

      <div className="updates-feed">
        <h4>📝 Recent Updates ({updates.length})</h4>
        <div className="updates-list">
          {updates.length === 0 ? (
            <div className="no-updates">
              <span>Waiting for updates...</span>
            </div>
          ) : (
            updates.map(update => (
              <div key={update.id} className="update-item">
                <div className="update-header">
                  <span 
                    className="update-icon"
                    style={{ color: getUpdateColor(update.type) }}
                  >
                    {getUpdateIcon(update.type)}
                  </span>
                  <span className="update-type">
                    {update.type.replace(/_/g, ' ')}
                  </span>
                  <span className="update-time">
                    {formatUpdateTime(update.timestamp)}
                  </span>
                </div>
                <div className="update-content">
                  {update.type === 'trending_topics_update' && (
                    <div>
                      Top trending: {update.data.trending_topics?.[0]?.topic || 'N/A'}
                    </div>
                  )}
                  {update.type === 'new_connections_discovered' && (
                    <div>
                      {update.data.connections?.length || 0} new story connections found
                    </div>
                  )}
                  {update.type === 'breaking_news_alert' && (
                    <div className="breaking-news-content">
                      <strong>🚨 {update.data.headline}</strong>
                      <br />
                      <small>Impact: {update.data.estimated_impact}</small>
                    </div>
                  )}
                  {update.type === 'geographic_activity_update' && (
                    <div>
                      Activity in {update.data.hotspots?.length || 0} regions
                    </div>
                  )}
                  {update.type === 'story_update' && (
                    <div>
                      Story updated: {update.data.story?.title?.slice(0, 40) || 'Unknown'}...
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default RealTimeUpdates;