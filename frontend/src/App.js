import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './App.css';

const App = () => {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [], metadata: {} });
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [days, setDays] = useState(3);
  const [section, setSection] = useState('');
  const [sources, setSources] = useState('guardian,nyt');
  const [complexityLevel, setComplexityLevel] = useState(3);
  const [geographicFocus, setGeographicFocus] = useState('');
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [showFeedback, setShowFeedback] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [feedbackData, setFeedbackData] = useState({
    rating: 5,
    comments: '',
    email: '',
    features_used: [],
    most_interesting_connection: '',
    suggested_improvements: ''
  });
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [analytics, setAnalytics] = useState({ stories_viewed: 0, connections_explored: 0 });
  const svgRef = useRef();

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  const sections = [
    '', 'world', 'politics', 'business', 'technology', 'sport', 
    'culture', 'science', 'environment', 'education', 'society'
  ];

  const sourceOptions = [
    { value: 'guardian,nyt', label: '📰 Both Sources (Recommended)' },
    { value: 'guardian', label: '🇬🇧 The Guardian Only' },
    { value: 'nyt', label: '🇺🇸 New York Times Only' }
  ];

  const complexityOptions = [
    { value: 1, label: '📱 Simple - Headlines & Key Facts', description: 'Perfect for quick updates' },
    { value: 2, label: '📝 Basic - Context & Background', description: 'Essential information with context' },
    { value: 3, label: '📊 Moderate - Analysis & Connections', description: 'Balanced depth and accessibility' },
    { value: 4, label: '🔍 Detailed - In-depth Analysis', description: 'Comprehensive coverage with expert insights' },
    { value: 5, label: '🎓 Expert - Theories & Implications', description: 'Academic-level analysis and implications' }
  ];

  const geographicOptions = [
    { value: '', label: '🌍 Global Perspective' },
    { value: 'US', label: '🇺🇸 United States Focus' },
    { value: 'Europe', label: '🇪🇺 European Focus' },
    { value: 'Asia', label: '🌏 Asian Focus' },
    { value: 'Americas', label: '🌎 Americas Focus' }
  ];

  useEffect(() => {
    loadAdvancedKnowledgeGraph();
  }, []);

  const trackAnalytics = async (action, metadata = {}) => {
    try {
      const analyticsData = {
        session_id: sessionId,
        action: action,
        story_id: metadata.story_id || null,
        connection_id: metadata.connection_id || null,
        timestamp: new Date().toISOString(),
        metadata: metadata
      };
      
      // Update local analytics
      if (action === 'view_story') {
        setAnalytics(prev => ({ ...prev, stories_viewed: prev.stories_viewed + 1 }));
      } else if (action === 'explore_connection') {
        setAnalytics(prev => ({ ...prev, connections_explored: prev.connections_explored + 1 }));
      }
      
      // Send to backend (optional, will fail gracefully in demo)
      fetch(`${backendUrl}/api/v3/analytics/track`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analyticsData)
      }).catch(() => {}); // Silent fail for demo
    } catch (error) {
      console.log('Analytics tracking (demo mode):', action);
    }
  };

  const loadAdvancedKnowledgeGraph = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        days: days,
        sources: sources,
        complexity_level: complexityLevel,
        session_id: sessionId,
        max_articles: '20'
      });
      
      if (section) params.append('section', section);
      if (geographicFocus) params.append('geographic_focus', geographicFocus);

      const response = await fetch(`${backendUrl}/api/v4/knowledge-graph/ultimate?${params}`);
      
      if (!response.ok) {
        throw new Error('API error, loading demo data');
      }
      
      const data = await response.json();
      setGraphData(data);
      renderAdvancedGraph(data);
      
      trackAnalytics('load_advanced_graph', {
        sources: sources,
        complexity_level: complexityLevel,
        total_articles: data.metadata?.total_articles || 0
      });
      
    } catch (error) {
      console.error('Error loading advanced graph:', error);
      // Fallback to production demo
      loadProductionDemo();
    } finally {
      setLoading(false);
    }
  };

  const loadProductionDemo = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/v4/demo/ultimate`);
      const data = await response.json();
      setGraphData(data);
      renderAdvancedGraph(data);
      
      trackAnalytics('load_demo', { demo_type: 'production' });
    } catch (error) {
      console.error('Error loading demo:', error);
    }
  };

  const searchAdvancedNews = async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    try {
      const params = new URLSearchParams({
        query: searchQuery,
        days: days,
        sources: sources,
        complexity_level: complexityLevel,
        session_id: sessionId,
        max_articles: '15'
      });

      const response = await fetch(`${backendUrl}/api/v4/news/search?${params}`);
      const data = await response.json();
      
      setGraphData(data);
      renderAdvancedGraph(data);
      
      trackAnalytics('advanced_search', {
        query: searchQuery,
        sources: sources,
        results_count: data.metadata?.total_articles || 0
      });
      
    } catch (error) {
      console.error('Error searching:', error);
      loadProductionDemo();
    } finally {
      setLoading(false);
    }
  };

  const submitEnhancedFeedback = async () => {
    try {
      const enhancedFeedback = {
        ...feedbackData,
        session_id: sessionId,
        features_used: [
          'multi_source_integration',
          'complexity_adaptation',
          'advanced_visualization',
          'ai_analysis'
        ].filter(feature => {
          // Basic feature usage detection
          if (feature === 'multi_source_integration' && sources.includes(',')) return true;
          if (feature === 'complexity_adaptation' && complexityLevel !== 3) return true;
          if (feature === 'advanced_visualization') return analytics.stories_viewed > 0;
          if (feature === 'ai_analysis') return analytics.connections_explored > 0;
          return false;
        })
      };
      
      const response = await fetch(`${backendUrl}/api/v3/feedback/enhanced`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(enhancedFeedback)
      });
      
      if (response.ok) {
        alert('🙏 Thank you for your detailed feedback! Your insights help us improve the platform.');
      } else {
        alert('Thank you for your feedback! (Demo mode - feedback noted locally)');
      }
      
      setShowFeedback(false);
      setFeedbackData({
        rating: 5,
        comments: '',
        email: '',
        features_used: [],
        most_interesting_connection: '',
        suggested_improvements: ''
      });
      
      trackAnalytics('submit_enhanced_feedback', { rating: feedbackData.rating });
      
    } catch (error) {
      alert('Thanks for your feedback! (Demo mode)');
      setShowFeedback(false);
    }
  };

  const renderAdvancedGraph = (data) => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    if (!data.nodes || data.nodes.length === 0) return;

    // Massive scale canvas - larger for hundreds of bubbles
    const width = 2000;  // Increased for massive scale
    const height = 1200; // Increased for massive scale
    
    svg.attr('width', width).attr('height', height);

    // Filter nodes for massive scale rendering
    const articleNodes = data.nodes.filter(d => d.type === 'article');
    const clusterNodes = data.nodes.filter(d => d.type === 'topic_cluster' || d.type === 'geographic_cluster');
    
    console.log(`🚀 MASSIVE SCALE: Rendering ${articleNodes.length} articles, ${clusterNodes.length} clusters, ${data.edges.length} connections`);

    // Optimized force simulation for massive datasets
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges)
        .id(d => d.id)
        .distance(d => {
          if (d.type && d.type.includes('belongs_to')) return 60;  // Cluster membership
          return 100 + (1 - (d.strength || 0.5)) * 80;  // Adaptive distance based on connection strength
        })
        .strength(d => {
          if (d.type && d.type.includes('belongs_to')) return 0.1;
          return (d.confidence || 0.5) * 0.4;  // Strength based on confidence
        })
      )
      .force('charge', d3.forceManyBody()
        .strength(d => {
          if (d.type === 'article') return -300;  // Reduced for performance
          if (d.type === 'topic_cluster') return -500;
          if (d.type === 'geographic_cluster') return -400;
          return -200;
        })
      )
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide()
        .radius(d => (d.size || 20) + 5)
      )
      // Topic clustering forces for organization
      .force('x', d3.forceX(d => {
        if (d.type === 'article' && d.cluster_x) {
          return d.cluster_x * width;
        }
        return width / 2;
      }).strength(0.05))  // Gentle clustering
      .force('y', d3.forceY(d => {
        if (d.type === 'article' && d.cluster_y) {
          return d.cluster_y * height;
        }
        return height / 2;
      }).strength(0.05));

    const container = svg.append('g');
    
    // Enhanced zoom for massive scale
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])  // Allow zooming out more for massive view
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });
    
    svg.call(zoom);

    // Efficient edge rendering for massive datasets
    const linkGroup = container.append('g').attr('class', 'links');
    const links = linkGroup.selectAll('.link')
      .data(data.edges)
      .enter()
      .append('line')
      .attr('class', 'link')
      .style('stroke', d => {
        if (d.type && d.type.includes('belongs_to')) return '#e0e0e0';
        
        const causalColors = {
          'causal': '#e74c3c',
          'economic_causal': '#f39c12',
          'political_causal': '#3498db',
          'social_causal': '#9b59b6',
          'environmental_causal': '#27ae60',
          'indirect_causal': '#e67e22'
        };
        return causalColors[d.type] || '#95a5a6';
      })
      .style('stroke-width', d => {
        if (d.type && d.type.includes('belongs_to')) return 1;
        return Math.max(1, Math.min(4, (d.width || 2)));  // Capped for performance
      })
      .style('stroke-opacity', d => {
        if (d.type && d.type.includes('belongs_to')) return 0.1;
        return Math.max(0.2, Math.min(0.8, d.opacity || 0.6));
      })
      .style('stroke-dasharray', d => {
        if (d.stroke_style === 'dashed') return '4,2';
        if (d.stroke_style === 'dotted') return '2,2';
        return 'none';
      });

    // Node groups for massive rendering
    const nodeGroup = container.append('g').attr('class', 'nodes');
    const nodes = nodeGroup.selectAll('.node')
      .data(data.nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer');

    // Topic and geographic cluster backgrounds
    const clusters = nodes.filter(d => d.type === 'topic_cluster' || d.type === 'geographic_cluster');
    
    clusters.append('circle')
      .attr('r', d => Math.min(60, d.size || 40))
      .style('fill', d => d.color)
      .style('opacity', 0.1)
      .style('stroke', d => d.color)
      .style('stroke-width', 2)
      .style('stroke-dasharray', d => d.type === 'geographic_cluster' ? '8,4' : '4,4');

    clusters.append('text')
      .text(d => `${d.title} (${d.story_count || 0})`)
      .style('font-size', '11px')
      .style('font-weight', 'bold')
      .style('fill', d => d.color)
      .style('text-anchor', 'middle')
      .style('pointer-events', 'none');

    // Article bubbles optimized for massive scale
    const articles = nodes.filter(d => d.type === 'article');
    
    articles.append('circle')
      .attr('r', d => {
        // Adaptive sizing for massive scale
        const titleLength = (d.title || '').length;
        return Math.max(12, Math.min(25, titleLength * 0.4));  // Smaller for massive view
      })
      .style('fill', d => d.color || '#3498db')
      .style('stroke', d => d.source === 'nyt' ? '#2c3e50' : '#3498db')
      .style('stroke-width', 1.5)
      .style('opacity', 0.9)
      .style('filter', 'drop-shadow(0px 1px 3px rgba(0,0,0,0.3))');

    // Source indicators
    articles.append('circle')
      .attr('r', 6)
      .attr('cx', d => {
        const radius = Math.max(12, Math.min(25, (d.title || '').length * 0.4));
        return radius - 4;
      })
      .attr('cy', d => {
        const radius = Math.max(12, Math.min(25, (d.title || '').length * 0.4));
        return -(radius - 4);
      })
      .style('fill', d => d.source === 'nyt' ? '#2c3e50' : '#3498db')
      .style('stroke', '#fff')
      .style('stroke-width', 1);

    // Headlines - adaptive for massive scale
    articles.append('text')
      .text(d => {
        const title = d.title || 'Untitled';
        // Shorter truncation for massive scale
        return title.length > 40 ? title.substring(0, 37) + '...' : title;
      })
      .style('font-size', '9px')  // Smaller font for massive scale
      .style('font-weight', '600')
      .style('fill', '#2c3e50')
      .style('text-anchor', 'middle')
      .attr('dy', d => {
        const radius = Math.max(12, Math.min(25, (d.title || '').length * 0.4));
        return radius + 15;
      })
      .style('pointer-events', 'none');

    // Efficient interaction system for massive scale
    articles
      .on('mouseover', function(event, d) {
        // Simplified hover for performance
        d3.select(this).select('circle')
          .transition().duration(100)
          .style('stroke-width', 3)
          .attr('r', d => {
            const titleLength = (d.title || '').length;
            return Math.max(15, Math.min(30, titleLength * 0.5));
          });
        
        // Lightweight tooltip
        const tooltip = container.append('g')
          .attr('class', 'quick-tooltip')
          .attr('transform', `translate(${d.x + 30}, ${d.y - 25})`);
        
        tooltip.append('rect')
          .attr('width', 220)
          .attr('height', 50)
          .attr('rx', 4)
          .style('fill', 'rgba(0,0,0,0.9)')
          .style('stroke', d.source === 'nyt' ? '#2c3e50' : '#3498db');
        
        tooltip.append('text')
          .attr('x', 8)
          .attr('y', 18)
          .style('fill', '#fff')
          .style('font-size', '10px')
          .style('font-weight', 'bold')
          .text(`${d.source?.toUpperCase()} • ${d.topic_cluster || 'News'}`);
        
        tooltip.append('text')
          .attr('x', 8)
          .attr('y', 38)
          .style('fill', '#4CAF50')
          .style('font-size', '9px')
          .text('👆 Click for details');
      })
      .on('mouseout', function(event, d) {
        d3.select(this).select('circle')
          .transition().duration(50)
          .style('stroke-width', 1.5)
          .attr('r', d => {
            const titleLength = (d.title || '').length;
            return Math.max(12, Math.min(25, titleLength * 0.4));
          });
        
        container.selectAll('.quick-tooltip').remove();
      })
      .on('click', function(event, d) {
        setSelectedNode(d);
        trackAnalytics('view_story', { 
          story_id: d.id, 
          source: d.source,
          topic_cluster: d.topic_cluster 
        });
      });

    // Optimized drag for massive scale
    const drag = d3.drag()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.05).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x; d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      });

    articles.call(drag);

    // High-performance tick function
    simulation.on('tick', () => {
      links
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      nodes.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Improved simulation performance
    simulation.alpha(0.3).alphaDecay(0.02); // Slower decay for better layout
    
    // Add performance monitoring
    let tickCount = 0;
    const maxTicks = 300; // Limit ticks for massive scale
    
    simulation.on('tick.counter', () => {
      tickCount++;
      if (tickCount > maxTicks) {
        simulation.stop();
        console.log(`🎯 MASSIVE SCALE: Simulation completed after ${tickCount} ticks`);
      }
    });

    console.log(`📊 MASSIVE SCALE: Graph rendered successfully - ${articleNodes.length} bubbles connected by ${data.edges.length} relationships`);
  };

  return (
    <div className="app">
      {/* Enhanced Onboarding Modal */}
      {showOnboarding && (
        <div className="modal-overlay">
          <div className="onboarding-modal advanced">
            <div className="modal-header">
              <h2>🚀 Welcome to Advanced News Intelligence</h2>
              <p>Multi-source AI analysis with unprecedented depth and customization</p>
            </div>
            
            <div className="onboarding-features">
              <div className="feature-highlight">
                <span className="feature-icon">🌐</span>
                <div className="feature-content">
                  <h3>Multi-Source Integration</h3>
                  <p>Combine The Guardian and New York Times for comprehensive coverage</p>
                </div>
              </div>
              
              <div className="feature-highlight">
                <span className="feature-icon">🧠</span>
                <div className="feature-content">
                  <h3>Advanced AI Analysis</h3>
                  <p>Confidence scoring, evidence assessment, and sophisticated relationship detection</p>
                </div>
              </div>
              
              <div className="feature-highlight">
                <span className="feature-icon">⚙️</span>
                <div className="feature-content">
                  <h3>Complexity Adaptation</h3>
                  <p>Content adapts from simple headlines to expert-level analysis based on your preference</p>
                </div>
              </div>
            </div>
            
            <div className="onboarding-actions">
              <button 
                className="demo-button advanced"
                onClick={() => {
                  setShowOnboarding(false);
                  loadAdvancedKnowledgeGraph();
                  trackAnalytics('complete_onboarding', { version: 'advanced' });
                }}
              >
                🎯 Explore Advanced Features
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Feedback Modal */}
      {showFeedback && (
        <div className="modal-overlay">
          <div className="feedback-modal enhanced">
            <div className="modal-header">
              <h2>🎯 Advanced Feedback</h2>
              <p>Help us refine the future of news intelligence</p>
              <button className="close-button" onClick={() => setShowFeedback(false)}>×</button>
            </div>
            
            <div className="feedback-content enhanced">
              <div className="feedback-section">
                <label>Overall Experience (1-10)</label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={feedbackData.rating}
                  onChange={(e) => setFeedbackData({...feedbackData, rating: e.target.value})}
                  className="rating-slider"
                />
                <span className="rating-display">{feedbackData.rating}/10</span>
              </div>
              
              <div className="feedback-section">
                <label>Most Interesting Connection</label>
                <input
                  type="text"
                  value={feedbackData.most_interesting_connection}
                  onChange={(e) => setFeedbackData({...feedbackData, most_interesting_connection: e.target.value})}
                  placeholder="e.g., 'EU climate policy affecting US interest rates'"
                  className="feedback-input"
                />
              </div>
              
              <div className="feedback-section">
                <label>What worked best for you?</label>
                <textarea
                  value={feedbackData.comments}
                  onChange={(e) => setFeedbackData({...feedbackData, comments: e.target.value})}
                  placeholder="Multi-source integration, complexity levels, AI explanations, etc."
                  className="feedback-textarea"
                />
              </div>
              
              <div className="feedback-section">
                <label>Suggestions for Improvement</label>
                <textarea
                  value={feedbackData.suggested_improvements}
                  onChange={(e) => setFeedbackData({...feedbackData, suggested_improvements: e.target.value})}
                  placeholder="New features, better visualizations, additional data sources..."
                  className="feedback-textarea"
                />
              </div>
              
              <div className="feedback-section">
                <label>Email (optional, for feature updates)</label>
                <input
                  type="email"
                  value={feedbackData.email}
                  onChange={(e) => setFeedbackData({...feedbackData, email: e.target.value})}
                  placeholder="your.email@example.com"
                  className="feedback-input"
                />
              </div>
              
              <button className="submit-feedback-button enhanced" onClick={submitEnhancedFeedback}>
                🚀 Submit Advanced Feedback
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="header advanced">
        <div className="header-content">
          <div className="header-left">
            <h1 className="title">News Intelligence Platform</h1>
            <p className="subtitle">Multi-Source AI Analysis • Advanced Visualization • Adaptive Complexity</p>
          </div>
          <div className="header-right">
            <div className="analytics-display">
              <div className="analytics-item">
                <span className="analytics-number">{analytics.stories_viewed}</span>
                <span className="analytics-label">Stories Viewed</span>
              </div>
              <div className="analytics-item">
                <span className="analytics-number">{analytics.connections_explored}</span>
                <span className="analytics-label">Connections Explored</span>
              </div>
            </div>
            <button 
              className="feedback-trigger advanced"
              onClick={() => setShowFeedback(true)}
            >
              💡 Share Insights
            </button>
          </div>
        </div>
      </header>

      {/* Advanced Controls */}
      <div className="controls advanced">
        <div className="search-section">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && searchAdvancedNews()}
            placeholder="Advanced search across multiple sources..."
            className="search-input advanced"
          />
          <button onClick={searchAdvancedNews} className="search-button advanced">
            🔍 AI-Powered Search
          </button>
        </div>

        <div className="filters advanced">
          <div className="filter-row primary">
            <div className="filter-group">
              <label>🗞️ News Sources:</label>
              <select value={sources} onChange={(e) => setSources(e.target.value)}>
                {sourceOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <label>📅 Time Range:</label>
              <select value={days} onChange={(e) => setDays(parseInt(e.target.value))}>
                <option value={1}>Last 24 hours</option>
                <option value={3}>Last 3 days</option>
                <option value={7}>Last week</option>
                <option value={14}>Last 2 weeks</option>
              </select>
            </div>

            <div className="filter-group">
              <label>📰 Section:</label>
              <select value={section} onChange={(e) => setSection(e.target.value)}>
                <option value="">All sections</option>
                {sections.slice(1).map(sec => (
                  <option key={sec} value={sec}>
                    {sec.charAt(0).toUpperCase() + sec.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="filter-row secondary">
            <div className="filter-group complexity">
              <label>🎓 Complexity Level:</label>
              <select 
                value={complexityLevel} 
                onChange={(e) => setComplexityLevel(parseInt(e.target.value))}
                className="complexity-select"
              >
                {complexityOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <span className="complexity-description">
                {complexityOptions.find(opt => opt.value === complexityLevel)?.description}
              </span>
            </div>

            <div className="filter-group">
              <label>🌍 Geographic Focus:</label>
              <select value={geographicFocus} onChange={(e) => setGeographicFocus(e.target.value)}>
                {geographicOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="filter-actions">
            <button onClick={loadAdvancedKnowledgeGraph} className="refresh-button advanced">
              🔄 Regenerate Analysis
            </button>
            <button onClick={loadProductionDemo} className="demo-button compact">
              🎭 Load Demo
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content advanced">
        {/* Graph Container */}
        <div className="graph-container advanced">
          {loading && (
            <div className="loading-overlay advanced">
              <div className="loading-content">
                <div className="loading-spinner advanced"></div>
                <h3>🤖 Advanced AI Analysis in Progress</h3>
                <div className="loading-steps">
                  <div className="loading-step">📊 Fetching from multiple sources...</div>
                  <div className="loading-step">🧠 Analyzing story relationships...</div>
                  <div className="loading-step">⚖️ Calculating confidence scores...</div>
                  <div className="loading-step">🎨 Generating visualization...</div>
                </div>
              </div>
            </div>
          )}
          <svg ref={svgRef} className="graph-svg advanced"></svg>
          
          {/* Enhanced Info Panel */}
          {graphData.metadata && (
            <div className="graph-info advanced">
              <div className="info-header">📊 Analysis Summary</div>
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">📰 Articles:</span>
                  <span className="info-value">{graphData.metadata.total_articles || 0}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">🔗 Connections:</span>
                  <span className="info-value">{graphData.metadata.total_connections || 0}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">📡 Sources:</span>
                  <span className="info-value">{graphData.metadata.total_sources || 1}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">🤖 AI Status:</span>
                  <span className="info-value">
                    {graphData.metadata.ai_analysis_enabled ? '✅ Active' : '❌ Offline'}
                  </span>
                </div>
              </div>
              
              {graphData.metadata.advanced_features && (
                <div className="advanced-features">
                  <div className="features-label">🚀 Advanced Features:</div>
                  <div className="features-list">
                    {Object.entries(graphData.metadata.advanced_features).map(([feature, enabled]) => (
                      <span key={feature} className={`feature-badge ${enabled ? 'enabled' : 'disabled'}`}>
                        {feature.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Enhanced Story Detail Panel */}
        {selectedNode && selectedNode.type === 'article' && (
          <div className="detail-panel advanced">
            <div className="detail-header advanced">
              <div className="header-main">
                <div className="source-indicator">
                  <span className={`source-badge ${selectedNode.source}`}>
                    {selectedNode.source === 'nyt' ? '🇺🇸 NYT' : '🇬🇧 Guardian'}
                  </span>
                  <span className="complexity-indicator">
                    Level {selectedNode.complexity_level || 3}/5
                  </span>
                </div>
                <h3>{selectedNode.title}</h3>
              </div>
              <button 
                className="close-button"
                onClick={() => setSelectedNode(null)}
              >
                ×
              </button>
            </div>
            
            <div className="detail-content advanced">
              <div className="story-meta advanced">
                <span className="section-badge">{selectedNode.section}</span>
                <span className="date">
                  {new Date(selectedNode.publication_date).toLocaleDateString()}
                </span>
                {selectedNode.author && (
                  <span className="author">By {selectedNode.author}</span>
                )}
                <span className="read-time">
                  📖 {selectedNode.read_time_minutes || 3} min read
                </span>
              </div>

              {selectedNode.sentiment_score !== undefined && (
                <div className="sentiment-analysis">
                  <h4>📊 Sentiment Analysis</h4>
                  <div className="sentiment-bar">
                    <div 
                      className="sentiment-fill"
                      style={{
                        width: `${((selectedNode.sentiment_score + 1) / 2) * 100}%`,
                        backgroundColor: selectedNode.sentiment_score > 0 ? '#27ae60' : selectedNode.sentiment_score < 0 ? '#e74c3c' : '#95a5a6'
                      }}
                    ></div>
                  </div>
                  <span className="sentiment-label">
                    {selectedNode.sentiment_score > 0.2 ? '😊 Positive' : 
                     selectedNode.sentiment_score < -0.2 ? '😟 Negative' : '😐 Neutral'}
                  </span>
                </div>
              )}

              <div className="ai-analysis-badge advanced">
                <span className="ai-icon">🤖</span>
                <span>AI-Generated Analysis • Complexity Level {selectedNode.complexity_level || 3}</span>
              </div>

              <div className="story-lede">
                <h4>📰 Lede</h4>
                <p>{selectedNode.lede}</p>
              </div>

              <div className="story-nutgraf">
                <h4>🎯 Why This Matters</h4>
                <p>{selectedNode.nutgraf}</p>
              </div>

              <div className="story-summary">
                <h4>📋 AI Summary</h4>
                <p>{selectedNode.summary}</p>
              </div>

              {selectedNode.entities && selectedNode.entities.length > 0 && (
                <div className="entities-section">
                  <h4>🏷️ Key Entities</h4>
                  <div className="entities-list">
                    {selectedNode.entities.map((entity, index) => (
                      <span key={index} className="entity-tag">{entity}</span>
                    ))}
                  </div>
                </div>
              )}

              <div className="story-engagement">
                <h4>📱 Social Media Preview</h4>
                <p className="engagement-preview advanced">{selectedNode.engagement_preview}</p>
              </div>

              <div className="story-actions advanced">
                <a 
                  href={selectedNode.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="read-full-button advanced"
                >
                  📖 Read Full Article
                </a>
                <button 
                  className="connection-button"
                  onClick={() => {
                    // Highlight connections
                    trackAnalytics('explore_connection', { 
                      story_id: selectedNode.id,
                      action: 'highlight_connections' 
                    });
                  }}
                >
                  🔗 Show All Connections
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Enhanced Legend for Causality */}
      <div className="legend advanced">
        <h4>🔗 Causal Connection Types</h4>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#e74c3c', height: '4px'}}></div>
            <span>⚡ Direct Causal</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#f39c12', height: '4px'}}></div>
            <span>💰 Economic Causal</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#3498db', height: '3px'}}></div>
            <span>🏛️ Political Causal</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#9b59b6', height: '3px'}}></div>
            <span>👥 Social Causal</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#27ae60', height: '3px'}}></div>
            <span>🌍 Environmental Causal</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#e67e22', height: '2px'}}></div>
            <span>🔄 Indirect Causal</span>
          </div>
        </div>
        <div className="legend-explanation advanced">
          <p className="legend-note">
            📏 <strong>Line thickness = causality strength</strong><br/>
            🎯 <strong>Full headlines displayed in bubbles</strong><br/>
            🏷️ <strong>Stories clustered by topic (Business, Politics, Culture, etc.)</strong><br/>
            ✨ <strong>Hover over bubbles to see expanded summaries</strong><br/>
            🔗 <strong>Every story connects to others - showing the web of causality</strong>
          </p>
        </div>
      </div>
    </div>
  );
};

export default App;