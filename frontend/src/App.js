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

    const width = 1400;  // Reduced from 1600
    const height = 800;  // Reduced from 1000
    
    svg.attr('width', width).attr('height', height);

    // Simplified, faster force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges)
        .id(d => d.id)
        .distance(120)  // Fixed distance for speed
        .strength(0.6)  // Fixed strength for speed
      )
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => (d.size || 25) + 8))
      .alpha(0.3)  // Start with lower alpha for faster settling
      .alphaDecay(0.05); // Faster decay

    // Container with simplified zoom
    const container = svg.append('g');
    const zoom = d3.zoom()
      .scaleExtent([0.5, 3])  // Reduced zoom range for performance
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });
    
    svg.call(zoom);

    // Simplified links - no complex hover effects initially
    const links = container.selectAll('.link')
      .data(data.edges)
      .enter()
      .append('line')
      .attr('class', 'link')
      .style('stroke', d => {
        if (d.type === 'belongs_to_cluster') return '#e0e0e0';
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
      .style('stroke-width', d => Math.min(8, d.width || 2))  // Cap width for performance
      .style('stroke-opacity', d => d.opacity || 0.6);

    // Simplified nodes - reduced DOM complexity
    const nodes = container.selectAll('.node')
      .data(data.nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer');

    // Topic cluster nodes - simplified
    nodes.filter(d => d.type === 'topic_cluster')
      .append('circle')
      .attr('r', d => Math.min(50, d.size || 40))  // Cap size
      .style('fill', d => d.color)
      .style('opacity', 0.15)
      .style('stroke', d => d.color)
      .style('stroke-width', 1);

    // Article nodes - optimized for speed
    const articleNodes = nodes.filter(d => d.type === 'article');
    
    articleNodes.append('circle')
      .attr('r', d => Math.max(25, Math.min(50, d.title.length * 0.6)))  // Smaller, capped sizes
      .style('fill', d => d.color || '#3498db')
      .style('stroke', d => d.source === 'nyt' ? '#2c3e50' : '#3498db')
      .style('stroke-width', 2)
      .style('opacity', 0.9);

    // Simplified source indicators
    articleNodes.append('circle')
      .attr('r', 8)
      .attr('cx', d => Math.max(25, Math.min(50, d.title.length * 0.6)) - 6)
      .attr('cy', d => -(Math.max(25, Math.min(50, d.title.length * 0.6)) - 6))
      .style('fill', d => d.source === 'nyt' ? '#2c3e50' : '#3498db')
      .style('stroke', '#fff')
      .style('stroke-width', 2);

    // Simplified headlines using text instead of foreignObject for better performance
    articleNodes.append('text')
      .text(d => {
        // Intelligent truncation for performance
        const title = d.title || '';
        return title.length > 60 ? title.substring(0, 57) + '...' : title;
      })
      .style('font-size', '11px')
      .style('font-weight', '600')
      .style('fill', '#2c3e50')
      .style('text-anchor', 'middle')
      .attr('dy', d => Math.max(25, Math.min(50, d.title.length * 0.6)) + 20)
      .style('pointer-events', 'none')
      .call(wrap, 120); // Text wrapping function

    // Topic cluster labels - simplified
    nodes.filter(d => d.type === 'topic_cluster')
      .append('text')
      .text(d => `${d.title} (${d.story_count})`)
      .style('font-size', '12px')
      .style('font-weight', 'bold')
      .style('fill', d => d.color)
      .style('text-anchor', 'middle')
      .style('pointer-events', 'none');

    // Optimized hover effects - only for article nodes
    articleNodes
      .on('mouseover', function(event, d) {
        // Simplified hover effect
        d3.select(this).select('circle')
          .transition().duration(150)
          .attr('r', d => Math.max(30, Math.min(60, d.title.length * 0.8)))
          .style('stroke-width', 3);
        
        // Simple tooltip - no complex DOM manipulation
        const tooltip = container.append('g')
          .attr('class', 'simple-tooltip')
          .attr('transform', `translate(${d.x + 40}, ${d.y - 30})`);
        
        tooltip.append('rect')
          .attr('width', 300)
          .attr('height', 80)
          .attr('rx', 6)
          .style('fill', 'rgba(0,0,0,0.9)')
          .style('stroke', d.source === 'nyt' ? '#2c3e50' : '#3498db');
        
        tooltip.append('text')
          .attr('x', 10)
          .attr('y', 20)
          .style('fill', '#fff')
          .style('font-size', '12px')
          .style('font-weight', 'bold')
          .text(d.source?.toUpperCase() || 'NEWS');
        
        tooltip.append('text')
          .attr('x', 10)
          .attr('y', 40)
          .style('fill', '#fff')
          .style('font-size', '11px')
          .text(d.title.substring(0, 35) + (d.title.length > 35 ? '...' : ''));
        
        tooltip.append('text')
          .attr('x', 10)
          .attr('y', 65)
          .style('fill', '#4CAF50')
          .style('font-size', '10px')
          .text('👆 Click for details');
      })
      .on('mouseout', function(event, d) {
        d3.select(this).select('circle')
          .transition().duration(100)
          .attr('r', d => Math.max(25, Math.min(50, d.title.length * 0.6)))
          .style('stroke-width', 2);
        
        container.selectAll('.simple-tooltip').remove();
      })
      .on('click', function(event, d) {
        setSelectedNode(d);
        trackAnalytics('view_story', { 
          story_id: d.id, 
          source: d.source,
          topic_cluster: d.topic_cluster 
        });
      });

    // Simplified drag behavior
    const drag = d3.drag()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.1).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x; d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      });

    articleNodes.call(drag);

    // Optimized simulation tick
    simulation.on('tick', () => {
      links
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      nodes.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Stop simulation early for better performance
    simulation.tick(50); // Pre-compute 50 ticks
    simulation.stop();
  };

  // Text wrapping helper function
  function wrap(text, width) {
    text.each(function() {
      const text = d3.select(this);
      const words = text.text().split(/\s+/).reverse();
      let word;
      let line = [];
      let lineNumber = 0;
      const lineHeight = 1.1;
      const y = text.attr("y");
      const dy = parseFloat(text.attr("dy"));
      let tspan = text.text(null).append("tspan").attr("x", 0).attr("y", y).attr("dy", dy + "em");
      
      while (word = words.pop()) {
        line.push(word);
        tspan.text(line.join(" "));
        if (tspan.node().getComputedTextLength() > width) {
          line.pop();
          tspan.text(line.join(" "));
          line = [word];
          tspan = text.append("tspan").attr("x", 0).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
        }
      }
    });
  }

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