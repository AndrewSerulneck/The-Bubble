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

      const response = await fetch(`${backendUrl}/api/v3/knowledge-graph/advanced?${params}`);
      
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
      const response = await fetch(`${backendUrl}/api/v3/demo/production`);
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

      const response = await fetch(`${backendUrl}/api/v3/news/search?${params}`);
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

    const width = 1400;
    const height = 900;
    
    svg.attr('width', width).attr('height', height);

    // Enhanced force simulation with more sophisticated physics
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges)
        .id(d => d.id)
        .distance(d => {
          if (d.type === 'belongs_to_section' || d.type === 'belongs_to_source') return 60;
          return 120 + (1 - d.strength) * 150;
        })
        .strength(d => {
          if (d.type === 'belongs_to_section' || d.type === 'belongs_to_source') return 0.2;
          return d.confidence ? d.strength * d.confidence : d.strength * 0.7;
        })
      )
      .force('charge', d3.forceManyBody()
        .strength(d => d.type === 'article' ? -400 : -200)
      )
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide()
        .radius(d => (d.size || 20) + 5)
      )
      .force('x', d3.forceX(width / 2).strength(0.1))
      .force('y', d3.forceY(height / 2).strength(0.1));

    // Container with zoom
    const container = svg.append('g');

    // Enhanced zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });
    
    svg.call(zoom);

    // Gradient definitions for enhanced visuals
    const defs = svg.append('defs');
    
    // Gradient for confidence-based connections
    const gradients = ['high-confidence', 'medium-confidence', 'low-confidence'];
    gradients.forEach((grad, i) => {
      const gradient = defs.append('linearGradient')
        .attr('id', grad)
        .attr('gradientUnits', 'userSpaceOnUse');
      
      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', d3.schemeCategory10[i])
        .attr('stop-opacity', 0.8);
      
      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', d3.schemeCategory10[i])
        .attr('stop-opacity', 0.3);
    });

    // Enhanced links with confidence visualization
    const links = container.selectAll('.link')
      .data(data.edges)
      .enter()
      .append('line')
      .attr('class', 'link')
      .style('stroke', d => {
        if (d.type === 'belongs_to_section' || d.type === 'belongs_to_source') return '#ddd';
        
        const colors = {
          'economic': '#f39c12',
          'political': '#3498db',
          'social': '#e74c3c',
          'environmental': '#27ae60',
          'causal': '#9b59b6',
          'thematic': '#e67e22',
          'geographic': '#1abc9c'
        };
        return colors[d.type] || '#95a5a6';
      })
      .style('stroke-width', d => d.width || 2)
      .style('stroke-opacity', d => d.opacity || 0.6)
      .style('stroke-dasharray', d => {
        if (d.type === 'belongs_to_section' || d.type === 'belongs_to_source') return '5,5';
        if (d.confidence && d.confidence < 0.5) return '10,5'; // Dashed for low confidence
        return 'none';
      })
      .on('mouseover', function(event, d) {
        if (d.explanation) {
          // Show connection tooltip
          const tooltip = container.append('g')
            .attr('class', 'connection-tooltip')
            .attr('transform', `translate(${event.offsetX}, ${event.offsetY})`);
          
          const rect = tooltip.append('rect')
            .attr('width', 300)
            .attr('height', 80)
            .attr('rx', 8)
            .style('fill', 'rgba(0,0,0,0.9)')
            .style('stroke', '#fff');
          
          tooltip.append('text')
            .attr('x', 10)
            .attr('y', 20)
            .style('fill', 'white')
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .text(`${d.type.toUpperCase()} Connection`);
          
          tooltip.append('text')
            .attr('x', 10)
            .attr('y', 40)
            .style('fill', '#ccc')
            .style('font-size', '10px')
            .text(`Strength: ${d.strength?.toFixed(1) || 'N/A'} | Confidence: ${d.confidence?.toFixed(1) || 'N/A'}`);
          
          const explanation = d.explanation.length > 50 
            ? d.explanation.substring(0, 47) + '...' 
            : d.explanation;
          
          tooltip.append('text')
            .attr('x', 10)
            .attr('y', 60)
            .style('fill', 'white')
            .style('font-size', '10px')
            .text(explanation);
        }
      })
      .on('mouseout', function() {
        container.selectAll('.connection-tooltip').remove();
      });

    // Enhanced nodes with source indicators
    const nodes = container.selectAll('.node')
      .data(data.nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer');

    // Node circles with enhanced styling
    nodes.append('circle')
      .attr('r', d => d.size || 20)
      .style('fill', d => d.color || '#3498db')
      .style('stroke', d => {
        if (d.source === 'nyt') return '#2c3e50';
        if (d.source === 'guardian') return '#3498db';
        return '#fff';
      })
      .style('stroke-width', d => d.type === 'article' ? 3 : 2)
      .style('opacity', d => d.type === 'section' || d.type === 'source' ? 0.8 : 0.95)
      .style('filter', 'drop-shadow(0px 2px 6px rgba(0,0,0,0.3))');

    // Source indicator for articles
    nodes.filter(d => d.type === 'article' && d.source)
      .append('circle')
      .attr('r', 8)
      .attr('cx', d => (d.size || 20) - 5)
      .attr('cy', d => -((d.size || 20) - 5))
      .style('fill', d => d.source === 'nyt' ? '#2c3e50' : '#3498db')
      .style('stroke', '#fff')
      .style('stroke-width', 2);

    // Enhanced labels with source information
    nodes.filter(d => d.type === 'article')
      .append('text')
      .text(d => {
        const title = d.title || '';
        const truncated = title.length > 35 ? title.substring(0, 35) + '...' : title;
        return truncated;
      })
      .style('font-size', '11px')
      .style('fill', '#2c3e50')
      .style('text-anchor', 'middle')
      .attr('dy', d => (d.size || 20) + 15)
      .style('pointer-events', 'none')
      .style('font-weight', '600');

    // Source labels for article nodes
    nodes.filter(d => d.type === 'article' && d.source)
      .append('text')
      .text(d => d.source.toUpperCase())
      .style('font-size', '8px')
      .style('fill', '#666')
      .style('text-anchor', 'middle')
      .attr('dy', d => (d.size || 20) + 28)
      .style('pointer-events', 'none')
      .style('font-weight', 'bold');

    // Section and source node labels
    nodes.filter(d => d.type === 'section' || d.type === 'source')
      .append('text')
      .text(d => d.title)
      .style('font-size', d => d.type === 'source' ? '14px' : '12px')
      .style('font-weight', 'bold')
      .style('fill', '#2c3e50')
      .style('text-anchor', 'middle')
      .attr('dy', 5)
      .style('pointer-events', 'none');

    // Enhanced hover effects
    nodes
      .on('mouseover', function(event, d) {
        const node = d3.select(this);
        
        node.select('circle')
          .transition()
          .duration(200)
          .attr('r', (d.size || 20) * 1.2)
          .style('stroke-width', d.type === 'article' ? 5 : 3)
          .style('filter', 'drop-shadow(0px 4px 12px rgba(0,0,0,0.4))');
        
        // Highlight connected links
        links.style('stroke-opacity', link => 
          link.source.id === d.id || link.target.id === d.id ? 1 : 0.1
        );

        // Enhanced tooltip for articles
        if (d.type === 'article') {
          const tooltip = container.append('g')
            .attr('class', 'article-tooltip')
            .attr('transform', `translate(${d.x + 40}, ${d.y - 40})`);
          
          const rect = tooltip.append('rect')
            .attr('width', 350)
            .attr('height', 120)
            .attr('rx', 10)
            .style('fill', 'rgba(0,0,0,0.95)')
            .style('stroke', d.source === 'nyt' ? '#2c3e50' : '#3498db')
            .style('stroke-width', 2);
          
          // Source badge
          tooltip.append('rect')
            .attr('x', 10)
            .attr('y', 10)
            .attr('width', 50)
            .attr('height', 20)
            .attr('rx', 10)
            .style('fill', d.source === 'nyt' ? '#2c3e50' : '#3498db');
          
          tooltip.append('text')
            .attr('x', 35)
            .attr('y', 23)
            .style('fill', 'white')
            .style('font-size', '10px')
            .style('font-weight', 'bold')
            .style('text-anchor', 'middle')
            .text(d.source?.toUpperCase() || 'NEWS');
          
          // Title
          tooltip.append('text')
            .attr('x', 70)
            .attr('y', 25)
            .style('fill', 'white')
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .text(d.title.substring(0, 35) + (d.title.length > 35 ? '...' : ''));
          
          // Metadata
          tooltip.append('text')
            .attr('x', 10)
            .attr('y', 50)
            .style('fill', '#ccc')
            .style('font-size', '10px')
            .text(`${d.section} • ${d.read_time_minutes || 3} min read • Complexity: ${d.complexity_level || 3}/5`);
          
          // Summary
          const summary = d.summary || d.lede || 'Click for full analysis';
          const truncatedSummary = summary.length > 70 ? summary.substring(0, 67) + '...' : summary;
          
          tooltip.append('text')
            .attr('x', 10)
            .attr('y', 75)
            .style('fill', 'white')
            .style('font-size', '11px')
            .text(truncatedSummary);
          
          // Call to action
          tooltip.append('text')
            .attr('x', 10)
            .attr('y', 100)
            .style('fill', '#4CAF50')
            .style('font-size', '10px')
            .style('font-weight', 'bold')
            .text('👆 Click for detailed AI analysis');
        }
      })
      .on('mouseout', function(event, d) {
        const node = d3.select(this);
        
        node.select('circle')
          .transition()
          .duration(200)
          .attr('r', d.size || 20)
          .style('stroke-width', d.type === 'article' ? 3 : 2)
          .style('filter', 'drop-shadow(0px 2px 6px rgba(0,0,0,0.3))');
        
        // Reset link opacity
        links.style('stroke-opacity', d => d.opacity || 0.6);
        
        // Remove tooltips
        container.selectAll('.article-tooltip').remove();
        container.selectAll('.connection-tooltip').remove();
      })
      .on('click', function(event, d) {
        if (d.type === 'article') {
          setSelectedNode(d);
          trackAnalytics('view_story', { 
            story_id: d.id, 
            source: d.source,
            complexity_level: d.complexity_level 
          });
        }
      });

    // Drag behavior
    const drag = d3.drag()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    nodes.call(drag);

    // Update positions
    simulation.on('tick', () => {
      links
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      nodes.attr('transform', d => `translate(${d.x},${d.y})`);
    });
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

      {/* Enhanced Legend */}
      <div className="legend advanced">
        <h4>🎨 Advanced Connection Types</h4>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#f39c12'}}></div>
            <span>💰 Economic</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#3498db'}}></div>
            <span>🏛️ Political</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#e74c3c'}}></div>
            <span>👥 Social</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#27ae60'}}></div>
            <span>🌍 Environmental</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#9b59b6'}}></div>
            <span>⚡ Causal</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#1abc9c'}}></div>
            <span>📍 Geographic</span>
          </div>
        </div>
        <div className="legend-explanation advanced">
          <p className="legend-note">
            📏 <strong>Line thickness = connection strength</strong><br/>
            ⚖️ <strong>Solid lines = high confidence • Dashed = lower confidence</strong><br/>
            🏷️ <strong>Source badges indicate article origins</strong>
          </p>
        </div>
      </div>
    </div>
  );
};

export default App;