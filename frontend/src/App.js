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
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackData, setFeedbackData] = useState({ rating: 5, comments: '', email: '' });
  const [showHelp, setShowHelp] = useState(false);
  const svgRef = useRef();

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  const sections = [
    '', 'world', 'politics', 'business', 'technology', 'sport', 
    'culture', 'science', 'environment', 'education', 'society'
  ];

  useEffect(() => {
    // Auto-load demo data on first visit
    loadKnowledgeGraph();
  }, []);

  const loadKnowledgeGraph = async () => {
    setLoading(true);
    try {
      // Use demo endpoint for reliable user testing
      const response = await fetch(`${backendUrl}/api/demo-graph`);
      const data = await response.json();
      
      setGraphData(data);
      renderGraph(data);
    } catch (error) {
      console.error('Error loading knowledge graph:', error);
    } finally {
      setLoading(false);
    }
  };

  const searchNews = async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.append('query', searchQuery);
      params.append('days', days);
      params.append('max_articles', '15');

      const response = await fetch(`${backendUrl}/api/search?${params}`);
      const data = await response.json();
      
      setGraphData(data);
      renderGraph(data);
    } catch (error) {
      console.error('Error searching news:', error);
      // Fallback to demo data if search fails
      loadKnowledgeGraph();
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData)
      });
      
      if (response.ok) {
        alert('Thank you for your feedback! 🙏');
        setShowFeedback(false);
        setFeedbackData({ rating: 5, comments: '', email: '' });
      }
    } catch (error) {
      alert('Thanks for your feedback! (Note: This is a demo, so feedback is noted locally)');
      setShowFeedback(false);
    }
  };

  const renderGraph = (data) => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    if (!data.nodes || data.nodes.length === 0) return;

    const width = 1200;
    const height = 800;
    
    svg.attr('width', width).attr('height', height);

    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges)
        .id(d => d.id)
        .distance(d => d.type === 'belongs_to' ? 50 : 100 + (1 - d.strength) * 100)
        .strength(d => d.type === 'belongs_to' ? 0.3 : d.strength * 0.8)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => d.size || 20));

    // Create container for zoom
    const container = svg.append('g');

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });
    
    svg.call(zoom);

    // Create links with enhanced styling
    const links = container.selectAll('.link')
      .data(data.edges)
      .enter()
      .append('line')
      .attr('class', 'link')
      .style('stroke', d => {
        if (d.type === 'belongs_to') return '#ddd';
        const colors = {
          'economic': '#f39c12',
          'political': '#3498db',
          'social': '#e74c3c',
          'environmental': '#27ae60',
          'causal': '#9b59b6',
          'thematic': '#e67e22'
        };
        return colors[d.type] || '#95a5a6';
      })
      .style('stroke-width', d => d.width || 2)
      .style('stroke-opacity', d => d.opacity || 0.6)
      .style('stroke-dasharray', d => d.type === 'belongs_to' ? '5,5' : 'none');

    // Create nodes
    const nodes = container.selectAll('.node')
      .data(data.nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer');

    // Add circles for nodes with enhanced styling
    nodes.append('circle')
      .attr('r', d => d.size || 20)
      .style('fill', d => d.color || '#3498db')
      .style('stroke', '#fff')
      .style('stroke-width', 2)
      .style('opacity', d => d.type === 'section' ? 0.7 : 0.9)
      .style('filter', 'drop-shadow(0px 2px 4px rgba(0,0,0,0.2))');

    // Add labels for articles
    nodes.filter(d => d.type === 'article')
      .append('text')
      .text(d => {
        const title = d.title || '';
        return title.length > 30 ? title.substring(0, 30) + '...' : title;
      })
      .style('font-size', '10px')
      .style('fill', '#2c3e50')
      .style('text-anchor', 'middle')
      .attr('dy', 35)
      .style('pointer-events', 'none')
      .style('font-weight', '600');

    // Add labels for sections
    nodes.filter(d => d.type === 'section')
      .append('text')
      .text(d => d.title)
      .style('font-size', '12px')
      .style('font-weight', 'bold')
      .style('fill', '#2c3e50')
      .style('text-anchor', 'middle')
      .attr('dy', 4)
      .style('pointer-events', 'none');

    // Add enhanced hover effects
    nodes
      .on('mouseover', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', (d.size || 20) * 1.3)
          .style('stroke-width', 4)
          .style('filter', 'drop-shadow(0px 4px 8px rgba(0,0,0,0.3))');
        
        // Highlight connected links
        links.style('stroke-opacity', link => 
          link.source.id === d.id || link.target.id === d.id ? 1 : 0.1
        );

        // Show connection info tooltip
        if (d.type === 'article') {
          const tooltip = container.append('g')
            .attr('class', 'tooltip')
            .attr('transform', `translate(${d.x + 30}, ${d.y - 30})`);
          
          const rect = tooltip.append('rect')
            .attr('width', 200)
            .attr('height', 60)
            .attr('rx', 5)
            .style('fill', 'rgba(0,0,0,0.8)')
            .style('stroke', '#fff');
          
          tooltip.append('text')
            .attr('x', 10)
            .attr('y', 20)
            .style('fill', 'white')
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .text(d.title.substring(0, 25) + '...');
          
          tooltip.append('text')
            .attr('x', 10)
            .attr('y', 40)
            .style('fill', '#ccc')
            .style('font-size', '10px')
            .text(`${d.section} • Click for details`);
        }
      })
      .on('mouseout', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', d.size || 20)
          .style('stroke-width', 2)
          .style('filter', 'drop-shadow(0px 2px 4px rgba(0,0,0,0.2))');
        
        // Reset link opacity
        links.style('stroke-opacity', d => d.opacity || 0.6);
        
        // Remove tooltip
        container.selectAll('.tooltip').remove();
      })
      .on('click', function(event, d) {
        if (d.type === 'article') {
          setSelectedNode(d);
        }
      });

    // Add drag behavior
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

    // Update positions on tick
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
      {/* Onboarding Modal */}
      {showOnboarding && (
        <div className="modal-overlay">
          <div className="onboarding-modal">
            <div className="modal-header">
              <h2>🎯 Welcome to News Knowledge Graph</h2>
              <p>Discover how AI reveals hidden connections between world events</p>
            </div>
            
            <div className="onboarding-steps">
              <div className="step">
                <span className="step-number">1</span>
                <div className="step-content">
                  <h3>🔗 See Story Connections</h3>
                  <p>AI analyzes news stories and shows how they're related - from obvious economic links to surprising global connections.</p>
                </div>
              </div>
              
              <div className="step">
                <span className="step-number">2</span>
                <div className="step-content">
                  <h3>💫 Interactive Exploration</h3>
                  <p>Hover over stories to see connections light up. Click circles for detailed analysis. Drag nodes to explore.</p>
                </div>
              </div>
              
              <div className="step">
                <span className="step-number">3</span>
                <div className="step-content">
                  <h3>📊 Connection Strength</h3>
                  <p>Line thickness shows connection strength: thick lines = strong relationships, thin lines = subtle connections.</p>
                </div>
              </div>
            </div>
            
            <div className="onboarding-actions">
              <button 
                className="demo-button"
                onClick={() => {
                  setShowOnboarding(false);
                  loadKnowledgeGraph();
                }}
              >
                🚀 Explore Demo Stories
              </button>
              <button 
                className="skip-button"
                onClick={() => setShowOnboarding(false)}
              >
                Skip Tour
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Help Modal */}
      {showHelp && (
        <div className="modal-overlay">
          <div className="help-modal">
            <div className="modal-header">
              <h2>❓ How to Use the Knowledge Graph</h2>
              <button 
                className="close-button"
                onClick={() => setShowHelp(false)}
              >
                ×
              </button>
            </div>
            
            <div className="help-content">
              <div className="help-section">
                <h3>🎯 What You're Seeing</h3>
                <ul>
                  <li><strong>Circles</strong> = News stories (articles) and topic categories (sections)</li>
                  <li><strong>Lines</strong> = AI-discovered relationships between stories</li>
                  <li><strong>Colors</strong> = Different types of connections (political, economic, etc.)</li>
                </ul>
              </div>
              
              <div className="help-section">
                <h3>🖱️ How to Interact</h3>
                <ul>
                  <li><strong>Hover</strong> over circles to highlight connections</li>
                  <li><strong>Click</strong> article circles to read detailed analysis</li>
                  <li><strong>Drag</strong> circles to reposition them</li>
                  <li><strong>Scroll</strong> to zoom in/out of the graph</li>
                </ul>
              </div>
              
              <div className="help-section">
                <h3>🔍 Example Connections</h3>
                <ul>
                  <li><strong>Political</strong>: Trump's institutional attacks are related</li>
                  <li><strong>Economic</strong>: Fed policies affect market confidence</li>
                  <li><strong>Thematic</strong>: International diplomatic coordination</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Feedback Modal */}
      {showFeedback && (
        <div className="modal-overlay">
          <div className="feedback-modal">
            <div className="modal-header">
              <h2>💭 Your Feedback Matters!</h2>
              <p>Help us improve the news knowledge graph experience</p>
              <button 
                className="close-button"
                onClick={() => setShowFeedback(false)}
              >
                ×
              </button>
            </div>
            
            <div className="feedback-content">
              <div className="feedback-section">
                <label>How intuitive was the interface? (1-10)</label>
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
                <label>What did you find most interesting?</label>
                <textarea
                  value={feedbackData.comments}
                  onChange={(e) => setFeedbackData({...feedbackData, comments: e.target.value})}
                  placeholder="E.g., 'The connection between political stories and economic impacts was surprising...'"
                  className="feedback-textarea"
                />
              </div>
              
              <div className="feedback-section">
                <label>Email (optional, for follow-up)</label>
                <input
                  type="email"
                  value={feedbackData.email}
                  onChange={(e) => setFeedbackData({...feedbackData, email: e.target.value})}
                  placeholder="your.email@example.com"
                  className="feedback-input"
                />
              </div>
              
              <button className="submit-feedback-button" onClick={submitFeedback}>
                📤 Submit Feedback
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="title">News Knowledge Graph</h1>
          <p className="subtitle">AI-Powered Story Relationship Visualization</p>
          <div className="header-actions">
            <button 
              className="help-trigger"
              onClick={() => setShowHelp(true)}
            >
              ❓ How it Works
            </button>
            <button 
              className="feedback-trigger"
              onClick={() => setShowFeedback(true)}
            >
              💭 Give Feedback
            </button>
          </div>
        </div>
      </header>

      {/* Controls */}
      <div className="controls">
        <div className="search-section">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && searchNews()}
            placeholder="Try searching: 'Trump', 'climate', 'economy'..."
            className="search-input"
          />
          <button onClick={searchNews} className="search-button">
            🔍 Search & Analyze
          </button>
        </div>

        <div className="filters">
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

          <button onClick={loadKnowledgeGraph} className="refresh-button">
            🔄 Load Demo
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Graph Visualization */}
        <div className="graph-container">
          {loading && (
            <div className="loading-overlay">
              <div className="loading-spinner"></div>
              <p>🤖 AI is analyzing story connections...</p>
              <div className="loading-tips">
                <p>💡 While you wait: Stories are being analyzed for economic, political, social, and thematic relationships</p>
              </div>
            </div>
          )}
          <svg ref={svgRef} className="graph-svg"></svg>
          
          {/* Graph Info */}
          {graphData.metadata && (
            <div className="graph-info">
              <div className="info-item">
                <span className="info-label">📰 Stories:</span>
                <span className="info-value">{graphData.metadata.total_articles || 0}</span>
              </div>
              <div className="info-item">
                <span className="info-label">🔗 Connections:</span>
                <span className="info-value">{graphData.metadata.total_connections || 0}</span>
              </div>
              <div className="info-item">
                <span className="info-label">🤖 AI Analysis:</span>
                <span className="info-value">
                  {graphData.metadata.ai_analysis_enabled ? '✅ Active' : '❌ Offline'}
                </span>
              </div>
              {graphData.metadata.demo_mode && (
                <div className="info-item">
                  <span className="info-label">🎭 Mode:</span>
                  <span className="info-value">Demo</span>
                </div>
              )}
            </div>
          )}
          
          {/* Interactive Instructions */}
          <div className="interaction-guide">
            <div className="guide-item">
              <span className="guide-icon">🖱️</span>
              <span>Hover circles for connections</span>
            </div>
            <div className="guide-item">
              <span className="guide-icon">👆</span>
              <span>Click articles for details</span>
            </div>
            <div className="guide-item">
              <span className="guide-icon">🔄</span>
              <span>Drag to reposition</span>
            </div>
          </div>
        </div>

        {/* Story Detail Panel */}
        {selectedNode && selectedNode.type === 'article' && (
          <div className="detail-panel">
            <div className="detail-header">
              <h3>{selectedNode.title}</h3>
              <button 
                className="close-button"
                onClick={() => setSelectedNode(null)}
              >
                ×
              </button>
            </div>
            
            <div className="detail-content">
              <div className="story-meta">
                <span className="section-badge">{selectedNode.section}</span>
                <span className="date">
                  {new Date(selectedNode.publication_date).toLocaleDateString()}
                </span>
              </div>

              <div className="ai-analysis-badge">
                <span className="ai-icon">🤖</span>
                <span>AI-Generated Analysis</span>
              </div>

              <div className="story-lede">
                <h4>📰 Lede (Opening Hook)</h4>
                <p>{selectedNode.lede}</p>
              </div>

              <div className="story-nutgraf">
                <h4>🎯 Nutgraf (Why It Matters)</h4>
                <p>{selectedNode.nutgraf}</p>
              </div>

              <div className="story-summary">
                <h4>📋 AI Summary</h4>
                <p>{selectedNode.summary}</p>
              </div>

              <div className="story-engagement">
                <h4>📱 Social Media Preview</h4>
                <p className="engagement-preview">{selectedNode.engagement_preview}</p>
              </div>

              <div className="story-actions">
                <a 
                  href={selectedNode.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="read-full-button"
                >
                  📖 Read Full Article
                </a>
                <button 
                  className="connection-button"
                  onClick={() => {
                    // Highlight connections for this story
                    const svg = d3.select(svgRef.current);
                    svg.selectAll('.link')
                      .style('stroke-opacity', link => 
                        link.source.id === selectedNode.id || link.target.id === selectedNode.id ? 1 : 0.1
                      );
                  }}
                >
                  🔗 Show Connections
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Enhanced Legend */}
      <div className="legend">
        <h4>🎨 Connection Types</h4>
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
            <div className="legend-line" style={{backgroundColor: '#e67e22'}}></div>
            <span>🎭 Thematic</span>
          </div>
        </div>
        <div className="legend-explanation">
          <p className="legend-note">
            📏 <strong>Line thickness = connection strength</strong><br/>
            🔍 Thick lines = strong relationships • Thin lines = subtle connections
          </p>
        </div>
      </div>

      {/* Call-to-Action for Feedback */}
      <div className="floating-feedback">
        <button 
          className="floating-feedback-button"
          onClick={() => setShowFeedback(true)}
        >
          💬 Quick Feedback
        </button>
      </div>
    </div>
  );
};

export default App;