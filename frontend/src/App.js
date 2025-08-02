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
  const svgRef = useRef();

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  const sections = [
    '', 'world', 'politics', 'business', 'technology', 'sport', 
    'culture', 'science', 'environment', 'education', 'society'
  ];

  useEffect(() => {
    loadKnowledgeGraph();
  }, []);

  const loadKnowledgeGraph = async () => {
    setLoading(true);
    try {
      // Use the demo endpoint for reliable data
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
    } finally {
      setLoading(false);
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

    // Create links
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

    // Add circles for nodes
    nodes.append('circle')
      .attr('r', d => d.size || 20)
      .style('fill', d => d.color || '#3498db')
      .style('stroke', '#fff')
      .style('stroke-width', 2)
      .style('opacity', d => d.type === 'section' ? 0.7 : 0.9);

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
      .style('pointer-events', 'none');

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

    // Add hover effects and click handlers
    nodes
      .on('mouseover', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', (d.size || 20) * 1.2)
          .style('stroke-width', 3);
        
        // Highlight connected links
        links.style('stroke-opacity', link => 
          link.source.id === d.id || link.target.id === d.id ? 1 : 0.2
        );
      })
      .on('mouseout', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', d.size || 20)
          .style('stroke-width', 2);
        
        // Reset link opacity
        links.style('stroke-opacity', d => d.opacity || 0.6);
      })
      .on('click', function(event, d) {
        setSelectedNode(d);
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
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="title">News Knowledge Graph</h1>
          <p className="subtitle">AI-Powered Story Relationship Visualization</p>
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
            placeholder="Search news stories..."
            className="search-input"
          />
          <button onClick={searchNews} className="search-button">
            Search
          </button>
        </div>

        <div className="filters">
          <div className="filter-group">
            <label>Time Range:</label>
            <select value={days} onChange={(e) => setDays(parseInt(e.target.value))}>
              <option value={1}>Last 24 hours</option>
              <option value={3}>Last 3 days</option>
              <option value={7}>Last week</option>
              <option value={14}>Last 2 weeks</option>
            </select>
          </div>

          <div className="filter-group">
            <label>Section:</label>
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
            Refresh Graph
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
              <p>Analyzing stories with AI...</p>
            </div>
          )}
          <svg ref={svgRef} className="graph-svg"></svg>
          
          {/* Graph Info */}
          {graphData.metadata && (
            <div className="graph-info">
              <div className="info-item">
                <span className="info-label">Articles:</span>
                <span className="info-value">{graphData.metadata.total_articles || 0}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Connections:</span>
                <span className="info-value">{graphData.metadata.total_connections || 0}</span>
              </div>
              <div className="info-item">
                <span className="info-label">AI Analysis:</span>
                <span className="info-value">
                  {graphData.metadata.ai_analysis_enabled ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            </div>
          )}
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

              <div className="story-lede">
                <h4>Lede</h4>
                <p>{selectedNode.lede}</p>
              </div>

              <div className="story-nutgraf">
                <h4>Nutgraf</h4>
                <p>{selectedNode.nutgraf}</p>
              </div>

              <div className="story-summary">
                <h4>Summary</h4>
                <p>{selectedNode.summary}</p>
              </div>

              <div className="story-engagement">
                <h4>Social Preview</h4>
                <p className="engagement-preview">{selectedNode.engagement_preview}</p>
              </div>

              <div className="story-actions">
                <a 
                  href={selectedNode.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="read-full-button"
                >
                  Read Full Article
                </a>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="legend">
        <h4>Connection Types</h4>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#f39c12'}}></div>
            <span>Economic</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#3498db'}}></div>
            <span>Political</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#e74c3c'}}></div>
            <span>Social</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#27ae60'}}></div>
            <span>Environmental</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#9b59b6'}}></div>
            <span>Causal</span>
          </div>
          <div className="legend-item">
            <div className="legend-line" style={{backgroundColor: '#e67e22'}}></div>
            <span>Thematic</span>
          </div>
        </div>
        <p className="legend-note">
          Line thickness = connection strength • Click nodes for details • Drag to reposition
        </p>
      </div>
    </div>
  );
};

export default App;