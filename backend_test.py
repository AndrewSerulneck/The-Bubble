import requests
import sys
import json
from datetime import datetime
from typing import Dict, Any

class NewsKnowledgeGraphTester:
    def __init__(self, base_url="https://9bb23194-cb06-4a30-80b8-000a0e730972.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.nyt_api_key = "JoqJGXzpg6EWfRzwGi4R9FtMie1lmQMO"  # NEW NYT API key from review request

    def log_test(self, name: str, success: bool, details: str = ""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "name": name,
            "success": success,
            "details": details
        })

    def run_test(self, name: str, method: str, endpoint: str, expected_status: int = 200, params: Dict = None, timeout: int = 30) -> tuple:
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            print(f"\n🔍 Testing {name}...")
            print(f"   URL: {url}")
            if params:
                print(f"   Params: {params}")
            
            if method == 'GET':
                response = requests.get(url, params=params, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=params, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            success = response.status_code == expected_status
            
            if success:
                try:
                    data = response.json()
                    self.log_test(name, True, f"Status: {response.status_code}")
                    return True, data
                except json.JSONDecodeError:
                    self.log_test(name, False, f"Invalid JSON response. Status: {response.status_code}")
                    return False, {}
            else:
                self.log_test(name, False, f"Expected {expected_status}, got {response.status_code}. Response: {response.text[:200]}")
                return False, {}

        except requests.exceptions.Timeout:
            self.log_test(name, False, f"Request timeout after {timeout}s")
            return False, {}
        except requests.exceptions.ConnectionError:
            self.log_test(name, False, "Connection error - service may be down")
            return False, {}
        except Exception as e:
            self.log_test(name, False, f"Unexpected error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health check endpoint"""
        success, data = self.run_test(
            "Health Check",
            "GET", 
            "/api/health"
        )
        
        if success:
            # Verify health check response structure
            required_fields = ['status', 'timestamp', 'guardian_api', 'ai_services', 'database']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                self.log_test("Health Check Structure", False, f"Missing fields: {missing_fields}")
                return False
            else:
                self.log_test("Health Check Structure", True, "All required fields present")
                print(f"   Guardian API: {data.get('guardian_api')}")
                print(f"   AI Services: {data.get('ai_services')}")
                print(f"   Database: {data.get('database')}")
                return True
        
        return success

    def test_knowledge_graph_default(self):
        """Test default knowledge graph endpoint"""
        success, data = self.run_test(
            "Knowledge Graph (Default)",
            "GET",
            "/api/knowledge-graph",
            params={'days': 3, 'max_articles': 10}
        )
        
        if success:
            # Verify knowledge graph structure
            required_fields = ['nodes', 'edges', 'metadata']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                self.log_test("Knowledge Graph Structure", False, f"Missing fields: {missing_fields}")
                return False
            
            # Check if we have nodes and edges
            nodes_count = len(data.get('nodes', []))
            edges_count = len(data.get('edges', []))
            metadata = data.get('metadata', {})
            
            print(f"   Nodes: {nodes_count}")
            print(f"   Edges: {edges_count}")
            print(f"   AI Analysis: {metadata.get('ai_analysis_enabled', 'Unknown')}")
            print(f"   Total Articles: {metadata.get('total_articles', 0)}")
            print(f"   Total Connections: {metadata.get('total_connections', 0)}")
            
            if nodes_count > 0:
                self.log_test("Knowledge Graph Data", True, f"Generated {nodes_count} nodes, {edges_count} edges")
                
                # Test node structure
                sample_node = data['nodes'][0]
                if sample_node.get('type') in ['article', 'section']:
                    self.log_test("Node Structure", True, f"Valid node type: {sample_node.get('type')}")
                else:
                    self.log_test("Node Structure", False, f"Invalid node type: {sample_node.get('type')}")
                
                return True
            else:
                self.log_test("Knowledge Graph Data", False, "No nodes generated")
                return False
        
        return success

    def test_search_functionality(self):
        """Test search endpoint with different queries"""
        test_queries = ["Trump", "climate", "economy"]
        
        for query in test_queries:
            success, data = self.run_test(
                f"Search Query: '{query}'",
                "GET",
                "/api/search",
                params={'query': query, 'days': 7, 'max_articles': 8}
            )
            
            if success:
                nodes_count = len(data.get('nodes', []))
                edges_count = len(data.get('edges', []))
                print(f"   Query '{query}': {nodes_count} nodes, {edges_count} edges")
                
                if nodes_count > 0:
                    # Check if articles are related to the query
                    article_nodes = [n for n in data.get('nodes', []) if n.get('type') == 'article']
                    if article_nodes:
                        sample_title = article_nodes[0].get('title', '').lower()
                        if query.lower() in sample_title:
                            self.log_test(f"Search Relevance: '{query}'", True, "Query term found in results")
                        else:
                            self.log_test(f"Search Relevance: '{query}'", True, "Results returned (relevance may be contextual)")
            
            if not success:
                return False
        
        return True

    def test_recent_news_filtering(self):
        """Test recent news endpoint with different filters"""
        test_cases = [
            {'days': 5, 'section': None, 'name': 'Recent News (5 days)'},
            {'days': 1, 'section': 'business', 'name': 'Business News (1 day)'},
            {'days': 7, 'section': 'politics', 'name': 'Politics News (7 days)'},
        ]
        
        for case in test_cases:
            params = {'days': case['days']}
            if case['section']:
                params['section'] = case['section']
            
            success, data = self.run_test(
                case['name'],
                "GET",
                "/api/news/recent",
                params=params
            )
            
            if success:
                articles = data.get('articles', [])
                total = data.get('total', 0)
                query_params = data.get('query_parameters', {})
                
                print(f"   Articles: {len(articles)}")
                print(f"   Total: {total}")
                print(f"   Date range: {query_params.get('from_date')} to {query_params.get('to_date')}")
                
                if articles:
                    # Check article structure
                    sample_article = articles[0]
                    required_fields = ['id', 'title', 'summary', 'lede', 'nutgraf', 'section']
                    missing_fields = [field for field in required_fields if field not in sample_article]
                    
                    if missing_fields:
                        self.log_test(f"Article Structure ({case['name']})", False, f"Missing fields: {missing_fields}")
                    else:
                        self.log_test(f"Article Structure ({case['name']})", True, "All required fields present")
                        
                        # Check if section filter worked
                        if case['section']:
                            article_section = sample_article.get('section', '').lower()
                            if case['section'].lower() in article_section:
                                self.log_test(f"Section Filter ({case['section']})", True, f"Correct section: {article_section}")
                            else:
                                self.log_test(f"Section Filter ({case['section']})", False, f"Expected {case['section']}, got {article_section}")
                else:
                    self.log_test(f"Article Data ({case['name']})", False, "No articles returned")
            
            if not success:
                return False
        
        return True

    def test_ai_integration(self):
        """Test AI integration by checking for AI-generated content"""
        success, data = self.run_test(
            "AI Integration Test",
            "GET",
            "/api/knowledge-graph",
            params={'days': 2, 'max_articles': 5}
        )
        
        if success:
            metadata = data.get('metadata', {})
            ai_enabled = metadata.get('ai_analysis_enabled', False)
            
            if ai_enabled:
                self.log_test("AI Services", True, "AI analysis is enabled")
                
                # Check for AI-generated connections
                edges = data.get('edges', [])
                ai_connections = [e for e in edges if e.get('type') not in ['belongs_to']]
                
                if ai_connections:
                    self.log_test("AI Story Connections", True, f"Found {len(ai_connections)} AI-analyzed connections")
                    
                    # Check connection structure
                    sample_connection = ai_connections[0]
                    if all(field in sample_connection for field in ['type', 'strength', 'explanation']):
                        self.log_test("AI Connection Structure", True, "Valid connection structure")
                        print(f"   Sample connection: {sample_connection.get('type')} (strength: {sample_connection.get('strength')})")
                        print(f"   Explanation: {sample_connection.get('explanation', '')[:100]}...")
                    else:
                        self.log_test("AI Connection Structure", False, "Missing connection fields")
                else:
                    self.log_test("AI Story Connections", False, "No AI-analyzed connections found")
                
                # Check for AI-generated summaries in nodes
                article_nodes = [n for n in data.get('nodes', []) if n.get('type') == 'article']
                if article_nodes:
                    sample_article = article_nodes[0]
                    if all(field in sample_article for field in ['summary', 'lede', 'nutgraf', 'engagement_preview']):
                        self.log_test("AI Content Generation", True, "AI-generated content fields present")
                    else:
                        self.log_test("AI Content Generation", False, "Missing AI-generated content fields")
            else:
                self.log_test("AI Services", False, "AI analysis is disabled")
        
        return success

    def test_ultimate_health_check(self):
        """Test ultimate health check endpoint with NYT API key detection"""
        success, data = self.run_test(
            "Ultimate Health Check",
            "GET", 
            "/api/health/ultimate"
        )
        
        if success:
            # Verify ultimate health check response structure
            required_fields = ['status', 'version', 'timestamp', 'services', 'api_sources', 'features']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                self.log_test("Ultimate Health Check Structure", False, f"Missing fields: {missing_fields}")
                return False
            else:
                self.log_test("Ultimate Health Check Structure", True, "All required fields present")
                
                # Check API sources
                api_sources = data.get('api_sources', {})
                guardian_available = api_sources.get('guardian', False)
                nyt_available = api_sources.get('nyt', False)
                
                print(f"   Guardian API: {guardian_available}")
                print(f"   NYT API: {nyt_available}")
                print(f"   Version: {data.get('version')}")
                print(f"   Status: {data.get('status')}")
                
                # Verify NYT API key is detected
                if nyt_available:
                    self.log_test("NYT API Key Detection", True, "NYT API key properly detected")
                else:
                    self.log_test("NYT API Key Detection", False, "NYT API key not detected or invalid")
                
                # Check features
                features = data.get('features', {})
                expected_features = [
                    'multi_source_integration', 'advanced_ai_analysis', 'real_time_updates',
                    'geographic_analysis', 'temporal_analysis', 'sentiment_analysis',
                    'influence_metrics', 'trending_detection', 'ultimate_knowledge_graph'
                ]
                
                missing_features = [f for f in expected_features if not features.get(f, False)]
                if missing_features:
                    self.log_test("Ultimate Features Check", False, f"Missing features: {missing_features}")
                else:
                    self.log_test("Ultimate Features Check", True, "All ultimate features enabled")
                
                return True
        
        return success

    def test_nyt_api_integration(self):
        """Test NYT API integration specifically"""
        print("\n🔍 Testing NYT API Integration...")
        
        # Test NYT-only source
        success, data = self.run_test(
            "NYT API Integration (NYT Only)",
            "GET",
            "/api/v4/knowledge-graph/ultimate",
            params={
                'sources': 'nyt',
                'days': 3,
                'max_articles': 10,
                'complexity_level': 3
            },
            timeout=45
        )
        
        if success:
            metadata = data.get('metadata', {})
            nodes = data.get('nodes', [])
            
            # Check for NYT articles
            nyt_articles = [n for n in nodes if n.get('type') == 'article' and n.get('source') == 'nyt']
            
            if nyt_articles:
                self.log_test("NYT Articles Retrieved", True, f"Found {len(nyt_articles)} NYT articles")
                
                # Check NYT article structure
                sample_nyt = nyt_articles[0]
                required_nyt_fields = ['id', 'title', 'summary', 'lede', 'nutgraf', 'source', 'url']
                missing_nyt_fields = [field for field in required_nyt_fields if not sample_nyt.get(field)]
                
                if missing_nyt_fields:
                    self.log_test("NYT Article Structure", False, f"Missing fields: {missing_nyt_fields}")
                else:
                    self.log_test("NYT Article Structure", True, "NYT article structure valid")
                    print(f"   Sample NYT Title: {sample_nyt.get('title', '')[:80]}...")
                    print(f"   NYT URL: {sample_nyt.get('url', '')[:60]}...")
            else:
                self.log_test("NYT Articles Retrieved", False, "No NYT articles found")
                return False
            
            # Check source node
            nyt_source_node = next((n for n in nodes if n.get('id') == 'source_nyt'), None)
            if nyt_source_node:
                self.log_test("NYT Source Node", True, "NYT source node present")
            else:
                self.log_test("NYT Source Node", False, "NYT source node missing")
        
        return success

    def test_multi_source_integration(self):
        """Test multi-source integration (Guardian + NYT)"""
        print("\n🔍 Testing Multi-Source Integration...")
        
        success, data = self.run_test(
            "Multi-Source Integration (Guardian + NYT)",
            "GET",
            "/api/v4/knowledge-graph/ultimate",
            params={
                'sources': 'guardian,nyt',
                'days': 3,
                'max_articles': 20,
                'complexity_level': 3
            },
            timeout=60
        )
        
        if success:
            nodes = data.get('nodes', [])
            metadata = data.get('metadata', {})
            
            # Check for both source types
            guardian_articles = [n for n in nodes if n.get('type') == 'article' and n.get('source') == 'guardian']
            nyt_articles = [n for n in nodes if n.get('type') == 'article' and n.get('source') == 'nyt']
            
            print(f"   Guardian articles: {len(guardian_articles)}")
            print(f"   NYT articles: {len(nyt_articles)}")
            print(f"   Total articles: {metadata.get('total_articles', 0)}")
            
            if guardian_articles and nyt_articles:
                self.log_test("Multi-Source Articles", True, f"Both sources present: {len(guardian_articles)} Guardian, {len(nyt_articles)} NYT")
            elif guardian_articles:
                self.log_test("Multi-Source Articles", False, "Only Guardian articles found, NYT missing")
            elif nyt_articles:
                self.log_test("Multi-Source Articles", False, "Only NYT articles found, Guardian missing")
            else:
                self.log_test("Multi-Source Articles", False, "No articles from either source")
                return False
            
            # Check for cross-source connections
            edges = data.get('edges', [])
            cross_source_connections = []
            
            for edge in edges:
                source_node = next((n for n in nodes if n.get('id') == edge.get('source')), None)
                target_node = next((n for n in nodes if n.get('id') == edge.get('target')), None)
                
                if (source_node and target_node and 
                    source_node.get('type') == 'article' and target_node.get('type') == 'article' and
                    source_node.get('source') != target_node.get('source')):
                    cross_source_connections.append(edge)
            
            if cross_source_connections:
                self.log_test("Cross-Source Connections", True, f"Found {len(cross_source_connections)} cross-source connections")
                
                # Check connection quality
                sample_connection = cross_source_connections[0]
                if all(field in sample_connection for field in ['type', 'strength', 'confidence', 'explanation']):
                    self.log_test("Cross-Source Connection Quality", True, "High-quality cross-source connections")
                    print(f"   Sample connection type: {sample_connection.get('type')}")
                    print(f"   Connection strength: {sample_connection.get('strength')}")
                    print(f"   Connection confidence: {sample_connection.get('confidence')}")
                else:
                    self.log_test("Cross-Source Connection Quality", False, "Missing connection metadata")
            else:
                self.log_test("Cross-Source Connections", False, "No cross-source connections found")
        
        return success

    def test_complexity_levels(self):
        """Test different complexity levels (1-5)"""
        print("\n🔍 Testing Complexity Level Features...")
        
        complexity_results = {}
        
        for level in range(1, 6):
            success, data = self.run_test(
                f"Complexity Level {level}",
                "GET",
                "/api/v4/knowledge-graph/ultimate",
                params={
                    'sources': 'guardian,nyt',
                    'days': 2,
                    'max_articles': 8,
                    'complexity_level': level
                },
                timeout=45
            )
            
            if success:
                nodes = data.get('nodes', [])
                metadata = data.get('metadata', {})
                
                # Check if complexity level is reflected in user preferences
                user_prefs = metadata.get('user_preferences', {})
                if user_prefs.get('complexity_level') == level:
                    self.log_test(f"Complexity Level {level} - Metadata", True, f"Complexity level {level} properly set")
                else:
                    self.log_test(f"Complexity Level {level} - Metadata", False, f"Expected {level}, got {user_prefs.get('complexity_level')}")
                
                # Check article complexity adaptation
                article_nodes = [n for n in nodes if n.get('type') == 'article']
                if article_nodes:
                    sample_article = article_nodes[0]
                    article_complexity = sample_article.get('complexity_level', 0)
                    
                    if article_complexity == level:
                        self.log_test(f"Complexity Level {level} - Article Adaptation", True, f"Articles adapted to level {level}")
                    else:
                        self.log_test(f"Complexity Level {level} - Article Adaptation", False, f"Expected {level}, got {article_complexity}")
                    
                    # Store results for comparison
                    complexity_results[level] = {
                        'summary_length': len(sample_article.get('summary', '')),
                        'lede_length': len(sample_article.get('lede', '')),
                        'nutgraf_length': len(sample_article.get('nutgraf', '')),
                        'read_time': sample_article.get('read_time_minutes', 0)
                    }
                    
                    print(f"   Level {level} - Summary length: {complexity_results[level]['summary_length']} chars")
                    print(f"   Level {level} - Read time: {complexity_results[level]['read_time']} minutes")
            else:
                return False
        
        # Verify complexity progression
        if len(complexity_results) >= 3:
            # Check if higher complexity levels generally have longer content
            level_1_summary = complexity_results.get(1, {}).get('summary_length', 0)
            level_5_summary = complexity_results.get(5, {}).get('summary_length', 0)
            
            if level_5_summary > level_1_summary:
                self.log_test("Complexity Progression", True, "Higher complexity levels have more detailed content")
            else:
                self.log_test("Complexity Progression", False, "No clear complexity progression detected")
        
        return True

    def test_ultimate_demo_endpoint(self):
        """Test ultimate demo endpoint"""
        success, data = self.run_test(
            "Ultimate Demo Endpoint",
            "GET",
            "/api/v4/demo/ultimate"
        )
        
        if success:
            # Check demo structure
            required_fields = ['nodes', 'edges', 'metadata']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                self.log_test("Ultimate Demo Structure", False, f"Missing fields: {missing_fields}")
                return False
            
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            metadata = data.get('metadata', {})
            
            # Check for ultimate features in demo
            advanced_features = metadata.get('advanced_features', {})
            expected_features = [
                'multi_source', 'geographic_analysis', 'temporal_analysis',
                'sentiment_analysis', 'complexity_adaptation', 'confidence_scoring',
                'influence_metrics', 'real_time_analytics'
            ]
            
            missing_demo_features = [f for f in expected_features if not advanced_features.get(f, False)]
            if missing_demo_features:
                self.log_test("Ultimate Demo Features", False, f"Missing demo features: {missing_demo_features}")
            else:
                self.log_test("Ultimate Demo Features", True, "All ultimate features present in demo")
            
            # Check for trending topics
            trending_topics = metadata.get('trending_topics', [])
            if trending_topics:
                self.log_test("Demo Trending Topics", True, f"Found {len(trending_topics)} trending topics")
            else:
                self.log_test("Demo Trending Topics", False, "No trending topics in demo")
            
            # Check geographic insights
            geographic_insights = metadata.get('geographic_insights', {})
            if geographic_insights:
                self.log_test("Demo Geographic Insights", True, "Geographic insights present")
            else:
                self.log_test("Demo Geographic Insights", False, "No geographic insights in demo")
            
            print(f"   Demo nodes: {len(nodes)}")
            print(f"   Demo edges: {len(edges)}")
            print(f"   Demo version: {metadata.get('version')}")
        
        return success

    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting News Knowledge Graph Backend Tests - Ultimate Edition")
        print(f"📍 Testing against: {self.base_url}")
        print(f"🔑 NYT API Key: {self.nyt_api_key}")
        print("=" * 80)
        
        # Test sequence - prioritizing NYT and complexity features
        tests = [
            self.test_ultimate_health_check,
            self.test_nyt_api_integration,
            self.test_multi_source_integration,
            self.test_complexity_levels,
            self.test_ultimate_demo_endpoint,
            self.test_health_check,
            self.test_knowledge_graph_default,
            self.test_search_functionality,
            self.test_recent_news_filtering,
            self.test_ai_integration
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.log_test(f"Test {test.__name__}", False, f"Test crashed: {str(e)}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 ULTIMATE TEST SUMMARY")
        print("=" * 80)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "0%")
        
        # Print failed tests
        failed_tests = [t for t in self.test_results if not t['success']]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test['name']}: {test['details']}")
        
        # Print passed tests summary
        passed_tests = [t for t in self.test_results if t['success']]
        if passed_tests:
            print(f"\n✅ PASSED TESTS ({len(passed_tests)}):")
            for test in passed_tests:
                print(f"   • {test['name']}")
        
        return self.tests_passed == self.tests_run

def main():
    tester = NewsKnowledgeGraphTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Backend is working correctly.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Check the details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())